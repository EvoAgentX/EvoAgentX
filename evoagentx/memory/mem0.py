"""
Mem0 memory bridge for CustomizeMemoryAgent.

This module adapts the legacy MemoryManager/LongTermMemory stack so it can
participate in the new memory lifecycle (retrieve -> inject -> execute ->
reflect -> persist) exposed by CustomizeMemoryAgent.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from uuid import uuid4

from ..agents.customize_memory_agent import (
    BaseMemoryBackend,
    BaseMemoryPolicy,
    CustomizeMemoryAgent,
    MemoryOperation,
    MemoryOrchestrator,
    MemoryRunContext,
)
from ..core.logging import logger
from ..core.message import Message, MessageType
from ..models.model_configs import LLMConfig
from ..prompts.template import StringTemplate
from ..storages.base import StorageHandler
from .memory_manager import MemoryManager


class Mem0MemoryBackend(BaseMemoryBackend):
    """
    Bridge between the legacy mem0 memory stack and CustomizeMemoryAgent.

    The backend reuses MemoryManager for retrieval, injection-ready summaries,
    and persistence operations while aligning with the lifecycle hooks of
    CustomizeMemoryAgent.
    """

    name: str = "mem0"
    priority: int = 50
    supported_operations: Tuple[MemoryOperation, ...] = (
        MemoryOperation.RETRIEVE,
        MemoryOperation.INJECT,
        MemoryOperation.REFLECT,
        MemoryOperation.PERSIST,
    )

    memory_manager: MemoryManager
    max_short_term_messages: int = 6
    persist_user_queries: bool = True
    persist_agent_responses: bool = True
    default_top_k: int = 5
    operations_field: str = "memory_operations"

    def init_module(self) -> None:
        super().init_module()
        if not getattr(self, "memory_manager", None):
            raise ValueError("Mem0MemoryBackend requires an initialised MemoryManager.")
        if not getattr(self.memory_manager, "memory", None):
            raise ValueError("MemoryManager must expose a LongTermMemory via `memory`.")
        logger.debug(
            "Mem0MemoryBackend ready (default corpus: %s)",
            getattr(self.memory_manager.memory, "default_corpus_id", None),
        )

    async def on_retrieve(
        self,
        agent: CustomizeMemoryAgent,
        run_context: MemoryRunContext,
        query: str,
        top_k: int,
        **_: Any,
    ) -> Optional[Dict[str, Any]]:
        conversation_id = self._resolve_conversation_id(run_context)
        run_context.metadata.setdefault("conversation_id", conversation_id)
        if run_context.inputs is not None:
            run_context.inputs.setdefault("conversation_id", conversation_id)

        corpus_id = self._resolve_corpus_id(run_context)
        if corpus_id:
            run_context.metadata.setdefault("corpus_id", corpus_id)

        filters = self._collect_metadata_filters(run_context, corpus_id)
        limit = top_k or run_context.top_k or self.default_top_k

        results = await self.memory_manager.memory.search_async(
            query=query,
            n=limit,
            metadata_filters=filters or None,
        )

        pairs = list(results)
        long_texts = [self._chunk_to_text(chunk) for chunk, _ in pairs]
        memory_ids = [memory_id for _, memory_id in pairs]
        short_term_context = self._build_short_term_context(agent)
        user_prompt = self._extract_user_prompt(run_context, fallback=query)

        return {
            "conversation_id": conversation_id,
            "corpus_id": corpus_id,
            "metadata_filters": filters,
            "memory_ids": memory_ids,
            "long_term_results": pairs,
            "long_term_texts": long_texts,
            "short_term_context": short_term_context,
            "user_prompt": user_prompt,
            "top_k": limit,
        }

    async def on_inject(
        self,
        agent: CustomizeMemoryAgent,
        run_context: MemoryRunContext,
        retrieved: Dict[str, Any],
        **_: Any,
    ) -> str:
        del agent, run_context

        long_texts = retrieved.get("long_term_texts") or []
        short_term_context = retrieved.get("short_term_context") or ""
        user_prompt = retrieved.get("user_prompt") or ""

        summary = ""
        try:
            summary = await self.memory_manager.summarize_memories(
                long_term_texts=long_texts,
                short_term_context=short_term_context,
                user_prompt=user_prompt,
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning("Mem0MemoryBackend summarisation failed: %s", exc)

        retrieved["summary"] = summary

        blocks: List[str] = ["### Long-Term Memories"]
        if long_texts:
            for idx, text in enumerate(long_texts, start=1):
                blocks.append(f"{idx}. {text}")
        else:
            blocks.append("(No relevant long-term memories retrieved)")

        blocks.append("")
        blocks.append("### Short-Term Context")
        blocks.append(short_term_context or "(No recent short-term context)")

        if summary:
            blocks.append("")
            blocks.append("### Memory Summary")
            blocks.append(summary)

        if user_prompt:
            blocks.append("")
            blocks.append("### User Prompt")
            blocks.append(user_prompt)

        return "\n".join(blocks).strip()

    async def on_reflect(
        self,
        agent: CustomizeMemoryAgent,
        run_context: MemoryRunContext,
        execution_result: Message,
        **_: Any,
    ) -> Optional[Dict[str, Any]]:
        retrieved = run_context.retrieved.get(self.name) or {}
        conversation_id = retrieved.get("conversation_id") or self._resolve_conversation_id(run_context)
        run_context.metadata.setdefault("conversation_id", conversation_id)

        corpus_id = retrieved.get("corpus_id") or run_context.metadata.get("corpus_id")
        if corpus_id:
            run_context.metadata.setdefault("corpus_id", corpus_id)

        plan: Dict[str, Any] = {
            "conversation_id": conversation_id,
            "corpus_id": corpus_id,
            "metadata_filters": retrieved.get("metadata_filters") or {},
            "memory_ids": retrieved.get("memory_ids") or [],
            "summary": retrieved.get("summary"),
            "add": [],
            "update": [],
            "delete": [],
        }

        user_prompt = self._extract_user_prompt(run_context, fallback=retrieved.get("user_prompt"))
        if self.persist_user_queries and user_prompt:
            plan["add"].append(
                Message(
                    content=user_prompt,
                    agent="user",
                    msg_type=MessageType.REQUEST,
                    conversation_id=conversation_id,
                )
            )

        if self.persist_agent_responses and execution_result:
            response_text = self._to_text(execution_result.content)
            if response_text:
                response_msg = Message(
                    content=response_text,
                    agent=execution_result.agent or agent.name,
                    action=execution_result.action,
                    prompt=execution_result.prompt,
                    next_actions=execution_result.next_actions,
                    msg_type=MessageType.RESPONSE,
                    wf_goal=execution_result.wf_goal,
                    wf_task=execution_result.wf_task,
                    wf_task_desc=execution_result.wf_task_desc,
                    conversation_id=conversation_id,
                )
                if plan["memory_ids"]:
                    response_msg.memory_ids = plan["memory_ids"]
                plan["add"].append(response_msg)

        custom_plan = self._extract_custom_operations(run_context, conversation_id)
        plan["add"].extend(custom_plan["add"])
        plan["update"].extend(custom_plan["update"])
        plan["delete"].extend(custom_plan["delete"])

        if not any(plan[key] for key in ("add", "update", "delete")):
            return None
        return plan

    async def on_persist(
        self,
        agent: CustomizeMemoryAgent,
        run_context: MemoryRunContext,
        reflection: Dict[str, Any],
        **_: Any,
    ) -> Dict[str, Any]:
        del agent, run_context
        if not reflection:
            return {}

        memory = getattr(self.memory_manager, "memory", None)
        corpus_id = reflection.get("corpus_id")
        if memory and corpus_id and getattr(memory, "default_corpus_id", None) != corpus_id:
            memory.default_corpus_id = corpus_id

        results: Dict[str, Any] = {}

        add_payloads = reflection.get("add") or []
        if add_payloads:
            results["add"] = await self.memory_manager.handle_memory(action="add", data=add_payloads)

        update_payloads = reflection.get("update") or []
        if update_payloads:
            results["update"] = await self.memory_manager.handle_memory(action="update", data=update_payloads)

        delete_payloads = reflection.get("delete") or []
        if delete_payloads:
            results["delete"] = await self.memory_manager.handle_memory(action="delete", data=delete_payloads)

        return results

    # --------------------------------------------------------------------- helpers

    def _resolve_conversation_id(self, run_context: MemoryRunContext) -> str:
        inputs = run_context.inputs or {}
        for candidate in (
            inputs.get("conversation_id"),
            run_context.metadata.get("conversation_id"),
        ):
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
        default_corpus = getattr(self.memory_manager.memory, "default_corpus_id", None)
        if default_corpus:
            return str(default_corpus)
        return str(uuid4())

    def _resolve_corpus_id(self, run_context: MemoryRunContext) -> Optional[str]:
        inputs = run_context.inputs or {}
        corpus_id = inputs.get("corpus_id")
        if isinstance(corpus_id, str) and corpus_id.strip():
            return corpus_id.strip()
        metadata_corpus = run_context.metadata.get("corpus_id")
        if isinstance(metadata_corpus, str) and metadata_corpus.strip():
            return metadata_corpus.strip()
        return getattr(self.memory_manager.memory, "default_corpus_id", None)

    def _collect_metadata_filters(
        self,
        run_context: MemoryRunContext,
        corpus_id: Optional[str],
    ) -> Dict[str, Any]:
        filters: Dict[str, Any] = {}
        for source in (run_context.inputs, run_context.metadata):
            if not isinstance(source, dict):
                continue
            for key in ("metadata_filters", "memory_filters", "memory_metadata_filters"):
                value = source.get(key)
                if isinstance(value, dict):
                    filters.update(value)
        if corpus_id and "corpus_id" not in filters:
            filters["corpus_id"] = corpus_id
        return filters

    def _build_short_term_context(self, agent: CustomizeMemoryAgent) -> str:
        window = max(0, self.max_short_term_messages)
        short_term = getattr(agent, "short_term_memory", None)
        if short_term is None or window == 0:
            return ""

        snippets: List[str] = []
        for message in short_term.get(n=window):
            msg_type = message.msg_type or MessageType.UNKNOWN
            if msg_type in (MessageType.REQUEST, MessageType.INPUT):
                role = "User"
            elif message.agent:
                role = message.agent
            else:
                role = "Agent"
            snippets.append(f"{role}: {self._to_text(message.content)}")
        return "\n".join(snippets).strip()

    def _extract_user_prompt(
        self,
        run_context: MemoryRunContext,
        *,
        fallback: Optional[str] = None,
    ) -> Optional[str]:
        inputs = run_context.inputs or {}
        candidates = (
            "user_prompt",
            "user_input",
            "prompt",
            "question",
            "task",
            "query",
            "input",
            "instruction",
        )
        for key in candidates:
            candidate = inputs.get(key)
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
        if isinstance(run_context.query, str) and run_context.query.strip():
            return run_context.query.strip()
        if fallback and isinstance(fallback, str) and fallback.strip():
            return fallback.strip()
        return None

    def _extract_custom_operations(
        self,
        run_context: MemoryRunContext,
        conversation_id: str,
    ) -> Dict[str, List[Any]]:
        raw = None
        if run_context.metadata and self.operations_field in run_context.metadata:
            raw = run_context.metadata[self.operations_field]
        elif run_context.inputs and self.operations_field in run_context.inputs:
            raw = run_context.inputs[self.operations_field]
        return self._normalise_operations(raw, conversation_id)

    def _normalise_operations(
        self,
        raw: Any,
        conversation_id: str,
    ) -> Dict[str, List[Any]]:
        plan = {"add": [], "update": [], "delete": []}
        if raw is None:
            return plan

        if isinstance(raw, dict) and "action" not in raw:
            add_values: Iterable[Any] = self._ensure_iterable(raw.get("add")) + self._ensure_iterable(raw.get("create"))
            for value in add_values:
                plan["add"].append(self._coerce_message(value, conversation_id, MessageType.REQUEST))

            for value in self._ensure_iterable(raw.get("update")):
                normalised = self._coerce_update(value, conversation_id)
                if normalised:
                    plan["update"].append(normalised)

            delete_values: Iterable[Any] = self._ensure_iterable(raw.get("delete")) + self._ensure_iterable(raw.get("remove"))
            for value in delete_values:
                target = self._coerce_delete(value)
                if target:
                    plan["delete"].append(target)
            return plan

        for entry in self._ensure_iterable(raw):
            if isinstance(entry, dict):
                action = entry.get("action")
                if action in ("add", "create"):
                    payload = entry.get("message") or entry.get("content") or entry.get("payload")
                    if payload is None:
                        continue
                    msg_type = entry.get("msg_type", MessageType.REQUEST)
                    if isinstance(msg_type, str):
                        try:
                            msg_type = MessageType(msg_type)
                        except ValueError:
                            msg_type = MessageType.REQUEST
                    elif not isinstance(msg_type, MessageType):
                        msg_type = MessageType.REQUEST
                    plan["add"].append(
                        self._coerce_message(
                            payload,
                            conversation_id,
                            msg_type,
                            agent_name=entry.get("agent"),
                        )
                    )
                elif action == "update":
                    normalised = self._coerce_update(entry, conversation_id)
                    if normalised:
                        plan["update"].append(normalised)
                elif action == "delete":
                    target = self._coerce_delete(entry)
                    if target:
                        plan["delete"].append(target)
            else:
                plan["add"].append(self._coerce_message(entry, conversation_id, MessageType.REQUEST))
        return plan

    def _chunk_to_text(self, chunk: Any) -> str:
        if chunk is None:
            return ""
        if hasattr(chunk, "text"):
            return str(chunk.text)
        if hasattr(chunk, "content"):
            return self._to_text(chunk.content)
        return self._to_text(chunk)

    def _coerce_message(
        self,
        payload: Any,
        conversation_id: str,
        default_type: MessageType,
        *,
        agent_name: Optional[str] = None,
    ) -> Message:
        if isinstance(payload, Message):
            if conversation_id and not payload.conversation_id:
                payload.conversation_id = conversation_id
            if payload.msg_type in (None, MessageType.UNKNOWN):
                payload.msg_type = default_type
            if agent_name and not payload.agent:
                payload.agent = agent_name
            return payload

        if isinstance(payload, dict) and "content" in payload and "class_name" not in payload:
            data = dict(payload)
            msg_type = data.get("msg_type")
            if isinstance(msg_type, str):
                try:
                    data["msg_type"] = MessageType(msg_type)
                except ValueError:
                    data["msg_type"] = default_type
            elif not isinstance(msg_type, MessageType):
                data["msg_type"] = default_type
            data.setdefault(
                "agent",
                agent_name or ("user" if data["msg_type"] in (MessageType.REQUEST, MessageType.INPUT) else None) or "assistant",
            )
            data.setdefault("conversation_id", conversation_id)
            data["content"] = self._to_text(data.get("content"))
            return Message(**data)

        derived_agent = agent_name or ("user" if default_type in (MessageType.REQUEST, MessageType.INPUT) else None) or "assistant"
        return Message(
            content=self._to_text(payload),
            agent=derived_agent,
            msg_type=default_type,
            conversation_id=conversation_id,
        )

    def _coerce_update(self, payload: Any, conversation_id: str) -> Optional[Tuple[str, Message]]:
        if isinstance(payload, tuple) and len(payload) == 2:
            memory_id, message_payload = payload
        elif isinstance(payload, dict):
            memory_id = payload.get("memory_id") or payload.get("id") or payload.get("target")
            message_payload = payload.get("message") or payload.get("content") or payload.get("payload")
            if memory_id is None:
                return None
        else:
            return None

        msg_type = None
        if isinstance(payload, dict):
            raw = payload.get("msg_type")
            if isinstance(raw, str):
                try:
                    msg_type = MessageType(raw)
                except ValueError:
                    msg_type = None
            elif isinstance(raw, MessageType):
                msg_type = raw

        message = self._coerce_message(
            message_payload,
            conversation_id,
            msg_type or MessageType.REQUEST,
            agent_name=(payload.get("agent") if isinstance(payload, dict) else None),
        )
        return str(memory_id), message

    def _coerce_delete(self, payload: Any) -> Optional[str]:
        if isinstance(payload, str):
            return payload
        if isinstance(payload, (int, float)):
            return str(payload)
        if isinstance(payload, dict):
            target = payload.get("memory_id") or payload.get("id") or payload.get("target")
            if target is not None:
                return str(target)
        return None

    def _to_text(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, (int, float, bool)):
            return str(value)
        if isinstance(value, (list, tuple)):
            try:
                return json.dumps(value, ensure_ascii=False)
            except TypeError:
                return ", ".join(self._to_text(item) for item in value)
        if isinstance(value, dict):
            return json.dumps(value, ensure_ascii=False)
        if hasattr(value, "model_dump"):
            try:
                return json.dumps(value.model_dump(), ensure_ascii=False)
            except TypeError:
                return str(value)
        if hasattr(value, "dict"):
            try:
                return json.dumps(value.dict(), ensure_ascii=False)
            except TypeError:
                return str(value)
        return str(value)

    @staticmethod
    def _ensure_iterable(value: Any) -> List[Any]:
        if value is None:
            return []
        if isinstance(value, (list, tuple, set)):
            return list(value)
        return [value]


class Mem0MemoryPolicy(BaseMemoryPolicy):
    """
    Policy that keeps mem0 prompts backward compatible by populating both
    `memory_context` and `context` with the injected content.
    """

    def aggregate_injections(
        self,
        agent: CustomizeMemoryAgent,
        run_context: MemoryRunContext,
        injections: Sequence[Any],
    ) -> Dict[str, Any]:
        inputs = super().aggregate_injections(agent=agent, run_context=run_context, injections=injections)
        memory_context = inputs.get("memory_context")
        if memory_context and "context" not in inputs:
            inputs["context"] = memory_context
        return inputs


def create_mem0_agent(
    memory_manager: MemoryManager,
    *,
    name: str = "mem0_agent",
    description: str = "CustomizeMemoryAgent backed by mem0 memory integration.",
    prompt: Optional[str] = None,
    llm_config: Optional[LLMConfig] = None,
    backend_name: str = "mem0",
    backend_priority: int = 50,
    short_term_window: int = 6,
    default_top_k: int = 5,
    persist_user_queries: bool = True,
    persist_agent_responses: bool = True,
    memory_policy: Optional[BaseMemoryPolicy] = None,
    memory_orchestrator: Optional[MemoryOrchestrator] = None,
    inputs: Optional[List[dict]] = None,
    outputs: Optional[List[dict]] = None,
    storage_handler: Optional[StorageHandler] = None,
    **agent_kwargs: Any,
) -> CustomizeMemoryAgent:
    backend = Mem0MemoryBackend(
        name=backend_name,
        priority=backend_priority,
        memory_manager=memory_manager,
        max_short_term_messages=short_term_window,
        persist_user_queries=persist_user_queries,
        persist_agent_responses=persist_agent_responses,
        default_top_k=default_top_k,
    )
    policy = memory_policy or Mem0MemoryPolicy(default_top_k=default_top_k)
    orchestrator = memory_orchestrator or MemoryOrchestrator()

    base_prompt = prompt or (
        "You are a memory-augmented assistant. Use `memory_context` to ground your response "
        "and keep the conversation consistent.\n\n"
        "User request:\n{user_input}\n\n"
        "Memory context:\n{memory_context}"
    )
    prompt_template = StringTemplate(instruction=base_prompt)

    if inputs is None:
        inputs = [
            {"name": "user_input", "type": "str", "description": "Primary user instruction."},
            {
                "name": "memory_context",
                "type": "str",
                "description": "Injected context from mem0 backend.",
                "required": False,
            },
        ]

    if outputs is None:
        outputs = [{"name": "response", "type": "str", "description": "Agent response."}]

    storage = storage_handler or getattr(memory_manager.memory, "storage_handler", None)

    agent_kwargs.setdefault("parse_mode", "str")
    agent_kwargs.setdefault("is_human", False)

    return CustomizeMemoryAgent(
        name=name,
        description=description,
        prompt=None,
        prompt_template=prompt_template,
        llm_config=llm_config,
        inputs=inputs,
        outputs=outputs,
        memory_backends={backend.name: backend},
        memory_policies={"default": policy},
        memory_orchestrator=orchestrator,
        storage_handler=storage,
        **agent_kwargs,
    )


__all__ = ["Mem0MemoryBackend", "Mem0MemoryPolicy", "create_mem0_agent"]
