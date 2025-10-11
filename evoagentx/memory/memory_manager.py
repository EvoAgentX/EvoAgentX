import json
from uuid import uuid4
from datetime import datetime
from typing import Union, List, Dict, Any, Optional, Tuple

from pydantic import Field

from .long_term_memory import LongTermMemory
from ..rag.schema import Query
from ..core.logging import logger
from ..core.module import BaseModule
from ..core.message import Message, MessageType
from ..models.base_model import BaseLLM
from ..prompts.memory.manager import MANAGER_PROMPT


class MemoryManager(BaseModule):
    """
    The Memory Manager organizes and manages LongTermMemory data at a higher level.
    It retrieves data, processes it with optional LLM-based action inference, and stores new or updated data.
    It creates Message objects for agent use, combining user prompts with memory context.

    Attributes:
        memory (LongTermMemory): The LongTermMemory instance for storing and retrieving messages.
        llm (Optional[BaseLLM]): LLM for deciding memory operations.
        use_llm_management (bool): Toggle LLM-based memory management.
    """
    memory: LongTermMemory = Field(..., description="Long-term memory instance")
    llm: Optional[BaseLLM] = Field(default=None, description="LLM for deciding memory operations")
    use_llm_management: bool = Field(default=True, description="Toggle LLM-based memory management")

    def init_module(self):
        """
        Initialize MemoryManager by ensuring LongTermMemory and LLM are ready.
        Called automatically by BaseModule when the module is created.
        """
        logger.info("[MemoryManager] 🔧 Initializing module...")
        if not self.memory:
            raise ValueError("[MemoryManager] ❌ LongTermMemory instance is missing.")
        else:
            logger.info("[MemoryManager] ✅ LongTermMemory loaded successfully.")

        if not self.llm:
            logger.warning("[MemoryManager] ⚠️ No LLM configured. Summarization will be skipped.")
        else:
            logger.info(f"[MemoryManager] ✅ LLM loaded: {self.llm.__class__.__name__}")

        logger.info("[MemoryManager] 🎯 init_module completed.")
    
    async def _prompt_llm_for_memory_operation(
        self,
        input_data: Dict[str, Any],
        relevant_data: List[Tuple[Message, str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Prompt the LLM to decide memory operation (add, update, delete)
        and return a JSON **list** of operations (even if only one).
        Key points:
        - Do not read/verify the memory_id in the LLM results at all
        - Only use the action + message of LLM
        - Automatically retrieve the true memory_id for update/delete
        """
        if not self.llm or not self.use_llm_management:
            logger.warning("[MemoryManager] ⚠️ No LLM or management disabled, bypassing to input_data.")
            return [input_data] 

        # Convert relevant memories to JSON strings to assist LLM in judgment
        relevant_data_str = '\n'.join([
            json.dumps({"message": msg.to_dict(), "memory_id": mid})
            for msg, mid in (relevant_data or [])
        ])
        prompt = MANAGER_PROMPT \
            .replace("<<INPUT_DATA>>", json.dumps(input_data, ensure_ascii=False)) \
            .replace("<<RELEVANT_DATA>>", relevant_data_str)

        try:
            # Call LLM for decision making
            response = self.llm.generate(prompt=prompt)
            raw_text = response.content.replace("```json", "").replace("```", "").strip()
            logger.debug(f"[MemoryManager] 🔎 Raw LLM output: {raw_text}")

            parsed = json.loads(raw_text)
            logger.debug(f"[MemoryManager] ✅ Parsed type: {type(parsed)}")

            # Convert to list uniformly
            if isinstance(parsed, dict):
                parsed_list = [parsed]
            elif isinstance(parsed, list):
                parsed_list = parsed
            else:
                raise ValueError(f"Unexpected LLM output type: {type(parsed)}")

            validated: List[Dict[str, Any]] = []
            for item in parsed_list:
                if not isinstance(item, dict):
                    logger.warning("[MemoryManager] ⚠️ Skipping non-dict item in parsed_list.")
                    continue

                action = item.get("action")
                if action not in ["add", "update", "delete"]:
                    logger.warning(f"[MemoryManager] ⚠️ Invalid action: {action}")
                    continue

                # ✅ Automatically retrieve the true memory_id 
                if action in ["update", "delete"]:
                    msg_content = None
                    if isinstance(item.get("message"), dict):
                        msg_content = item["message"].get("content")
                    elif isinstance(item.get("message"), Message):
                        msg_content = item["message"].content

                    if msg_content:
                        try:
                            search_results = await self.memory.search_async(
                                query=msg_content,
                                n=1  # ← Use the real retrieval interface
                            )
                            if search_results:
                                matched_mid = search_results[0][1]
                                item["memory_id"] = matched_mid
                                logger.info(
                                    f"[MemoryManager] 🔗 Auto-filled memory_id for {action}: {matched_mid}"
                                )
                            else:
                                logger.warning(
                                    f"[MemoryManager] ⚠️ No memory found to auto-fill memory_id for {action}"
                                )
                        except Exception as e:
                            logger.error(f"[MemoryManager] ❌ Auto-search failed: {e}")

                # ✅ For add/update, message is required
                if action in ["add", "update"] and not item.get("message"):
                    logger.warning(f"[MemoryManager] ⚠️ message required for {action}")
                    continue
                # ✅ For update/delete, memory_id must be auto-filled
                if action in ["update", "delete"] and not item.get("memory_id"):
                    logger.warning(f"[MemoryManager] ⚠️ memory_id still missing after auto-search for {action}")
                    continue

                logger.info(
                    f"[MemoryManager] 🟢 Ready to execute {action.upper()} | memory_id={item.get('memory_id')} | "
                    f"message_content={item.get('message',{}).get('content') if item.get('message') else None}"
                )


                validated.append({
                    "action": action,
                    "memory_id": item.get("memory_id") if action in ["update", "delete"] else None,
                    "message": item.get("message") if action in ["add", "update"] else None
                })

            if not validated:
                logger.warning("[MemoryManager] ⚠️ No valid operations parsed, fallback to input_data.")
                return [input_data]

            return validated

        except Exception as e:
            logger.error(f"[MemoryManager] ❌ LLM failed to generate valid memory operation: {str(e)}")
            return [input_data]  # Fallback to list with one item
        # pass
        
    async def summarize_memories(
        self,
        long_term_texts: List[str],
        short_term_context: str,
        user_prompt: str = ""
    ) -> str:
        """
        Summarize the [long-term memory + short-term memory + user input] using LLM.
        Return a concise summary text.
        """
        if not self.llm:
            logger.warning("[MemoryManager] No LLM configured, skip summarization.")
            return "(No LLM, skip summary.)"

        # Construct summary prompt
        context_for_summary = "\n".join(long_term_texts) if long_term_texts else "(No long-term memories)"
        short_context = short_term_context or "(No short-term context)"
        prompt = (
            "Summarize the following information into a concise context summary:\n\n"
            f"Long-Term Memories:\n{context_for_summary}\n\n"
            f"Short-Term Context:\n{short_context}\n\n"
            f"User Prompt:\n{user_prompt}\n\n"
            "Summary:"
        )

        logger.info("[MemoryManager] 🔎 Sending memories to LLM for summarization...")
        try:
            response = await self.llm.async_generate(prompt=prompt)
            logger.info("[MemoryManager] ✅ Summary generated successfully.")
            return response.content.strip()
        except Exception as e:
            logger.error(f"[MemoryManager] ❌ Summarization failed: {e}")
            return "(Summary generation failed.)"


    async def handle_memory(
        self,
        action: str,
        user_prompt: Optional[Union[str, Message, Query]] = None,
        data: Optional[Union[Message, str, List[Union[Message, str]], Dict, List[Tuple[str, Union[Message, str]]]]] = None,
        top_k: Optional[int] = None,
        metadata_filters: Optional[Dict] = None
    ) -> Union[List[str], List[Tuple[Message, str]], List[bool], Message, None]:
        """
        Handle memory operations based on the specified action, with optional LLM inference.

        Args:
            action (str): The memory operation ("add", "search", "get", "update", "delete", "clear", "save", "load", "create_message").
            user_prompt (Optional[Union[str, Message, Query]]): The user prompt or query to process with memory data.
            data (Optional): Input data for the operation (e.g., messages, memory IDs, updates).
            top_k (Optional[int]): Number of results to retrieve for search operations.
            metadata_filters (Optional[Dict]): Filters for memory retrieval.

        Returns:
            Union[List[str], List[Tuple[Message, str]], List[bool], Message, None]: Result of the operation.
        """
        if action not in ["add", "search", "get", "update", "delete", "clear", "save", "load", "create_message"]:
            logger.error(f"Invalid action: {action}")
            raise ValueError(f"Invalid action: {action}")

        if action == "add":
            if not data:
                logger.warning("No data provided for add operation")
                return []
            if not isinstance(data, list):
                data = [data]

            messages = [
                Message(
                    content=msg if isinstance(msg, str) else msg.content,
                    msg_type=MessageType.REQUEST if isinstance(msg, str) else msg.msg_type,
                    timestamp=datetime.now().isoformat() if isinstance(msg, str) else msg.timestamp,
                    agent="user" if isinstance(msg, str) else msg.agent,
                    message_id=str(uuid4()) if isinstance(msg, str) or not msg.message_id else msg.message_id
                )
                for msg in data
            ]

            input_data = [
                {
                    "action": "add",
                    "message": msg.to_dict()
                }
                for msg in messages
            ]

            if self.use_llm_management and self.llm:
                llm_decisions = await self._prompt_llm_for_memory_operation(input_data)

                final_messages: list[Message] = []
                for decision, msg in zip(llm_decisions, messages):
                    action_from_llm = decision.get("action")

                    if action_from_llm == "update":
                        # The LLM deems an update necessary → After retrieving the real ID, it is handed over to the UPDATE
                        logger.info("[MemoryManager] 🔁 LLM chose UPDATE for memory")
                        return await self.handle_memory(
                            action="update",
                            data={
                                # memory_id will be automatically retrieved and filled by _prompt_llm_for_memory_operation
                                "memory_id": decision.get("memory_id"),
                                "message": decision.get("message") or msg.to_dict()
                            }
                        )

                    if action_from_llm == "delete":
                        logger.info("[MemoryManager] 🔁 LLM chose DELETE for memory")
                        return await self.handle_memory(
                            action="delete",
                            data=decision.get("memory_id")
                        )

                    if action_from_llm == "add":
                        final_messages.append(msg)
                    else:
                        logger.info(f"[MemoryManager] ⚠️ Unexpected action: {action_from_llm}")

                # ⚡️Directly writes to the database and returns the true memory_id
                return self.memory.add(final_messages) if final_messages else []

            # If no LLM management, write directly
            return self.memory.add(messages)


        elif action == "search":
            if not user_prompt:
                logger.warning("No user_prompt provided for search operation")
                return []
            if isinstance(user_prompt, Message):
                user_prompt = user_prompt.content
            return await self.memory.search_async(user_prompt, top_k, metadata_filters)

        elif action == "get":
            if not data:
                logger.warning("No memory IDs provided for get operation")
                return []
            return await self.memory.get(data, return_chunk=False)

        elif action == "update":
            if not data:
                logger.warning("No updates provided for update operation")
                return []

            # 🔧 If it's a dict, correctly destructure memory_id and message
            if isinstance(data, dict):
                updates = [(data["memory_id"], Message(
                    content=data["message"]["content"] if isinstance(data["message"], dict) else data["message"].content,
                    msg_type=MessageType.REQUEST,
                    timestamp=datetime.now().isoformat(),
                    agent="user" if isinstance(data["message"], dict) else data["message"].agent,
                    message_id=str(uuid4())
                ))]
            else:
                updates = [
                    (mid, Message(
                        content=msg if isinstance(msg, str) else msg.content,
                        msg_type=MessageType.REQUEST if isinstance(msg, str) else msg.msg_type,
                        timestamp=datetime.now().isoformat(),
                        agent="user" if isinstance(msg, str) else msg.agent,
                        message_id=str(uuid4()) if isinstance(msg, str) or not msg.message_id else msg.message_id
                    ))
                    for mid, msg in (data if isinstance(data, list) else [data])
                ]
            input_data = [
                {
                    "action": "update",
                    "memory_id": mid,
                    "message": msg.to_dict()
                } for mid, msg in updates
            ]
            if self.use_llm_management and self.llm:
                existing_memories = await self.memory.get([mid for mid, _ in updates])
                llm_decisions = await self._prompt_llm_for_memory_operation(input_data, relevant_data=existing_memories)
                final_updates = []
                for decision, (mid, msg) in zip(llm_decisions, updates):
                    if decision.get("action") != "update":
                        logger.info(f"LLM rejected updating memory {mid}: {decision}")
                        continue
                    final_updates.append((mid, msg))
                    logger.info(f"[MemoryManager] 🚀 Executing UPDATE: {[(mid, msg.content) for mid, msg in final_updates]}")
                return await self.memory.update(final_updates) if final_updates else [False] * len(updates)
            
            return self.memory.update(updates)

        elif action == "delete":
            if not data:
                logger.warning("No memory IDs provided for delete operation")
                return []
            memory_ids = data if isinstance(data, list) else [data]
            if self.use_llm_management and self.llm:
                input_data = [{"action": "delete", "memory_id": mid} for mid in memory_ids]
                existing_memories = await self.memory.get(memory_ids)
                llm_decisions = await self._prompt_llm_for_memory_operation(input_data, relevant_data=existing_memories)
                valid_memory_ids = [decision.get("memory_id") for decision in llm_decisions if decision.get("action") == "delete"]
                return await self.memory.delete(valid_memory_ids) if valid_memory_ids else [False] * len(memory_ids)
            return await self.memory.delete(memory_ids)

        elif action == "clear":
            self.memory.clear()
            return None

        elif action == "save":
            self.memory.save(data)
            return None

        elif action == "load":
            return self.memory.load(data)

        elif action == "create_message":
            if not user_prompt:
                logger.warning("No user_prompt provided for create_message operation")
                return None
            if isinstance(user_prompt, Query):
                user_prompt = user_prompt.query_str
            elif isinstance(user_prompt, Message):
                user_prompt = user_prompt.content
            memories = await self.memory.search_async(user_prompt, top_k, metadata_filters)
            context = "\n".join([msg.content for msg, _ in memories])
            memory_ids = [mid for _, mid in memories]
            combined_content = f"User Prompt: {user_prompt}\nContext: {context}" if context else user_prompt
            return Message(
                content=combined_content,
                msg_type=MessageType.REQUEST,
                timestamp=datetime.now().isoformat(),
                agent="user",
                memory_ids=memory_ids
            )

    async def create_conversation_message(
        self,
        user_prompt: str,
        conversation_id: str,
        short_term_context: str = "",
        top_k: int | None = None,
        metadata_filters: dict | None = None,
        corpus_id: str | None = None,
    ) -> Message:
        """
        Build a Message that contains:
        1. Retrieved Long-Term Memories
        2. Short-Term Context
        3. Current User Prompt
        Each part is clearly separated for better LLM understanding.
        """

        # ✅ Build metadata filter correctly
        history_filter = {}
        if corpus_id:
            history_filter["corpus_id"] = corpus_id
        if metadata_filters:
            history_filter.update(metadata_filters)

        # 🔍 Retrieve long-term memory history
        history_results = await self.memory.search_async(
            query=user_prompt,
            n=top_k or 10,
            metadata_filters=history_filter,
        )

        # Collect retrieved long-term memory texts and their IDs
        long_term_texts: list[str] = []
        memory_ids: list[str] = []
        for chunk, mid in history_results:
            # Compatible with TextChunk or Message
            long_term_texts.append(chunk.text if hasattr(chunk, "text") else chunk.content)
            memory_ids.append(mid)

        combined_parts: list[str] = []

        # 1️⃣ Long-Term Memories
        if long_term_texts:
            combined_parts.append("### Long-Term Memories")
            for i, c in enumerate(long_term_texts, 1):
                combined_parts.append(f"{i}. {c}")
        else:
            combined_parts.append("### Long-Term Memories\n(No relevant long-term memories retrieved)")

        # 2️⃣ Short-Term Context
        if short_term_context:
            combined_parts.append("\n### Short-Term Context")
            combined_parts.append(short_term_context)
        else:
            combined_parts.append("\n### Short-Term Context\n(No recent short-term context)")

        # 🆕 3️⃣ Generate Summary of Memories
        logger.info("[MemoryManager] 📝 Generating summary for conversation message...")
        summary_text = await self.summarize_memories(
            long_term_texts=long_term_texts,
            short_term_context=short_term_context,
            user_prompt=user_prompt
        )
        combined_parts.append("\n### Memory Summary")
        combined_parts.append(summary_text)

        # 4️⃣ User Prompt
        combined_parts.append("\n### User Prompt")
        combined_parts.append(user_prompt)

        combined_content = "\n".join(combined_parts)

        logger.info("[MemoryManager] ✅ create_conversation_message assembled successfully.")

        return Message(
            content=combined_content,
            msg_type=MessageType.REQUEST,
            timestamp=datetime.now().isoformat(),
            agent="user",
            conversation_id=conversation_id,
            memory_ids=memory_ids,
        )
