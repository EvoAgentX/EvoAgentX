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
        pass
    
    async def _prompt_llm_for_memory_operation(self, input_data: Dict[str, Any], relevant_data: List[Tuple[Message, str]] = None) -> Dict[str, Any]:
        """Prompt the LLM to decide memory operation (add, update, delete) and return structured JSON."""
        if not self.llm or not self.use_llm_management:
            return input_data  # Bypass LLM if disabled or no LLM provided

        relevant_data_str = '\n'.join([json.dumps({"message": msg.to_dict(), "memory_id": mid}) for msg, mid in (relevant_data or [])])
        prompt = MANAGER_PROMPT.replace("<<INPUT_DATA>>", json.dumps(input_data, ensure_ascii=False)).replace("<<RELEVANT_DATA>>", relevant_data_str)

        logger.info(f"Memory Manager LLM Prompt: \n\n{prompt}")
        try:
            response = self.llm.generate(prompt=prompt)
            parsed = json.loads(response.content.replace("```json", "").replace("```", "").strip())
            if parsed["action"] not in ["add", "update", "delete"]:
                raise ValueError(f"Invalid action: {parsed['action']}")
            if parsed["action"] in ["update", "delete"] and not parsed.get("memory_id"):
                raise ValueError(f"memory_id required for {parsed['action']}")
            if parsed["action"] in ["add", "update"] and not parsed.get("message"):
                raise ValueError(f"message required for {parsed['action']}")
            return parsed
        except Exception as e:
            logger.error(f"LLM failed to generate valid memory operation: {str(e)}")
            return input_data  # Fallback to input data on error

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
                ) for msg in data
            ]
            input_data = [
                {
                    "action": "add",
                    "memory_id": str(uuid4()),
                    "message": msg.to_dict()
                } for msg in messages
            ]
            if self.use_llm_management and self.llm:
                llm_decisions = await self._prompt_llm_for_memory_operation(input_data)
                final_messages = []
                final_memory_ids = []
                for decision, msg in zip(llm_decisions, messages):
                    if decision.get("action") != "add":
                        logger.info(f"LLM rejected adding memory: {decision}")
                        continue
                    final_messages.append(msg)
                    final_memory_ids.append(decision.get("memory_id"))
                return self.memory.add(final_messages) if final_messages else []
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
            updates = [
                (mid, Message(
                    content=msg if isinstance(msg, str) else msg.content,
                    msg_type=MessageType.REQUEST if isinstance(msg, str) else msg.msg_type,
                    timestamp=datetime.now().isoformat(),
                    agent="user" if isinstance(msg, str) else msg.agent,
                    message_id=str(uuid4()) if isinstance(msg, str) or not msg.message_id else msg.message_id
                )) for mid, msg in (data if isinstance(data, list) else [data])
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
                return self.memory.update(final_updates) if final_updates else [False] * len(updates)
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
                return self.memory.delete(valid_memory_ids) if valid_memory_ids else [False] * len(memory_ids)
            return self.memory.delete(memory_ids)

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
        user_prompt: Union[str, Message],
        conversation_id: str,
        top_k: Optional[int] = None,
        metadata_filters: Optional[Dict] = None
    ) -> Message:
        """
        Create a Message combining user prompt with conversation history and relevant memories.

        Args:
            user_prompt (Union[str, Message]): The user's input prompt or message.
            conversation_id (str): ID of the conversation thread.
            top_k (Optional[int]): Number of results to retrieve.
            metadata_filters (Optional[Dict]): Filters for memory retrieval.

        Returns:
            Message: A new Message object with user prompt, history, and memory context.
        """
        if isinstance(user_prompt, Message):
            user_prompt = user_prompt.content

        # Retrieve conversation history
        history_filter = {"corpus_id": conversation_id}
        if metadata_filters:
            history_filter.update(metadata_filters)
        history_results = await self.memory.search_async(
            query=user_prompt, n=top_k or 10, metadata_filters=history_filter
        )
        history = "\n".join([f"{msg.content}" for msg, _ in history_results])

        # Combine prompt, history, and context
        combined_content = (
            f"User Prompt: \n{user_prompt}\n"
            f"Conversation History: \n\n{history or 'No history available'}\n"
        )
        return Message(
            content=combined_content,
            msg_type=MessageType.REQUEST,
            timestamp=datetime.now().isoformat(),
            agent="user",
            memory_ids=user_prompt.message_id if isinstance(user_prompt, Message) else str(uuid4())
        )