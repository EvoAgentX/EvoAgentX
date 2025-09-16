import json
import asyncio
from uuid import uuid4
from pydantic import Field
from datetime import datetime
from typing import Optional, List, Tuple, Dict, Union

from evoagentx.agents import Agent
from evoagentx.core.parser import Parser
from evoagentx.models import BaseLLM
from evoagentx.core.logging import logger
from evoagentx.models import OpenAILLMConfig
from evoagentx.storages.base import StorageHandler
from evoagentx.core.message import Message, MessageType
from evoagentx.memory.memory_manager import MemoryManager
from evoagentx.memory.long_term_memory import LongTermMemory
from evoagentx.actions.action import Action, ActionInput, ActionOutput
from evoagentx.rag.rag_config import RAGConfig


class MemoryActionInput(ActionInput):
    user_prompt: str = Field(description="The user's input prompt")
    conversation_id: Optional[str] = Field(default=None, description="ID for tracking conversation")
    top_k: Optional[int] = Field(default=5, description="Number of memory results to retrieve")
    metadata_filters: Optional[Dict] = Field(default=None, description="Filters for memory retrieval")


class MemoryActionOutput(ActionOutput):
    response: str = Field(description="The agent's response based on memory and prompt")


class MemoryAction(Action):
    def __init__(
        self,
        name: str = "MemoryAction",
        description: str = "Action that processes user input with long-term memory context",
        prompt: str = "Based on the following context and user prompt, provide a relevant response:\n\nContext: {context}\n\nUser Prompt: {user_prompt}\n\n",
        inputs_format: ActionInput = None,
        outputs_format: ActionOutput = None,
        **kwargs
    ):
        inputs_format = inputs_format or MemoryActionInput
        outputs_format = outputs_format or MemoryActionOutput
        super().__init__(
            name=name,
            description=description,
            prompt=prompt,
            inputs_format=inputs_format,
            outputs_format=outputs_format,
            **kwargs
        )

    def execute(self, llm: BaseLLM | None = None, 
                inputs: Dict | None = None, 
                sys_msg: str | None = None, 
                return_prompt: bool = False, 
                memory_manager: Optional[MemoryManager] = None,
                **kwargs
    ) -> Parser | Tuple[Parser | str] | None:
        return asyncio.run(self.async_execute(llm, inputs, sys_msg, return_prompt, memory_manager, **kwargs))

    async def async_execute(
        self,
        llm: Optional["BaseLLM"] = None,
        inputs: Optional[Dict] = None,
        sys_msg: Optional[str] = None,
        return_prompt: bool = False,
        memory_manager: Optional[MemoryManager] = None,
        **kwargs
    ) -> Union[MemoryActionOutput, tuple]:
        if not memory_manager:
            logger.error("MemoryManager is required for MemoryAction execution")
            raise ValueError("MemoryManager is required for MemoryAction")

        action_input = self.inputs_format(**inputs)
        user_prompt = action_input.user_prompt
        conversation_id = action_input.conversation_id
        if not conversation_id:
            conversation_id = str(uuid4())
            logger.warning("No conversation_id provided; generated a new UUID4 for this session")
        top_k = action_input.top_k
        metadata_filters = action_input.metadata_filters

        message = await memory_manager.create_conversation_message(
            user_prompt=user_prompt,
            conversation_id=conversation_id,
            top_k=top_k,
            metadata_filters=metadata_filters
        )

        action_input_attrs = self.inputs_format.get_attrs()
        action_input_data = {attr: getattr(action_input, attr, "undefined") for attr in action_input_attrs}
        action_input_data["context"] = message.content
        prompt = self.prompt.format(**action_input_data)
        logger.info(f"The New Created Message by LongTermMemory:\n\n{prompt}")

        output = await llm.async_generate(
            prompt=prompt,
            system_message=sys_msg,
            parser=self.outputs_format,
            parse_mode='str'
        )
        
        response_message = Message(
            content=output.content,
            msg_type=MessageType.RESPONSE,
            timestamp=datetime.now().isoformat(),
            conversation_id=conversation_id,
            memory_ids=message.memory_ids
        )
        memory_ids = await memory_manager.handle_memory(
            action="add",
            data=response_message,
        )

        # Prepare the final output
        final_output = self.outputs_format(
            response=output.content,
            memory_ids=memory_ids
        )

        if return_prompt:
            return final_output, prompt
        return final_output


class MemoryAgent(Agent):
    memory_manager: Optional[MemoryManager] = Field(default=None, description="Manager for long-term memory operations")
    inputs: List[Dict] = Field(default_factory=list, description="Input specifications for the memory action")
    outputs: List[Dict] = Field(default_factory=list, description="Output specifications for the memory action")

    def __init__(
        self,
        name: str = "MemoryAgent",
        description: str = "An agent that uses long-term memory to provide context-aware responses",
        inputs: Optional[List[Dict]] = None,
        outputs: Optional[List[Dict]] = None,
        llm_config: Optional[OpenAILLMConfig] = None,
        storage_handler: Optional[StorageHandler] = None,
        rag_config: Optional[RAGConfig] = None,
        conversation_id: Optional[str] = None,
        system_prompt: Optional[str] = None,
        prompt: str = "Based on the following context and user prompt, provide a relevant response:\n\nContext: {context}\n\nUser Prompt: {user_prompt}",
        **kwargs
    ):
        # Define inputs and outputs inspired by CustomizeAgent
        inputs = inputs or []
        outputs = outputs or []

        # Initialize base Agent with provided parameters
        super().__init__(
            name=name,
            description=description,
            llm_config=llm_config,
            system_prompt=system_prompt,
            storage_handler=storage_handler,
            inputs=inputs,
            outputs=outputs,
            **kwargs
        )

        self.long_term_memory = LongTermMemory(
            storage_handler=storage_handler,
            rag_config=rag_config,
            default_corpus_id=conversation_id
        )
        self.memory_manager = MemoryManager(
            memory=self.long_term_memory,
            llm=llm_config.get_llm() if llm_config else None,
            use_llm_management=True
        )

        # Initialize inputs and outputs
        self.inputs = inputs
        self.outputs = outputs

        # Initialize actions list and add MemoryAction
        self.actions = []
        self._action_map = {}
        memory_action = MemoryAction(
            name="MemoryAction",
            description="Action that processes user input with long-term memory context",
            prompt=prompt,
            inputs_format=MemoryActionInput,
            outputs_format=MemoryActionOutput
        )
        self.add_action(memory_action)

    def _create_output_message(
        self,
        action_output,
        action_name: str,
        action_input_data: Optional[Dict],
        prompt: str,
        return_msg_type: MessageType = MessageType.RESPONSE,
        **kwargs
    ) -> Message:
        msg = super()._create_output_message(
            action_output=action_output,
            action_name=action_name,
            action_input_data=action_input_data,
            prompt=prompt,
            return_msg_type=return_msg_type,
            **kwargs
        )

        if action_input_data and "user_prompt" in action_input_data:
            user_msg = Message(
                content=action_input_data["user_prompt"],
                msg_type=MessageType.REQUEST,
                conversation_id=msg.conversation_id
            )
            asyncio.create_task(self.memory_manager.handle_memory(action="add", data=user_msg))

        response_msg = Message(
            content=action_output.response if hasattr(action_output, "response") else str(action_output),
            msg_type=MessageType.RESPONSE,
            conversation_id=msg.conversation_id
        )
        asyncio.create_task(self.memory_manager.handle_memory(action="add", data=response_msg))

        return msg

    async def async_execute(
        self,
        action_name: str,
        msgs: Optional[List[Message]] = None,
        action_input_data: Optional[Dict] = None,
        return_msg_type: Optional[MessageType] = MessageType.RESPONSE,
        return_action_input_data: Optional[bool] = False,
        **kwargs
    ) -> Union[Message, Tuple[Message, Dict]]:
        """
        Execute an action asynchronously with memory management.

        Args:
            action_name: Name of the action to execute
            msgs: Optional list of messages providing context
            action_input_data: Optional input data for the action
            return_msg_type: Message type for the return message
            return_action_input_data: Whether to return the action input data
            **kwargs: Additional parameters

        Returns:
            Message or tuple: The execution result, optionally with input data
        """
        action, action_input_data = self._prepare_execution(
            action_name=action_name,
            msgs=msgs,
            action_input_data=action_input_data,
            **kwargs
        )

        # Execute action with memory_manager
        execution_results = await action.async_execute(
            llm=self.llm,
            inputs=action_input_data,
            sys_msg=self.system_prompt,
            return_prompt=True,
            memory_manager=self.memory_manager,
            **kwargs
        )
        action_output, prompt = execution_results

        message = self._create_output_message(
            action_output=action_output,
            prompt=prompt,
            action_name=action_name,
            return_msg_type=return_msg_type,
            action_input_data=action_input_data,
            **kwargs
        )
        if return_action_input_data:
            return message, action_input_data
        return message

    def execute(
        self,
        action_name: str,
        msgs: Optional[List[Message]] = None,
        action_input_data: Optional[Dict] = None,
        return_msg_type: Optional[MessageType] = MessageType.RESPONSE,
        return_action_input_data: Optional[bool] = False,
        **kwargs
    ) -> Union[Message, Tuple[Message, Dict]]:
        """
        Execute an action synchronously with memory management.

        Args:
            action_name: Name of the action to execute
            msgs: Optional list of messages providing context
            action_input_data: Optional input data for the action
            return_msg_type: Message type for the return message
            return_action_input_data: Whether to return the action input data
            **kwargs: Additional parameters

        Returns:
            Message or tuple: The execution result, optionally with input data
        """
        action, action_input_data = self._prepare_execution(
            action_name=action_name,
            msgs=msgs,
            action_input_data=action_input_data,
            **kwargs
        )

        # Execute action with memory_manager
        execution_results = action.execute(
            llm=self.llm,
            inputs=action_input_data,
            sys_msg=self.system_prompt,
            return_prompt=True,
            memory_manager=self.memory_manager,
            **kwargs
        )
        action_output, prompt = execution_results

        message = self._create_output_message(
            action_output=action_output,
            prompt=prompt,
            action_name=action_name,
            return_msg_type=return_msg_type,
            action_input_data=action_input_data,
            **kwargs
        )
        if return_action_input_data:
            return message, action_input_data
        return message

    def chat(
        self,
        user_prompt: str,
        *,
        conversation_id: Optional[str] = None,
        top_k: Optional[int] = None,
        metadata_filters: Optional[dict] = None,
        return_message: bool = True,
        **kwargs
    ):
        action_input_data = {
            "user_prompt": user_prompt,
            "conversation_id": conversation_id or self._default_conversation_id(),
            "top_k": top_k if top_k is not None else 3,
            "metadata_filters": metadata_filters or {},
        }
        msg = self.execute(
            action_name="MemoryAction",
            action_input_data=action_input_data,
            return_msg_type=MessageType.RESPONSE,
            **kwargs
        )
        return msg if return_message else (getattr(msg, "content", None) or str(msg))


    async def async_chat(
        self,
        user_prompt: str,
        *,
        conversation_id: Optional[str] = None,
        top_k: Optional[int] = None,
        metadata_filters: Optional[dict] = None,
        return_message: bool = True,
        **kwargs
    ):
        action_input_data = {
            "user_prompt": user_prompt,
            "conversation_id": conversation_id or self._default_conversation_id(),
            "top_k": top_k if top_k is not None else 3,
            "metadata_filters": metadata_filters or {},
        }
        msg = await self.async_execute(
            action_name="MemoryAction",
            action_input_data=action_input_data,
            return_msg_type=MessageType.RESPONSE,
            **kwargs
        )
        return msg if return_message else (getattr(msg, "content", None) or str(msg))


    def _default_conversation_id(self) -> str:
        """
        Session scope: By default, a new uuid4() is returned (new session).
        User/global scope: Reuse LongTermMemory.default_corpus_id (stable namespace).
        Note: The final ID is still uniformly managed by MemoryAgent._prepare_execution() (which will override based on the scope).
        """
        scope = getattr(self, "conversation_scope", "session")
        if scope == "session":
            return str(uuid4())
        return getattr(getattr(self, "long_term_memory", None), "default_corpus_id", None) or "global_corpus"
    
    async def interactive_chat(
        self,
        conversation_id: Optional[str] = None,
        top_k: int = 3,
        metadata_filters: Optional[dict] = None
    ):
        """
        In interactive chat, each round of input will:
        1. Retrieve from memory
        2. Generate a response based on historical context
        3. Write the input/output to long-term memory and refresh the index 
        """
        conversation_id = conversation_id or self._default_conversation_id()
        metadata_filters = metadata_filters or {}

        print("💬 MemoryAgent has been started (type 'exit' to quit)\n")

        while True:
            user_prompt = input("You: ").strip()
            if user_prompt.lower() in ["exit", "quit"]:
                print("🔚 Conversation ended")
                break

            # Retrieve historical context
            retrieved_memories = await self.memory_manager.handle_memory(
                action="search",
                user_prompt=user_prompt,
                top_k=top_k,
                metadata_filters=metadata_filters
            )

            context_texts = []
            for msg, _ in retrieved_memories:
                if hasattr(msg, "content") and msg.content:
                    context_texts.append(msg.content)
            context_str = "\n".join(context_texts)

            # if context_str:
            #     print(f"📖 Retrieved context from memory:\n{context_str}\n")

            # Concatenate the historical context into the user input and invoke async_chat
            full_prompt = f"Context:\n{context_str}\n\nUser: {user_prompt}" if context_str else user_prompt
            msg = await self.async_chat(
                user_prompt=full_prompt,
                conversation_id=conversation_id,
                top_k=top_k,
                metadata_filters=metadata_filters
            )

            print(f"Agent: {msg.content}\n")

            # Refresh the index to ensure it can be retrieved in the next round
            if hasattr(self.memory_manager, "handle_memory_flush"):
                await self.memory_manager.handle_memory_flush()
            else:
                await asyncio.sleep(0.1)



    def save_module(self, path: str, ignore: List[str] = ["llm", "llm_config", "memory_manager"], **kwargs) -> str:
        """
        Save the agent's configuration to a JSON file, excluding memory_manager by default.

        Args:
            path: File path to save the configuration
            ignore: List of keys to exclude from the saved configuration
            **kwargs: Additional parameters for saving

        Returns:
            str: The path where the configuration was saved
        """
        return super().save_module(path=path, ignore=ignore, **kwargs)

    @classmethod
    def from_file(cls, path: str, llm_config: OpenAILLMConfig, storage_handler: Optional[StorageHandler] = None, rag_config: Optional[RAGConfig] = None, **kwargs) -> "MemoryAgent":
        """
        Load a MemoryAgent from a JSON configuration file.

        Args:
            path: Path to the JSON configuration file
            llm_config: LLM configuration
            storage_handler: Optional storage handler
            rag_config: Optional RAG configuration
            **kwargs: Additional parameters

        Returns:
            MemoryAgent: The loaded agent instance
        """
        with open(path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return cls(
            name=config.get("name", "MemoryAgent"),
            description=config.get("description", "An agent that uses long-term memory"),
            llm_config=llm_config,
            storage_handler=storage_handler,
            rag_config=rag_config,
            system_prompt=config.get("system_prompt"),
            prompt=config.get("prompt"),
            use_long_term_memory=config.get("use_long_term_memory", True),
            **kwargs
        )