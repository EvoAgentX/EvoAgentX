import os
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
from evoagentx.storages.storages_config import VectorStoreConfig, DBConfig, StoreConfig
from evoagentx.rag.rag_config import ReaderConfig, ChunkerConfig, IndexConfig, RetrievalConfig, EmbeddingConfig, RAGConfig


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
        conversation_id = action_input.conversation_id or str(uuid4())
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
            **kwargs
        )
        if return_action_input_data:
            return message, action_input_data
        return message

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


# Test Helper
async def add_memory_history(memory_agent):
    from datetime import datetime, timedelta
    messages = [
        Message(
            content="Python is great for scripting and automation tasks.",
            msg_type=MessageType.INPUT,
            timestamp=(datetime.now() - timedelta(days=1)).isoformat(),
            agent="User",
        ),
        Message(
            content="Yes, Python's simplicity and extensive libraries make it ideal for scripting.",
            msg_type=MessageType.RESPONSE,
            timestamp=(datetime.now() - timedelta(days=1)).isoformat(),
            agent="MemoryAgent",
        ),
        Message(
            content="What are some popular Python libraries for data analysis?",
            msg_type=MessageType.INPUT,
            timestamp=(datetime.now() - timedelta(days=1)).isoformat(),
            agent="User",
        ),
        Message(
            content="Popular Python libraries for data analysis include Pandas, NumPy, and Matplotlib.",
            msg_type=MessageType.RESPONSE,
            timestamp=(datetime.now() - timedelta(days=1)).isoformat(),
            agent="MemoryAgent",
        )
    ]

    for message in messages:
        await memory_agent.memory_manager.handle_memory(
            action="add",
            data=message,
        )

# Example usage
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    load_dotenv()
    # from evoagentx.models import OpenRouterConfig, OpenRouterLLM

    # OPEN_ROUNTER_API_KEY = os.environ["OPEN_ROUNTER_API_KEY"]
    # config = OpenRouterConfig(
    #     openrouter_key=OPEN_ROUNTER_API_KEY,
    #     temperature=0.3,
    #     model="qwen/qwen3-235b-a22b:free",
    # )
    # llm = OpenRouterLLM(config=config)


    from evoagentx.models import OpenAILLMConfig, OpenAILLM
    OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
    # Configure the model
    config = OpenAILLMConfig(
        model="gpt-4o-mini",  
        openai_key=OPENAI_API_KEY,
        temperature=0.3,
    )

    # Initialize the model
    llm = OpenAILLM(config=config)

    # Initialize StorageHandler
    store_config = StoreConfig(
        dbConfig=DBConfig(
            db_name="sqlite",
            path="./debug/data/hotpotqa/cache/test_hotpotQA.sql"
        ),
        vectorConfig=VectorStoreConfig(
            vector_name="faiss",
            dimensions=768,    # 1536: text-embedding-ada-002, 384: bge-small-en-v1.5, 768: nomic-embed-text
            index_type="flat_l2",
        ),
        graphConfig=None,
        # graphConfig=None,
        path="./debug/data/hotpotqa/cache/indexing"
    )
    storage_handler = StorageHandler(storageConfig=store_config)

    embedding=EmbeddingConfig(
            provider="huggingface",
            model_name=r"debug/bge-small-en-v1.5",
            device="cpu"
    )

    rag_config = RAGConfig(
        reader=ReaderConfig(
            recursive=False, exclude_hidden=True,
            num_files_limit=None, custom_metadata_function=None,
            extern_file_extractor=None,
            errors="ignore", encoding="utf-8"
        ),
        chunker=ChunkerConfig(
            strategy="simple",
            chunk_size=512,
            chunk_overlap=0,
            max_chunks=None
        ),
        embedding=embedding,
        index=IndexConfig(index_type="vector"),
        retrieval=RetrievalConfig(
            retrivel_type="vector",
            postprocessor_type="simple",
            top_k=2,  # Retrieve top-10 contexts
            similarity_cutoff=0.3,
            keyword_filters=None,
            metadata_filters=None
        )
    )

    # Initialize MemoryAgent
    agent = MemoryAgent(
        llm=llm,
        rag_config=rag_config,
        storage_handler=storage_handler,
        name="MemoryAgent",
        description="An agent that uses long-term memory for context-aware responses",
    )

    # Add History
    asyncio.run(add_memory_history(agent))

    # Example execution
    result = agent.execute(
        action_name="MemoryAction",
        action_input_data={
            "user_prompt": "What did we discuss about Python yesterday?",
            "conversation_id": agent.memory_manager.memory.default_corpus_id,
            "top_k": 3,
            "metadata_filters": {}
        }
    )
    print(f"Response: {result}")