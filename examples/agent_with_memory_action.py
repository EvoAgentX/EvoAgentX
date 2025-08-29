import os
from pydantic import Field
from datetime import datetime
from dotenv import load_dotenv
from typing import Optional, List, Dict

from evoagentx.agents import Agent
from evoagentx.rag import RAGConfig
from evoagentx.storages.base import StorageHandler
from evoagentx.models import BaseLLM, OpenRouterConfig, OpenRouterLLM
from evoagentx.core.message import Message, MessageType
from evoagentx.memory.long_term_memory import LongTermMemory
from evoagentx.actions import Action, ActionInput, ActionOutput
from evoagentx.storages.storages_config import StoreConfig, VectorStoreConfig, DBConfig
from evoagentx.rag.rag_config import RAGConfig, ReaderConfig, IndexConfig, ChunkerConfig, RetrievalConfig, EmbeddingConfig

load_dotenv()

class AddMemoriesInput(ActionInput):
    messages: List[Dict] = Field(description="List of messages to add, each with content and optional attributes")

class AddMemoriesOutput(ActionOutput):
    memory_ids: List[str] = Field(description="List of memory IDs for added messages")

class AddMemories(Action):
    def __init__(
        self,
        name: str = "AddMemories",
        description: str = "Add multiple messages to long-term memory",
        prompt: str = "Add the following messages to memory: {messages}",
        inputs_format: ActionInput = None,
        outputs_format: ActionOutput = None,
        **kwargs
    ):
        inputs_format = inputs_format or AddMemoriesInput
        outputs_format = outputs_format or AddMemoriesOutput
        super().__init__(
            name=name,
            description=description,
            prompt=prompt,
            inputs_format=inputs_format,
            outputs_format=outputs_format,
            **kwargs
        )

    def execute(
        self,
        llm: Optional[BaseLLM] = None,
        inputs: Optional[Dict] = None,
        sys_msg: Optional[str] = None,
        return_prompt: bool = False,
        memory: Optional[LongTermMemory] = None,
        **kwargs
    ) -> AddMemoriesOutput:
        if memory is None:
            raise ValueError("LongTermMemory instance required")
        action_input_attrs = self.inputs_format.get_attrs()
        action_input_data = {attr: inputs.get(attr, []) for attr in action_input_attrs}
        messages = [
            Message(
                content=msg.get("content", ""),
                action=msg.get("action"),
                wf_goal=msg.get("wf_goal"),
                timestamp=msg.get("timestamp", datetime.now().isoformat()),
                agent=msg.get("agent", "user"),
                msg_type=msg.get("msg_type", MessageType.REQUEST),
                prompt=msg.get("prompt"),
                next_actions=msg.get("next_actions"),
                wf_task=msg.get("wf_task"),
                wf_task_desc=msg.get("wf_task_desc"),
                message_id=msg.get("message_id")
            )
            for msg in action_input_data["messages"]
        ]
        memory_ids = memory.add(messages)
        output = AddMemoriesOutput(memory_ids=memory_ids)
        if return_prompt:
            prompt = self.prompt.format(messages=[msg.model_dump() for msg in messages])
            return output, prompt
        return output

class DeleteMemoriesInput(ActionInput):
    memory_ids: List[str] = Field(description="List of memory IDs to delete")

class DeleteMemoriesOutput(ActionOutput):
    successes: List[bool] = Field(description="List of success statuses for deletions")

class DeleteMemories(Action):
    def __init__(
        self,
        name: str = "DeleteMemories",
        description: str = "Delete multiple memories by IDs",
        prompt: str = "Delete the memories with IDs: {memory_ids}",
        inputs_format: ActionInput = None,
        outputs_format: ActionOutput = None,
        **kwargs
    ):
        inputs_format = inputs_format or DeleteMemoriesInput
        outputs_format = outputs_format or DeleteMemoriesOutput
        super().__init__(
            name=name,
            description=description,
            prompt=prompt,
            inputs_format=inputs_format,
            outputs_format=outputs_format,
            **kwargs
        )

    def execute(
        self,
        llm: Optional[BaseLLM] = None,
        inputs: Optional[Dict] = None,
        sys_msg: Optional[str] = None,
        return_prompt: bool = False,
        memory: Optional[LongTermMemory] = None,
        **kwargs
    ) -> DeleteMemoriesOutput:
        if memory is None:
            raise ValueError("LongTermMemory instance required")
        action_input_attrs = self.inputs_format.get_attrs()
        action_input_data = {attr: inputs.get(attr, []) for attr in action_input_attrs}
        successes = memory.delete(action_input_data["memory_ids"])
        output = DeleteMemoriesOutput(successes=successes)
        if return_prompt:
            prompt = self.prompt.format(memory_ids=action_input_data["memory_ids"])
            return output, prompt
        return output

class UpdateMemoriesInput(ActionInput):
    updates: List[Dict] = Field(description="List of updates, each with memory_id and message attributes")

class UpdateMemoriesOutput(ActionOutput):
    successes: List[bool] = Field(description="List of success statuses for updates")

class UpdateMemories(Action):
    def __init__(
        self,
        name: str = "UpdateMemories",
        description: str = "Update multiple memories by IDs",
        prompt: str = "Update the memories with the following data: {updates}",
        inputs_format: ActionInput = None,
        outputs_format: ActionOutput = None,
        **kwargs
    ):
        inputs_format = inputs_format or UpdateMemoriesInput
        outputs_format = outputs_format or UpdateMemoriesOutput
        super().__init__(
            name=name,
            description=description,
            prompt=prompt,
            inputs_format=inputs_format,
            outputs_format=outputs_format,
            **kwargs
        )

    def execute(
        self,
        llm: Optional[BaseLLM] = None,
        inputs: Optional[Dict] = None,
        sys_msg: Optional[str] = None,
        return_prompt: bool = False,
        memory: Optional[LongTermMemory] = None,
        **kwargs
    ) -> UpdateMemoriesOutput:
        if memory is None:
            raise ValueError("LongTermMemory instance required")
        action_input_attrs = self.inputs_format.get_attrs()
        action_input_data = {attr: inputs.get(attr, []) for attr in action_input_attrs}
        updates = [
            (
                update["memory_id"],
                Message(
                    content=update.get("content", ""),
                    action=update.get("action"),
                    wf_goal=update.get("wf_goal"),
                    timestamp=update.get("timestamp", datetime.now().isoformat()),
                    agent=update.get("agent", "user"),
                    msg_type=update.get("msg_type", MessageType.REQUEST),
                    prompt=update.get("prompt"),
                    next_actions=update.get("next_actions"),
                    wf_task=update.get("wf_task"),
                    wf_task_desc=update.get("wf_task_desc"),
                    message_id=update.get("message_id")
                )
            )
            for update in action_input_data["updates"]
        ]
        successes = memory.update(updates)
        output = UpdateMemoriesOutput(successes=successes)
        if return_prompt:
            prompt = self.prompt.format(updates=[{"memory_id": mid, "message": msg.model_dump()} for mid, msg in updates])
            return output, prompt
        return output

class SearchMemoriesInput(ActionInput):
    query: str = Field(description="Query string to search memories")
    top_k: Optional[int] = Field(default=5, description="Number of results to return")
    metadata_filters: Optional[Dict] = Field(default=None, description="Metadata filters (e.g., agent, msg_type)")

class SearchMemoriesOutput(ActionOutput):
    results: List[Dict] = Field(description="List of retrieved messages with their memory IDs")

class SearchMemories(Action):
    def __init__(
        self,
        name: str = "SearchMemories",
        description: str = "Search memories by query and metadata filters",
        prompt: str = "Search memories with query: {query}, filters: {metadata_filters}",
        inputs_format: ActionInput = None,
        outputs_format: ActionOutput = None,
        **kwargs
    ):
        inputs_format = inputs_format or SearchMemoriesInput
        outputs_format = outputs_format or SearchMemoriesOutput
        super().__init__(
            name=name,
            description=description,
            prompt=prompt,
            inputs_format=inputs_format,
            outputs_format=outputs_format,
            **kwargs
        )

    def execute(
        self,
        llm: Optional[BaseLLM] = None,
        inputs: Optional[Dict] = None,
        sys_msg: Optional[str] = None,
        return_prompt: bool = False,
        memory: Optional[LongTermMemory] = None,
        **kwargs
    ) -> SearchMemoriesOutput:
        if memory is None:
            raise ValueError("LongTermMemory instance required")
        action_input_attrs = self.inputs_format.get_attrs()
        action_input_data = {attr: inputs.get(attr, None) for attr in action_input_attrs}
        results = memory.search(
            query=action_input_data["query"],
            n=action_input_data["top_k"],
            metadata_filters=action_input_data["metadata_filters"]
        )
        output = SearchMemoriesOutput(
            results=[{"message": msg.model_dump(), "memory_id": mid} for msg, mid in results]
        )
        if return_prompt:
            prompt = self.prompt.format(
                query=action_input_data["query"],
                metadata_filters=action_input_data["metadata_filters"] or {}
            )
            return output, prompt
        return output

def main():
    OPEN_ROUNTER_API_KEY = os.environ["OPEN_ROUNTER_API_KEY"]
    config = OpenRouterConfig(
        openrouter_key=OPEN_ROUNTER_API_KEY,
        temperature=0.3,
        model="google/gemini-2.5-flash-lite-preview-06-17",
    )
    llm = OpenRouterLLM(config=config)

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

    memory = LongTermMemory(
        storage_handler=storage_handler,
        rag_config=rag_config,
    )

    # Define the agent
    memory_agent = Agent(
        name="MemoryAgent",
        description="An agent that manages long-term memory operations",
        actions=[
            AddMemories(),
            SearchMemories(),
            UpdateMemories(),
            DeleteMemories(),
        ],
        llm_config=config
    )

    actions = memory_agent.get_all_actions()
    print(f"Available actions of agent {memory_agent.name}:")
    for action in actions:
        print(f"- {action.name}: {action.description}")

    messages = [
        {
            "content": "Schedule a meeting with Alice on Monday",
            "action": "schedule",
            "wf_goal": "plan_meeting",
            "agent": "user",
            "msg_type": MessageType.REQUEST.value,
            "wf_task": "schedule_meeting",
            "wf_task_desc": "Schedule a meeting with a colleague",
            "message_id": "msg_001"
        },
        {
            "content": "Send report to Bob by Friday",
            "action": "send",
            "wf_goal": "submit_report",
            "agent": "user",
            "msg_type": MessageType.REQUEST.value,
            "wf_task": "send_report",
            "wf_task_desc": "Send a report to a colleague",
            "message_id": "msg_002"
        }
    ]
    add_result = memory_agent.execute(
        action_name="AddMemories",
        action_input_data={"messages": messages},
        memory=memory
    )
    print("\nAdded memories:")
    print(f"Memory IDs: {add_result.content.memory_ids}")

    # Search memories
    search_result = memory_agent.execute(
        action_name="SearchMemories",
        action_input_data={
            "query": "meeting",
            "top_k": 2,
            "metadata_filters": {"agent": "user"}
        },
        memory=memory
    )
    print("\nSearch results:")
    for result in search_result.content.results:
        print(f"- Memory ID: {result['memory_id']}, Message: {result['message'].content}")

    # Update memories
    updates = [
        {
            "memory_id": add_result.content.memory_ids[0],
            "content": "Reschedule meeting with Alice to Tuesday",
            "action": "reschedule",
            "wf_goal": "plan_meeting",
            "agent": "user",
            "msg_type": MessageType.REQUEST.value,
            "wf_task": "reschedule_meeting",
            "wf_task_desc": "Reschedule a meeting with a colleague",
            "message_id": "msg_001_updated"
        }
    ]
    update_result = memory_agent.execute(
        action_name="UpdateMemories",
        action_input_data={"updates": updates},
        memory=memory
    )
    print("\nUpdate results:")
    print(f"Successes: {update_result.content.successes}")

    # Delete memories
    delete_result = memory_agent.execute(
        action_name="DeleteMemories",
        action_input_data={"memory_ids": add_result.content.memory_ids},
        memory=memory
    )
    print("\nDelete results:")
    print(f"Successes: {delete_result.content.successes}")

    # Verify the delete operation
    new_search_result = memory_agent.execute(
        action_name="SearchMemories",
        action_input_data={
            "query": "meeting",
            "top_k": 2,
            "metadata_filters": {"agent": "user"}
        },
        memory=memory
    )
    print("\nSearch results:")
    for result in new_search_result.content.results:
        print(f"- Memory ID: {result['memory_id']}, Message: {result['message'].content}")

if __name__ == "__main__":
    main()