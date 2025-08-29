import os
from pydantic import Field
from datetime import datetime
from dotenv import load_dotenv
from typing import Optional, List, Dict

from evoagentx.agents import Agent
from evoagentx.rag import RAGConfig
from evoagentx.rag.schema import Query
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
        successes = memory.delete(action_input_data["memory_ids"], delete_from_db=True)
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
    OPEN_ROUNTER_API_KEY = os.environ.get("OPEN_ROUNTER_API_KEY")
    if not OPEN_ROUNTER_API_KEY:
        raise ValueError("OPEN_ROUNTER_API_KEY not set in environment")

    config = OpenRouterConfig(
        openrouter_key=OPEN_ROUNTER_API_KEY,
        temperature=0.3,
        model="google/gemini-2.5-pro-exp-03-25",
    )
    llm = OpenRouterLLM(config=config)

    store_config = StoreConfig(
        dbConfig=DBConfig(
            db_name="sqlite",
            path="./debug/data/hotpotqa/cache/test_hotpotqa.sql"
        ),
        vectorConfig=VectorStoreConfig(
            vector_name="faiss",
            dimensions=384,  # bge-small-en-v1.5
            index_type="flat_l2",
        ),
        graphConfig=None,
        path="./debug/data/hotpotqa/cache/indexing"
    )
    storage_handler = StorageHandler(storageConfig=store_config)

    embedding = EmbeddingConfig(
        provider="huggingface",
        model_name=r"debug/weights/bge-small-en-v1.5",
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
            top_k=2,
            similarity_cutoff=0.3,
            keyword_filters=None,
            metadata_filters=None
        )
    )

    memory = LongTermMemory(
        storage_handler=storage_handler,
        rag_config=rag_config,
        llm=llm,
        use_llm_management=False
    )
    memory.init_module()

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

    print(f"Available actions of agent {memory_agent.name}:")
    for action in memory_agent.get_all_actions():
        print(f"- {action.name}: {action.description}")

    # Test 1: Add memories
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
    print("\nTest 1: Added memories")
    print(f"Memory IDs: {add_result.content.memory_ids}")

    # Test 2: Search memories with string query
    search_result = memory_agent.execute(
        action_name="SearchMemories",
        action_input_data={
            "query": "meeting",
            "top_k": 1,
            "metadata_filters": {"agent": "user"}
        },
        memory=memory
    )
    print("\nTest 2: Search results (string query)")
    for result in search_result.content.results:
        print(f"- Memory ID: {result['memory_id']}, Content: {result['message'].content}")

    # Test 3: Search memories with Query object
    query = Query(
        query_str="report",
        top_k=1,
        metadata_filters={"msg_type": MessageType.REQUEST.value}
    )
    search_result_query = memory_agent.execute(
        action_name="SearchMemories",
        action_input_data={
            "query": query.query_str,
            "top_k": query.top_k,
            "metadata_filters": query.metadata_filters
        },
        memory=memory
    )
    print("\nTest 3: Search results (Query object)")
    for result in search_result_query.content.results:
        print(f"- Memory ID: {result['memory_id']}, Content: {result['message'].content}")

    # Test 4: Update memories
    updates = [
        {
            "memory_id": add_result.content.memory_ids[0] if add_result.content.memory_ids else "",
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
    print("\nTest 4: Update results")
    print(f"Successes: {update_result.content.successes}")

    # Test 5: Save memories
    memory.save()
    print("\nTest 5: Saved memories to database")

    # Test 6: Clear memories (in-memory only)
    memory.clear()
    search_after_clear = memory_agent.execute(
        action_name="SearchMemories",
        action_input_data={"query": "meeting", "top_k": 1},
        memory=memory
    )
    print("\nTest 6: Search after clear (in-memory)")
    print(f"Results: {len(search_after_clear.content.results)} memories found")

    # Test 7: Load memories
    loaded_ids = memory.load()
    print("\nTest 7: Loaded memories")
    print(f"Loaded {len(loaded_ids)} memory IDs: {loaded_ids}")

    # Test 8: Delete memories
    delete_result = memory_agent.execute(
        action_name="DeleteMemories",
        action_input_data={"memory_ids": add_result.content.memory_ids},
        memory=memory
    )
    print("\nTest 8: Delete results")
    print(f"Successes: {delete_result.content.successes}")

    # Test 9: Clear all (including database)
    memory.clear()
    search_after_full_clear = memory_agent.execute(
        action_name="SearchMemories",
        action_input_data={"query": "meeting", "top_k": 1},
        memory=memory
    )
    print("\nTest 9: Search after full clear")
    print(f"Results: {len(search_after_full_clear.content.results)} memories found")

if __name__ == "__main__":
    main()