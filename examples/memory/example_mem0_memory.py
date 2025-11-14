"""
Demonstration of the mem0-backed CustomizeMemoryAgent.

The script wires LongTermMemory + MemoryManager into the new mem0 backend,
showing how short/long-term memories are injected and how CRUD operations can
be triggered through the agent interface.
"""

import asyncio
import os
from typing import Tuple

from evoagentx.memory.mem0 import create_mem0_agent
from evoagentx.memory.memory_manager import MemoryManager
from evoagentx.memory.long_term_memory import LongTermMemory
from evoagentx.models import OpenAILLM, OpenAILLMConfig
from evoagentx.rag.rag_config import (
    ChunkerConfig,
    EmbeddingConfig,
    IndexConfig,
    RAGConfig,
    ReaderConfig,
    RetrievalConfig,
)
from evoagentx.storages.base import StorageHandler
from evoagentx.storages.storages_config import DBConfig, StoreConfig, VectorStoreConfig


async def bootstrap_memory_stack(llm: OpenAILLM) -> Tuple[LongTermMemory, MemoryManager, StorageHandler]:
    """Prepare LongTermMemory + MemoryManager backed by a FAISS vector index."""
    store_config = StoreConfig(
        dbConfig=DBConfig(db_name="sqlite", path="./debug/data/mem0_demo.sql"),
        vectorConfig=VectorStoreConfig(vector_name="faiss", dimensions=768, index_type="flat_l2"),
        graphConfig=None,
        path="./debug/data/mem0_demo_index",
    )
    storage_handler = StorageHandler(storageConfig=store_config)

    embedding = EmbeddingConfig(
        provider="huggingface",
        model_name="BAAI/bge-small-en-v1.5",
        device="cpu",
    )
    rag_config = RAGConfig(
        reader=ReaderConfig(recursive=False, exclude_hidden=True),
        chunker=ChunkerConfig(strategy="simple", chunk_size=512, chunk_overlap=0),
        embedding=embedding,
        index=IndexConfig(index_type="vector"),
        retrieval=RetrievalConfig(
            retrivel_type="vector",
            postprocessor_type="simple",
            top_k=3,
            similarity_cutoff=0.3,
        ),
    )

    long_term_memory = LongTermMemory(
        storage_handler=storage_handler,
        rag_config=rag_config,
    )
    memory_manager = MemoryManager(
        memory=long_term_memory,
        llm=llm,
        use_llm_management=True,
    )
    return long_term_memory, memory_manager, storage_handler


async def main():
    # 1. Boot model
    llm_config = OpenAILLMConfig(
        model="gpt-4o-mini",
        openai_key=os.environ["OPENAI_API_KEY"],
        temperature=0.1,
    )
    llm = OpenAILLM(config=llm_config)

    # 2. Initialise memory stack and agent
    _, memory_manager, storage_handler = await bootstrap_memory_stack(llm)

    agent = create_mem0_agent(
        memory_manager=memory_manager,
        name="Mem0CustomizeAgent",
        description="Mem0-backed memory agent demo.",
        llm_config=llm_config,
        llm=llm,
        storage_handler=storage_handler,
        short_term_window=4,
    )

    conversation_id = "mem0-demo-session"
    print("Running mem0 memory demo...\n")

    # Step 1 - Add a memory via natural instruction
    prompt_create = "Please remember that the price of an apple is 1 yuan."
    response_create = await agent(
        inputs={"user_input": prompt_create, "conversation_id": conversation_id},
    )
    print("Step 1 - Add memory")
    print(f"User prompt: {prompt_create}")
    print(f"Agent response: {response_create.content}\n")

    # Capture the memory id for later update/delete operations
    search_results = await memory_manager.handle_memory(
        action="search",
        user_prompt="apple price",
        top_k=1,
    )
    memory_id = search_results[0][1] if search_results else None
    if memory_id:
        print(f"Indexed memory id: {memory_id}\n")

    # Step 2 - Retrieve and inject context
    prompt_query = "What price did we store for apples?"
    response_query = await agent(
        inputs={"user_input": prompt_query, "conversation_id": conversation_id},
    )
    print("Step 2 - Retrieve memory")
    print(f"User prompt: {prompt_query}")
    print(f"Agent response: {response_query.content}\n")

    # Step 3 - Update the stored memory via memory_operations
    if memory_id:
        prompt_update = "Update the memory so the apple price is now 2 yuan."
        response_update = await agent(
            inputs={
                "user_input": prompt_update,
                "conversation_id": conversation_id,
                "memory_operations": {
                    "update": [
                        {
                            "memory_id": memory_id,
                            "message": {
                                "content": "The price of an apple is 2 yuan.",
                                "msg_type": "response",
                            },
                        }
                    ]
                },
            },
        )
        print("Step 3 - Update memory")
        print(f"User prompt: {prompt_update}")
        print(f"Agent response: {response_update.content}\n")

    # Step 4 - Ask again to confirm updated context
    prompt_confirm = "Remind me: how much does an apple cost now?"
    response_confirm = await agent(
        inputs={"user_input": prompt_confirm, "conversation_id": conversation_id},
    )
    print("Step 4 - Confirm update")
    print(f"User prompt: {prompt_confirm}")
    print(f"Agent response: {response_confirm.content}\n")

    # Step 5 - Delete the memory explicitly
    if memory_id:
        prompt_delete = "Remove the stored apple price from memory."
        response_delete = await agent(
            inputs={
                "user_input": prompt_delete,
                "conversation_id": conversation_id,
                "memory_operations": {
                    "delete": [memory_id],
                },
            },
        )
        print("Step 5 - Delete memory")
        print(f"User prompt: {prompt_delete}")
        print(f"Agent response: {response_delete.content}\n")

    # Step 6 - Query once more to show the memory has been removed
    prompt_final = "Do we still know the apple price?"
    response_final = await agent(
        inputs={"user_input": prompt_final, "conversation_id": conversation_id},
    )
    print("Step 6 - Post-delete query")
    print(f"User prompt: {prompt_final}")
    print(f"Agent response: {response_final.content}\n")


if __name__ == "__main__":
    asyncio.run(main())
