import asyncio
import os
from evoagentx.models import OpenAILLMConfig, OpenAILLM
from evoagentx.storages.base import StorageHandler
from evoagentx.storages.storages_config import VectorStoreConfig, DBConfig, StoreConfig
from evoagentx.rag.rag_config import (
    ReaderConfig, ChunkerConfig, IndexConfig, RetrievalConfig, EmbeddingConfig, RAGConfig
)
from evoagentx.agents.long_term_memory_agent import MemoryAgent


async def main():
    # === 1. Initialize LLM ===
    llm_config = OpenAILLMConfig(
        model="gpt-4o-mini",
        openai_key=os.environ["OPENAI_API_KEY"],
        temperature=0.1
    )
    llm = OpenAILLM(config=llm_config)

    # === 2. Setup Storage and RAG Config ===
    store_config = StoreConfig(
        dbConfig=DBConfig(db_name="sqlite", path="./debug/data/memory_crud_fixed.sql"),
        vectorConfig=VectorStoreConfig(vector_name="faiss", dimensions=768, index_type="flat_l2"),
        graphConfig=None,
        path="./debug/data/memory_crud_fixed_index"
    )
    storage_handler = StorageHandler(storageConfig=store_config)

    embedding = EmbeddingConfig(
        provider="huggingface",
        model_name="BAAI/bge-small-en-v1.5",
        device="cpu"
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
            similarity_cutoff=0.3
        )
    )

    # === 3. Initialize MemoryAgent ===
    agent = MemoryAgent(
        llm=llm,
        llm_config=llm_config,
        rag_config=rag_config,
        storage_handler=storage_handler,
        name="MemoryAgent",
        description="CRUD example for memory agent",
    )

    # === 4. Define structured CRUD prompts ===
    prompts = [
        "Memory instruction: Please remember that the price of an apple is 1 yuan.",       # Create
        "Memory instruction: Update the existing memory about the apple price. The new price of an apple is 2 yuan.",  # Update
        "Memory instruction: Delete the memory about the apple price, because the information is incorrect."          # Delete
    ]

    print("🚀 Starting MemoryAgent CRUD workflow...\n")

    for step, prompt in enumerate(prompts, 1):
        print(f"\n🧠 Step {step}: {prompt}")
        try:
            msg = await agent.async_chat(user_prompt=prompt, top_k=3)
            print(f"🤖 Agent response: {msg.content}\n")
        except Exception as e:
            print(f"❌ Error in step {step}: {e}")

        if hasattr(agent.memory_manager, "handle_memory_flush"):
            await agent.memory_manager.handle_memory_flush()
        else:
            await asyncio.sleep(0.1)

if __name__ == "__main__":
    asyncio.run(main())
