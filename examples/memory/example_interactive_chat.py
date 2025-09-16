import asyncio
import os
import pytest
from evoagentx.models import OpenAILLMConfig, OpenAILLM
from evoagentx.storages.base import StorageHandler
from evoagentx.storages.storages_config import VectorStoreConfig, DBConfig, StoreConfig
from evoagentx.rag.rag_config import (
    ReaderConfig, ChunkerConfig, IndexConfig, RetrievalConfig, EmbeddingConfig, RAGConfig
)
from evoagentx.agents.long_term_memory_agent import MemoryAgent

async def main():
    # Initialize the LLM
    llm_config = OpenAILLMConfig(
        model="gpt-4o-mini",
        openai_key=os.environ["OPENAI_API_KEY"],
        temperature=0.1
    )
    llm = OpenAILLM(config=llm_config)

    # Storage and RAG configuration
    store_config = StoreConfig(
        dbConfig=DBConfig(db_name="sqlite", path="./debug/data/memory_interactive.sql"),
        vectorConfig=VectorStoreConfig(vector_name="faiss", dimensions=768, index_type="flat_l2"),
        graphConfig=None,
        path="./debug/data/memory_interactive_index"
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
        retrieval=RetrievalConfig(retrivel_type="vector", postprocessor_type="simple", top_k=3, similarity_cutoff=0.3)
    )

    # Initialize MemoryAgent
    agent = MemoryAgent(
        llm=llm,
        rag_config=rag_config,
        storage_handler=storage_handler,
        name="MemoryAgent",
        description="Interactive memory chat test",
    )

    # Start interactive_chat
    await agent.interactive_chat(top_k=3)

if __name__ == "__main__":
    asyncio.run(main())
