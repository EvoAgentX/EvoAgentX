import json
import hashlib
import asyncio
from uuid import uuid4
from typing import Union, List, Dict, Optional, Tuple

from pydantic import Field

from .memory import BaseMemory
from evoagentx.rag import RAGConfig, RAGEngine
from evoagentx.rag.schema import Corpus, Chunk, ChunkMetadata, Query, RagResult
from evoagentx.storages.base import StorageHandler
from evoagentx.core.message import Message
from evoagentx.core.logging import logger

class LongTermMemory(BaseMemory):
    """
    Manages long-term storage and retrieval of memories, integrating with RAGEngine for indexing
    and StorageHandler for persistence.
    """
    storage_handler: StorageHandler = Field(..., description="Handler for persistent storage")
    rag_config: RAGConfig = Field(..., description="Configuration for RAG engine")
    rag_engine: RAGEngine = Field(default=None, description="RAG engine for indexing and retrieval")
    memory_table: str = Field(default="memory", description="Database table for storing memories")
    default_corpus_id: Optional[str] = Field(default=None, description="Default corpus ID for memory indexing")

    def init_module(self):
        """Initialize the RAG engine and memory indices."""
        super().init_module()
        if self.rag_engine is None:
            self.rag_engine = RAGEngine(config=self.rag_config, storage_handler=self.storage_handler)
        if self.default_corpus_id is None:
            self.default_corpus_id = str(uuid4())
        logger.info(f"Initialized LongTermMemory with corpus_id {self.default_corpus_id}")

    def _create_memory_chunk(self, message: Message, memory_id: str) -> Chunk:
        """Convert a Message to a Chunk for RAG indexing."""
        metadata = ChunkMetadata(
            corpus_id=self.default_corpus_id,
            memory_id=memory_id,
            timestamp=message.timestamp,
            action=message.action,
            wf_goal=message.wf_goal,
            agent=message.agent,
            msg_type=message.msg_type.value if message.msg_type else None,
            prompt=message.prompt,
            next_actions=message.next_actions,
            wf_task=message.wf_task,
            wf_task_desc=message.wf_task_desc,
            message_id=message.message_id,
            content=json.dumps(message.content),
        )
        return Chunk(
            chunk_id=memory_id,
            text=str(message.content),
            metadata=metadata,
            start_char_idx=0,
            end_char_idx=len(str(message.content)),
        )

    def _chunk_to_message(self, chunk: Chunk) -> Message:
        """Convert a Chunk to a Message object."""
        return Message(
            content=chunk.metadata.content,
            action=chunk.metadata.action,
            wf_goal=chunk.metadata.wf_goal,
            timestamp=chunk.metadata.timestamp,
            agent=chunk.metadata.agent,
            msg_type=chunk.metadata.msg_type,
            prompt=chunk.metadata.prompt,
            next_actions=chunk.metadata.next_actions,
            wf_task=chunk.metadata.wf_task,
            wf_task_desc=chunk.metadata.wf_task_desc,
            message_id=chunk.metadata.message_id,
        )

    def add(self, messages: Union[Message, str, List[Union[Message, str]]]) -> List[str]:
        """Store messages in memory and index them in RAGEngine, returning memory_ids."""
        if not isinstance(messages, list):
            messages = [messages]
        messages = [Message(content=msg) if isinstance(msg, str) else msg for msg in messages]
        messages = [msg for msg in messages if msg.content]  # Filter out empty messages

        if not messages:
            logger.warning("No valid messages to add")
            return []

        # Hash-based deduplication
        existing_hashes = {
            record["content_hash"]
            for record in self.storage_handler.load(tables=[self.memory_table]).get(self.memory_table, [])
            if "content_hash" in record
        }
        memory_ids = [str(uuid4()) for _ in messages]
        final_messages = []
        final_memory_ids = []
        final_chunks = []

        for msg, memory_id in zip(messages, memory_ids):
            content_hash = hashlib.sha256(str(msg.content).encode()).hexdigest()
            if content_hash in existing_hashes:
                logger.info(f"Duplicate message found (hash): {msg.content[:50]}...")
                existing_id = next(
                    (r["memory_id"] for r in self.storage_handler.load(tables=[self.memory_table]).get(self.memory_table, [])
                     if r.get("content_hash") == content_hash), None
                )
                if existing_id:
                    final_memory_ids.append(existing_id)
                    continue
            final_messages.append(msg)
            final_memory_ids.append(memory_id)
            chunk = self._create_memory_chunk(msg, memory_id)
            chunk.metadata.content_hash = content_hash
            final_chunks.append(chunk)

        if not final_chunks:
            logger.info("No messages added after deduplication")
            return final_memory_ids

        # Add to in-memory messages
        for msg in final_messages:
            super().add_message(msg)

        # Index in RAGEngine
        corpus = Corpus(chunks=final_chunks, corpus_id=self.default_corpus_id)
        chunk_ids = self.rag_engine.add(index_type=self.rag_config.index.index_type, nodes=corpus, corpus_id=self.default_corpus_id)
        if not chunk_ids:
            logger.error("Failed to index memories")
            return final_memory_ids

        return final_memory_ids

    async def get(self, memory_ids: Union[str, List[str]], return_chunk: bool = True) -> List[Tuple[Union[Chunk, Message], str]]:
        """Retrieve memories by memory_ids, returning (Message/Chunk, memory_id) tuples."""
        if not isinstance(memory_ids, list):
            memory_ids = [memory_ids]

        if not memory_ids:
            logger.warning("No memory_ids provided for get")
            return []

        try:
            chunks = await self.rag_engine.aget(
                corpus_id=self.default_corpus_id,
                index_type=self.rag_config.index.index_type,
                node_ids=memory_ids
            )
            results = [(self._chunk_to_message(chunk), chunk.metadata.memory_id) if not return_chunk else (chunk, chunk.metadata.memory_id)
                       for chunk in chunks if chunk]
            logger.info(f"Retrieved {len(results)} memories for memory_ids: {memory_ids}")
            return results
        except Exception as e:
            logger.error(f"Failed to get memories: {str(e)}")
            return []

    def delete(self, memory_ids: Union[str, List[str]]) -> List[bool]:
        """Delete memories by memory_ids, returning success status for each."""
        if not isinstance(memory_ids, list):
            memory_ids = [memory_ids]

        if not memory_ids:
            logger.warning("No memory_ids provided for deletion")
            return []

        successes = [False] * len(memory_ids)
        valid_memory_ids = []

        existing_chunks = asyncio.run(self.get(memory_ids, return_chunk=True))
        for idx, (chunk, mid) in enumerate(existing_chunks):
            if chunk:
                valid_memory_ids.append(mid)
                super().remove_message(self._chunk_to_message(chunk))
                successes[idx] = True

        if not valid_memory_ids:
            logger.info("No memories found for deletion")
            return successes

        # Remove from RAG index
        self.rag_engine.delete(
            corpus_id=self.default_corpus_id,
            index_type=self.rag_config.index.index_type,
            node_ids=valid_memory_ids
        )

        return successes

    def update(self, updates: Union[Tuple[str, Union[Message, str]], List[Tuple[str, Union[Message, str]]]]) -> List[bool]:
        """Update memories with new content, returning success status for each."""
        if not isinstance(updates, list):
            updates = [updates]
        updates = [(mid, Message(content=msg) if isinstance(msg, str) else msg) for mid, msg in updates]
        updates_dict = {mid: msg for mid, msg in updates if msg.content}

        if not updates_dict:
            logger.warning("No valid updates provided")
            return []

        memory_ids = list(updates_dict.keys())
        existing_memories = asyncio.run(self.get(memory_ids, return_chunk=False))
        existing_dict = {mid: msg for msg, mid in existing_memories}

        successes = [False] * len(updates)
        final_updates = []
        final_memory_ids = []

        for mid, msg in updates_dict.items():
            if mid not in existing_dict:
                logger.warning(f"No memory found with memory_id {mid}")
                continue
            final_updates.append((mid, msg))
            final_memory_ids.append(mid)
            successes[memory_ids.index(mid)] = True
            super().remove_message(existing_dict[mid])

        if not final_updates:
            logger.info("No memories updated")
            return successes

        chunks = [self._create_memory_chunk(msg, mid) for mid, msg in final_updates]
        for msg in [msg for _, msg in final_updates]:
            super().add_message(msg)

        corpus = Corpus(chunks=chunks, corpus_id=self.default_corpus_id)
        chunk_ids = self.rag_engine.add(index_type=self.rag_config.index.index_type, nodes=corpus, corpus_id=self.default_corpus_id)
        if not chunk_ids:
            logger.error(f"Failed to update memories in RAG index: {final_memory_ids}")
            return [False] * len(updates)

        return successes

    async def search_async(self, query: Union[str, Query], n: Optional[int] = None,
                          metadata_filters: Optional[Dict] = None, return_chunk=False) -> List[Tuple[Message, str]]:
        """Retrieve messages from RAG index asynchronously based on a query, returning messages and memory_ids."""
        if isinstance(query, str):
            query_obj = Query(
                query_str=query,
                top_k=n or self.rag_config.retrieval.top_k,
                metadata_filters=metadata_filters or {}
            )
        else:
            query_obj = query
            query_obj.top_k = n or self.rag_config.retrieval.top_k
            if metadata_filters:
                query_obj.metadata_filters = {**query_obj.metadata_filters, **metadata_filters} if query_obj.metadata_filters else metadata_filters

        try:
            result: RagResult = await self.rag_engine.query_async(query_obj, corpus_id=self.default_corpus_id)
            if return_chunk:
                return [(chunk, chunk.metadata.memory_id) for chunk in result.corpus.chunks]
            else:
                messages = [(self._chunk_to_message(chunk), chunk.metadata.memory_id) for chunk in result.corpus.chunks]
            logger.info(f"Retrieved {len(messages)} memories for query: {query_obj.query_str}")
            return messages[:n] if n else messages
        except Exception as e:
            logger.error(f"Failed to search memories: {str(e)}")
            return []

    def search(self, query: Union[str, Query], n: Optional[int] = None,
               metadata_filters: Optional[Dict] = None) -> List[Tuple[Message, str]]:
        """Synchronous wrapper for searching memories."""
        return asyncio.run(self.search_async(query, n, metadata_filters))

    def clear(self) -> None:
        """Clear all messages and indices."""
        super().clear()
        self.rag_engine.clear(corpus_id=self.default_corpus_id)
        logger.info(f"Cleared LongTermMemory with corpus_id {self.default_corpus_id}")

    def save(self, save_path: Optional[str] = None) -> None:
        """Save all indices and memory data to database."""
        self.rag_engine.save(output_path=save_path, corpus_id=self.default_corpus_id, table=self.memory_table)

    def load(self, save_path: Optional[str] = None) -> List[str]:
        """Load memory data from database and reconstruct indices, returning memory_ids."""
        return self.rag_engine.load(source=save_path, corpus_id=self.default_corpus_id, table=self.memory_table)