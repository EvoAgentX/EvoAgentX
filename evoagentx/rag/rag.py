import os
import json
import asyncio
from uuid import uuid4
from datetime import datetime
from typing import List, Union, Optional, Sequence, Dict, Any, Tuple

from llama_index.core.schema import NodeWithScore, TextNode, RelatedNodeInfo

from .rag_config import RAGConfig
from .readers import LLamaIndexReader
from .indexings import IndexFactory, BaseIndexWrapper
from .chunkers import ChunkFactory
from .embeddings import EmbeddingFactory, EmbeddingProvider
from .retrievers import RetrieverFactory, BaseRetrieverWrapper
from .postprocessors import PostprocessorFactory
from .indexings.base import IndexType
from .retrievers.base import RetrieverType
from .schema import Chunk, Corpus, ChunkMetadata, IndexMetadata, Query, RagResult
from evoagentx.storages.base import StorageHandler
from evoagentx.storages.schema import IndexStore
from evoagentx.models.base_model import BaseLLM
from evoagentx.core.logging import logger


class RAGEngine:
    def __init__(self, config: RAGConfig, storage_handler: StorageHandler, llm: Optional[BaseLLM] = None):
        self.config = config
        self.storage_handler = storage_handler  # Maybe reinit the vector_store by the load funcion.
        self.embedding_factory = EmbeddingFactory()
        self.index_factory = IndexFactory()
        self.chunk_factory = ChunkFactory()
        self.retriever_factory = RetrieverFactory()
        self.postprocessor_factory = PostprocessorFactory()
        self.llm = llm  # LLM for entity extractor

        # Initialize reader
        self.reader = LLamaIndexReader(
            recursive=self.config.reader.recursive,
            exclude_hidden=self.config.reader.exclude_hidden,
            num_workers=self.config.num_workers,
            num_files_limits=self.config.reader.num_files_limit,
            custom_metadata_function=self.config.reader.custom_metadata_function,
            extern_file_extractor=self.config.reader.extern_file_extractor,
            errors=self.config.reader.errors,
            encoding=self.config.reader.encoding
        )

        # Initialize embedding model. 
        self.embed_model = self.embedding_factory.create(
            provider=self.config.embedding.provider,
            model_config=self.config.embedding.model_dump(exclude_unset=True),
        )

        # Dynamic Check the dimensions in StorageHandler
        if (self.storage_handler.vector_store is not None) and (self.embed_model.dimensions is not None):
            if self.storage_handler.storageConfig.vectorConfig.dimensions != self.embed_model.dimensions:
                logger.warning("The dimensions in vector_store is not equal with embed_model. Reiniliaze vector_store.")
                self.storage_handler.storageConfig.vectorConfig.dimensions = self.embed_model.dimensions
                self.storage_handler._init_vector_store()

        # Initialize chunker
        self.chunker = self.chunk_factory.create(
            strategy=self.config.chunker.strategy,
            embed_model=self.embed_model.get_embedding_model(),
            chunker_config={
                "chunk_size": self.config.chunker.chunk_size,
                "chunk_overlap": self.config.chunker.chunk_overlap,
                "max_chunks": self.config.chunker.max_chunks
            }
        )

        # Initialize indices and retrievers
        self.indices: Dict[str, Dict[str, BaseIndexWrapper]] = {}  # Nested: {corpus_id: {index_type: index}}
        self.retrievers: Dict[str, Dict[str, BaseRetrieverWrapper]] = {}  # Nested: {corpus_id: {index_type: retriever}}

    def read(self, file_paths: Union[Sequence[str], str], 
             exclude_files: Optional[Union[str, List, Tuple, Sequence]] = None,
             filter_file_by_suffix: Optional[Union[str, List, Tuple, Sequence]] = None,
             merge_by_file: bool = False,
             show_progress: bool = False,
             corpus_id: str = None) -> Corpus:
        """Load and chunk documents from files.

        Reads files from specified paths, processes them into documents, and chunks them into a Corpus.

        Args:
            file_paths (Union[Sequence[str], str]): Path(s) to files or directories.
            exclude_files (Optional[Union[str, List, Tuple, Sequence]]): Files to exclude.
            filter_file_by_suffix (Optional[Union[str, List, Tuple, Sequence]]): Filter files by suffix (e.g., '.pdf').
            merge_by_file (bool): Merge documents by file.
            show_progress (bool): Show loading progress.
            corpus_id (Optional[str]): Identifier for the corpus. Defaults to a UUID if None.

        Returns:
            Corpus: The chunked corpus containing processed document chunks.

        Raises:
            Exception: If document reading or chunking fails.
        """
        try:
            corpus_id = corpus_id or str(uuid4())
            documents = self.reader.load(
                file_paths=file_paths,
                exclude_files=exclude_files,
                filter_file_by_suffix=filter_file_by_suffix,
                merge_by_file=merge_by_file,
                show_progress=show_progress
            )
            corpus = self.chunker.chunk(documents)
            corpus.corpus_id = corpus_id
            logger.info(f"Read {len(documents)} documents and created {len(corpus.chunks)} chunks for corpus {corpus_id}")
            return corpus
        except Exception as e:
            logger.error(f"Failed to read documents for corpus {corpus_id}: {str(e)}")
            raise

    def add(self, index_type: str, nodes: Union[Corpus, List[NodeWithScore], List[TextNode]], 
            corpus_id: str = None) -> Sequence[str]:
        """Add nodes to an index for a specific corpus.

        Initializes an index if it doesn't exist and inserts nodes, updating metadata with corpus_id and index_type.

        Args:
            index_type (str): Type of index (e.g., VECTOR, GRAPH).
            nodes (Union[Corpus, List[NodeWithScore], List[TextNode]]): Nodes or Corpus to add.
            corpus_id (str, optional): Identifier for the corpus. Defaults to a UUID if None.

        Return:
            return a sequence with id of each added node.
            
        Raises:
            Exception: If index creation or node insertion fails.
        """
        try:
            corpus_id = corpus_id or str(uuid4())
            if corpus_id not in self.indices:
                self.indices[corpus_id] = {}
                self.retrievers[corpus_id] = {}

            if index_type not in self.indices[corpus_id]:
                index = self.index_factory.create(
                    index_type=index_type,
                    embed_model=self.embed_model.get_embedding_model(),
                    storage_handler=self.storage_handler,
                    index_config=self.config.index.model_dump(exclude_unset=True) if self.config.index else {},
                    llm=self.llm,
                )
                self.indices[corpus_id][index_type] = index
                self.retrievers[corpus_id][index_type] = self.retriever_factory.create(
                    retriever_type=index_type,
                    llm=self.llm,
                    index=index.get_index(),
                    graph_store=index.get_index().storage_context.graph_store,
                    embed_model=self.embed_model.get_embedding_model(),
                    query=Query(query_str="", top_k=self.config.retrieval.top_k if self.config.retrieval else 5),
                    storage_handler=self.storage_handler,
                )

            nodes_to_insert = nodes.to_llama_nodes() if isinstance(nodes, Corpus) else nodes
            for node in nodes_to_insert:
                node.metadata.update({"corpus_id": corpus_id, "index_type": index_type})
            nodes_ids = self.indices[corpus_id][index_type].insert_nodes(nodes_to_insert)
            logger.info(f"Added {len(nodes_to_insert)} nodes to {index_type} index for corpus {corpus_id}")
            return nodes_ids
        except Exception as e:
            logger.error(f"Failed to add nodes to {index_type} index for corpus {corpus_id}: {str(e)}")
            return []

    def delete(self, corpus_id: str, index_type: Optional[str] = None, 
               node_ids: Optional[Union[str, List[str]]] = None, 
               metadata_filters: Optional[Dict[str, Any]] = None) -> None:
        """Delete nodes or an entire index from a corpus.

        Removes specific nodes by ID or metadata filters, or deletes the entire index if no filters are provided.

        Args:
            corpus_id (str): Identifier for the corpus.
            index_type (Optional[IndexType]): Specific index type to delete from. If None, affects all indices.
            node_ids (Union[str, Optional[List[str]]]): List of node IDs to delete.
            metadata_filters (Optional[Dict[str, Any]]): Metadata filters to select nodes for deletion.

        Raises:
            Exception: If deletion fails.
        """
        try:
            if corpus_id not in self.indices:
                logger.warning(f"No indices found for corpus {corpus_id}")
                return

            target_indices = [index_type] if index_type else self.indices[corpus_id].keys()
            for idx_type in list(target_indices):  # Use list to avoid runtime modification issues
                if idx_type not in self.indices[corpus_id]:
                    logger.warning(f"Index type {idx_type} not found for corpus {corpus_id}")
                    continue

                index = self.indices[corpus_id][idx_type]
                if node_ids or metadata_filters:
                    # Convert single node_id to list for consistency
                    node_ids_list = [node_ids] if isinstance(node_ids, str) else node_ids
                    index.delete_nodes(node_ids=node_ids_list, metadata_filters=metadata_filters)
                    logger.info(f"Deleted nodes from {idx_type} index for corpus {corpus_id}")
                else:
                    # Delete entire index
                    index.clear()
                    del self.indices[corpus_id][idx_type]
                    del self.retrievers[corpus_id][idx_type]
                    logger.info(f"Deleted entire {idx_type} index for corpus {corpus_id}")

            # Clean up corpus if no indices remain
            if not self.indices[corpus_id]:
                del self.indices[corpus_id]
                del self.retrievers[corpus_id]
                logger.info(f"Removed empty corpus {corpus_id}")

        except Exception as e:
            logger.error(f"Failed to delete from corpus {corpus_id}, index {index_type}: {str(e)}")
            raise

    def clear(self, corpus_id: Optional[str] = None) -> None:
        """Clear all indices for a specific corpus or all corpora.

        Args:
            corpus_id (Optional[str]): Specific corpus to clear. If None, clears all corpora.

        Raises:
            Exception: If clearing fails.
        """
        try:
            target_corpora = [corpus_id] if corpus_id else list(self.indices.keys())
            for cid in target_corpora:
                if cid not in self.indices:
                    logger.warning(f"No indices found for corpus {cid}")
                    continue

                for idx_type in list(self.indices[cid].keys()):
                    index = self.indices[cid][idx_type]
                    index.clear()
                    del self.indices[cid][idx_type]
                    del self.retrievers[cid][idx_type]
                    logger.info(f"Cleared {idx_type} index for corpus {cid}")

                # Clean up corpus if no indices remain
                del self.indices[cid]
                del self.retrievers[cid]
                logger.info(f"Cleared corpus {cid}")

        except Exception as e:
            logger.error(f"Failed to clear indices for corpus {corpus_id or 'all'}: {str(e)}")
            raise

    def save(self, output_path: Optional[str] = None, corpus_id: Optional[str] = None, 
                index_type: Optional[str] = None, table: Optional[str] = None, 
                graph_exported: bool = False) -> None:
        """Save indices to files or database.

        Serializes corpus chunks to JSONL files and metadata to JSON files if output_path is provided,
        or saves to the SQLite database via StorageHandler if output_path is None.

        Args:
            output_path (Optional[str]): Directory to save JSONL and JSON files. If None, saves to database.
            corpus_id (Optional[str]): Specific corpus to save. If None, saves all corpora.
            index_type (Optional[str]): Specific index type to save. If None, saves all indices.
            table (Optional[str]): Database table name for index data. Defaults to 'indexing' if None.
            graph_exported (bool): If True, export graph nodes and relations for graph indices. Defaults to False.

        Raises:
            Exception: If saving fails or file operations encounter errors.
        """
        try:
            target_corpora = [corpus_id] if corpus_id else list(self.indices.keys())
            table = table or "indexing"

            for cid in target_corpora:
                if cid not in self.indices:
                    logger.warning(f"No indices found for corpus {cid}")
                    continue

                target_indices = [index_type] if index_type and index_type in self.indices[cid] else self.indices[cid].keys()
                for idx_type in target_indices:
                    index = self.indices[cid][idx_type]

                    # Skip saving for graph indices unless graph_exported is True
                    if idx_type == IndexType.GRAPH and not graph_exported:
                        logger.warning(f"Skipping save for graph index {idx_type} in corpus {cid} as graph_exported is False")
                        continue

                    # For graph indices, include kg_nodes and kg_rels if graph_exported is True
                    if idx_type == IndexType.GRAPH and graph_exported:
                        index.build_kv_store()  # FIXME: Computer's memory may increase dramatically

                    # Convert index nodes to Corpus
                    chunks = [
                        Chunk.from_llama_node(node_data)
                        for node_id, node_data in index.id_to_node.items()
                    ]
                    corpus = Corpus(chunks=chunks, corpus_id=cid)

                    # Prepare metadata
                    vector_config = self.storage_handler.storageConfig.vectorConfig.model_dump() if self.storage_handler.storageConfig.vectorConfig else {}
                    graph_config = self.storage_handler.storageConfig.graphConfig.model_dump() if self.storage_handler.storageConfig.graphConfig else {}
                    metadata = IndexMetadata(
                        corpus_id=cid,
                        index_type=idx_type,
                        collection_name=vector_config.get("qdrant_collection_name", "default_collection"),
                        dimension=self.embed_model.dimensions,
                        vector_db_type=vector_config.get("vector_name", None),
                        graph_db_type=graph_config.get("graph_name", None),
                        embedding_model_name=self.config.embedding.model_name,
                        date=str(datetime.now()),
                    )

                    if output_path:
                        # File-based saving
                        os.makedirs(output_path, exist_ok=True)
                        safe_cid = "".join(c if c.isalnum() or c in ["-", "_"] else "_" for c in cid)
                        safe_idx_type = "".join(c if c.isalnum() or c in ["-", "_"] else "_" for c in idx_type)
                        nodes_file = os.path.join(output_path, f"{safe_cid}_{safe_idx_type}_nodes.jsonl")
                        metadata_file = os.path.join(output_path, f"{safe_cid}_{safe_idx_type}_metadata.json")

                        # Save corpus as JSONL
                        corpus.to_jsonl(nodes_file, indent=0)
                        logger.info(f"Saved {len(corpus.chunks)} chunks to {nodes_file}")

                        # Save metadata as JSON
                        with open(metadata_file, "w", encoding="utf-8") as f:
                            json.dump(metadata.model_dump(), f, indent=2, ensure_ascii=False)
                        logger.info(f"Saved metadata to {metadata_file}")
                    else:
                        # Database saving
                        index_data = {
                            "corpus_id": cid,
                            "content": corpus.model_dump(),
                            "date": str(datetime.now()),
                            "metadata": metadata.model_dump()
                        }
                        self.storage_handler.save_index(index_data, table=table)
                        logger.info(f"Saved {idx_type} index with {len(corpus.chunks)} chunks for corpus {cid} to database table {table}")

        except Exception as e:
            logger.error(f"Failed to save indices for corpus {corpus_id or 'all'}: {str(e)}")
            raise

    def load(self, source: Optional[str] = None, corpus_id: Optional[str] = None, 
             index_type: Optional[str] = None, table: Optional[str] = None) -> Sequence[str]:
        """Load indices from files or database.

        Reconstructs indices and retrievers from JSONL/JSON files or SQLite database records.
        Validates the embedding model name and dimension before reinitializing the embedding model.

        Args:
            source (Optional[str]): Directory containing JSONL/JSON files. If None, loads from database.
            corpus_id (Optional[str]): Specific corpus to load. If None, loads all corpora.
            index_type (Optional[str]): Specific index type to load. If None, loads all indices.
            table (Optional[str]): Database table name for index data. Defaults to 'indexing' if None.

        Returns:
            The Sequence with id of loaded chunk.
        
        Raises:
            Exception: If loading fails due to file or database errors, invalid data, or unsupported embedding model/dimension.
        
        Warning:
            Try to call this function may cause some Bugs, when you load the nodes from file or database storage systems at twice. 
            Because All the indexing share the same storage backend from storageHandler.
            For example:
            The vector database (.e.g Faiss) can insert again, even thougt there is a same node.
        """
        try:
            table = table or "indexing"
            config_dimension = self.storage_handler.storageConfig.vectorConfig.dimensions
            loaded_chunk_ids: List[str] = []

            if source:
                # File-based loading
                if not os.path.exists(source):
                    logger.error(f"Source directory {source} does not exist")
                    raise FileNotFoundError(f"Source directory {source} does not exist")

                for file_name in os.listdir(source):
                    if not file_name.endswith("_metadata.json"):
                        continue
                    parts = file_name.split("_")
                    if len(parts) < 3:
                        logger.warning(f"Skipping invalid metadata file: {file_name}")
                        continue
                    cid = "_".join(parts[:-2])
                    idx_type = parts[-2]
                    
                    if (corpus_id and corpus_id != cid) or (index_type and index_type != idx_type):
                        continue

                    metadata_file = os.path.join(source, file_name)
                    nodes_file = os.path.join(source, f"{cid}_{idx_type}_nodes.jsonl")

                    # Load metadata
                    with open(metadata_file, "r", encoding="utf-8") as f:
                        metadata = IndexMetadata.model_validate(json.load(f))

                    # Validate embedding model
                    if not self.embed_model.validate_model(self.config.embedding.provider, metadata.embedding_model_name):
                        raise ValueError(
                            f"Embedding model '{metadata.embedding_model_name}' is not supported by provider '{self.config.embedding.provider}'. "
                            f"Supported models: {EmbeddingProvider.SUPPORTED_MODELS.get(self.config.embedding.provider, [])}"
                        )

                    # Validate dimension
                    if metadata.dimension != config_dimension:
                        raise ValueError(
                            f"Embedding dimension {metadata.dimension} in metadata does not match configured dimension {config_dimension}."
                        )

                    # Load corpus
                    if not os.path.exists(nodes_file):
                        logger.warning(f"Nodes file {nodes_file} not found for metadata {metadata_file}")
                        continue
                    corpus = Corpus.from_jsonl(nodes_file, corpus_id=cid)

                    # Reinitialize embedding model if needed
                    if metadata.embedding_model_name != self.config.embedding.model_name:
                        logger.info(f"Reinitializing embedding model to {metadata.embedding_model_name}")
                        self.embed_model = self.embedding_factory.create(
                            provider=self.config.embedding.provider,
                            model_config=self.config.embedding.model_dump(exclude_unset=True)
                        )

                    # Load index
                    chunk_ids = self._load_index(corpus, cid, idx_type)
                    loaded_chunk_ids.extend(chunk_ids)
                    logger.info(f"Loaded {idx_type} index with {len(corpus.chunks)} chunks for corpus {cid} from {nodes_file}")
            else:
                # Database loading
                records = self.storage_handler.load(tables=[table]).get(table, [])

                if not records:
                    logger.warning(f"No records found in table {table}")
                    return

                for record in records:
                    parsed = self.storage_handler.parse_result(record, IndexStore)
                    cid = parsed["corpus_id"]
                    idx_type = parsed["metadata"]["index_type"]
                    if (corpus_id and corpus_id != cid) or (index_type and index_type != idx_type):
                        continue

                    # Reconstruct corpus
                    corpus = Corpus(
                        chunks=[
                            Chunk(
                                chunk_id=chunk["chunk_id"],
                                text=chunk["text"],
                                metadata=ChunkMetadata.model_validate(chunk["metadata"]),
                                embedding=chunk["embedding"],
                                start_char_idx=chunk["start_char_idx"],
                                end_char_idx=chunk["end_char_idx"],
                                excluded_embed_metadata_keys=chunk["excluded_embed_metadata_keys"],
                                excluded_llm_metadata_keys=chunk["excluded_llm_metadata_keys"],
                                relationships={k: RelatedNodeInfo(**v) for k, v in chunk["relationships"].items()}
                            ) for chunk in parsed["content"]["chunks"]
                        ],
                        corpus_id=cid,
                        metadata=IndexMetadata.model_validate(parsed["metadata"])
                    )

                    # Validate embedding model
                    metadata = IndexMetadata.model_validate(parsed["metadata"])
                    if not self.embed_model.validate_model(self.config.embedding.provider, metadata.embedding_model_name):
                        raise ValueError(
                            f"Embedding model '{metadata.embedding_model_name}' is not supported by provider '{self.config.embedding.provider}'. "
                            f"Supported models: {EmbeddingProvider.SUPPORTED_MODELS.get(self.config.embedding.provider, [])}"
                        )

                    # Validate dimension
                    if metadata.dimension != config_dimension:
                        raise ValueError(
                            f"Embedding dimension {metadata.dimension} in metadata does not match configured dimension {config_dimension}."
                        )

                    # Reinitialize embedding model if needed
                    if metadata.embedding_model_name != self.config.embedding.model_name:
                        logger.info(f"Reinitializing embedding model to {metadata.embedding_model_name}")
                        self.embed_model = self.embedding_factory.create(
                            provider=self.config.embedding.provider,
                            model_config=self.config.embedding.model_dump(exclude_unset=True)
                        )

                    # Load index
                    chunk_ids = self._load_index(corpus, cid, idx_type)
                    loaded_chunk_ids.extend(chunk_ids)
                    logger.info(f"Loaded {idx_type} index with {len(corpus.chunks)} chunks for corpus {cid} from database table {table}")
            
            return loaded_chunk_ids
        except Exception as e:
            logger.error(f"Failed to load indices: {str(e)}")
            raise

    def _load_index(self, corpus: Corpus, corpus_id: str, index_type: str) -> Sequence[str]:
        """Helper method to load an index and its retriever."""
        try:
            if corpus_id not in self.indices:
                self.indices[corpus_id] = {}
                self.retrievers[corpus_id] = {}

            if index_type not in self.indices[corpus_id]:
                index = self.index_factory.create(
                    index_type=index_type,
                    embed_model=self.embed_model.get_embedding_model(),
                    storage_handler=self.storage_handler,
                    index_config=self.config.index.model_dump(exclude_unset=True) if self.config.index else {},
                    llm=self.llm
                )
                self.indices[corpus_id][index_type] = index
                
                retriever_type = RetrieverType.GRAPH if index_type == IndexType.GRAPH else RetrieverType.VECTOR
                self.retrievers[corpus_id][index_type] = self.retriever_factory.create(
                    retriever_type=retriever_type,
                    llm=self.llm,
                    index=index.get_index(),
                    graph_store=index.get_index().storage_context.graph_store,
                    embed_model=self.embed_model.get_embedding_model(),
                    query=Query(query_str="", top_k=self.config.retrieval.top_k if self.config.retrieval else 5),
                    storage_handler=self.storage_handler
                )

            nodes = corpus.to_llama_nodes()
            chunk_ids = self.indices[corpus_id][index_type].load(nodes)
            logger.info(f"Inserted {len(nodes)} nodes into {index_type} index for corpus {corpus_id}")
            return chunk_ids
        except Exception as e:
            logger.error(f"Failed to load index for corpus {corpus_id}, index_type {index_type}: {str(e)}")
            raise

    async def aget(self, corpus_id: str, index_type: str, node_ids: List[str]) -> List[Chunk]:
        """Retrieve chunks by node_ids from the index."""
        try:
            chunks = await self.indices[corpus_id][index_type].get(node_ids=node_ids)
            logger.info(f"Retrieved {len(chunks)} chunks for node_ids: {node_ids}")
            return chunks
        except Exception as e:
            logger.error(f"Failed to get chunks: {str(e)}")
            return []

    async def query_async(self, query: Union[str, Query], corpus_id: Optional[str] = None,
                        query_transforms: Optional[List] = None) -> RagResult:
        """Execute a query across indices and return processed results asynchronously.

        Performs query preprocessing, asynchronous retrieval, and post-processing.

        Args:
            query (Union[str, Query]): Query string or Query object.
            corpus_id (Optional[str]): Specific corpus to query. If None, queries all corpora.
            query_transforms (Optional[List]): Query Transforms is used to augment query in pre-processing.

        Returns:
            RagResult: Retrieved chunks with scores and metadata.

        Raises:
            Exception: If query processing fails.
        """
        try:
            if isinstance(query, str):
                query = Query(query_str=query, top_k=self.config.retrieval.top_k)
            
            if not self.indices or (corpus_id and corpus_id not in self.indices):
                logger.warning(f"No indices found for corpus {corpus_id or 'any'}")
                return RagResult(corpus=Corpus(chunks=[]), scores=[], metadata={"query": query.query_str})

            # Pre-Processing
            if query_transforms and query_transforms is not None:
                for t in query_transforms:
                    query = t(query)

            results = []
            target_corpora = [corpus_id] if corpus_id else self.indices.keys()

            # Create all retrieval tasks
            tasks = []
            for cid in target_corpora:
                for idx_type, retriever in self.retrievers[cid].items():
                    if query.metadata_filters and query.metadata_filters.get("index_type") and \
                    query.metadata_filters["index_type"] != idx_type:
                        continue
                    
                    task = retriever.aretrieve(
                        Query(
                            query_str=query.query_str,
                            top_k=query.top_k or self.config.retrieval.top_k,
                            similarity_cutoff=query.similarity_cutoff,
                            keyword_filters=query.keyword_filters,
                            metadata_filters=query.metadata_filters
                        )
                    )
                    tasks.append((task, cid, idx_type))

            # Run all retrievals concurrently
            retrieval_tasks = [task for task, _, _ in tasks]
            retrieval_results = await asyncio.gather(*retrieval_tasks, return_exceptions=True)

            # Process results
            for (_, cid, idx_type), result in zip(tasks, retrieval_results):
                if isinstance(result, Exception):
                    logger.error(f"Retrieval failed for {idx_type} in corpus {cid}: {str(result)}")
                else:
                    results.append(result)
                    logger.info(f"Retrieved {len(result.corpus.chunks)} chunks from {idx_type} retriever for corpus {cid}")

            if not results:
                return RagResult(corpus=Corpus(chunks=[]), scores=[], metadata={"query": query.query_str})

            # Check the 'similarity_cutoff' and 'keyword_filters' in query
            query.similarity_cutoff = self.config.retrieval.similarity_cutoff if query.similarity_cutoff is None else query.similarity_cutoff
            query.keyword_filters = self.config.retrieval.keyword_filters if query.keyword_filters is None else query.keyword_filters

            postprocessor = self.postprocessor_factory.create(
                self.config.retrieval.postprocessor_type,
                query=query
            )
            final_result = postprocessor.postprocess(query, results)

            if query.metadata_filters:
                final_result.corpus.chunks = [
                    chunk for chunk in final_result.corpus.chunks
                    if all(chunk.metadata.model_dump().get(k) == v for k, v in query.metadata_filters.items())
                ]
                final_result.scores = [chunk.metadata.similarity_score for chunk in final_result.corpus.chunks]
                logger.info(f"Applied metadata filters, retained {len(final_result.corpus.chunks)} chunks")

            logger.info(f"Query returned {len(final_result.corpus.chunks)} chunks after post-processing")
            return final_result
        except Exception as e:
            logger.error(f"Query failed: {str(e)}")
            raise

    def query(self, query: Union[str, Query], corpus_id: Optional[str] = None,
            query_transforms: Optional[List] = None) -> RagResult:
        """Synchronous wrapper for the async query method."""
        return asyncio.run(self.query_async(query, corpus_id, query_transforms))