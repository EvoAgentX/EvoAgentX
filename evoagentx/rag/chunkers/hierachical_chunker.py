from typing import List, Optional, Dict
from concurrent.futures import ThreadPoolExecutor

from llama_index.core.node_parser import HierarchicalNodeParser

from .base import BaseChunker, ChunkingStrategy
from .simple_chunker import SimpleChunker
from evoagentx.core.logging import logger
from evoagentx.rag.schema import Document, Corpus, Chunk


class HierarchicalChunker(BaseChunker):
    """Enhanced hierarchical chunker with dynamic hierarchy level assignment.

    Creates a multi-level hierarchy of chunks with full node relationships:
    - SOURCE: The source document.
    - PREVIOUS/NEXT: Sequential nodes in the document.
    - PARENT/CHILD: Hierarchical relationships.

    Supports custom level parsers or default chunk sizes, with dynamic hierarchy level
    assignment based on node parser IDs. Uses multi-threading and async parsing.

    Attributes:
        level_parsers (Dict[str, BaseChunker]): Custom parsers for each hierarchy level.
        chunk_sizes (List[int]): Chunk sizes for default parsers (e.g., [2048, 512, 128]).
        chunk_overlap (int): Overlap between adjacent chunks.
        parser (HierarchicalNodeParser): LlamaIndex parser for hierarchical chunking.
        include_metadata (bool): Whether to include metadata in nodes.
        include_prev_next_rel (bool): Whether to include previous/next node relationships.
        max_workers (int): Maximum number of threads for parallel processing.
        parser_to_level (Dict[str, int]): Mapping of node_parser_id to hierarchy level.
    """

    def __init__(
        self,
        level_parsers: Dict[str, BaseChunker] = None,
        chunk_sizes: Optional[List[int]] = None,
        chunk_overlap: int = 20,
        include_metadata: bool = True,
        include_prev_next_rel: bool = True,
        max_workers: int = 4,
    ):
        """Initialize the HierarchicalChunker.

        Args:
            level_parsers (Dict[str, BaseChunker], optional): Custom parsers for hierarchy levels.
            chunk_sizes (List[int], optional): Chunk sizes for default parsers (default: [2048, 512, 128]).
            chunk_overlap (int): Overlap between adjacent chunks (default: 20).
            include_metadata (bool): Include metadata in nodes (default: True).
            include_prev_next_rel (bool): Include prev/next relationships (default: True).
            max_workers (int): Maximum number of threads for parallel processing (default: 4).
        """
        self.level_parsers = level_parsers or {}
        self.chunk_sizes = chunk_sizes or [2048, 512, 128]
        self.chunk_overlap = chunk_overlap
        self.include_metadata = include_metadata
        self.include_prev_next_rel = include_prev_next_rel
        self.max_workers = max_workers

        node_parser_ids = None
        node_parser_map = None

        if not self.level_parsers:
            # Default to SimpleChunker for each chunk size
            node_parser_ids = [f"chunk_size_{size}" for size in self.chunk_sizes]
            node_parser_map = {
                node_id: SimpleChunker(
                    chunk_size=size,
                    chunk_overlap=chunk_overlap,
                    include_metadata=include_metadata,
                    include_prev_next_rel=include_prev_next_rel,
                ).parser
                for size, node_id in zip(self.chunk_sizes, node_parser_ids)
            }
        else:
            if chunk_sizes is not None:
                raise ValueError("If level_parsers is provided, chunk_sizes should be None.")
            node_parser_ids = list(self.level_parsers.keys())
            node_parser_map = {k: v.parser for k, v in self.level_parsers.items()}

        # Map node_parser_id to hierarchy level (1-based)
        self.parser_to_level = {pid: idx + 1 for idx, pid in enumerate(node_parser_ids)}

        self.parser = HierarchicalNodeParser.from_defaults(
            chunk_sizes=None,
            chunk_overlap=self.chunk_overlap,
            node_parser_ids=node_parser_ids,
            node_parser_map=node_parser_map,
            include_metadata=include_metadata,
            include_prev_next_rel=include_prev_next_rel,
        )

    def _process_document(self, doc: Document, custom_metadata: Dict = None) -> List[Chunk]:
        """Process a single document into chunks in a thread.

        Args:
            doc (Document): The document to chunk.
            custom_metadata (Dict, optional): User-defined metadata for sections.

        Returns:
            List[Chunk]: List of Chunk objects with metadata.
        """
        try:
            llama_doc = doc.to_llama_document()
            llama_doc.metadata["doc_id"] = doc.doc_id

            nodes = self.parser.get_nodes_from_documents([llama_doc])

            chunks = []

            for i, node in enumerate(nodes):
                chunk = Chunk.from_llama_node(node)
                
                chunk.metadata.chunking_strategy = ChunkingStrategy.HIERARCHICAL
                chunks.extend([chunk])
                
            logger.debug(f"Processed document {doc.doc_id} into {len(chunks)} chunks")
            return chunks
        except Exception as e:
            logger.error(f"Failed to process document {doc.doc_id}: {str(e)}")
            return []

    def chunk(self, documents: List[Document], **kwargs) -> Corpus:
        """Chunk documents using hierarchical strategy with dynamic chunk size adjustment.

        Args:
            documents (List[Document]): List of Document objects to chunk.
            **kwargs: Additional parameters, e.g., custom_metadata for section titles.

        Returns:
            Corpus: A collection of hierarchically organized chunks.
        """
        if not documents:
            logger.info("No documents provided, returning empty Corpus")
            return Corpus(chunks=[])

        chunks = []
        custom_metadata = kwargs.get("custom_metadata", {})
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_doc = {
                executor.submit(self._process_document, doc, custom_metadata): doc
                for doc in documents
            }
            for future in future_to_doc:
                doc = future_to_doc[future]
                try:
                    chunks.extend(future.result())
                except Exception as e:
                    logger.error(f"Error processing document {doc.doc_id}: {str(e)}")

        logger.info(f"Chunked {len(documents)} documents into {len(chunks)} chunks")
        return Corpus(chunks=chunks)