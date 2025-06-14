import asyncio
from typing import List
from concurrent.futures import ThreadPoolExecutor

from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.node_parser import SemanticSplitterNodeParser

from evoagentx.core.logging import logger
from .base import BaseChunker, ChunkingStrategy
from evoagentx.rag.schema import Document, Corpus, Chunk

class SemanticChunker(BaseChunker):
    """Chunker that splits documents based on semantic similarity.

    Uses LlamaIndex's SemanticChunker with an embedding model to create chunks that preserve
    semantic coherence, ideal for improving retrieval accuracy in RAG pipelines.

    Attributes:
        embed_model (BaseEmbedding): The embedding model for semantic similarity.
        parser (SemanticChunker): The LlamaIndex parser for semantic chunking.
    """

    def __init__(self, embed_model: BaseEmbedding, similarity_threshold: float = 0.7, max_workers=4, **kwargs):
        """Initialize the SemanticChunker.

        Args:
            embed_model_name (BaseEmbedding): the embedding model.
            similarity_threshold (float, optional): Threshold for semantic similarity to split chunks (default: 0.7).
        """
        self.embed_model = embed_model
        self.parser = SemanticSplitterNodeParser(
            embed_model=self.embed_model,
            similarity_threshold=similarity_threshold
        )
        self.max_workers = max_workers

    def _process_document(self, doc: Document) -> List[Chunk]:
        """Process a single document into chunks.

        Args:
            doc (Document): The document to chunk.

        Returns:
            List[Chunk]: List of Chunk objects with metadata.
        """
        try:
            llama_doc = doc.to_llama_document()
            llama_doc.metadata["doc_id"] = doc.doc_id

            nodes = asyncio.run(self.parser.aget_nodes_from_documents([llama_doc]))

            # Convert nodes to Chunks
            chunks = []
            for idx, node in enumerate(nodes):
                chunk = Chunk.from_llama_node(node)
                
                chunk.metadata.chunking_strategy = ChunkingStrategy.SIMPLE
                chunks.extend([chunk])
            logger.debug(f"Processed document {doc.doc_id} into {len(chunks)} chunks")
            return chunks
        except Exception as e:
            logger.error(f"Failed to process document {doc.doc_id}: {str(e)}")
            return []

    def chunk(self, documents: List[Document], **kwargs) -> Corpus:
        """Chunk documents based on semantic similarity.

        Args:
            documents (List[Document]): List of Document objects to chunk.
            **kwargs: Additional parameters (e.g., max_chunk_size).

        Returns:
            Corpus: A collection of Chunk objects with semantic metadata.
        """
        if not documents:
            logger.info("No documents provided, returning empty Corpus")
            return Corpus([])

        chunks = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all documents for processing
            future_to_doc = {executor.submit(self._process_document, doc): doc for doc in documents}
            for future in future_to_doc:
                doc = future_to_doc[future]
                try:
                    chunks.extend(future.result())
                except Exception as e:
                    logger.error(f"Error processing document {doc.doc_id}: {str(e)}")

        logger.info(f"Chunked {len(documents)} documents into {len(chunks)} chunks")
        return Corpus(chunks=chunks)