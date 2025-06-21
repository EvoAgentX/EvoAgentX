from enum import Enum
from typing import List
from abc import ABC, abstractmethod

from evoagentx.rag.schema import Document, Corpus


class ChunkingStrategy(str, Enum):
    SIMPLE = "simple"
    SEMANTIC = "semantic"
    HIERARCHICAL = "hierarchical"

class BaseChunker(ABC):
    """Abstract base class for chunking documents into smaller segments.

    This class defines the interface for chunking strategies in the RAG pipeline,
    converting Documents into a Corpus of Chunks.
    """

    @abstractmethod
    def chunk(self, documents: List[Document], **kwargs) -> Corpus:
        """Chunk documents into a Corpus of Chunks.

        Args:
            documents (List[Document]): List of Document objects to chunk.
            **kwargs: Additional parameters specific to the chunking strategy.

        Returns:
            Corpus: A collection of Chunk objects.
        """
        pass