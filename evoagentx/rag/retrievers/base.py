import logging
from enum import Enum
from abc import ABC, abstractmethod

from llama_index.core.retrievers import BaseRetriever

from ..schema import RagQuery, RagResult


class RetrieverType(str, Enum):
    VECTOR = "vector"
    GRAPH = "graph"


class BaseRetrieverWrapper(ABC):
    """Base interface for retriever wrappers."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    @abstractmethod
    def retrieve(self, query: RagQuery) -> RagResult:
        """Retrieve results for a query."""
        pass
    
    @abstractmethod
    def get_retriever(self) -> BaseRetriever:
        """Return the LlamaIndex-compatible retriever."""
        pass