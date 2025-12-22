from enum import Enum
from abc import ABC, abstractmethod
from typing import List, Any, Dict, Optional

from llama_index.core.storage import StorageContext
from llama_index.core.indices.base import BaseIndex


class IndexType(str, Enum):
    VECTOR = "vector"
    GRAPH = "graph"
    SUMMARY = "summary"
    TREE = "tree"

class BaseIndexWrapper(ABC):
    """Base interface for index wrappers."""

    @abstractmethod
    def _create_storage_context(self) -> StorageContext:
        """Create the LlamaIndex-compatible stroage"""
        pass

    @abstractmethod
    def get_index(self) -> BaseIndex:
        """Return the LlamaIndex-compatible index."""
        pass

    @abstractmethod
    def insert_nodes(self, nodes: List[Any]):
        """Insert nodes into the index."""
        pass

    @abstractmethod
    def delete_nodes(self, node_ids: Optional[List[str]] = None, metadata_filters: Optional[Dict[str, Any]] = None) -> None:
        """Delete nodes from the index by node IDs or metadata filters."""
        pass

    @abstractmethod
    def load(self) -> None:
        """Load all nodes from the index."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all nodes from the index."""
        pass