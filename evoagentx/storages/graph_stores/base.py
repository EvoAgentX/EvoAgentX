from enum import Enum
from typing import Dict
from abc import ABC, abstractmethod

from llama_index.core.graph_stores.simple import GraphStore


class GraphStoreType(str, Enum):
    NEO4J = "neo4j"

class GraphStoreBase(ABC):
    """Base interface for graph stores."""
    
    @abstractmethod
    def get_graph_store(self) -> GraphStore:
        """Return the LlamaIndex-compatible graph store."""
        pass

    @property
    def supports_vector_queries(self):
        NotImplementedError()

    @abstractmethod
    def clear(self) -> None:
        """Clear the node and relation in the graph database."""
        pass

    @abstractmethod
    def aload(self) -> None:
        """Asynchronously load a single node into the graph database."""
        pass

    @abstractmethod
    def build_kv_store(self) -> Dict:
        """Exported all the nodes and relations from graph database into python Dict for saving to file or database."""
        pass