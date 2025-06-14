from typing import Dict, Any

from llama_index.core.graph_stores.types import GraphStore

from .base import GraphStoreType, GraphStoreBase
from .neo4j import Neo4jGraphStoreWrapper
from evoagentx.core.logging import logger

__all__ = ['GraphStoreBase', 'Neo4jGraphStoreWrapper', 'GraphStoreFactory']

class GraphStoreFactory:
    """Factory for creating graph stores."""

    def create(
        self,
        store_type: str,
        store_config: Dict[str, Any] = None
    ) -> GraphStore:
        """Create a graph store based on configuration.
        
        Args:
            store_type (str): The type of graph store (e.g., 'neo4j').
            store_config (Dict[str, Any], optional): Store configuration.
            
        Returns:
            GraphStore: A LlamaIndex-compatible graph store.
            
        Raises:
            ValueError: If the store type or configuration is invalid.
        """
        store_config = store_config or {}
        if store_type == GraphStoreType.NEO4J.value:
            required_fields = ["uri", "username", "password"]
            if not all(field in store_config for field in required_fields):
                raise ValueError("Neo4j requires uri, username, and password")
            graph_store = Neo4jGraphStoreWrapper(**store_config).get_graph_store()
        else:
            raise ValueError(f"Unsupported graph store type: {store_type}")
        logger.info(f"Created graph store: {store_type}")
        return graph_store