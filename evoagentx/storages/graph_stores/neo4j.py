from typing import List, Any, Dict

from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.core.graph_stores.types import PropertyGraphStore

from .base import GraphStoreBase
from evoagentx.core.logging import logger


class Neo4jGraphStoreWrapper(GraphStoreBase):
    """Wrapper for Neo4j graph store."""
    
    def __init__(
        self,
        uri: str,
        username: str,
        password: str,
        database: str = "neo4j",
        **kwargs
    ):
        try:
            self.graph_store = Neo4jPropertyGraphStore(
                url=uri,
                username=username,
                password=password,
                database=database,
                **kwargs
            )
        except Exception as e:
            raise ValueError(f"Failed to connect to Neo4j: {str(e)}")
    
    def get_graph_store(self) -> PropertyGraphStore:
        return self.graph_store
    
    def add_triples(self, triples: List[Dict[str, Any]]):
        try:
            for triple in triples:
                self.graph_store.upsert_triplet(
                    subject=triple["subject"],
                    relation=triple["relation"],
                    object_=triple["object"]
                )
            logger.info(f"Added {len(triples)} triples to Neo4j store")
        except Exception as e:
            logger.error(f"Failed to add triples: {str(e)}")
            raise
    
    def query(self, query: str) -> List[Any]:
        try:
            results = self.graph_store.structured_query(query)
            logger.info(f"Queried Neo4j store, retrieved {len(results)} results")
            return results
        except Exception as e:
            logger.error(f"Query failed: {str(e)}")
            raise