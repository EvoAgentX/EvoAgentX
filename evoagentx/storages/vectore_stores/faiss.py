from typing import Union, Literal

import faiss
from llama_index.core.schema import BaseNode
from llama_index.vector_stores.faiss import FaissMapVectorStore

from .base import VectorStoreBase
from evoagentx.core.logging import logger


class FaissVectorStoreWrapper(VectorStoreBase):
    """Wrapper for FAISS vector store."""
    
    def __init__(self, dimensions: int = 1536, 
                 metrics: Union[Literal["flat_l2", "ivf_flat"]] = "flat_l2", **kwargs):
        self.dimensions = dimensions
        self.metrics = metrics
        self.faiss_index = self._create_index()
        self.vector_store = FaissMapVectorStore(faiss_index=faiss.IndexIDMap2(self.faiss_index))

    def _create_index(self) -> faiss.Index:
        if self.metrics == "flat_l2":
            return faiss.IndexFlatL2(self.dimensions)
        elif self.metrics == "ivf_flat":
            quantizer = faiss.IndexFlatL2(self.dimensions)
            return faiss.IndexIVFFlat(quantizer, self.dimensions, 100)
        else:
            raise ValueError(f"Unsupported FAISS index type: {self.metrics}")
        
    def get_vector_store(self) -> FaissMapVectorStore:
        return self.vector_store

    async def aload(self, node: BaseNode) -> None:
        """
        Asynchronously load a single node into the FAISS vector store.

        Checks if a node with the same ID already exists in the FAISS vector store. If it does not exist,
        inserts the node with its embedding. Handles both Chunk and BaseNode types.

        Args:
            node (Union[Chunk, BaseNode]): The node to load, either a Chunk or a LlamaIndex BaseNode.
        """
        try:
            if not isinstance(node, BaseNode):
                raise ValueError(f"Unsupported node type: {type(node)}. Must be Chunk or BaseNode.")

            existing_ids = self.vector_store._faiss_index.id_map
            node_id_int = self.vector_store._get_node_id_int(node.id)
            if node_id_int in existing_ids:
                logger.info(f"Node with ID {node.id} already exists in FAISS vector store, skipping insertion.")
                return

            if not node.embedding or len(node.embedding) != self.dimension:
                raise ValueError(f"Node {node.id} has invalid or missing embedding. Expected dimension: {self.dimension}")


            self.vector_store.add([node])

            logger.info(f"Inserted node with ID {node.id} into FAISS vector store.")

        except Exception as e:
            logger.error(f"Failed to load node with ID {node.id} into FAISS vector store: {str(e)}")
            raise