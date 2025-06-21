from typing import List, Dict, Any, Union, Optional

from llama_index.core.schema import BaseNode
from llama_index.core import VectorStoreIndex
from llama_index.core.storage import StorageContext
from llama_index.core.embeddings import BaseEmbedding

from .base import BaseIndexWrapper, IndexType
from evoagentx.rag.schema import Chunk
from evoagentx.core.logging import logger
from evoagentx.storages.base import StorageHandler

class VectorIndexing(BaseIndexWrapper):
    """Wrapper for LlamaIndex VectorStoreIndex."""

    def __init__(
        self,
        embed_model: BaseEmbedding,
        storage_handler: StorageHandler,
        index_config: Dict[str, Any] = None
    ):
        super().__init__()
        self.index_type = IndexType.VECTOR
        self.embed_model = embed_model
        self.storage_handler = storage_handler
        # create a storage_context for llama_index
        self._create_storage_context()
        # for caching llama_index node
        self.id_to_node = dict()

        self.index_config = index_config or {}
        try:
            self.index = VectorStoreIndex(
                nodes=[],
                embed_model=self.embed_model,
                storage_context=self.storage_context,
                show_progress=self.index_config.get("show_progress", False)
            )
        except Exception as e:
            logger.error(f"Failed to initialize VectorStoreIndex: {str(e)}")
            raise

    def _create_storage_context(self, ):
        # Construct a storage_context for llama_index
        assert self.storage_handler.vector_store is not None, "VectorIndexing must init a vector backend in 'storageHandler'"
        self.storage_context = StorageContext.from_defaults(
            vector_store=self.storage_handler.vector_store.get_vector_store()
        )

    def get_index(self) -> VectorStoreIndex:
        return self.index
    
    def insert_nodes(self, nodes: List[Union[Chunk, BaseNode]]):
        try:
            filted_node = []
            # only add the unique node
            for node in nodes:
                converted = node.to_llama_node() if isinstance(node, Chunk) else node
                if converted.id_ in self.id_to_node:
                    continue
                filted_node.extend([converted])

            # TODO: find a better way to manage the node
            # Caching the node 
            nodes_with_embedding = self.index._get_node_with_embedding(nodes=filted_node)
            for node in nodes_with_embedding:
                self.id_to_node[node.node_id] = node.model_copy()
            self.index.insert_nodes(nodes_with_embedding)
            logger.info(f"Inserted {len(nodes_with_embedding)} nodes into VectorStoreIndex")
        except Exception as e:
            logger.error(f"Failed to insert nodes: {str(e)}")
            raise

    def delete_nodes(self, node_ids: Optional[List[str]] = None, 
                     metadata_filters: Optional[Dict[str, Any]] = None) -> None:
        """Delete nodes from the index by node IDs or metadata filters."""
        try:
            if node_ids:
                for node_id in node_ids:
                    if node_id in self.id_to_node:
                        self.index.delete_nodes([node_id], delete_from_docstore=True)
                        self.id_to_node.pop(node_id)
                        logger.info(f"Deleted node {node_id} from VectorStoreIndex")

            elif metadata_filters:
                nodes_to_delete = []
                for node_id, node in self.id_to_node.items():
                    if all(node.metadata.get(k) == v for k, v in metadata_filters.items()):
                        nodes_to_delete.append(node_id)
                if nodes_to_delete:
                    self.index.delete_nodes(nodes_to_delete, delete_from_docstore=True)
                    
                    for node_id in nodes_to_delete:
                        del self.id_to_node[node_id]
                    logger.info(f"Deleted {len(nodes_to_delete)} nodes matching metadata filters from VectorStoreIndex")
            else:
                logger.warning("No node_ids or metadata_filters provided for deletion")
        except Exception as e:
            logger.error(f"Failed to delete nodes: {str(e)}")
            raise

    def clear(self) -> None:
        """Clear all nodes from the index."""
        try:

            node_ids = list(self.id_to_node.keys())
            self.index.delete_nodes(node_ids, delete_from_docstore=True)
            self.id_to_node.clear()
            logger.info("Cleared all nodes from VectorStoreIndex")
        except Exception as e:
            logger.error(f"Failed to clear index: {str(e)}")
            raise