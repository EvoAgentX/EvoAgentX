import asyncio
from typing import List, Dict, Any, Union, Optional, Sequence

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
    
    def insert_nodes(self, nodes: List[Union[Chunk, BaseNode]]) -> Sequence[str]:
        """
        Insert or update nodes into the vector index.

        Converts Chunk objects to LlamaIndex nodes, serializes metadata as JSON strings, and inserts
        them into the VectorStoreIndex. Nodes are cached in id_to_node for quick access.

        Args:
            nodes (List[Union[Chunk, BaseNode]]): List of nodes to insert, either Chunk or BaseNode.
        
        Returns:

        """
        try:
            filtered_nodes = []
            # only add the unique node
            for node in nodes:
                converted = node.to_llama_node() if isinstance(node, Chunk) else node
                if converted.id_ in self.id_to_node:
                    continue
                filtered_nodes.extend([converted])

            # TODO: find a better way to manage the node
            # Caching the node 
            nodes_with_embedding = self.index._get_node_with_embedding(nodes=filtered_nodes)
            for node in nodes_with_embedding:
                self.id_to_node[node.node_id] = node.model_copy()
            self.index.insert_nodes(nodes_with_embedding)
            logger.info(f"Inserted {len(nodes_with_embedding)} nodes into VectorStoreIndex")
            return list([n.node_id for n in filtered_nodes])
        
        except Exception as e:
            logger.error(f"Failed to insert nodes: {str(e)}")
            return []

    def delete_nodes(self, node_ids: Optional[List[str]] = None, 
                     metadata_filters: Optional[Dict[str, Any]] = None) -> None:
        """
        Delete nodes from the vector index based on node IDs or metadata filters.

        Removes specified nodes from the index and the id_to_node cache. If metadata_filters are
        provided, nodes matching the filters are deleted.

        Args:
            node_ids (Optional[List[str]]): List of node IDs to delete. Defaults to None.
            metadata_filters (Optional[Dict[str, Any]]): Metadata filters to select nodes for deletion. Defaults to None.
        """
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

    async def aload(self, nodes: List[Union[Chunk, BaseNode]]) -> None:
        """
        Asynchronously load nodes into the vector index and its backend store.

        Caches nodes in id_to_node and loads them into the FAISS vector store, ensuring
        no duplicates are inserted by relying on the backend's duplicate checking.

        Args:
            nodes (List[Union[Chunk, BaseNode]]): The nodes to load.
        """
        try:
            filtered_nodes = [node.to_llama_node() if isinstance(node, Chunk) else node for node in nodes]

            tasks = []
            for node in filtered_nodes:
                node_id = node.id if hasattr(node, "id") else node.id_
                self.id_to_node[node_id] = node.model_copy()

                if self.storage_handler.vector_store is not None:
                    tasks.append(self.storage_handler.vector_store.aload(node))

            # Execute all vector store loading tasks concurrently
            await asyncio.gather(*tasks, return_exceptions=True)
            logger.info(f"Loaded {len(filtered_nodes)} nodes into cache and vector store.")

        except Exception as e:
            logger.error(f"Failed to load nodes into VectorStoreIndex: {str(e)}")
            raise

    def load(self, nodes: List[Union[Chunk, BaseNode]]) -> None:
        """
        Synchronously load nodes into the vector index.

        Args:
            nodes (List[Union[Chunk, BaseNode]]): The nodes to load.
        """
        asyncio.run(self.aload(nodes))

    def clear(self) -> None:
        """
        Clear all nodes from the vector index and its cache.

        Deletes all nodes from the VectorStoreIndex and clears the id_to_node cache.
        """
        try:

            node_ids = list(self.id_to_node.keys())
            self.index.delete_nodes(node_ids, delete_from_docstore=False)
            self.id_to_node.clear()
            self.index.storage_context.docstore._kvstore._collections_mappings.clear()
            logger.info("Cleared all nodes from VectorStoreIndex")
        except Exception as e:
            logger.error(f"Failed to clear index: {str(e)}")
            raise