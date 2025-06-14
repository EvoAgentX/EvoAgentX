from typing import Dict, Any, Optional

from llama_index.core.embeddings import BaseEmbedding

from evoagentx.core.logging import logger
from evoagentx.storages.base import StorageHandler
from .base import IndexType, BaseIndexWrapper
from .vector_index import VectorIndexing
from .graph_index import GraphIndexing
from .summary_index import SummaryIndexing
from .tree_index import TreeIndexing

__all__ = ['VectorIndexing', 'GraphIndexing', 'SummaryIndexing', 'TreeIndexing', 'IndexFactory', 'BaseIndexWrapper']

class IndexFactory:
    """Factory for creating LlamaIndex indices."""
    
    def create(
        self,
        index_type: IndexType,
        embed_model: BaseEmbedding,
        storage_handler: StorageHandler,
        index_config: Dict[str, Any] = None,
        node_parser: Optional[Any] = None  # Unused, kept for compatibility
    ) -> BaseIndexWrapper:
        """Create an index based on configuration.
        
        Args:
            index_type (IndexType): The type of index to create.
            embed_model (BaseEmbedding): Embedding model for the index.
            storage_context (StorageContext): Storage context for persistence.
            index_config (Dict[str, Any], optional): Index-specific configuration.
            node_parser (Any, optional): Node parser (unused, kept for compatibility).
            
        Returns:
            BaseIndexWrapper: A wrapped LlamaIndex index.
            
        Raises:
            ValueError: If the index type or configuration is invalid.
        """
        index_config = index_config or {}
        
        if index_type == IndexType.VECTOR:
            index = VectorIndexing(
                embed_model=embed_model,
                storage_handler=storage_handler,
                index_config=index_config
            )
        elif index_type == IndexType.GRAPH:
            raise NotImplementedError()
        elif index_type == IndexType.SUMMARY:
            raise NotImplementedError()
        elif index_type == IndexType.TREE:
            raise NotImplementedError()
        else:
            raise ValueError(f"Unsupported index type: {index_type}")
        
        logger.info(f"Created index: {index_type}")
        return index