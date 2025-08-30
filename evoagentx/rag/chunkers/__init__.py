from typing import Dict, Any

from llama_index.core.embeddings import BaseEmbedding

from .base import BaseChunker, ChunkingStrategy
from .simple_chunker import SimpleChunker
from .semantic_chunker import SemanticChunker
from .hierachical_chunker import HierarchicalChunker
from evoagentx.core.logging import logger

__all__ = ['SimpleChunker', 'SemanticChunker', 'HierarchicalChunker', 'ChunkFactory', 'BaseChunker']


class ChunkFactory:
    """Factory for creating chunkers based on configuration."""
    
    def create(
        self,
        strategy: ChunkingStrategy,
        embed_model: BaseEmbedding = None,
        chunker_config: Dict[str, Any] = None
    ) -> BaseChunker:
        """Create a chunker based on strategy and configuration.
        
        Args:
            strategy (ChunkingStrategy): The chunking strategy.
            embed_model (BaseEmbedding, optional): Embedding model for semantic chunking.
            chunker_config (Dict[str, Any], optional): Chunker configuration.
            
        Returns:
            BaseChunker: A chunker instance.
            
        Raises:
            ValueError: If the strategy or configuration is invalid.
        """
        chunker_config = chunker_config or {}
        
        if strategy == ChunkingStrategy.SIMPLE:
            chunker = SimpleChunker(
                chunk_size=chunker_config.get("chunk_size", 1024),
                chunk_overlap=chunker_config.get("chunk_overlap", 20),
                max_workers=chunker_config.get("max_workers", 2)
            )
        elif strategy == ChunkingStrategy.SEMANTIC:
            if not embed_model:
                raise ValueError("Embed model required for semantic chunking")
            chunker = SemanticChunker(
                embed_model=embed_model,
                similarity_threshold=chunker_config.get("similarity_threshold", 0.7),
                max_workers=chunker_config.get("max_workers", 2)
            )
        elif strategy == ChunkingStrategy.HIERARCHICAL:
            chunker = HierarchicalChunker(
                chunk_sizes=chunker_config.get("chunk_sizes", [2048, 512, 128]),
                chunk_overlap=chunker_config.get("chunk_overlap", 20)
            )
        else:
            raise ValueError(f"Unsupported chunking strategy: {strategy}")
        
        logger.info(f"Created chunker for strategy: {strategy}")
        return chunker