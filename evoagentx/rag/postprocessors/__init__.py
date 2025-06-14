from typing import Optional

from evoagentx.rag.schema import RagQuery
from evoagentx.core.logging import logger
from .simple_reranker import SimpleReranker
from .base import BasePostprocessor, RerankerType


__all__ = ['PostprocessorFactory', 'SimpleReranker', 'BasePostprocessor']

class PostprocessorFactory:
    """Factory for creating post-processors."""
    
    def create(
        self,
        postprocessor_type: str,
        query: Optional[RagQuery] = None
    ) -> BasePostprocessor:
        """Create a post-processor based on configuration.
        
        Args:
            postprocessor_type (str): Type of post-processor (e.g., 'simple', 'bge').
            query (Query, optional): Query for configuration.
            
        Returns:
            BasePostprocessor: A post-processor instance.
            
        Raises:
            ValueError: If the post-processor type or configuration is invalid.
        """
        if postprocessor_type == RerankerType.SIMPLE:
            if not query:
                raise ValueError("Query required for reranker")
            postprocessor = SimpleReranker(
                similarity_cutoff=query.similarity_cutoff,
                keyword_filters=query.keyword_filters
            )
        else:
            raise ValueError(f"Unsupported post-processor type: {postprocessor_type}")
        
        logger.info(f"Created post-processor: {postprocessor_type}")
        return postprocessor