from typing import Dict, Any

from evoagentx.core.logging import logger
from .base import EmbeddingProvider, BaseEmbeddingWrapper
from .openai_embedding import OpenAIEmbeddingWrapper
from .huggingface_embedding import HuggingFaceEmbeddingWrapper
from .ollama_embedding import OllamaEmbeddingWrapper
from .voyage import VoyageEmbeddingWrapper

__all__ = [
    'OpenAIEmbeddingWrapper',
    'HuggingFaceEmbeddingWrapper',
    'OllamaEmbeddingWrapper',
    'VoyageEmbeddingWrapper',
    'EmbeddingFactory',
    'BaseEmbedding',
    'EmbeddingProvider'
]

class EmbeddingFactory:
    """Factory for creating embedding models based on configuration."""
    
    def create(
        self,
        provider: EmbeddingProvider,
        model_config: Dict[str, Any] = None
    ) -> BaseEmbeddingWrapper:
        """Create an embedding model based on the provider and configuration.
        
        Args:
            provider (EmbeddingProvider): The embedding provider (e.g., OpenAI, HuggingFace, Ollama).
            model_config (Dict[str, Any], optional): Configuration for the embedding model.
            
        Returns:
            BaseEmbeddingWrapper: A LlamaIndex-compatible embedding model wrapper.
            
        Raises:
            ValueError: If the provider or configuration is invalid.
        """
        model_config = model_config or {}
        model_config.pop("provider")    # filter the provider key

        if provider == EmbeddingProvider.OPENAI:
            wrapper = OpenAIEmbeddingWrapper(**model_config)
        elif provider == EmbeddingProvider.HUGGINGFACE:
            wrapper = HuggingFaceEmbeddingWrapper(**model_config)
        elif provider == EmbeddingProvider.OLLAMA:
            wrapper = OllamaEmbeddingWrapper(**model_config)
        elif provider == EmbeddingProvider.VOYAGE:
            wrapper = VoyageEmbeddingWrapper(**model_config)
        else:
            raise ValueError(f"Unsupported embedding provider: {provider}")
        
        logger.info(f"Created embedding model for provider: {provider}")
        return wrapper