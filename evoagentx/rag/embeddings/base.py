import os
from enum import Enum
from typing import Dict, List

from llama_index.core.embeddings import BaseEmbedding


# Mapping of supported models for each provider
SUPPORTED_MODELS: Dict[str, List[str]] = {
    "openai": [
        "text-embedding-ada-002",
        "text-embedding-3-small",
        "text-embedding-3-large"
    ],
    "huggingface": [
        "sentence-transformers/all-MiniLM-L6-v2",
        "sentence-transformers/all-mpnet-base-v2",
        "sentence-transformers/multi-qa-mpnet-base-dot-v1",
        "BAAI/bge-small-en-v1.5",
        "BAAI/bge-large-en-v1.5",
    ],
    "ollama": [
        "nomic-embed-text",
        "mxbai-embed-large",
        "bge-m3",
        "all-minilm"
        "snowflake-arctic-embed"
    ]
}


class EmbeddingProvider(str, Enum):
    OPENAI = "openai"
    HUGGINGFACE = "huggingface"
    OLLAMA = "ollama"

    @classmethod
    def validate_model(cls, provider: str, model_name: str) -> bool:
        """Validate if the model is supported for the given provider.

        Args:
            provider (str): The embedding provider (e.g., 'openai', 'huggingface', 'ollama').
            model_name (str): The name of the embedding model to validate.

        Returns:
            bool: True if the model is supported or provider is 'custom', False otherwise.

        Raises:
            ValueError: If the provider is invalid.
        """
        if provider not in SUPPORTED_MODELS:
            raise ValueError(f"Unsupported provider: {provider}")
        
        # Handle the local model.
        if provider == "huggingface":
            if os.path.exists(model_name):
                return True
            return model_name in SUPPORTED_MODELS.get(provider, [])

        return model_name in SUPPORTED_MODELS.get(provider, [])


class BaseEmbeddingWrapper:
    """Base interface for embedding wrappers."""
    
    def get_embedding_model(self) -> BaseEmbedding:
        """Return the LlamaIndex-compatible embedding model."""
        raise NotImplementedError()

    def validate_model(self, provider: EmbeddingProvider, model_name: str) -> bool:
        """Validate if the model is supported for the given provider.

        Args:
            provider (EmbeddingProvider): The embedding provider.
            model_name (str): The name of the embedding model to validate.

        Returns:
            bool: True if the model is supported, False otherwise.
        """
        return EmbeddingProvider.validate_model(provider, model_name)
    
    @property
    def dimensions(self) -> int:
        raise NotImplementedError()