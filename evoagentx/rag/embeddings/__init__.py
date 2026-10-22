from importlib import import_module
from typing import Any

from evoagentx.core.logging import logger

from .base import BaseEmbeddingWrapper, EmbeddingProvider

__all__ = [
    'AzureOpenAIEmbeddingWrapper',
    'BaseEmbeddingWrapper',
    'EmbeddingFactory',
    'EmbeddingProvider',
    'HuggingFaceEmbeddingWrapper',
    'OllamaEmbeddingWrapper',
    'OpenAIEmbeddingWrapper',
    'VoyageEmbeddingWrapper'
]

_EXPORTS = {
    "OpenAIEmbeddingWrapper": ".openai_embedding",
    "AzureOpenAIEmbeddingWrapper": ".azure_openai_embedding",
    "HuggingFaceEmbeddingWrapper": ".huggingface_embedding",
    "OllamaEmbeddingWrapper": ".ollama_embedding",
    "VoyageEmbeddingWrapper": ".voyage",
}


def __getattr__(name: str):
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))


def _get_wrapper(name: str):
    return globals().get(name) or __getattr__(name)


class EmbeddingFactory:
    """Factory for creating embedding models based on configuration."""
    
    def create(
        self,
        provider: EmbeddingProvider,
        model_config: dict[str, Any] | None = None
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
            wrapper = _get_wrapper("OpenAIEmbeddingWrapper")(**model_config)
        elif provider == EmbeddingProvider.AZURE_OPENAI:
            wrapper = _get_wrapper("AzureOpenAIEmbeddingWrapper")(**model_config)
        elif provider == EmbeddingProvider.HUGGINGFACE:
            wrapper = _get_wrapper("HuggingFaceEmbeddingWrapper")(**model_config)
        elif provider == EmbeddingProvider.OLLAMA:
            wrapper = _get_wrapper("OllamaEmbeddingWrapper")(**model_config)
        elif provider == EmbeddingProvider.VOYAGE:
            wrapper = _get_wrapper("VoyageEmbeddingWrapper")(**model_config)
        else:
            raise ValueError(f"Unsupported embedding provider: {provider}")
        
        logger.info(f"Created embedding model for provider: {provider}")
        return wrapper
