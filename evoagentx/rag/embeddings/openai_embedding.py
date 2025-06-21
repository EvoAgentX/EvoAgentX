import os
import warnings
from typing import List, Optional, Dict

from openai import OpenAI
from llama_index.core.embeddings import BaseEmbedding

from evoagentx.core.logging import logger
from .base import BaseEmbeddingWrapper, EmbeddingProvider, SUPPORTED_MODELS

# Mapping of default embedding dimensions for OpenAI models
MODEL_DIMENSIONS = {
    "text-embedding-ada-002": 1536,
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072
}

SUPPORTED_DIMENSIONS = ["text-embedding-3-small", "text-embedding-3-large",]

class OpenAIEmbedding(BaseEmbedding):
    """OpenAI embedding model compatible with LlamaIndex BaseEmbedding."""
    
    api_key: str
    client: OpenAI = None
    base_url: str = "https://api.openai.com/v1"
    model_name: str = "text-embedding-3-small"
    embed_batch_size: int = 10
    dimensions: Optional[int] = None
    kwargs: Optional[Dict] = {}
    
    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        api_key: str = None,
        dimensions: int = None,
        base_url: str = None,
        **kwargs
    ):
        api_key = api_key or os.getenv("OPENAI_API_KEY") or ""
        super().__init__(api_key=api_key, model_name=model_name, embed_batch_size=10)
        base_url = (
            base_url
            or os.getenv("OPENAI_API_BASE")
            or os.getenv("OPENAI_BASE_URL")
            or "https://api.openai.com/v1"
        )
        if os.environ.get("OPENAI_API_BASE"):
            warnings.warn(
                "The environment variable 'OPENAI_API_BASE' is deprecated and will be removed in the 0.1.80. "
                "Please use 'OPENAI_BASE_URL' instead.",
                DeprecationWarning,
            )
        self.base_url = base_url
        self.dimensions = dimensions
        self.kwargs = kwargs

        if not EmbeddingProvider.validate_model(EmbeddingProvider.OPENAI, model_name):
            raise ValueError(f"Unsupported OpenAI model: {model_name}. Supported models: {SUPPORTED_MODELS['openai']}")
        # Check for the dimensions support
        if dimensions is not None and model_name not in SUPPORTED_DIMENSIONS:
            logger.warning(
                f"Dimensions parameter is not supported for model {model_name}. "
                f"Only '{[k for k in SUPPORTED_DIMENSIONS.keys()]}' support custom dimensions. Ignoring dimensions parameter."
            )
            self.dimensions = None
        elif dimensions is None and model_name in SUPPORTED_DIMENSIONS:
            self.dimensions = dimensions or MODEL_DIMENSIONS.get(model_name)

        try:
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            logger.debug(f"Initialized OpenAI embedding model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {str(e)}")
            raise

    def _get_query_embedding(self, query: str) -> List[float]:
        """Get embedding for a query string."""
        try:
            query = query.replace("\n", " ")
            response = self.client.embeddings.create(
                input=[query],
                model=self.model_name,
                dimensions=self.dimensions,
                **self.kwargs
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Failed to encode query: {str(e)}")
            raise

    def _get_text_embedding(self, text: str) -> List[float]:
        """Get embedding for a text string."""
        try:
            text = text.replace("\n", " ")
            response = self.client.embeddings.create(
                input=[text],
                model=self.model_name,
                dimensions=self.dimensions,
                **self.kwargs
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Failed to encode text: {str(e)}")
            raise

    async def _aget_query_embedding(self, query: str) -> List[float]:
        """Asynchronous query embedding."""
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        """Asynchronous text embedding."""
        return self._get_text_embedding(text)

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for a list of texts synchronously."""
        try:
            texts = [text.replace("\n", " ") for text in texts]
            response = self.client.embeddings.create(
                input=texts,
                model=self.model_name,
                dimensions=self.dimensions,
                **self.kwargs
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            logger.error(f"Failed to encode texts: {str(e)}")
            raise

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Asynchronous batch text embedding."""
        return self._get_text_embeddings(texts)


class OpenAIEmbeddingWrapper(BaseEmbeddingWrapper):
    """Wrapper for OpenAI embedding models."""
    
    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        api_key: str = None,
        dimensions: int = None,
        base_url: str = None,
        **kwargs
    ):
        self.model_name = model_name
        self.api_key = api_key
        self._dimensions = dimensions
        self.base_url = base_url
        self.kwargs = kwargs
        self._embedding_model = None

    def get_embedding_model(self) -> BaseEmbedding:
        """Return the LlamaIndex-compatible embedding model."""
        if self._embedding_model is None:
            try:
                self._embedding_model = OpenAIEmbedding(
                    model_name=self.model_name,
                    api_key=self.api_key,
                    dimensions=self._dimensions,
                    base_url=self.base_url,
                    **self.kwargs
                )
                logger.debug(f"Initialized OpenAI embedding wrapper for model: {self.model_name}")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI embedding wrapper: {str(e)}")
                raise
        return self._embedding_model
    
    @property
    def dimensions(self) -> int:
        """Return the embedding dimensions."""
        return self._embedding_model or MODEL_DIMENSIONS.get(self.model_name, None)