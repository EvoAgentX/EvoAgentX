import os
from typing import Any, Dict, List, Optional

from llama_index.core.embeddings import BaseEmbedding
from llama_index.embeddings.azure_openai import (
    AzureOpenAIEmbedding as LlamaAzureEmbedding,
)

from evoagentx.core.logging import logger
from .base import BaseEmbeddingWrapper, EmbeddingProvider, SUPPORTED_MODELS

MODEL_DIMENSIONS: Dict[str, int] = {
    "text-embedding-ada-002": 1536,
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
}

SUPPORTED_DIMENSIONS = [
    "text-embedding-3-small",
    "text-embedding-3-large",
]


class AzureOpenAIEmbedding(BaseEmbedding):
    """Azure OpenAI embedding implementation compatible with LlamaIndex."""

    api_key: Optional[str] = None
    model_name: str = "text-embedding-3-small"
    azure_endpoint: Optional[str] = None
    api_version: Optional[str] = None
    deployment_name: Optional[str] = None
    embed_batch_size: int = 10
    dimensions: Optional[int] = None
    kwargs: Dict[str, Any] = {}

    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        azure_endpoint: Optional[str] = None,
        api_version: Optional[str] = None,
        deployment_name: Optional[str] = None,
        dimensions: Optional[int] = None,
        embed_batch_size: int = 10,
        **kwargs: Any,
    ) -> None:
        api_key = (
            api_key
            or os.getenv("AZURE_OPENAI_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or ""
        )
        super().__init__(api_key=api_key, model_name=model_name, embed_batch_size=embed_batch_size)

        self.model_name = model_name
        self.azure_endpoint = azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        self.api_version = api_version or os.getenv("AZURE_OPENAI_API_VERSION")
        self.deployment_name = (
            deployment_name
            or os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT")
            or model_name
        )
        self.dimensions = dimensions
        self.embed_batch_size = embed_batch_size
        self.kwargs = kwargs or {}

        if not EmbeddingProvider.validate_model(EmbeddingProvider.AZURE_OPENAI, model_name):
            raise ValueError(
                "Unsupported Azure OpenAI model: "
                f"{model_name}. Supported models: {SUPPORTED_MODELS['azure_openai']}"
            )

        if self.dimensions is not None and model_name not in SUPPORTED_DIMENSIONS:
            logger.warning(
                "Dimensions parameter is not supported for model %s. Only %s support custom "
                "dimensions. Ignoring provided value.",
                model_name,
                SUPPORTED_DIMENSIONS,
            )
            self.dimensions = None
        elif self.dimensions is None and model_name in SUPPORTED_DIMENSIONS:
            self.dimensions = MODEL_DIMENSIONS.get(model_name)

        try:
            self._embedding: LlamaAzureEmbedding = LlamaAzureEmbedding(
                model=self.model_name,
                azure_endpoint=self.azure_endpoint,
                azure_deployment=self.deployment_name,
                deployment_name=self.deployment_name,
                api_key=self.api_key,
                api_version=self.api_version,
                dimensions=self.dimensions,
                embed_batch_size=self.embed_batch_size,
                **self.kwargs,
            )
            if self.dimensions is None and hasattr(self._embedding, "dimensions"):
                self.dimensions = getattr(self._embedding, "dimensions")
            logger.debug(
                "Initialized Azure OpenAI embedding: model=%s deployment=%s",
                self.model_name,
                self.deployment_name,
            )
        except Exception as exc:
            logger.error("Failed to initialize Azure OpenAI embedding: %s", exc)
            raise

    def _get_query_embedding(self, query: str) -> List[float]:
        query = query.replace("\n", " ")
        try:
            return self._embedding.get_query_embedding(query)
        except Exception as exc:
            logger.error("Failed to encode query with Azure OpenAI: %s", exc)
            raise

    def _get_text_embedding(self, text: str) -> List[float]:
        text = text.replace("\n", " ")
        try:
            return self._embedding.get_text_embedding(text)
        except Exception as exc:
            logger.error("Failed to encode text with Azure OpenAI: %s", exc)
            raise

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        cleaned = [text.replace("\n", " ") for text in texts]
        try:
            return self._embedding.get_text_embedding_batch(cleaned)
        except Exception as exc:
            logger.error("Failed to encode texts with Azure OpenAI: %s", exc)
            raise


class AzureOpenAIEmbeddingWrapper(BaseEmbeddingWrapper):
    """Wrapper for Azure OpenAI embedding models."""

    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        azure_endpoint: Optional[str] = None,
        api_version: Optional[str] = None,
        deployment_name: Optional[str] = None,
        dimensions: Optional[int] = None,
        embed_batch_size: int = 10,
        **kwargs: Any,
    ) -> None:
        self.model_name = model_name
        self.api_key = api_key
        self.azure_endpoint = azure_endpoint
        self.api_version = api_version
        self.deployment_name = deployment_name or model_name
        self.kwargs = kwargs or {}
        self.embed_batch_size = embed_batch_size
        self._dimensions = MODEL_DIMENSIONS.get(self.model_name) or dimensions
        self._embedding_model: Optional[AzureOpenAIEmbedding] = None
        self._dimensions = self._dimensions or dimensions

    def get_embedding_model(self) -> BaseEmbedding:
        if self._embedding_model is None:
            try:
                self._embedding_model = AzureOpenAIEmbedding(
                    model_name=self.model_name,
                    api_key=self.api_key,
                    azure_endpoint=self.azure_endpoint,
                    api_version=self.api_version,
                    deployment_name=self.deployment_name,
                    dimensions=self._dimensions,
                    embed_batch_size=self.embed_batch_size,
                    **self.kwargs,
                )
                logger.debug(
                    "Initialized Azure OpenAI embedding wrapper for model %s",
                    self.model_name,
                )
            except Exception as exc:
                logger.error("Failed to initialize Azure OpenAI embedding wrapper: %s", exc)
                raise
        return self._embedding_model

    @property
    def dimensions(self) -> Optional[int]:
        return self._embedding_model.dimensions if self._embedding_model else self._dimensions
