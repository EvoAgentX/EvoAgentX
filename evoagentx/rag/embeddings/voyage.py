import os
import asyncio
from typing import List, Optional, Dict, Any, Union

import voyageai
from llama_index.core.embeddings import BaseEmbedding
from PIL import Image

from evoagentx.core.logging import logger
from .base import BaseEmbeddingWrapper, EmbeddingProvider


class VoyageEmbedding(BaseEmbedding):
    """Voyage AI multimodal embedding model compatible with LlamaIndex BaseEmbedding."""
    
    api_key: str = ""
    client: Optional[voyageai.AsyncClient] = None
    model_name: str = "voyage-multimodal-3"
    embed_batch_size: int = 10
    _dimension: Optional[int] = None
    
    def __init__(
        self,
        model_name: str = "voyage-multimodal-3",
        api_key: str = None,
        **kwargs
    ):
        api_key = api_key or os.getenv("VOYAGE_API_KEY") or ""
        if not api_key:
            raise ValueError("Voyage API key is required. Set VOYAGE_API_KEY environment variable or pass api_key parameter.")
        
        super().__init__(model_name=model_name, embed_batch_size=10, api_key=api_key)
        self.client = voyageai.AsyncClient(api_key=api_key)
        
        # Set dimension based on model
        if "voyage-multimodal-3" in model_name:
            self._dimension = 1024
        else:
            self._dimension = 1024  # Default for other models
        
        logger.debug(f"Initialized Voyage embedding model: {model_name}")

    async def _async_embed_documents(self, documents: List[Any]) -> List[List[float]]:
        """Async method to embed documents (images or text)."""
        try:
            # Handle different input types
            inputs = []
            for doc in documents:
                if isinstance(doc, str):
                    # Text document
                    inputs.append({"content": [{"type": "text", "text": doc}]})
                elif isinstance(doc, Image.Image):
                    # PIL Image
                    inputs.append([doc])
                elif hasattr(doc, 'get_image'):
                    # ImageChunk or similar with get_image method
                    image = doc.get_image()
                    if image:
                        inputs.append([image])
                    else:
                        raise ValueError(f"Could not load image from document: {doc}")
                else:
                    # Assume it's already in the right format
                    inputs.append([doc])
            
            result = await self.client.multimodal_embed(
                inputs=inputs,
                model=self.model_name,
                input_type="document"
            )
            return result.embeddings
        except Exception as e:
            logger.error(f"Error embedding documents with Voyage: {str(e)}")
            raise

    async def _async_embed_query(self, query: Union[str, Dict, List]) -> List[float]:
        """Async method to embed a query."""
        try:
            # Handle different query formats
            if isinstance(query, str):
                formatted_query = {"content": [{"type": "text", "text": query}]}
            elif isinstance(query, dict):
                formatted_query = query
            elif isinstance(query, list):
                # Assume it's already formatted content
                formatted_query = {"content": query}
            else:
                formatted_query = {"content": [{"type": "text", "text": str(query)}]}
            
            result = await self.client.multimodal_embed(
                inputs=[formatted_query],
                model=self.model_name,
                input_type="query"
            )
            return result.embeddings[0]
        except Exception as e:
            logger.error(f"Error embedding query with Voyage: {str(e)}")
            raise

    def _get_query_embedding(self, query: str) -> List[float]:
        """Get embedding for a query string (sync wrapper)."""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(self._async_embed_query(query))

    def _get_text_embedding(self, text: str) -> List[float]:
        """Get embedding for a text string (sync wrapper)."""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(self._async_embed_documents([text]))[0]

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for a list of texts (sync wrapper)."""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(self._async_embed_documents(texts))

    def _get_image_embedding(self, image_node) -> List[float]:
        """Get embedding for an ImageNode."""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(self._async_embed_documents([image_node]))[0]

    def get_image_embedding(self, image: Union[Image.Image, Any]) -> List[float]:
        """Get embedding for an image."""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(self._async_embed_documents([image]))[0]

    async def _aget_query_embedding(self, query: str) -> List[float]:
        """Asynchronous query embedding."""
        return await self._async_embed_query(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        """Asynchronous text embedding."""
        return (await self._async_embed_documents([text]))[0]

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Asynchronous batch text embedding."""
        return await self._async_embed_documents(texts)

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        return self._dimension


class VoyageEmbeddingWrapper(BaseEmbeddingWrapper):
    """Wrapper for Voyage AI embedding models."""
    
    def __init__(
        self,
        model_name: str = "voyage-multimodal-3",
        api_key: str = None,
        **kwargs
    ):
        self.model_name = model_name
        self.api_key = api_key or os.getenv("VOYAGE_API_KEY")
        self.kwargs = kwargs
        
        if not self.api_key:
            raise ValueError("Voyage API key is required. Set VOYAGE_API_KEY environment variable or pass api_key parameter.")
        
        # Initialize the embedding model
        self._embedding_model = VoyageEmbedding(
            model_name=model_name,
            api_key=self.api_key,
            **kwargs
        )
        
        logger.info(f"Voyage embedding wrapper initialized with model: {model_name}")

    def get_embedding_model(self) -> BaseEmbedding:
        """Return the LlamaIndex-compatible embedding model."""
        return self._embedding_model

    def validate_model(self, provider: EmbeddingProvider, model_name: str) -> bool:
        """Validate if the model is supported for Voyage AI.
        
        Args:
            provider (EmbeddingProvider): The embedding provider.
            model_name (str): The name of the embedding model to validate.
            
        Returns:
            bool: True if the model is supported, False otherwise.
        """
        supported_models = [
            "voyage-multimodal-3",
        ]
        return model_name in supported_models

    @property
    def dimensions(self) -> int:
        """Return the embedding dimension."""
        return self._embedding_model.dimension
