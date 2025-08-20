from typing import List, Optional, Dict

import torch
from PIL import Image
from colpali_engine.models import BiQwen2_5, BiQwen2_5_Processor
from llama_index.core.embeddings import BaseEmbedding

from evoagentx.core.logging import logger
from .base import BaseEmbeddingWrapper


class MultimodalHuggingFaceEmbedding(BaseEmbedding):
    """Multimodal HuggingFace embedding model compatible with LlamaIndex BaseEmbedding."""
    
    model: Optional[object] = None
    processor: Optional[object] = None
    device: Optional[object] = None
    _dimension: int = None
    model_name: str = "nomic-ai/nomic-embed-multimodal-7b"
    embed_batch_size: int = 1
    device_str: Optional[str] = None
    model_kwargs: Dict = {}
    
    def __init__(
        self,
        model_name: str = "nomic-ai/nomic-embed-multimodal-7b",
        device: Optional[str] = None,
        **model_kwargs
    ):
        super().__init__(model_name=model_name, embed_batch_size=1)
        self.device_str = device
        self.model_kwargs = model_kwargs or {}



        try:
            self._initialize_model()
            logger.debug(f"Initialized multimodal HuggingFace embedding model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize multimodal HuggingFace embedding: {str(e)}")
            raise

    def _initialize_model(self):
        """Initialize the BiQwen2.5 model."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device_str:
            self.device = torch.device(self.device_str)
        
        self.model = BiQwen2_5.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16 if self.device.type == 'cuda' else torch.float32,
            device_map=self.device,
            **self.model_kwargs
        ).to(self.device).eval()
    
        self.processor = BiQwen2_5_Processor.from_pretrained(self.model_name)
        
        # Get actual dimension from model
        with torch.no_grad():
            dummy_text = self.processor.process_queries(["test"]).to(self.device)
            dummy_output = self.model(**dummy_text)
            self._dimension = dummy_output.shape[-1]

    async def embed_document(self, image):
        """Embed a document (image)."""
        with torch.inference_mode():
            batch_image = self.processor.process_images([image]).to(self.device)
            embedding = self.model(**batch_image)
            embedding = embedding.cpu().float().numpy().tolist()

        return embedding[0]

    async def embed_query(self, query: str) -> List[float]:
        """Embed a text query."""
        try:
            with torch.inference_mode():
                batch_query = self.processor.process_queries([query]).to(self.device)
                embedding = self.model(**batch_query)
                embedding = embedding.cpu().float().numpy().tolist()

            return embedding[0]
        except Exception as e:
            logger.error(f"Failed to embed query: {str(e)}")
            raise

    def _get_query_embedding(self, query: str) -> List[float]:
        """Get embedding for a query string."""
        try:
            import asyncio
            return asyncio.run(self.embed_query(query))
        except Exception as e:
            logger.error(f"Failed to encode query: {str(e)}")
            raise

    def _get_text_embedding(self, text: str) -> List[float]:
        """Get embedding for a text string."""
        import asyncio
        return asyncio.run(self.embed_query(text))

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for a list of texts."""
        embeddings = []
        for text in texts:
            embeddings.append(self._get_text_embedding(text))
        return embeddings
    
    def _get_image_embedding(self, image_node) -> List[float]:
        """Get embedding for an ImageNode."""
        # ImageNode.image contains the PIL Image
        return self.get_image_embedding(image_node.image)

    def get_image_embedding(self, image) -> List[float]:
        """Get embedding for an image."""
        import asyncio
        return asyncio.run(self.embed_document(image))

    async def _aget_query_embedding(self, query: str) -> List[float]:
        """Asynchronous query embedding."""
        return await self.embed_query(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        """Asynchronous text embedding."""
        return await self.embed_query(text)

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Asynchronous batch text embedding."""
        embeddings = []
        for text in texts:
            embeddings.append(await self.embed_query(text))
        return embeddings

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        return self._dimension


class MultimodalHuggingFaceEmbeddingWrapper(BaseEmbeddingWrapper):
    """Wrapper for multimodal HuggingFace embedding models."""
    
    def __init__(
        self,
        model_name: str = "nomic-ai/nomic-embed-multimodal-7b",
        device: Optional[str] = None,
        **model_kwargs
    ):
        self.model_name = model_name
        self.device = device
        self.model_kwargs = model_kwargs
        self._embedding_model = None
        self._embedding_model = self.get_embedding_model()

    def get_embedding_model(self) -> BaseEmbedding:
        """Return the LlamaIndex-compatible embedding model."""
        if self._embedding_model is None:
            try:
                self._embedding_model = MultimodalHuggingFaceEmbedding(
                    model_name=self.model_name,
                    device=self.device,
                    **self.model_kwargs
                )
                logger.debug(f"Initialized multimodal HuggingFace embedding wrapper for model: {self.model_name}")
            except Exception as e:
                logger.error(f"Failed to initialize multimodal HuggingFace embedding wrapper: {str(e)}")
                raise
        return self._embedding_model
    
    @property
    def dimensions(self) -> int:
        """Return the embedding dimensions."""
        return self._embedding_model.dimension

    def embed_image(self, image) -> List[float]:
        """Embed an image directly."""
        return self._embedding_model.get_image_embedding(image)
    
    def embed_text(self, text: str) -> List[float]:
        """Embed text directly."""
        return self._embedding_model._get_text_embedding(text)
