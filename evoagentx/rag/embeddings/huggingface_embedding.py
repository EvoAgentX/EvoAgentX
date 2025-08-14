from typing import List, Optional, Dict, ClassVar, Any

from llama_index.core.embeddings import BaseEmbedding

from evoagentx.core.logging import logger
from .base import BaseEmbeddingWrapper, EmbeddingProvider, SUPPORTED_MODELS


class HuggingFaceEmbedding(BaseEmbedding):
    """HuggingFace embedding model compatible with LlamaIndex BaseEmbedding."""
    
    model: ClassVar[Any] = None
    _dimension: Optional[int] = None
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embed_batch_size: int = 10
    device: Optional[str] = None
    normalize: bool = False
    model_kwargs: Dict[str, Any] = {}
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None,
        normalize: bool = False,
        **model_kwargs
    ):
        super().__init__(model_name=model_name, embed_batch_size=10)
        self.device = device
        self.normalize = normalize
        self.model_kwargs = model_kwargs or {}

        if not EmbeddingProvider.validate_model(EmbeddingProvider.HUGGINGFACE, model_name):
            raise ValueError(f"Unsupported HuggingFace model: {model_name}. Supported models: {SUPPORTED_MODELS['huggingface']}")
        
        # Lazy import and initialization
        self._initialize_model()

    def _initialize_model(self):
        """Lazy initialize the SentenceTransformer model."""
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name, device=self.device, **self.model_kwargs)
            self._dimension = self.model.get_sentence_embedding_dimension()
            logger.debug(f"Initialized HuggingFace embedding model: {self.model_name}")
        except ImportError:
            raise ImportError(
                "sentence-transformers package is required for HuggingFace embeddings. "
                "Please install it with: pip install sentence-transformers"
            )
        except Exception as e:
            logger.error(f"Failed to initialize HuggingFace embedding: {str(e)}")
            raise

    def _ensure_model_loaded(self):
        """Ensure the model is loaded before use."""
        if self.model is None:
            self._initialize_model()

    def _get_query_embedding(self, query: str) -> List[float]:
        """Get embedding for a query string."""
        self._ensure_model_loaded()
        try:
            embedding = self.model.encode(
                query,
                normalize_embeddings=self.normalize,
                convert_to_numpy=True
            ).tolist()
            return embedding
        except Exception as e:
            logger.error(f"Failed to encode query: {str(e)}")
            raise

    def _get_text_embedding(self, text: str) -> List[float]:
        """Get embedding for a text string."""
        self._ensure_model_loaded()
        try:
            embedding = self.model.encode(
                text,
                normalize_embeddings=self.normalize,
                convert_to_numpy=True
            ).tolist()
            return embedding
        except Exception as e:
            logger.error(f"Failed to encode text: {str(e)}")
            raise

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for a list of texts synchronously."""
        self._ensure_model_loaded()
        try:
            embeddings = self.model.encode(
                texts,
                normalize_embeddings=self.normalize,
                convert_to_numpy=True,
                batch_size=self.embed_batch_size
            ).tolist()
            return embeddings
        except Exception as e:
            logger.error(f"Failed to encode texts: {str(e)}")
            raise

    async def _aget_query_embedding(self, query: str) -> List[float]:
        """Asynchronous query embedding (falls back to sync)."""
        return self._get_query_embedding(query)

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        if self._dimension is None:
            self._ensure_model_loaded()
        return self._dimension


class HuggingFaceEmbeddingWrapper(BaseEmbeddingWrapper):
    """Wrapper for HuggingFace embedding models."""
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None,
        normalize: bool = True,
        **model_kwargs
    ):
        self.model_name = model_name
        self.device = device
        self.normalize = normalize
        self.model_kwargs = model_kwargs
        self._embedding_model = None

    def get_embedding_model(self) -> BaseEmbedding:
        """Return the LlamaIndex-compatible embedding model."""
        if self._embedding_model is None:
            try:
                self._embedding_model = HuggingFaceEmbedding(
                    model_name=self.model_name,
                    device=self.device,
                    normalize=self.normalize,
                    **self.model_kwargs
                )
                logger.debug(f"Initialized HuggingFace embedding wrapper for model: {self.model_name}")
            except Exception as e:
                logger.error(f"Failed to initialize HuggingFace embedding wrapper: {str(e)}")
                raise
        return self._embedding_model
    
    @property
    def dimensions(self) -> int:
        """Return the embedding dimensions."""
        return self.get_embedding_model().dimension