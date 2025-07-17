import sys
import subprocess
from typing import List, Dict, Optional

from llama_index.core.embeddings import BaseEmbedding

from evoagentx.core.logging import logger
from .base import BaseEmbeddingWrapper, EmbeddingProvider, SUPPORTED_MODELS

try:
    from ollama import Client
except ImportError:
    logger.warning("The 'ollama' library is not installed. Attempting to install it.")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "ollama"])
        from ollama import Client
    except subprocess.CalledProcessError:
        logger.error("Failed to install 'ollama'. Please install it manually using 'pip install ollama'.")
        raise ImportError("The 'ollama' library is required.")


MODEL_DIMENSIONS = {
    "nomic-embed-text": 384,
    # "mxbai-embed-large": ,
    # "bge-m3": ,
    # "all-minilm": ,
    # "snowflake-arctic-embed",
}

class OllamaEmbedding(BaseEmbedding):
    """Ollama embedding model compatible with LlamaIndex BaseEmbedding."""
    
    base_url: str = None
    client: Client = None
    model_name: str = "nomic-embed-text"
    embed_batch_size: int = 10
    embedding_dims: int = None
    kwargs: Optional[Dict] = {}
    
    def __init__(
        self,
        model_name: str = "nomic-embed-text",
        base_url: str = None,
        embedding_dims: int = None,
        **kwargs
    ):
        super().__init__(model_name=model_name, embed_batch_size=10)
        self.base_url = base_url or "http://localhost:11434"
        self.embedding_dims = embedding_dims or 512
        self.kwargs = kwargs

        if not EmbeddingProvider.validate_model(EmbeddingProvider.OLLAMA, model_name):
            raise ValueError(f"Unsupported Ollama model: {model_name}. Supported models: {SUPPORTED_MODELS['ollama']}")

        try:
            self.client = Client(host=self.base_url)
            self._ensure_model_exists()
            logger.debug(f"Initialized Ollama embedding model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Ollama client: {str(e)}")
            raise

    def _ensure_model_exists(self):
        """Ensure the specified model exists locally, pulling it if necessary."""
        try:
            local_models = self.client.list()["models"]
            if not any(model.get("name") == self.model_name for model in local_models):
                logger.info(f"Pulling Ollama model: {self.model_name}")
                self.client.pull(self.model_name)
        except Exception as e:
            logger.error(f"Failed to ensure Ollama model exists: {str(e)}")
            raise

    def _get_query_embedding(self, query: str) -> List[float]:
        """Get embedding for a query string."""
        try:
            response = self.client.embeddings(model=self.model_name, prompt=query, **self.kwargs)
            return response["embedding"]
        except Exception as e:
            logger.error(f"Failed to encode query: {str(e)}")
            raise

    def _get_text_embedding(self, text: str) -> List[float]:
        """Get embedding for a text string."""
        try:
            response = self.client.embeddings(model=self.model_name, prompt=text, **self.kwargs)
            return response["embedding"]
        except Exception as e:
            logger.error(f"Failed to encode text: {str(e)}")
            raise

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for a list of texts synchronously."""
        try:
            embeddings = []
            for i in range(0, len(texts), self.embed_batch_size):
                batch = texts[i:i + self.embed_batch_size]
                batch_embeddings = [self._get_text_embedding(text) for text in batch]
                embeddings.extend(batch_embeddings)
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
        return self.embedding_dims


class OllamaEmbeddingWrapper(BaseEmbeddingWrapper):
    """Wrapper for Ollama embedding models."""
    
    def __init__(
        self,
        model_name: str = "nomic-embed-text",
        base_url: str = None,
        dimensions: int = None,
        **kwargs
    ):
        self.model_name = model_name
        self.base_url = base_url
        self._dimensions = MODEL_DIMENSIONS.get(model_name, None) or dimensions 
        self.kwargs = kwargs
        self._embedding_model = None
        self._embedding_model = self.get_embedding_model()

    def get_embedding_model(self) -> BaseEmbedding:
        """Return the LlamaIndex-compatible embedding model."""
        if self._embedding_model is None:
            try:
                self._embedding_model = OllamaEmbedding(
                    model_name=self.model_name,
                    base_url=self.base_url,
                    embedding_dims=self._dimensions,
                    **self.kwargs
                )
                logger.debug(f"Initialized Ollama embedding wrapper for model: {self.model_name}")
            except Exception as e:
                logger.error(f"Failed to initialize Ollama embedding wrapper: {str(e)}")
                raise
        return self._embedding_model
    
    @property
    def dimensions(self) -> int:
        """Return the embedding dimensions."""
        return self._dimensions