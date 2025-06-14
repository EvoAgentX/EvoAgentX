from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from .base import BaseEmbeddingWrapper
from evoagentx.core.logging import logger

class HuggingFaceEmbeddingWrapper(BaseEmbeddingWrapper):
    """Wrapper for HuggingFace embedding models."""
    
    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en",
        **kwargs
    ):
        self.model_name = model_name
        self.kwargs = kwargs
    
    def get_embedding_model(self) -> HuggingFaceEmbedding:
        """Return the HuggingFace embedding model."""
        try:
            model = HuggingFaceEmbedding(
                model_name=self.model_name,
                **self.kwargs
            )
            logger.debug(f"Initialized HuggingFace embedding model: {self.model_name}")
            return model
        except Exception as e:
            logger.error(f"Failed to initialize HuggingFace embedding: {str(e)}")
            raise