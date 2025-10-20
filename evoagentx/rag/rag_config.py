from typing import Optional, Callable, Dict, Any, List

from pydantic import Field

from evoagentx.core.base_config import BaseConfig, BaseModule
from .indexings.base import IndexType
from .chunkers.base import ChunkingStrategy
from .embeddings.base import EmbeddingProvider
from .retrievers.base import RetrieverType
from .postprocessors.base import RerankerType


class ReaderConfig(BaseModule):
    """Configuration for document reading stage."""
    recursive: bool = Field(default=False, description="Whether to recursively read directories.")
    exclude_hidden: bool = Field(default=True, description="Exclude hidden files and directories.")
    num_files_limit: Optional[int] = Field(default=None, description="Maximum number of files to read.")
    custom_metadata_function: Optional[Callable] = Field(default=None, description="Custom function to extract metadata from files.")
    extern_file_extractor: Optional[Dict[str, Any]] = Field(default=None, description="External file extractors for specific file types.")
    errors: str = Field(default="ignore", description="Error handling strategy ('ignore', 'strict').")
    encoding: str = Field(default="utf-8", description="File encoding for reading.")


class ChunkerConfig(BaseModule):
    """Configuration for document chunking stage."""
    strategy: str = Field(default=ChunkingStrategy.SIMPLE, description="Chunking strategy (SIMPLE, SEMANTIC, HIERARCHICAL).")
    chunk_size: int = Field(default=1024, description="Maximum size of each chunk in characters.")
    chunk_overlap: int = Field(default=20, description="Overlap between chunks in characters.")
    max_chunks: Optional[int] = Field(default=None, description="Maximum number of chunks per document.")


class EmbeddingConfig(BaseModule):
    """Configuration for embedding stage."""
    provider: str = Field(default=EmbeddingProvider.OPENAI, description="Embedding provider (OPENAI, HUGGINGFACE).")
    model_name: str = Field(default="text-embedding-ada-002", description="Name of the embedding model.")
    api_key: Optional[str] = Field(default=None, description="API key for the embedding provider (if required).")
    api_url: str = Field(default="https://api.openai.com/v1", description="api url for embedding model.")
    dimensions: Optional[int] = Field(default=None, description="Dimensions of the embedding model.")
    normalize: Optional[bool] = Field(default=True, description="Whether to normalize the embedding model(huggingface).")
    device: Optional[str] = Field(default=None, description="The device was used for embedding model.")

class IndexConfig(BaseModule):
    """Configuration for indexing stage."""
    index_type: str = Field(default=IndexType.VECTOR, description="Index type (VECTOR, GRAPH, SUMMARY, TREE).")


class RetrievalConfig(BaseModule):
    """Configuration for retrieval stage.(pre-retrieve, retrieve, post-retrieve)"""
    # prepross_type: str = Field(default="")
    retrivel_type: str = Field(default=RetrieverType.VECTOR, description="The type of retriver for retrieve.")
    postprocessor_type: str = Field(default=RerankerType.SIMPLE, description="The type of postprocessor for retrieve.")
    top_k: int = Field(default=5, description="Number of top results to retrieve.")
    similarity_cutoff: Optional[float] = Field(default=0.7, description="Minimum similarity score for retrieved chunks.")
    keyword_filters: Optional[List[str]] = Field(default=None, description="Keywords to filter retrieved chunks.")
    metadata_filters: Optional[Dict[str, Any]] = Field(default=None, description="Metadata filters for retrieval.")


class RAGConfig(BaseConfig):
    """Configuration for the RAG pipeline."""
    modality: str = Field(default="text", description="RAG modality: 'text' for text-only, 'multimodal' for images.")
    num_workers: Optional[int] = Field(default=None, description="Number of workers for parallel processing (e.g., reading, retrieval).")
    reader: ReaderConfig = Field(default_factory=ReaderConfig, description="Configuration for document reading.")
    chunker: ChunkerConfig = Field(default_factory=ChunkerConfig, description="Configuration for document chunking.")
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig, description="Configuration for embeddings.")
    index: IndexConfig = Field(default_factory=IndexConfig, description="Configuration for indexing.")
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig, description="Configuration for retrieval.")