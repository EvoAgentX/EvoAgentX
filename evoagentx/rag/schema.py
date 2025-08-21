
import json
import hashlib
from abc import ABC, abstractmethod
from uuid import uuid4
from typing import List, Dict, Optional, Union, Any, TypeVar, Generic
from pathlib import Path

from pydantic import Field
from llama_index.core.schema import QueryBundle
from llama_index.core import Document as LlamaIndexDocument
from llama_index.core.schema import BaseNode, TextNode, ImageNode, RelatedNodeInfo

from evoagentx.core.base_config import BaseModule
from evoagentx.core.logging import logger


DEAFULT_EXCLUDED = ['file_name', 'file_type', 'file_size', 'page_count', 'creation_date', 
                        'last_modified_date', 'language', 'word_count', 'custom_fields', 'hash_doc']

# Generic type for chunk types
ChunkType = TypeVar('ChunkType', bound='BaseChunk')
class DocumentMetadata(BaseModule):
    """
    This class ensures type safety and validation for metadata associated with a document,
    such as file information, creation date, and custom fields.
    """

    file_name: Optional[str] = Field(default=None, description="The name of the document file, excluding the path.")
    file_path: Optional[str] = Field(default=None, description="The file path or URL where the document is stored.")
    file_type: Optional[str] = Field(default=None, description="The type of the document (e.g., '.pdf', '.docx', '.md', '.txt').")
    file_size: Optional[int] = Field(default=None, description="The size of the document.")
    page_count: Optional[int] = Field(default=None, description="The number of pages in the document, if applicable (e.g., for PDFs).")
    creation_date: Optional[str] = Field(default=None, description="The creation date and time of the document.")
    last_modified_date: Optional[str] = Field(default=None, description="The last modified date and time of the document.")
    language: Optional[str] = Field(default=None, description="The primary language of the document (e.g., 'en', 'zh').")
    word_count: Optional[int] = Field(default=None, description="The number of words in the document, calculated during initialization.")
    custom_fields: Dict[str, Any] = Field(default_factory=dict, description="A dictionary for storing additional user-defined metadata.")
    hash_doc: Optional[str] = Field(default=None, description="The hash code of this Document for deduplication")


class ChunkMetadata(DocumentMetadata):
    """
    This class holds metadata for a chunk, including its relationship to the parent document,
    chunking parameters, and retrieval-related information.
    """

    doc_id: str = Field(description="The unique identifier of the parent document.")
    corpus_id: Optional[str] = Field(default=None, description="The unique identifier of the Corpus(Indexing).")
    chunk_size: Optional[int] = Field(default=None, description="The size of the chunk in characters, if applicable.")
    chunk_overlap: Optional[int] = Field(default=None, description="The number of overlapping characters between adjacent chunks.")
    chunk_index: Optional[int] = Field(default=None, description="The index of the chunk within the parent document.")
    chunking_strategy: Optional[str] = Field(default=None, description="The strategy used to create the chunk (e.g., 'simple', 'semantic', 'tree').")
    similarity_score: Optional[float] = Field(default=None, description="Similarity score from retrieval.")


class IndexMetadata(BaseModule):
    corpus_id: str = Field(..., description="Identifier for the corpus")
    index_type: str = Field(..., description="Type of index (e.g., 'vector', 'graph', 'summary', 'tree')")
    collection_name: Optional[str] = Field(default="default_collection", description="Vector store collection name or FAISS file path")
    dimension: Optional[int] = Field(default=1536, description="Vector dimension")
    vector_db_type: Optional[str] = Field(default=None, description="Vector database type (e.g., 'faiss', 'qdrant', 'chroma')")
    graph_db_type: Optional[str] = Field(default=None, description="Graph database type (e.g., 'neo4j')")
    embedding_model_name: Optional[str] = Field(default=None, description="")
    date: Optional[str] = Field(default=None, description="Creation or last update date")


class Document(BaseModule):
    """A custom document class for managing documents in the RAG pipeline.

    Attributes:
        text (str): The full content of the document.
        doc_id (str): Unique identifier for the document.
        metadata (DocumentMetadata): Metadata including file info, creation date, etc.
        source (str): Source of the document (e.g., file path or URL).
        llama_doc (LlamaIndexDocument): Underlying LlamaIndex Document object.
    """

    def __init__(
        self,
        text: str,
        metadata: Optional[Union[Dict, DocumentMetadata]] = None,
        embedding: Optional[List[float]] = None,
        doc_id: Optional[str] = None,
        excluded_embed_metadata_keys: List[str] = DEAFULT_EXCLUDED,
        excluded_llm_metadata_keys: List[str] = DEAFULT_EXCLUDED,
        relationships: Dict[str, RelatedNodeInfo] = {}, 
        metadata_template: str = '{key}: {value}', 
        metadata_separator: str = '\n',
        text_template: str = '{metadata_str}\n\n{content}'
    ):
        metadata = (
            DocumentMetadata.model_validate(metadata) if isinstance(metadata, dict) else metadata or DocumentMetadata()
        )
        
        super().__init__(
            text=text.strip(),
            doc_id=doc_id or str(uuid4()),
            metadata=metadata,
            embedding=embedding,
            excluded_embed_metadata_keys=list(set(DEAFULT_EXCLUDED + excluded_embed_metadata_keys)),
            excluded_llm_metadata_keys=list(set(DEAFULT_EXCLUDED + excluded_llm_metadata_keys)),
            relationships=relationships,
            metadata_template=metadata_template,
            metadata_separator=metadata_separator,
            text_template=text_template,
        )
        self.metadata.word_count = len(self.text.split())

    def to_llama_document(self) -> LlamaIndexDocument:
        """Convert to LlamaIndex Document."""
        return LlamaIndexDocument(
            text=self.text,
            metadata=self.metadata.model_dump(),
            id_=self.doc_id,
            embedding=self.embedding,
            excluded_llm_metadata_keys=self.excluded_llm_metadata_keys,
            excluded_embed_metadata_keys=self.excluded_embed_metadata_keys,
            relationships=self.relationships,
            metadata_template=self.metadata_template,
            metadata_separator=self.metadata_separator,
            text_template=self.text_template,
        )

    @classmethod
    def from_llama_document(cls, llama_doc: LlamaIndexDocument) -> "Document":
        """Create Document from LlamaIndex Document."""
        metadata = DocumentMetadata.model_validate(llama_doc.metadata)
        return cls(
            text=llama_doc.text,
            metadata=metadata,
            doc_id=llama_doc.id_,
            embedding=llama_doc.embedding,
            excluded_llm_metadata_keys=llama_doc.excluded_llm_metadata_keys,
            excluded_embed_metadata_keys=llama_doc.excluded_llm_metadata_keys,
            relationships=llama_doc.relationships,
            metadata_template=llama_doc.metadata_template,
            metadata_separator=llama_doc.metadata_separator,
            text_template=llama_doc.text_template
        )

    def set_embedding(self, embedding: List[float]):
        """Set the embedding vector for the Document."""
        self.embedding = embedding

    def compute_hash(self) -> str:
        """Compute a hash of the document text for deduplication."""
        return hashlib.sha256(self.text.encode()).hexdigest()

    def get_fragment(self, max_length: int = 100) -> str:
        """Return a fragment of the document text."""
        return (self.text[:max_length] + "...") if len(self.text) > max_length else self.text

    def to_dict(self) -> Dict:
        """Convert document to dictionary for serialization."""
        return {
            "doc_id": self.doc_id,
            "text": self.text,
            "metadata": self.metadata.model_dump(),
            "embedding": self.embedding,
            "excluded_embed_metadata_keys": self.excluded_embed_metadata_keys,
            "excluded_llm_metadata_keys": self.excluded_llm_metadata_keys,
            "relationships": {str(k): v for k, v in self.relationships.items()},
            "metadata_template": self.metadata_template,
            "metadata_separator": self.metadata_separator,
            "text_template": self.text_template,
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert document to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def __str__(self) -> str:
        return (
            f"Document(id={self.doc_id}, embedding={self.embedding}, metadata={self.metadata.model_dump()}"
            f"fragment={self.get_fragment(max_length=300)})"
        )

    def __repr__(self) -> str:
        return (
            f"Document(doc_id={self.doc_id}, embedding={self.embedding}, metadata={self.metadata.model_dump()},"
            f"fragment={self.get_fragment(max_length=300)})"
        )


class BaseChunk(BaseModule, ABC):
    """Abstract base class for all chunk types.
    
    Provides common interface and functionality shared by text and image chunks.
    Subclasses implement specific data access patterns.
    """
    
    @property
    @abstractmethod
    def chunk_id(self) -> str:
        """Unique identifier for the chunk."""
        pass
    
    @property
    @abstractmethod
    def metadata(self):
        """Chunk metadata."""
        pass
    
    @abstractmethod
    def to_llama_node(self) -> BaseNode:
        """Convert to appropriate LlamaIndex Node."""
        pass
    
    @classmethod
    @abstractmethod
    def from_llama_node(cls, node: BaseNode) -> "BaseChunk":
        """Create chunk from LlamaIndex Node."""
        pass
    
    @abstractmethod
    def get_fragment(self, max_length: int = 100) -> str:
        """Return a fragment for display."""
        pass


# Backward compatibility alias
Chunk = TextChunk


class TextChunk(BaseChunk):
    """A text-based chunk implementing BaseChunk interface.
    
    Attributes:
        text (str): The text content of the chunk.
        chunk_id (str): Unique identifier for the chunk.
        metadata (ChunkMetadata): Metadata including chunk size, embedding, etc.
    """

    def __init__(
        self,
        text: str,
        chunk_id: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        start_char_idx: Optional[int] = None,
        end_char_idx: Optional[int] = None,
        excluded_embed_metadata_keys: List[str] = DEAFULT_EXCLUDED,
        excluded_llm_metadata_keys: List[str] = DEAFULT_EXCLUDED,
        text_template: str = '{metadata_str}\n\n{content}',
        relationships: Dict[str, RelatedNodeInfo] = {},
        metadata: Optional[Union[Dict, ChunkMetadata]] = None,
    ):
        metadata = (
            ChunkMetadata.model_validate(metadata) if isinstance(metadata, dict) else metadata or ChunkMetadata()
        )
        super().__init__(
            text=text.strip(),
            chunk_id=chunk_id or str(uuid4()),
            embedding=embedding,
            start_char_idx=start_char_idx,
            end_char_idx=end_char_idx,
            excluded_embed_metadata_keys=list(set(DEAFULT_EXCLUDED + excluded_embed_metadata_keys)),
            excluded_llm_metadata_keys=list(set(DEAFULT_EXCLUDED + excluded_llm_metadata_keys)),
            text_template=text_template,
            relationships=relationships,
            metadata=metadata,
        )
        self.metadata.word_count = len(self.text.split())

    def to_llama_node(self) -> TextNode:
        """Convert to LlamaIndex TextNode."""
        relationships = dict()
        for k, v in self.relationships.items():
            relationships[k] = v if isinstance(v, RelatedNodeInfo) else RelatedNodeInfo.from_dict(v)
        return TextNode(
            text=self.text,
            metadata=self.metadata.model_dump(),
            id_=self.chunk_id,
            embedding=self.embedding,
            start_char_idx=self.start_char_idx,
            end_char_idx=self.end_char_idx,
            excluded_llm_metadata_keys=self.excluded_llm_metadata_keys,
            excluded_embed_metadata_keys=self.excluded_embed_metadata_keys,
            text_template=self.text_template,
            relationships=relationships
        )

    @classmethod
    def from_llama_node(cls, node: TextNode) -> "TextChunk":
        """Create TextChunk from LlamaIndex TextNode."""
        metadata = ChunkMetadata.model_validate(node.metadata)
        return cls(
            chunk_id=node.id_,
            text=node.text,
            metadata=metadata,
            embedding=node.embedding,
            start_char_idx=getattr(node, "start_char_idx", None),
            end_char_idx=getattr(node, "end_char_idx", None),
            excluded_embed_metadata_keys=node.excluded_embed_metadata_keys,
            excluded_llm_metadata_keys=node.excluded_llm_metadata_keys,
            text_template=node.text_template,
            relationships=node.relationships
        )

    def get_fragment(self, max_length: int = 100) -> str:
        """Return a fragment of the text content."""
        return (self.text[:max_length] + "...") if len(self.text) > max_length else self.text


class ImageChunk(BaseChunk):
    """An image-based chunk with lazy loading.
    
    Attributes:
        image_path (str): Path to the image file.
        image_mimetype (Optional[str]): MIME type of the image.
        chunk_id (str): Unique identifier for the chunk.
        metadata (ChunkMetadata): Metadata including embedding, similarity scores, etc.
    """

    def __init__(
        self,
        image_path: str,
        image_mimetype: Optional[str] = None,
        chunk_id: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        excluded_embed_metadata_keys: List[str] = DEAFULT_EXCLUDED,
        excluded_llm_metadata_keys: List[str] = DEAFULT_EXCLUDED,
        text_template: str = '{metadata_str}\n\n{content}',
        relationships: Dict[str, RelatedNodeInfo] = {},
        metadata: Optional[Union[Dict, ChunkMetadata]] = None,
    ):
        metadata = (
            ChunkMetadata.model_validate(metadata) if isinstance(metadata, dict) else metadata or ChunkMetadata()
        )
        super().__init__(
            image_path=image_path,
            image_mimetype=image_mimetype,
            chunk_id=chunk_id or str(uuid4()),
            embedding=embedding,
            excluded_embed_metadata_keys=list(set(DEAFULT_EXCLUDED + excluded_embed_metadata_keys)),
            excluded_llm_metadata_keys=list(set(DEAFULT_EXCLUDED + excluded_llm_metadata_keys)),
            text_template=text_template,
            relationships=relationships,
            metadata=metadata,
        )
        
        # Private cache for lazy loading
        self._cached_image = None

    def get_image(self):
        """Load PIL Image on-demand with caching."""
        if self._cached_image is None:
            from PIL import Image
            try:
                self._cached_image = Image.open(self.image_path)
                logger.debug(f"Loaded image from {self.image_path}")
            except Exception as e:
                logger.error(f"Failed to load image from {self.image_path}: {str(e)}")
                return None
        return self._cached_image
    
    def get_image_bytes(self, format: str = "PNG") -> Optional[bytes]:
        """Get image as bytes for embedding or processing."""
        import io
        image = self.get_image()
        if image is None:
            return None
        
        img_bytes = io.BytesIO()
        image.save(img_bytes, format=format)
        return img_bytes.getvalue()

    def to_llama_node(self) -> ImageNode:
        """Convert to LlamaIndex ImageNode with on-demand image loading."""
        relationships = dict()
        for k, v in self.relationships.items():
            relationships[k] = v if isinstance(v, RelatedNodeInfo) else RelatedNodeInfo.from_dict(v)
        
        # Load image data on-demand for LlamaIndex compatibility
        image_data = self.get_image_bytes() if self.image_path else None
        
        return ImageNode(
            text="",  # No text content for pure image chunks
            image=image_data,       # Loaded on-demand
            image_path=self.image_path,
            image_mimetype=self.image_mimetype,
            metadata=self.metadata.model_dump(),
            id_=self.chunk_id,
            embedding=self.embedding,
            excluded_llm_metadata_keys=self.excluded_llm_metadata_keys,
            excluded_embed_metadata_keys=self.excluded_embed_metadata_keys,
            text_template=self.text_template,
            relationships=relationships
        )

    @classmethod
    def from_llama_node(cls, node: ImageNode) -> "ImageChunk":
        """Create ImageChunk from LlamaIndex ImageNode."""
        metadata = ChunkMetadata.model_validate(node.metadata)
        
        return cls(
            chunk_id=node.id_,
            image_path=node.image_path,
            image_mimetype=node.image_mimetype,
            metadata=metadata,
            embedding=node.embedding,
            excluded_embed_metadata_keys=node.excluded_embed_metadata_keys,
            excluded_llm_metadata_keys=node.excluded_llm_metadata_keys,
            text_template=node.text_template,
            relationships=node.relationships
        )

    def get_fragment(self, max_length: int = 100) -> str:
        """Return a fragment representing the image."""
        filename = Path(self.image_path).name
        return f"[Image: {filename}]"


class Corpus(BaseModule, Generic[ChunkType]):
    """A generic collection of document chunks for RAG processing.

    Attributes:
        corpus_id (str): The unique id for corpus.
        chunks (List[ChunkType]): List of chunks in the corpus.
        chunk_index (Dict[str, ChunkType]): Index of chunks by chunk_id for fast lookup.
        metadata (Optional[IndexMetadata]): the metadata for this corpus.
    """

    def __init__(self, chunks: List[ChunkType] = None, corpus_id: str = None, 
                 metadata:Optional[Union[IndexMetadata, Dict]]=None):
        corpus_id = uuid4() if corpus_id is None else corpus_id
        chunks = [] if chunks is None else chunks
        chunk_index = {} if chunks is None else {chunk.chunk_id: chunk for chunk in chunks}
        
        if metadata is None:
            metadata = {}
        elif isinstance(metadata, IndexMetadata):
            metadata = metadata.model_dump()
        super().__init__(
            corpus_id=corpus_id,
            chunks=chunks,
            chunk_index=chunk_index,
            metadata=metadata
        )

    def to_llama_nodes(self) -> List[BaseNode]:
        """Convert to list of LlamaIndex Nodes."""
        if not self.chunks:
            self.chunks = []
        return [chunk.to_llama_node() for chunk in self.chunks]

    @classmethod
    def from_llama_nodes(cls, nodes: List[BaseNode]) -> "Corpus":
        """Create a Corpus from a list of LlamaIndex Nodes.

        Args:
            nodes (List[BaseNode]): The LlamaIndex Nodes to convert.
            doc_id (str): The ID of the parent document.

        Returns:
            Corpus: A new Corpus instance.
        """
        return cls([Chunk.from_llama_node(node) for node in nodes])

    def add_chunk(self, batch_chunk: Union[ChunkType, List[ChunkType]]):
        """Add a batch chunk to the corpus and update index."""
        if not isinstance(batch_chunk, list):
            batch_chunk = [batch_chunk]

        for chunk in batch_chunk:
            self.chunks.append(chunk)
            self.chunk_index[chunk.chunk_id] = chunk

    def get_chunk(self, chunk_id: str) -> Optional[ChunkType]:
        """Retrieve a chunk by its ID."""
        return self.chunk_index.get(chunk_id)

    def remove_chunk(self, chunk_id: str):
        """Remove a chunk by its ID."""
        self.chunks = [chunk for chunk in self.chunks if chunk.chunk_id != chunk_id]
        self.chunk_index.pop(chunk_id, None)

    def filter_by_doc_id(self, doc_id: str) -> List[ChunkType]:
        """Filter chunks by parent document ID."""
        return [chunk for chunk in self.chunks if hasattr(chunk, 'doc_id') and chunk.doc_id == doc_id]

    def filter_by_similarity(self, threshold: float) -> List[ChunkType]:
        """Filter chunks by similarity score."""
        return [chunk for chunk in self.chunks if chunk.metadata.similarity_score and chunk.metadata.similarity_score >= threshold]

    def sort_by_similarity(self, reverse: bool = True) -> List[ChunkType]:
        """Sort chunks by similarity score (descending by default)."""
        return sorted(
            [chunk for chunk in self.chunks if chunk.metadata.similarity_score is not None],
            key=lambda x: x.metadata.similarity_score,
            reverse=reverse
        )

    def to_dict(self, round_trip=False) -> Dict:
        """Convert corpus to dictionary for serialization."""
        return [self.model_dump(round_trip=round_trip)]

    def to_json(self, indent: int = 2, round_trip=True) -> str:
        """Convert corpus to JSON string."""
        return json.dumps(self.to_dict(round_trip), indent=indent, ensure_ascii=False)

    def to_jsonl(self, output_path: str, indent: int = 0):
        with open(output_path, 'w', encoding='utf-8') as f:
            for chunk in self.chunks:
                json_str = chunk.to_json(indent=None)
                if '\n' in json_str:
                    # Log warning if JSON contains newlines, which breaks JSONL format
                    print(f"Chunk {chunk.chunk_id} contains newlines in JSON, which may break JSONL format.")
                f.write(json_str + '\n')

    @classmethod
    def from_jsonl(cls, input_path: str, corpus_id: Optional[str] = None) -> "Corpus":
        chunks = []
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                chunk_dict = json.loads(line.strip())
                metadata = ChunkMetadata.model_validate(chunk_dict["metadata"])
                chunk = Chunk(
                    chunk_id=chunk_dict["chunk_id"],
                    text=chunk_dict["text"],
                    metadata=metadata,
                    embedding=chunk_dict["embedding"],
                    start_char_idx=chunk_dict["start_char_idx"],
                    end_char_idx=chunk_dict["end_char_idx"],
                    excluded_embed_metadata_keys=chunk_dict["excluded_embed_metadata_keys"],
                    excluded_llm_metadata_keys=chunk_dict["excluded_llm_metadata_keys"],
                    relationships={
                        k: RelatedNodeInfo(**v) for k, v in chunk_dict["relationships"].items()
                    }
                )
                chunks.append(chunk)
        return cls(chunks=chunks, corpus_id=corpus_id)

    def __str__(self) -> str:
        stats = self.get_stats()
        return (
            f"Corpus(chunks={stats['chunk_count']}, unique_docs={stats['unique_docs']}, "
            f"avg_word_count={stats['avg_word_count']:.1f}, strategies={stats['strategies']})"
        )

    def __repr__(self) -> str:
        return f"Corpus(chunks={len(self.chunks)}, chunk_index_keys={list(self.chunk_index.keys())})"

    def __len__(self) -> int:
        return len(self.chunks)
    

class Query(BaseModule):
    """Represents a retrieval query."""
    
    query_str: str = Field(description="The query string.")
    top_k: Optional[int] = Field(default=None, description="Number of top results to retrieve.")
    custom_embedding_strs: Optional[List[str]] = Field(default=None, description="The List to store additional strings need to be embed with the query.")
    similarity_cutoff: Optional[float] = Field(default=None, description="Minimum similarity score.")
    keyword_filters: Optional[List[str]] = Field(default=None, description="Keywords to filter results.")
    metadata_filters: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata filters.")

    @property
    def embedding_strs(self) -> List[str]:
        """Use custom embedding strs if specified, otherwise use query str."""
        if self.custom_embedding_strs is None:
            if len(self.query_str) == 0:
                return []
            return [self.query_str]
        else:
            return self.custom_embedding_strs

    def to_QueryBundle(self):
        return QueryBundle(
            query_str=self.query_str,
            custom_embedding_strs=self.custom_embedding_strs
        )

class RagResult(BaseModule, Generic[ChunkType]):
    """Represents a generic retrieval result."""
    
    corpus: Corpus[ChunkType] = Field(description="Retrieved chunks.")
    scores: List[float] = Field(description="Similarity scores for each chunk.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional result metadata.")
    
    def get_top_chunks(self, limit: int = None) -> List[ChunkType]:
        """Get top chunks sorted by similarity score."""
        chunks = self.corpus.sort_by_similarity(reverse=True)
        return chunks[:limit] if limit else chunks

