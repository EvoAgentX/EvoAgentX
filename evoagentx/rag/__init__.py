"""RAG exports, loaded only when explicitly requested."""

from importlib import import_module

__all__ = [
    "Chunk",
    "ChunkMetadata",
    "Corpus",
    "HierarchicalChunker",
    "ImageChunk",
    "LLamaIndexReader",
    "MultimodalReader",
    "OpenAIEmbeddingWrapper",
    "Query",
    "RAGConfig",
    "RAGEngine",
    "RagResult",
    "SemanticChunker",
    "SimpleChunker",
    "SimpleReranker",
    "TextChunk",
    "VectorIndexing",
    "VectorRetriever",
    "VoyageEmbeddingWrapper",
]

_EXPORTS = {
    "RAGEngine": ".rag",
    "LLamaIndexReader": ".readers",
    "MultimodalReader": ".readers",
    "SimpleChunker": ".chunkers",
    "SemanticChunker": ".chunkers",
    "HierarchicalChunker": ".chunkers",
    "OpenAIEmbeddingWrapper": ".embeddings",
    "VoyageEmbeddingWrapper": ".embeddings",
    "VectorIndexing": ".indexings",
    "VectorRetriever": ".retrievers",
    "SimpleReranker": ".postprocessors",
    "RAGConfig": ".rag_config",
    "TextChunk": ".schema",
    "ImageChunk": ".schema",
    "Chunk": ".schema",
    "Corpus": ".schema",
    "RagResult": ".schema",
    "ChunkMetadata": ".schema",
    "Query": ".schema",
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
