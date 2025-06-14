# SearchEngine(RAG) Module Documentation

## Overview

The `SearchEngine` module is a core component of a Retrieval-Augmented Generation (RAG) system, designed to manage document indexing, storage, and retrieval for efficient information access. Built on top of LlamaIndex, it integrates with various storage backends (e.g., SQLite, FAISS) and supports multiple index types (e.g., vector, graph). The module is part of a larger framework for long-term memory management, enabling agents to process and query large datasets effectively.

### Purpose
The `SearchEngine` serves as the central interface for:
- **Document Processing**: Loading and chunking documents from various formats (e.g., PDF, text).
- **Indexing**: Creating and managing indices for efficient retrieval.
- **Retrieval**: Querying indexed data with support for metadata filtering and similarity-based search.
- **Storage Management**: Persisting indices to files or databases for scalability.

It is designed for integration with other components like `StorageHandler` and `MemoryManager`, making it suitable for agent-based systems requiring contextual knowledge retrieval.

### Key Features
- **Flexible Document Loading**: Supports loading files from directories with customizable filters (e.g., file suffixes, exclusion lists).
- **Chunking and Embedding**: Automatically chunks documents and generates embeddings using configurable models (e.g., OpenAI’s `text-embedding-ada-002`).
- **Multi-Index Support**: Handles different index types (vector, graph, summary, tree) for diverse retrieval needs.
- **Advanced Retrieval**: Supports metadata filters, similarity cutoffs, and keyword-based queries, with asynchronous and multi-threaded retrieval.
- **Storage Integration**: Seamlessly integrates with SQLite for metadata and vector stores like FAISS for embeddings.
- **Persistence**: Saves and loads indices to/from files or databases, ensuring data durability.
- **Error Handling**: Robust logging and exception handling for reliable operation.

## Usage Instructions

### Prerequisites
- Install required dependencies: `llama_index`, `pydantic`, and other libraries specified in the project.
- Configure environment variables (e.g., `OPENAI_API_KEY` for embedding models).
- Ensure a `StorageHandler` instance is available, configured with appropriate storage backends (e.g., SQLite, FAISS).

### Initialization
To use `SearchEngine`, initialize it with a `RAGConfig` and a `StorageHandler` instance. Optionally, provide a language model (`BaseLLM`) for advanced query processing.

```python
from evoagentx.rag.search_engine import SearchEngine
from evoagentx.rag.rag_config import RAGConfig, ReaderConfig, ChunkerConfig, EmbeddingConfig, IndexConfig, RetrievalConfig
from evoagentx.storages.base import StorageHandler
from evoagentx.storages.storages_config import StoreConfig, VectorStoreConfig, DBConfig

# Configure storage
store_config = StoreConfig(
    dbConfig=DBConfig(db_name="sqlite", path="./data/cache.db"),
    vectorConfig=VectorStoreConfig(vector_name="faiss", dimensions=1536, index_type="flat_l2"),
    graphConfig=None,
    path="./data/indexing"
)
storage_handler = StorageHandler(storageConfig=store_config)

# Configure RAG
rag_config = RAGConfig(
    reader=ReaderConfig(recursive=False, exclude_hidden=True),
    chunker=ChunkerConfig(strategy="simple", chunk_size=512, chunk_overlap=0),
    embedding=EmbeddingConfig(provider="openai", model_name="text-embedding-ada-002", api_key="your-api-key"),
    index=IndexConfig(index_type="vector"),
    retrieval=RetrievalConfig(retrivel_type="vector", postprocessor_type="simple", top_k=10, similarity_cutoff=0.3)
)

# Initialize SearchEngine
search_engine = SearchEngine(config=rag_config, storage_handler=storage_handler)
```

### Core Functionality

1. **Loading Documents**:
   - Use `read` to load and chunk documents from files or directories.
   - Example:
     ```python
     corpus = search_engine.read(
         file_paths="./data/docs",
         filter_file_by_suffix=[".txt", ".pdf"],
         merge_by_file=False,
         show_progress=True,
         corpus_id="doc_corpus"
     )
     ```

2. **Indexing Documents**:
   - Use `add` to index a corpus or list of nodes into a specific index type.
   - Example:
     ```python
     search_engine.add(index_type="vector", nodes=corpus, corpus_id="doc_corpus")
     ```

3. **Querying**:
   - Use `query` to retrieve relevant chunks based on a query string or `RagQuery` object.
   - Example:
     ```python
     from evoagentx.rag.schema import RagQuery
     query = RagQuery(query_str="What is the capital of France?", top_k=5)
     result = search_engine.query(query, corpus_id="doc_corpus")
     print(result.corpus.chunks)  # Retrieved chunks
     ```

4. **Deleting Nodes or Indices**:
   - Use `delete` to remove specific nodes or entire indices.
   - Example:
     ```python
     search_engine.delete(corpus_id="doc_corpus", node_ids=["node_1", "node_2"])
     search_engine.delete(corpus_id="doc_corpus", index_type="vector")  # Delete entire index
     ```

5. **Clearing Indices**:
   - Use `clear` to remove all indices for a corpus or all corpora.
   - Example:
     ```python
     search_engine.clear(corpus_id="doc_corpus")  # Clear specific corpus
     search_engine.clear()  # Clear all corpora
     ```

6. **Saving Indices**:
   - Use `save` to persist indices to files or a database.
   - Example:
     ```python
     search_engine.save(output_path="./data/indexing", corpus_id="doc_corpus", index_type="vector")
     search_engine.save(corpus_id="doc_corpus", table="indexing")  # Save to database
     ```

7. **Loading Indices**:
   - Use `load` to reconstruct indices from files or a database.
   - Example:
     ```python
     search_engine.load(source="./data/indexing", corpus_id="doc_corpus", index_type="vector")
     search_engine.load(corpus_id="doc_corpus", table="indexing")  # Load from database
     ```

### Example Usage
The following example demonstrates how to use `SearchEngine` with the HotPotQA dataset to index and query documents, and evaluate retrieval performance. [examples/search_engine.py](../../examples/search_engine.py)

```python
import os
import json
import logging
from typing import List, Dict
from collections import defaultdict
from dotenv import load_dotenv

from evoagentx.storages.base import StorageHandler
from evoagentx.rag.search_engine import SearchEngine
from evoagentx.storages.storages_config import VectorStoreConfig, DBConfig, StoreConfig
from evoagentx.rag.rag_config import RAGConfig, ReaderConfig, ChunkerConfig, IndexConfig, EmbeddingConfig, RetrievalConfig
from evoagentx.rag.schema import RagQuery, Corpus, Chunk, ChunkMetadata
from evoagentx.benchmark.hotpotqa import HotPotQA, download_raw_hotpotqa_data

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment
load_dotenv()

# Download datasets
download_raw_hotpotqa_data("hotpot_dev_distractor_v1.json", "./debug/data/hotpotqa")
datasets = HotPotQA("./debug/data/hotpotqa")

# Initialize StorageHandler
store_config = StoreConfig(
    dbConfig=DBConfig(
        db_name="sqlite",
        path="./debug/data/hotpotqa/cache/test_hotpotQA.sql"
    ),
    vectorConfig=VectorStoreConfig(
        vector_name="faiss",
        dimensions=1536,
        index_type="flat_l2",
    ),
    graphConfig=None,
    path="./debug/data/hotpotqa/cache/indexing"
)
storage_handler = StorageHandler(storageConfig=store_config)

# Initialize SearchEngine
rag_config = RAGConfig(
    reader=ReaderConfig(
        recursive=False, exclude_hidden=True,
        num_files_limit=None, custom_metadata_function=None,
        extern_file_extractor=None,
        errors="ignore", encoding="utf-8"
    ),
    chunker=ChunkerConfig(
        strategy="simple",
        chunk_size=512,
        chunk_overlap=0,
        max_chunks=None
    ),
    embedding=EmbeddingConfig(
        provider="openai",
        model_name="text-embedding-ada-002",
        api_key=os.environ["OPENAI_API_KEY"],
    ),
    index=IndexConfig(index_type="vector"),
    retrieval=RetrievalConfig(
        retrivel_type="vector",
        postprocessor_type="simple",
        top_k=10,  # Retrieve top-10 contexts
        similarity_cutoff=0.3,
        keyword_filters=None,
        metadata_filters=None
    )
)
search_engine = SearchEngine(config=rag_config, storage_handler=storage_handler)

def create_corpus_from_context(context: List[List], corpus_id: str) -> Corpus:
    """Convert HotPotQA context into a Corpus for indexing."""
    chunks = []
    for title, sentences in context:
        for idx, sentence in enumerate(sentences):
            chunk = Chunk(
                chunk_id=f"{title}_{idx}",
                text=sentence,
                metadata=ChunkMetadata(
                    doc_id=str(idx),
                    corpus_id=corpus_id
                ),
                start_char_idx=0,
                end_char_idx=len(sentence),
                excluded_embed_metadata_keys=[],
                excluded_llm_metadata_keys=[],
                relationships={}
            )
            chunk.metadata.title = title    # initilize a new attribute
            chunks.append(chunk)
    return Corpus(chunks=chunks, corpus_id=corpus_id)

def evaluate_retrieval(retrieved_chunks: List[Chunk], supporting_facts: List[List], top_k: int) -> Dict[str, float]:
    """Evaluate retrieved chunks against supporting facts."""
    # Ground-truth relevant sentences: set of (title, sentence_idx) tuples
    relevant = {(fact[0], fact[1]) for fact in supporting_facts}
    
    # Retrieved sentences: list of (title, sentence_idx) tuples
    retrieved = []
    for chunk in retrieved_chunks[:top_k]:
        title = chunk.metadata.title
        sentence_idx = int(chunk.metadata.doc_id)
        retrieved.append((title, sentence_idx))
    
    # Count hits
    hits = sum(1 for r in retrieved if r in relevant)

    # Compute metrics
    precision = hits / top_k if top_k > 0 else 0.0
    recall = hits / len(relevant) if len(relevant) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Compute MRR
    mrr = 0.0
    for rank, r in enumerate(retrieved, 1):
        if r in relevant:
            mrr = 1.0 / rank
            break
    
    # Compute Hit@K
    hit = 1.0 if hits > 0 else 0.0
    
    intersection = set((r[0], r[1]) for r in retrieved) & relevant
    union = set((r[0], r[1]) for r in retrieved) | relevant
    jaccard = len(intersection) / len(union) if union else 0.0

    return {
        "precision@k": precision,
        "recall@k": recall,
        "f1@k": f1,
        "mrr": mrr,
        "hit@k": hit,
        "jaccard": jaccard
    }

def run_evaluation(samples: List[Dict], top_k: int = 5) -> Dict[str, float]:
    """Run evaluation on HotPotQA samples."""
    metrics = defaultdict(list)
    
    for sample in samples:
        question = sample["question"]
        context = sample["context"]
        supporting_facts = sample["supporting_facts"]
        corpus_id = sample["_id"]
        
        logger.info(f"Processing sample: {corpus_id}, question: {question}")
        
        # Create and index corpus
        corpus = create_corpus_from_context(context, corpus_id)
        logger.info(f"Created corpus with {len(corpus.chunks)} chunks")
        search_engine.add(index_type="vector", nodes=corpus, corpus_id=corpus_id)
        
        # Query
        query = RagQuery(query_str=question, top_k=top_k)
        result = search_engine.query(query, corpus_id=corpus_id)
        retrieved_chunks = result.corpus.chunks
        logger.info(f"Retrieved {len(retrieved_chunks)} chunks for query")
        
        # Evaluate
        sample_metrics = evaluate_retrieval(retrieved_chunks, supporting_facts, top_k)
        for metric_name, value in sample_metrics.items():
            metrics[metric_name].append(value)
        logger.info(f"Metrics for sample {corpus_id}: {sample_metrics}")
        
        # Clear index to avoid memory issues
        search_engine.clear(corpus_id=corpus_id)
    
    # Aggregate metrics
    avg_metrics = {name: sum(values) / len(values) for name, values in metrics.items()}
    return avg_metrics


if __name__ == "__main__":
    # Run evaluation on a subset of samples
    samples = datasets._dev_data[:20]  # Limit to 20 samples for testing
    print(len(datasets._dev_data))

    avg_metrics = run_evaluation(samples, top_k=5)

    logger.info("Average Metrics:")
    for metric_name, value in avg_metrics.items():
        logger.info(f"{metric_name}: {value:.4f}")

    # Save results
    with open("./debug/data/hotpotqa/evaluation_results.json", "w") as f:
        json.dump(avg_metrics, f, indent=2)
```

### Interfaces

| Method | Description | Parameters | Returns |
| ------ | ----------- | ---------- | ------- |
| `__init__` | Initializes the SearchEngine with configuration and storage handler. | `config: RAGConfig`, `storage_handler: StorageHandler`, `llm: Optional[BaseLLM]` | None |
| `read` | Loads and chunks documents from files into a Corpus. | `file_paths`, `exclude_files`, `filter_file_by_suffix`, `merge_by_file`, `show_progress`, `corpus_id` | `Corpus` |
| `add` | Adds nodes to an index for a specific corpus. | `index_type`, `nodes`, `corpus_id` | None |
| `delete` | Deletes nodes or an entire index from a corpus. | `corpus_id`, `index_type`, `node_ids`, `metadata_filters` | None |
| `clear` | Clears all indices for a specific corpus or all corpora. | `corpus_id` | None |
| `save` | Saves indices to files or database. | `output_path`, `corpus_id`, `index_type`, `table` | None |
| `load` | Loads indices from files or database. | `source`, `corpus_id`, `index_type`, `table` | None |
| `query` | Executes a query and returns processed results. | `query`, `corpus_id` | `RagResult` |

### Notes
- **Storage Backend**: Ensure the `StorageHandler` is properly configured to handle vector and metadata storage.
- **Warning**: Loading indices multiple times may cause issues if the same nodes are inserted into the vector store (e.g., FAISS). Clear indices before reloading if necessary.