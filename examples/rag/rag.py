import os
import json
from typing import List, Dict
from collections import defaultdict
from dotenv import load_dotenv

from evoagentx.core.logging import logger
from evoagentx.storages.base import StorageHandler
from evoagentx.rag.rag import RAGEngine
from evoagentx.storages.storages_config import VectorStoreConfig, DBConfig, StoreConfig
from evoagentx.rag.rag_config import RAGConfig, ReaderConfig, ChunkerConfig, IndexConfig, EmbeddingConfig, RetrievalConfig
from evoagentx.rag.schema import Query, Corpus, Chunk, ChunkMetadata
from evoagentx.benchmark.hotpotqa import HotPotQA, download_raw_hotpotqa_data

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
        dimensions=768,    # 1536: text-embedding-ada-002, 384: bge-small-en-v1.5, 768: nomic-embed-text
        index_type="flat_l2",
    ),
    graphConfig=None,
    path="./debug/data/hotpotqa/cache/indexing"
)
storage_handler = StorageHandler(storageConfig=store_config)

# Initialize RAGEngine
# Define 3 embeddings models
"""
# For openai example
embedding=EmbeddingConfig(
        provider="openai",
        model_name="text-embedding-ada-002",
        api_key=os.environ["OPENAI_API_KEY"],
    )
# For huggingface example
embedding=EmbeddingConfig(
        provider="huggingface",
        model_name="debug/weights/bge-small-en-v1.5",
        device="cpu"
    )
# For ollama example
embedding=EmbeddingConfig(
        provider="ollama",
        model_name="nomic-embed-text",
        base_url="10.168.1.71:17174",
        dimensions=768
    )
"""
# For ollama example
embedding=EmbeddingConfig(
        provider="openai",
        model_name="text-embedding-ada-002",
        api_key=os.environ["OPENAI_API_KEY"],
    )

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
    embedding=embedding,
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
search_engine = RAGEngine(config=rag_config, storage_handler=storage_handler)

# Define Helper function and evaluation function
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
        query = Query(query_str=question, top_k=top_k)
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

    """
    Results using 20 samples:
        text-embedding-ada-002:
            precision@k:0.3400, recall@k:0.7117, f1@k:0.4539, mrr:0.9250, hit@k: 1.0000, jaccard:0.3089
        bge-small-en-v1.5:
            precision@k:0.3100, recall@k:0.6767, f1@k:0.4207, mrr: 0.7667, hit@k: 0.9500, jaccard:0.2837
        nomic-embed-text:
            precision@k:0.3500, recall@k:0.7367, f1@k: 0.4682, mrr:0.7958, hit@k: 0.9500, jaccard: 0.3268
    """