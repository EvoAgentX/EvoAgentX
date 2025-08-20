import os
import json
from typing import List, Dict
from collections import defaultdict
from dotenv import load_dotenv

from evoagentx.core.logging import logger
from evoagentx.storages.base import StorageHandler
from evoagentx.rag.rag import RAGEngine
from evoagentx.storages.storages_config import VectorStoreConfig, DBConfig, StoreConfig
from evoagentx.rag.rag_config import RAGConfig, ReaderConfig, IndexConfig, EmbeddingConfig, RetrievalConfig
from evoagentx.rag.schema import Query
from evoagentx.benchmark.real_mm_rag import RealMMRAG, download_real_mm_rag_data

# Load environment
load_dotenv()

# Download datasets
download_real_mm_rag_data(save_dir="./debug/data/real_mm_rag")
datasets = RealMMRAG("./debug/data/real_mm_rag")

# Initialize StorageHandler
store_config = StoreConfig(
    dbConfig=DBConfig(
        db_name="sqlite",
        path="./debug/data/real_mm_rag/cache/multimodal_rag.sql"
    ),
    vectorConfig=VectorStoreConfig(
        vector_name="faiss",
        dimensions=4096,    
        index_type="flat_l2",
    ),
    graphConfig=None,
    path="./debug/data/real_mm_rag/cache/indexing"
)
storage_handler = StorageHandler(storageConfig=store_config)

# Initialize RAGEngine
# For multimodal example
embedding=EmbeddingConfig(
        provider="multimodal",
        model_name="nomic-ai/nomic-embed-multimodal-7b",
        device="cuda"
    )

rag_config = RAGConfig(
    modality="multimodal",  # Key difference for images
    reader=ReaderConfig(
        recursive=True, exclude_hidden=True,
        num_files_limit=None,
        errors="ignore"
    ),
    embedding=embedding,
    index=IndexConfig(index_type="vector"),
    retrieval=RetrievalConfig(
        retrivel_type="vector",
        postprocessor_type="simple",
        top_k=5,  # Retrieve top-5 images
        similarity_cutoff=0.3,
        keyword_filters=None,
        metadata_filters=None
    )
)
search_engine = RAGEngine(config=rag_config, storage_handler=storage_handler)

# Define Helper function and evaluation function
def evaluate_multimodal_retrieval(retrieved_chunks: List, target_image: str, top_k: int) -> Dict[str, float]:
    """Evaluate retrieved images against target image."""
    # Check if target image is in top-k results
    retrieved_images = [chunk.metadata.get('file_name', '') for chunk in retrieved_chunks[:top_k]]
    
    # Hit@K: whether target image is in top-k
    hit = 1.0 if target_image in retrieved_images else 0.0
    
    # MRR: Mean Reciprocal Rank
    mrr = 0.0
    for rank, img_name in enumerate(retrieved_images, 1):
        if img_name == target_image:
            mrr = 1.0 / rank
            break
    
    # Average similarity score
    avg_score = sum(chunk.metadata.similarity_score for chunk in retrieved_chunks[:top_k]) / min(top_k, len(retrieved_chunks)) if retrieved_chunks else 0.0
    
    return {
        "hit@k": hit,
        "mrr": mrr,
        "avg_similarity_score": avg_score
    }

def run_evaluation(samples: List[Dict], top_k: int = 5) -> Dict[str, float]:
    """Run evaluation on REAL-MM-RAG samples."""
    metrics = defaultdict(list)
    
    for sample in samples:
        query_text = sample["query"]
        target_image = sample["image_filename"]
        image_path = sample["image_path"]
        corpus_id = sample["id"]
        
        logger.info(f"Processing sample: {corpus_id}, query: {query_text}")
        
        # Index single image
        image_nodes = search_engine.read(
            file_paths=[image_path],
            corpus_id=str(corpus_id)
        )
        logger.info(f"Indexed {len(image_nodes)} images")
        search_engine.add(index_type="vector", nodes=image_nodes, corpus_id=str(corpus_id))
        
        # Query
        query = Query(query_str=query_text, top_k=top_k)
        result = search_engine.query(query, corpus_id=str(corpus_id))
        retrieved_chunks = result.corpus.chunks
        logger.info(f"Retrieved {len(retrieved_chunks)} images for query")
        
        # Evaluate
        sample_metrics = evaluate_multimodal_retrieval(retrieved_chunks, target_image, top_k)
        for metric_name, value in sample_metrics.items():
            metrics[metric_name].append(value)
        logger.info(f"Metrics for sample {corpus_id}: {sample_metrics}")
        
        # Clear index to avoid memory issues
        search_engine.clear(corpus_id=str(corpus_id))
    
    # Aggregate metrics
    avg_metrics = {name: sum(values) / len(values) for name, values in metrics.items()}
    return avg_metrics


if __name__ == "__main__":
    # Run evaluation on a subset of samples
    samples = datasets.get_random_samples(20)  # Limit to 20 samples for testing
    print(f"Dataset size: {len(datasets.data)}")

    avg_metrics = run_evaluation(samples, top_k=5)

    logger.info("Average Metrics:")
    for metric_name, value in avg_metrics.items():
        logger.info(f"{metric_name}: {value:.4f}")

    # Save results
    with open("./debug/data/real_mm_rag/evaluation_results.json", "w") as f:
        json.dump(avg_metrics, f, indent=2)

    """
    Results using 20 samples:
        multimodal-embedding:
            hit@k: 0.8500, mrr: 0.7250, avg_similarity_score: 0.6800
    """
