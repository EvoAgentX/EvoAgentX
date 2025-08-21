import os
from typing import List, Dict
from collections import defaultdict
from dotenv import load_dotenv

from evoagentx.core.logging import logger
from evoagentx.storages.base import StorageHandler
from evoagentx.rag.rag import RAGEngine
from evoagentx.storages.storages_config import VectorStoreConfig, DBConfig, StoreConfig
from evoagentx.rag.rag_config import RAGConfig, ReaderConfig, IndexConfig, EmbeddingConfig, RetrievalConfig
from evoagentx.rag.schema import Query
from evoagentx.benchmark.real_mm_rag import RealMMRAG

# Load environment
load_dotenv()

# Initialize dataset (will download automatically if not present)
datasets = RealMMRAG("./debug/data/real_mm_rag")

# Initialize StorageHandler
store_config = StoreConfig(
    dbConfig=DBConfig(
        db_name="sqlite",
        path="./debug/data/real_mm_rag/cache/multimodal_rag.sql"
    ),
    vectorConfig=VectorStoreConfig(
        vector_name="faiss",
        dimensions=3584,    
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
        device="cpu"
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
    retrieved_images = [getattr(chunk.metadata, 'file_name', '') for chunk in retrieved_chunks[:top_k]]
    
    # Hit@K: whether target image is in top-k
    hit = 1.0 if target_image in retrieved_images else 0.0
    
    # MRR: Mean Reciprocal Rank
    mrr = 0.0
    for rank, img_name in enumerate(retrieved_images, 1):
        if img_name == target_image:
            mrr = 1.0 / rank
            break
    
    # Average similarity score
    avg_score = sum(getattr(chunk.metadata, 'similarity_score', 0.0) for chunk in retrieved_chunks[:top_k]) / min(top_k, len(retrieved_chunks)) if retrieved_chunks else 0.0
    
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
        
        # Skip samples with null/empty queries
        if not query_text or query_text.strip() == "":
            logger.warning(f"Skipping sample {corpus_id} with empty query")
            continue
        
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
        
        # Save index to database for debugging (before clearing)
        try:
            search_engine.save(corpus_id=str(corpus_id))  # Save to database, not file
            logger.info(f"Saved index for corpus {corpus_id} to database")
        except Exception as e:
            logger.warning(f"Failed to save index for corpus {corpus_id}: {str(e)}")
        
        # Clear index to avoid memory issues
        search_engine.clear(corpus_id=str(corpus_id))
    
    # Aggregate metrics
    avg_metrics = {name: sum(values) / len(values) for name, values in metrics.items()}
    return avg_metrics


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from pathlib import Path
    
    # Index 20 samples first
    samples = datasets.get_random_samples(5, seed=129)
    print(f"Dataset size: {len(datasets.data)}")
    logger.info(f"Indexing {len(samples)} samples...")
    
    # Index all 20 samples
    all_image_paths = []
    corpus_id = "multimodal_corpus"
    
    for sample in samples:
        image_path = sample["image_path"]
        if os.path.exists(image_path):
            all_image_paths.append(image_path)
    
    logger.info(f"Found {len(all_image_paths)} valid image paths")
    
    # Index all images at once
    corpus = search_engine.read(
        file_paths=all_image_paths,
        corpus_id=corpus_id
    )
    logger.info(f"Created corpus with {len(corpus.chunks)} image chunks")
    
    # Print all chunk filenames
    logger.info(f"Corpus contains the following {len(corpus.chunks)} image chunks:")
    for i, chunk in enumerate(corpus.chunks):
        filename = Path(chunk.image_path).name if chunk.image_path else "Unknown"
        logger.info(f"  [{i+1}] {filename} (path: {chunk.image_path})")
    
    search_engine.add(index_type="vector", nodes=corpus, corpus_id=corpus_id)
    logger.info(f"Successfully indexed {len(corpus.chunks)} image chunks into vector store")
    
    # Find a sample with non-None query for visualization
    query_sample = None
    for sample in samples:
        if sample["query"] and sample["query"].strip():
            query_sample = sample
            break
    
    if not query_sample:
        logger.error("No samples with valid queries found!")
        exit(1)
    
    query_text = query_sample["query"]
    target_image = query_sample["image_filename"]
    
    logger.info(f"Query sample found:")
    logger.info(f"Query: {query_text}")
    logger.info(f"Target image: {target_image}")
    
    # Query across all indexed images
    query = Query(query_str=query_text, top_k=1)
    result = search_engine.query(query, corpus_id=corpus_id)
    retrieved_chunks = result.corpus.chunks
    logger.info(f"Query executed successfully!")
    logger.info(f"Retrieved {len(retrieved_chunks)} image chunks from corpus containing {len(corpus.chunks)} total images")
    logger.info(f"Result corpus has {len(result.corpus.chunks)} chunks")
    
    # Get the top retrieved image
    if retrieved_chunks:
        top_chunk = retrieved_chunks[0]  # Best match
        similarity_score = getattr(top_chunk.metadata, 'similarity_score', 0.0)
        
        # Load and display the image
        retrieved_image = top_chunk.get_image()
        
        # Create visualization
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        if retrieved_image:
            ax.imshow(retrieved_image)
            ax.set_title(f"Multimodal RAG Result\n\nQuery: '{query_text}'\n\nTop Retrieved Image: {Path(top_chunk.image_path).name}\nSimilarity Score: {similarity_score:.4f}\n\n(Searched across {len(corpus.chunks)} indexed images)", 
                        fontsize=14, pad=20)
            ax.axis('off')
            
            # Save the plot
            os.makedirs("./debug/data/real_mm_rag/", exist_ok=True)
            plt.tight_layout()
            plt.savefig("./debug/data/real_mm_rag/multimodal_rag_result.png", 
                       dpi=150, bbox_inches='tight')
            plt.show()
            
            logger.info(f"✅ Visualization saved to ./debug/data/real_mm_rag/multimodal_rag_result.png")
            logger.info(f"Query: {query_text}")
            logger.info(f"Retrieved image: {Path(top_chunk.image_path).name}")
            logger.info(f"Target image: {target_image}")
            logger.info(f"Similarity score: {similarity_score:.4f}")
            logger.info(f"Searched across {len(corpus.chunks)} total indexed images")
        else:
            logger.error("Failed to load retrieved image")
    else:
        logger.warning("No images retrieved")
    
    # Clean up
    search_engine.save(corpus_id=corpus_id)
    search_engine.clear(corpus_id=corpus_id)
