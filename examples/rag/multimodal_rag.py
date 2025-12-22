import os
from typing import List, Dict
from dotenv import load_dotenv
from pathlib import Path

from evoagentx.core.logging import logger
from evoagentx.storages.base import StorageHandler
from evoagentx.rag.rag import RAGEngine
from evoagentx.storages.storages_config import VectorStoreConfig, DBConfig, StoreConfig
from evoagentx.rag.rag_config import RAGConfig, ReaderConfig, IndexConfig, EmbeddingConfig, RetrievalConfig
from evoagentx.rag.schema import Query, TextChunk
from evoagentx.benchmark.real_mm_rag import RealMMRAG
from evoagentx.models.openai_model import OpenAILLM
from evoagentx.models.model_configs import OpenAILLMConfig


# Load environment
load_dotenv()

def demonstrate_rag_to_generation_pipeline():
    """Simple demo: Index 20 docs, retrieve 5, generate answer."""
    print("🚀 EvoAgentX Multimodal RAG-to-Generation Pipeline")
    print("=" * 60)
    
    # Check if OpenAI API key is available
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        print("❌ OPENAI_API_KEY not found. Please set it to run this demo.")
        return

    # Check if VOYAGE API key is available
    voyage_key = os.getenv("VOYAGE_API_KEY")
    if not voyage_key:
        print("❌ VOYAGE_API_KEY not found. Please set it to run this demo.")
        return
    
    # Initialize dataset
    datasets = RealMMRAG("./debug/data/real_mm_rag")
    samples = datasets.get_random_samples(20, seed=42)  # Get 20 samples
    print(f"📊 Dataset loaded with {len(samples)} samples")
    
    # Setup storage and RAG engine
    store_config = StoreConfig(
        dbConfig=DBConfig(db_name="sqlite", path="./debug/data/real_mm_rag/cache/demo.sql"),
        vectorConfig=VectorStoreConfig(vector_name="faiss", dimensions=1024, index_type="flat_l2"),
        path="./debug/data/real_mm_rag/cache/indexing"
    )
    storage_handler = StorageHandler(storageConfig=store_config)
    
    rag_config = RAGConfig(
        modality="multimodal",
        reader=ReaderConfig(recursive=True, exclude_hidden=True, errors="ignore"),
        embedding=EmbeddingConfig(provider="voyage", model_name="voyage-multimodal-3", device="cpu" ,api_key=voyage_key),
        index=IndexConfig(index_type="vector"),
        retrieval=RetrievalConfig(retrivel_type="vector", top_k=5, similarity_cutoff=0.3)
    )
    search_engine = RAGEngine(config=rag_config, storage_handler=storage_handler)
    
    # Index 20 documents
    print("\n📚 Step 1: Indexing 20 documents...")
    corpus_id = "demo_corpus"
    valid_paths = [s["image_path"] for s in samples if os.path.exists(s["image_path"])][:20]
    
    if len(valid_paths) < 20:
        print(f"⚠️ Only found {len(valid_paths)} valid image paths, using those")
    
    corpus = search_engine.read(file_paths=valid_paths, corpus_id=corpus_id)
    search_engine.add(index_type="vector", nodes=corpus, corpus_id=corpus_id)
    print(f"✅ Indexed {len(corpus.chunks)} image documents")
    
    # Find a good query sample
    query_sample = next((s for s in samples if s["query"] and len(s["query"].strip()) > 10), None)
    if not query_sample:
        print("❌ No suitable query found in samples")
        return
    
    query_text = query_sample["query"]
    target_image = query_sample["image_filename"]
    
    print(f"\n🔍 Step 2: Querying with: '{query_text}'")
    print(f"🎯 Target document: {target_image}")
    
    # Retrieve 5 documents
    query = Query(query_str=query_text, top_k=5)
    result = search_engine.query(query, corpus_id=corpus_id)
    retrieved_chunks = result.corpus.chunks
    
    print(f"\n📄 Retrieved {len(retrieved_chunks)} documents:")
    retrieved_paths = []
    for i, chunk in enumerate(retrieved_chunks):
        filename = Path(chunk.image_path).name if chunk.image_path else "Unknown"
        similarity = getattr(chunk.metadata, 'similarity_score', 0.0)
        retrieved_paths.append(filename)
        print(f"  {i+1}. {filename} (similarity: {similarity:.3f})")
    
    # Generate answer using multimodal LLM
    print(f"\n🤖 Step 3: Generating answer with GPT-4o...")
    
    try:
        # Initialize LLM with proper configuration
        llm_config = OpenAILLMConfig(
            model="gpt-4o",
            openai_key=openai_key,
            temperature=0.1,
            max_tokens=300
        )
        llm = OpenAILLM(config=llm_config)
        
        print("✅ LLM initialized successfully")
        
        # Prepare content with text and retrieved images - LLM handles everything automatically  
        content = [TextChunk(text=f"Query: {query_text}\n\nAnalyze these retrieved images and answer the query:")]
        content.extend(retrieved_chunks[:3])  # Add top 3 retrieved images
        
        # Generate response - seamless multimodal generation
        response = llm.generate(messages=[
            {"role": "system", "content": "You are an expert image analyst. Answer queries based on provided images."},
            {"role": "user", "content": content}
        ])
        
        print("✅ Response generated successfully")
        answer = response.content
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"❌ Detailed error:")
        print(error_details)
        answer = f"Error in generation: {str(e)}"
    
    # Print final results
    print("\n" + "=" * 60)
    print("📋 FINAL RESULTS")
    print("=" * 60)
    print(f"🔍 QUERY: {query_text}")
    print(f"\n📄 RETRIEVED PATHS:")
    for i, path in enumerate(retrieved_paths):
        print(f"  {i+1}. {path}")
    print(f"\n🎯 TARGET DOCUMENT: {target_image}")
    print(f"\n🤖 GENERATED ANSWER:")
    print(answer)
    print("EXPECTED ANSWER:")
    print(query_sample["answer"])
    print("=" * 60)
    
    # Cleanup
    search_engine.clear(corpus_id=corpus_id)


if __name__ == "__main__":
    demonstrate_rag_to_generation_pipeline()
