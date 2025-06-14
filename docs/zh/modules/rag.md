# SearchEngine 模块文档

## 概述

`SearchEngine` 模块是检索增强生成（RAG）系统的核心组件，设计用于管理文档的索引、存储和检索，以实现高效的信息访问。基于 LlamaIndex 构建，它与多种存储后端（如 SQLite、FAISS）集成，并支持多种索引类型（如向量索引、图索引）。该模块是一个长期记忆管理框架的一部分，适用于需要高效知识检索的代理系统。

### 定位
`SearchEngine` 的主要作用是：
- **文档处理**：从多种格式（如 PDF、文本）加载并分块文档。
- **索引管理**：创建和管理索引以支持高效检索。
- **检索功能**：支持基于元数据过滤和相似性搜索的查询。
- **存储管理**：将索引持久化到文件或数据库以实现可扩展性。

它设计为与 `StorageHandler` 和 `MemoryManager` 等组件集成，适合需要上下文知识检索的代理系统。

### 核心功能
- **灵活的文档加载**：支持从目录加载文件，并提供可定制的过滤选项（如文件后缀、排除列表）。
- **分块与嵌入**：自动将文档分块并使用可配置的模型（如 OpenAI 的 `text-embedding-ada-002`）生成嵌入。
- **多索引支持**：支持多种索引类型（向量、图、摘要、树）以满足不同检索需求。
- **高级检索**：支持元数据过滤、相似性阈值和基于关键字的查询，采用异步和多线程检索。
- **存储集成**：与 SQLite（用于元数据）和 FAISS（用于向量嵌入）无缝集成。
- **持久化**：支持将索引保存到文件或数据库，确保数据持久性。
- **错误处理**：提供健壮的日志记录和异常处理，确保操作可靠性。

## 使用说明

### 前置条件
- 安装依赖：`llama_index`、`pydantic` 等项目要求的库。
- 配置环境变量（如 `OPENAI_API_KEY` 用于嵌入模型），参考[Installation Guide for EvoAgentX](./docs/installation.md)。
- 确保 `StorageHandler` 实例已配置好适当的存储后端（如 SQLite、FAISS）。

### 初始化
使用 `RAGConfig` 和 `StorageHandler` 初始化 `SearchEngine`，可选提供语言模型（`BaseLLM`）以支持高级查询处理。

```python
from evoagentx.rag.search_engine import SearchEngine
from evoagentx.rag.rag_config import RAGConfig, ReaderConfig, ChunkerConfig, EmbeddingConfig, IndexConfig, RetrievalConfig
from evoagentx.storages.base import StorageHandler
from evoagentx.storages.storages_config import StoreConfig, VectorStoreConfig, DBConfig

# 配置存储
store_config = StoreConfig(
    dbConfig=DBConfig(db_name="sqlite", path="./data/cache.db"),
    vectorConfig=VectorStoreConfig(vector_name="faiss", dimensions=1536, index_type="flat_l2"),
    graphConfig=None,
    path="./data/indexing"
)
storage_handler = StorageHandler(storageConfig=store_config)

# 配置 RAG
rag_config = RAGConfig(
    reader=ReaderConfig(recursive=False, exclude_hidden=True),
    chunker=ChunkerConfig(strategy="simple", chunk_size=512, chunk_overlap=0),
    embedding=EmbeddingConfig(provider="openai", model_name="text-embedding-ada-002", api_key="your-api-key"),
    index=IndexConfig(index_type="vector"),
    retrieval=RetrievalConfig(retrivel_type="vector", postprocessor_type="simple", top_k=10, similarity_cutoff=0.3)
)

# 初始化 SearchEngine
search_engine = SearchEngine(config=rag_config, storage_handler=storage_handler)
```

### 核心功能使用

1. **加载文档**：
   - 使用 `read` 方法从文件或目录加载并分块文档。
   - 示例：
     ```python
     corpus = search_engine.read(
         file_paths="./data/docs",
         filter_file_by_suffix=[".txt", ".pdf"],
         merge_by_file=False,
         show_progress=True,
         corpus_id="doc_corpus"
     )
     ```

2. **索引文档**：
   - 使用 `add` 方法将语料库或节点列表索引到指定索引类型。
   - 示例：
     ```python
     search_engine.add(index_type="vector", nodes=corpus, corpus_id="doc_corpus")
     ```

3. **查询**：
   - 使用 `query` 方法根据查询字符串或 `RagQuery` 对象检索相关分块。
   - 示例：
     ```python
     from evoagentx.rag.schema import RagQuery
     query = RagQuery(query_str="法国的首都是什么？", top_k=5)
     result = search_engine.query(query, corpus_id="doc_corpus")
     print(result.corpus.chunks)  # 检索到的分块
     ```

4. **删除节点或索引**：
   - 使用 `delete` 方法删除特定节点或整个索引。
   - 示例：
     ```python
     search_engine.delete(corpus_id="doc_corpus", node_ids=["node_1", "node_2"])
     search_engine.delete(corpus_id="doc_corpus", index_type="vector")  # 删除整个索引
     ```

5. **清除索引**：
   - 使用 `clear` 方法清除特定语料库或所有语料库的索引。
   - 示例：
     ```python
     search_engine.clear(corpus_id="doc_corpus")  # 清除特定语料库
     search_engine.clear()  # 清除所有语料库
     ```

6. **保存索引**：
   - 使用 `save` 方法将索引持久化到文件或数据库。
   - 示例：
     ```python
     search_engine.save(output_path="./data/indexing", corpus_id="doc_corpus", index_type="vector")
     search_engine.save(corpus_id="doc_corpus", table="indexing")  # 保存到数据库
     ```

7. **加载索引**：
   - 使用 `load` 方法从文件或数据库重建索引。
   - 示例：
     ```python
     search_engine.load(source="./data/indexing", corpus_id="doc_corpus", index_type="vector")
     search_engine.load(corpus_id="doc_corpus", table="indexing")  # 从数据库加载
     ```

### 使用示例
以下示例展示如何使用 `SearchEngine` 处理 HotPotQA 数据集，执行文档索引、查询和检索性能评估。[examples/search_engine.py](../../examples/search_engine.py)

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

### 接口列表

| 方法 | 描述 | 参数 | 返回值 |
| ------ | ----------- | ---------- | ------- |
| `__init__` | 初始化 SearchEngine，配置 RAG 和存储处理程序。 | `config: RAGConfig`, `storage_handler: StorageHandler`, `llm: Optional[BaseLLM]` | 无 |
| `read` | 从文件加载并分块文档到语料库。 | `file_paths`, `exclude_files`, `filter_file_by_suffix`, `merge_by_file`, `show_progress`, `corpus_id` | `Corpus` |
| `add` | 将节点添加到指定语料库的索引中。 | `index_type`, `nodes`, `corpus_id` | 无 |
| `delete` | 删除语料库中的节点或整个索引。 | `corpus_id`, `index_type`, `node_ids`, `metadata_filters` | 无 |
| `clear` | 清除特定语料库或所有语料库的索引。 | `corpus_id` | 无 |
| `save` | 将索引保存到文件或数据库。 | `output_path`, `corpus_id`, `index_type`, `table` | 无 |
| `load` | 从文件或数据库加载索引。 | `source`, `corpus_id`, `index_type`, `table` | 无 |
| `query` | 执行查询并返回处理后的结果。 | `query`, `corpus_id` | `RagResult` |

### 注意事项
- **存储后端**：确保 `StorageHandler` 正确配置以处理向量和元数据存储。
- **警告**：多次加载索引可能导致问题（如向量存储中重复插入节点）。在重新加载前清空索引。