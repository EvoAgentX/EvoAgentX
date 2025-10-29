import os
import json
import glob
from typing import List, Dict, Any

from dotenv import load_dotenv

from evoagentx.rag.schema import Query, Document, TextChunk, Corpus, ChunkMetadata
from evoagentx.rag.rag import RAGEngine
from evoagentx.rag.rag_config import RAGConfig, ReaderConfig, ChunkerConfig, EmbeddingConfig, IndexConfig, RetrievalConfig
from evoagentx.storages.base import StorageHandler
from evoagentx.storages.storages_config import StoreConfig, VectorStoreConfig, DBConfig

# 加载环境变量
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def extract_api_info_from_openapi(api_spec: Dict[str, Any]) -> List[Dict[str, str]]:
    functions = []
    
    # 提取基本信息
    info = api_spec.get("info", {})
    api_source = info.get("source")
    api_title = info.get("title")
    
    # 提取所有端点和操作信息
    paths = api_spec.get("paths", {})
    for path, methods in paths.items():
        for method, operation in methods.items():
            if isinstance(operation, dict):
                operation_id = operation.get("operationId", f"{method}_{path}")
                summary = operation.get("summary", "")
                op_description = operation.get("description", "")
                
                # 组合描述文本用于检索
                description_parts = []
                if summary:
                    description_parts.append(summary)
                if op_description:
                    description_parts.append(op_description)
                
                description = " ".join(description_parts)
                
                functions.append({
                    "api_source": api_source,
                    "api_title": api_title,
                    "function_name": operation_id,
                    "description": description
                })
    
    return functions


def load_api_pool(api_pool_dir: str) -> List[Dict[str, str]]:
    all_functions = []
    
    # 查找所有 JSON 文件
    json_files = glob.glob(os.path.join(api_pool_dir, "*.json"))
    
    print(f"发现 {len(json_files)} 个 API 文件")
    
    for json_file in json_files:
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                api_spec = json.load(f)
                functions = extract_api_info_from_openapi(api_spec)
                all_functions.extend(functions)
                print(f"已加载: {json_file}")
        except Exception as e:
            print(f"加载 {json_file} 失败: {e}")
    
    return all_functions


# 配置存储（SQLite 用于元数据，FAISS 用于向量）
store_config = StoreConfig(
    dbConfig=DBConfig(db_name="sqlite", path="./evoagentx/tools/api_pool/data/api_cache.db"),
    vectorConfig=VectorStoreConfig(vector_name="faiss", dimensions=1536, index_type="flat_l2"),
    graphConfig=None,
    path="./evoagentx/tools/api_pool/data/api_indexing"
)
storage_handler = StorageHandler(storageConfig=store_config)

# 配置 RAGEngine - 针对 API 检索优化
rag_config = RAGConfig(
    embedding=EmbeddingConfig(
        provider="openai", 
        model_name="text-embedding-3-small", 
        api_key=OPENAI_API_KEY
    ),
    index=IndexConfig(index_type="vector"),
    retrieval=RetrievalConfig(
        retrieval_type="vector", 
        postprocessor_type="simple", 
        top_k=5  # 返回前5个最相关的API函数
    )
)

# 初始化 RAGEngine
rag_engine = RAGEngine(config=rag_config, storage_handler=storage_handler)

print("API RAGEngine 已准备就绪！")


# 步骤 1：加载并索引 API Pool
api_pool_dir = "./evoagentx/tools/api_pool"
all_functions = load_api_pool(api_pool_dir)

# 保存为 JSON 文件用于备份和调试(实际检索时使用 metadata,不需要加载此文件)
os.makedirs("./evoagentx/tools/api_pool/data", exist_ok=True)
api_json_file = "./evoagentx/tools/api_pool/data/api_pool_functions.json"
with open(api_json_file, "w", encoding="utf-8") as f:
    json.dump(all_functions, f, ensure_ascii=False, indent=2)

print(f"\nAPI 信息已写入: {api_json_file}")

# 为每个函数创建独立的文档
documents = []
for i, func in enumerate(all_functions):
    # 直接使用描述作为文本内容,不需要 FUNCTION_ID 标记
    doc_text = func['description']
    doc = Document(
        text=doc_text,
        metadata=ChunkMetadata(
            doc_id=f"func_{i}",
            custom_fields={
                "function_id": i,
                "function_name": func["function_name"],
                "api_title": func["api_title"],
                "api_source": func["api_source"],
                "description": func["description"]  # 也在 metadata 中保存完整描述
            }
        )
    )
    documents.append(doc)

# 直接从 Document 对象创建 Corpus
print(f"\n开始创建 chunks...")
chunks = []
for i, doc in enumerate(documents):
    chunk = TextChunk(
        text=doc.text,
        metadata=doc.metadata,
        chunk_id=doc.doc_id
    )
    chunks.append(chunk)
    if (i + 1) % 10 == 0:
        print(f"  已创建 {i + 1}/{len(documents)} chunks")

print(f"Chunk 创建完成，共 {len(chunks)} 个")

corpus = Corpus(chunks=chunks, corpus_id="api_pool_corpus")
print(f"\n开始添加到向量索引（这可能需要一些时间，因为需要调用 OpenAI API 生成 embeddings）...")
rag_engine.add(index_type="vector", nodes=corpus, corpus_id="api_pool_corpus")
print("向量索引添加完成!")

print("API Pool 索引成功！")
print(f"共索引 {len(all_functions)} 个函数\n")

# 将索引保存到磁盘
rag_engine.save(output_path="./evoagentx/tools/api_pool/data/indexing", corpus_id="api_pool_corpus", index_type="vector")
print("索引已保存到 ./evoagentx/tools/api_pool/data/indexing")


def search_api_functions(query_str: str, top_k: int = 3) -> List[Dict[str, str]]:
    # RAG 检索
    query = Query(query_str=query_str, top_k=top_k)
    result = rag_engine.query(query, corpus_id="api_pool_corpus")
    
    functions = []
    for chunk in result.corpus.chunks:
        # 直接从 metadata 的 custom_fields 中获取 API 信息
        custom_fields = chunk.metadata.custom_fields
        if custom_fields:
            func_info = {
                "api_source": custom_fields.get("api_source"),
                "api_title": custom_fields.get("api_title"),
                "function_name": custom_fields.get("function_name"),
                "description": custom_fields.get("description", chunk.text)
            }
            functions.append(func_info)
    
    return functions


def get_api_spec_by_title(api_title: str, api_pool_dir: str = "./evoagentx/tools/api_pool") -> Dict[str, Any]:
    json_files = glob.glob(os.path.join(api_pool_dir, "*.json"))
    
    for json_file in json_files:
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                api_spec = json.load(f)
                if api_spec.get("info", {}).get("title") == api_title:
                    return api_spec
        except Exception as e:
            print(f"读取 {json_file} 失败: {e}")
    
    return None


def get_function_details(function_name: str, api_title: str) -> Dict[str, Any]:
    api_spec = get_api_spec_by_title(api_title)
    
    if not api_spec:
        return {"error": f"未找到 API: {api_title}"}
    
    # 查找对应的函数
    paths = api_spec.get("paths", {})
    for path, methods in paths.items():
        for method, operation in methods.items():
            if isinstance(operation, dict) and operation.get("operationId") == function_name:
                return {
                    "api_title": api_title,
                    "function_name": function_name,
                    "method": method.upper(),
                    "path": path,
                    "summary": operation.get("summary", ""),
                    "description": operation.get("description", ""),
                    "parameters": operation.get("parameters", []),
                }
    
    return {"error": f"未找到函数: {function_name}"}


# 步骤 2：测试查询 - 根据用户问题检索合适的 API 函数
test_queries = [
    "I need to get weather information for a city.",
    "How can I get weather forecast the day after tomorrow?",
]

print("\n" + "="*80)
print("测试 API 检索功能")
print("="*80)

for query_str in test_queries:
    print(f"\n用户查询: {query_str}")
    print("-" * 80)
    
    # 搜索相关函数
    functions = search_api_functions(query_str, top_k=2)
    
    print(f"找到 {len(functions)} 个相关函数：\n")
    
    for i, func in enumerate(functions, 1):
        print(f"候选 {i}:")
        print(f"  函数名: {func['function_name']}")
        print(f"  API来源: {func['api_source']}")
        print(f"  API名称: {func['api_title']}")
        print(f"  描述: {func['description']}")
        
        # 获取详细信息（包括参数）
        details = get_function_details(func['function_name'], func['api_title'])
        if "error" not in details:
            print(f"  方法: {details['method']} {details['path']}")
            if details['parameters']:
                print(f"  参数:")
                for param in details['parameters']:
                    required = "(必需)" if param.get('required') in ['true', True] else "(可选)"
                    print(f"    - {param.get('name')} {required}: {param.get('description', '')}")
        print()

# 注意：不清理索引，以便后续使用
# 如果需要清理，可以取消注释下面这行
# rag_engine.clear(corpus_id="api_pool_corpus")