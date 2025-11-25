import os
import json
import glob
from typing import List, Dict, Any

from dotenv import load_dotenv

from evoagentx.rag.schema import Query, Document, TextChunk, Corpus, ChunkMetadata
from evoagentx.rag.rag import RAGEngine
from evoagentx.rag.rag_config import RAGConfig, EmbeddingConfig, IndexConfig, RetrievalConfig
from evoagentx.storages.base import StorageHandler
from evoagentx.storages.storages_config import StoreConfig, VectorStoreConfig, DBConfig

# 加载环境变量
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def extract_mcp_info(mcp_spec: Dict[str, Any]) -> Dict[str, Any]:
    """从 MCP JSON 文件中提取信息"""
    return {
        "name": mcp_spec.get("name"),
        "config": mcp_spec.get("config"),
        "description": mcp_spec.get("description", "")
    }


def load_mcp_pool(mcp_pool_dir: str) -> List[Dict[str, Any]]:
    """加载 MCP Pool 目录中的所有 MCP 服务器"""
    all_mcps = []
    
    # 查找所有 JSON 文件
    json_files = glob.glob(os.path.join(mcp_pool_dir, "*.json"))
    
    print(f"发现 {len(json_files)} 个 MCP 文件")
    
    for json_file in json_files:
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                mcp_spec = json.load(f)
                mcp_info = extract_mcp_info(mcp_spec)
                all_mcps.append(mcp_info)
                print(f"已加载: {json_file} - {mcp_info['name']}")
        except Exception as e:
            print(f"加载 {json_file} 失败: {e}")
    
    return all_mcps


# 配置存储（SQLite 用于元数据，FAISS 用于向量）
store_config = StoreConfig(
    dbConfig=DBConfig(db_name="sqlite", path="./evoagentx/tools/mcp_pool/data/mcp_cache.db"),
    vectorConfig=VectorStoreConfig(vector_name="faiss", dimensions=1536, index_type="flat_l2"),
    graphConfig=None,
    path="./evoagentx/tools/mcp_pool/data/mcp_indexing"
)
storage_handler = StorageHandler(storageConfig=store_config)

# 配置 RAGEngine - 针对 MCP 检索优化
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
        top_k=5  # 返回前5个最相关的MCP服务器
    )
)

# 初始化 RAGEngine
rag_engine = RAGEngine(config=rag_config, storage_handler=storage_handler)

print("MCP RAGEngine 已准备就绪！")


# 步骤 1：加载并索引 MCP Pool
mcp_pool_dir = "./evoagentx/tools/mcp_pool"
all_mcps = load_mcp_pool(mcp_pool_dir)

# 保存为 JSON 文件用于备份和调试
os.makedirs("./evoagentx/tools/mcp_pool/data", exist_ok=True)
mcp_json_file = "./evoagentx/tools/mcp_pool/data/mcp_pool_servers.json"
with open(mcp_json_file, "w", encoding="utf-8") as f:
    json.dump(all_mcps, f, ensure_ascii=False, indent=2)

print(f"\nMCP 信息已写入: {mcp_json_file}")

# 为每个 MCP 服务器创建独立的文档
documents = []
for i, mcp in enumerate(all_mcps):
    # 使用 description 作为文本内容进行相似度计算
    doc_text = mcp['description']
    doc = Document(
        text=doc_text,
        metadata=ChunkMetadata(
            doc_id=f"mcp_{i}",
            custom_fields={
                "mcp_id": i,
                "name": mcp["name"],
                "config": mcp["config"],  # 保存完整的 config
                "description": mcp["description"]
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
    print(f"  已创建 {i + 1}/{len(documents)} chunks")

print(f"Chunk 创建完成，共 {len(chunks)} 个")

corpus = Corpus(chunks=chunks, corpus_id="mcp_pool_corpus")
print(f"\n开始添加到向量索引（这可能需要一些时间，因为需要调用 OpenAI API 生成 embeddings）...")
rag_engine.add(index_type="vector", nodes=corpus, corpus_id="mcp_pool_corpus")
print("向量索引添加完成!")

print("MCP Pool 索引成功！")
print(f"共索引 {len(all_mcps)} 个 MCP 服务器\n")

# 将索引保存到磁盘
rag_engine.save(output_path="./evoagentx/tools/mcp_pool/data/indexing", corpus_id="mcp_pool_corpus", index_type="vector")
print("索引已保存到 ./evoagentx/tools/mcp_pool/data/indexing")


def search_mcp_servers(query_str: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """搜索相关的 MCP 服务器"""
    # RAG 检索
    query = Query(query_str=query_str, top_k=top_k)
    result = rag_engine.query(query, corpus_id="mcp_pool_corpus")
    
    mcp_servers = []
    for chunk in result.corpus.chunks:
        # 从 metadata 的 custom_fields 中获取 MCP 信息
        custom_fields = chunk.metadata.custom_fields
        if custom_fields:
            mcp_info = {
                "name": custom_fields.get("name"),
                "config": custom_fields.get("config"),
                "description": custom_fields.get("description", chunk.text)
            }
            mcp_servers.append(mcp_info)
    
    return mcp_servers


def convert_to_mcp_config(mcp_servers: List[Dict[str, Any]]) -> Dict[str, Any]:
    """将 MCP 服务器列表转换为 mcp.config 格式"""
    mcp_config = {
        "mcpServers": {}
    }
    
    for mcp in mcp_servers:
        name = mcp["name"]
        config = mcp["config"]
        mcp_config["mcpServers"][name] = config
    
    return mcp_config


def save_mcp_config(mcp_servers: List[Dict[str, Any]], output_path: str):
    """保存为 mcp.config 格式的 JSON 文件"""
    mcp_config = convert_to_mcp_config(mcp_servers)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(mcp_config, f, ensure_ascii=False, indent=4)
    
    print(f"MCP 配置已保存到: {output_path}")
    return mcp_config


# 步骤 2：测试查询 - 根据用户需求检索合适的 MCP 服务器
test_queries = [
    "I need to search for job opportunities and get job details",
    "Help me find and explore career opportunities",
]

print("\n" + "="*80)
print("测试 MCP 服务器检索功能")
print("="*80)

for query_str in test_queries:
    print(f"\n用户查询: {query_str}")
    print("-" * 80)
    
    # 搜索相关 MCP 服务器
    mcp_servers = search_mcp_servers(query_str, top_k=2)
    
    print(f"找到 {len(mcp_servers)} 个相关 MCP 服务器：\n")
    
    for i, mcp in enumerate(mcp_servers, 1):
        print(f"候选 {i}:")
        print(f"  名称: {mcp['name']}")
        print(f"  描述: {mcp['description']}")
        print(f"  配置:")
        print(f"    命令: {mcp['config']['command']}")
        print(f"    参数: {mcp['config']['args']}")
        if mcp['config'].get('env'):
            print(f"    环境变量: {list(mcp['config']['env'].keys())}")
        print()
    
    # 转换为 mcp.config 格式
    mcp_config = convert_to_mcp_config(mcp_servers)
    print("转换为 mcp.config 格式:")
    print(json.dumps(mcp_config, indent=4, ensure_ascii=False))
    print()
    
    # 保存配置文件
    output_path = f"./evoagentx/tools/mcp_pool/data/mcp.config"
    save_mcp_config(mcp_servers, output_path)
    print()

# 注意：不清理索引，以便后续使用
# 如果需要清理，可以取消注释下面这行
# rag_engine.clear(corpus_id="mcp_pool_corpus")

