# 使用 RAGEngine 构建您的第一个 RAG 系统

在 EvoAgentX 中，`RAGEngine` 是一个强大的工具，用于构建检索增强生成（RAG）系统。它允许您加载文档、创建可搜索的索引，并检索相关信息来回答问题。本教程专为初学者设计，将指导您完成使用 `RAGEngine` 创建和使用 RAG 系统的基本步骤：

1. **设置 RAGEngine**：学习如何配置和初始化引擎。
2. **索引和查询文档**：加载一个简单数据集并查询以找到答案。
3. **保存和加载索引**：保存您的索引数据并在以后重复使用。

完成本教程后，您将能够设置一个基础的 RAG 系统，索引文档，查询它们，并持久化您的工作以供未来使用。我们将使用 HotPotQA 数据集的一个样本来进行测试。

## 1. 设置 RAGEngine

第一步是设置您的环境并初始化 `RAGEngine`。这包括安装依赖项、配置存储后端和设置嵌入模型。

### 安装依赖项
确保您已安装 EvoAgentX。您还需要一个 OpenAI API 密钥用于嵌入文本。

在您的项目目录中创建一个 `.env` 文件，并添加您的 OpenAI API 密钥：

```plaintext
OPENAI_API_KEY=您的-openai-api-密钥
```

### 配置环境
让我们编写一个 Python 脚本来设置 `RAGEngine`。我们将使用 SQLite 存储元数据，FAISS 存储向量嵌入。

```python
import os
from dotenv import load_dotenv
from evoagentx.rag.rag import RAGEngine
from evoagentx.rag.rag_config import RAGConfig, ReaderConfig, ChunkerConfig, EmbeddingConfig, IndexConfig, RetrievalConfig
from evoagentx.storages.base import StorageHandler
from evoagentx.storages.storages_config import StoreConfig, VectorStoreConfig, DBConfig

# 加载环境变量
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 配置存储（SQLite 用于元数据，FAISS 用于向量）
store_config = StoreConfig(
    dbConfig=DBConfig(db_name="sqlite", path="./data/cache.db"),
    vectorConfig=VectorStoreConfig(vector_name="faiss", dimensions=1536, index_type="flat_l2"),
    graphConfig=None,
    path="./data/indexing"
)
storage_handler = StorageHandler(storageConfig=store_config)

# 配置 RAGEngine
rag_config = RAGConfig(
    reader=ReaderConfig(recursive=False, exclude_hidden=True),
    chunker=ChunkerConfig(strategy="simple", chunk_size=512, chunk_overlap=50),
    embedding=EmbeddingConfig(provider="openai", model_name="text-embedding-ada-002", api_key=OPENAI_API_KEY),
    index=IndexConfig(index_type="vector"),
    retrieval=RetrievalConfig(retrieval_type="vector", postprocessor_type="simple", top_k=3, similarity_cutoff=0.3)
)

# 初始化 RAGEngine
rag_engine = RAGEngine(config=rag_config, storage_handler=storage_handler)

print("RAGEngine 已准备就绪！")
```

### 代码解析
- **环境设置**：我们从 `.env` 文件中加载 OpenAI API 密钥。
- **存储配置**：设置 SQLite 存储元数据，FAISS 存储向量嵌入。`dimensions=1536` 与 OpenAI 嵌入模型匹配。
- **RAG 配置**：配置 RAG 流程：
  - `ReaderConfig`：读取文件（稍后我们将使用 JSON 文件）。
  - `ChunkerConfig`：将文档分割成 512 字符的分块，50 字符重叠。
  - `EmbeddingConfig`：使用 OpenAI 的 `text-embedding-ada-002` 生成嵌入。
  - `IndexConfig`：创建向量索引。
  - `RetrievalConfig`：检索相似度得分高于 0.3 的前 3 个分块。
- **初始化**：创建 `RAGEngine` 实例，准备处理文档。

有关配置的更多详细信息，请参见 [RAGEngine 文档](../modules/rag.md)。

## 2. 索引和查询文档

现在，我们将索引 HotPotQA 数据集中的一个文档并查询它。我们将使用您提供的第一个 HotPotQA 示例，其中包含关于 Scott Derrickson 和 Ed Wood 的信息，并回答问题：“Scott Derrickson 和 Ed Wood 是同一个国籍吗？”

### 准备数据集
创建一个名为 `data` 的目录，并将 HotPotQA 示例保存为 `hotpotqa_sample.json`：

```json
{
  "_id": "5a8b57f25542995d1e6f1371",
  "answer": "yes",
  "question": "Were Scott Derrickson and Ed Wood of the same nationality?",
  "supporting_facts": [
    [
      "Scott Derrickson",
      0
    ],
    [
      "Ed Wood",
      0
    ]
  ],
  "context": [
    [
      "Ed Wood (film)",
      [
        "Ed Wood is a 1994 American biographical period comedy-drama film directed and produced by Tim Burton, and starring Johnny Depp as cult filmmaker Ed Wood.",
        " The film concerns the period in Wood's life when he made his best-known films as well as his relationship with actor Bela Lugosi, played by Martin Landau.",
        " Sarah Jessica Parker, Patricia Arquette, Jeffrey Jones, Lisa Marie, and Bill Murray are among the supporting cast."
      ]
    ],
    [
      "Scott Derrickson",
      [
        "Scott Derrickson (born July 16, 1966) is an American director, screenwriter and producer.",
        " He lives in Los Angeles, California.",
        " He is best known for directing horror films such as \"Sinister\", \"The Exorcism of Emily Rose\", and \"Deliver Us From Evil\", as well as the 2016 Marvel Cinematic Universe installment, \"Doctor Strange.\""
      ]
    ],
    [
      "Woodson, Arkansas",
      [
        "Woodson is a census-designated place (CDP) in Pulaski County, Arkansas, in the United States.",
        " Its population was 403 at the 2010 census.",
        " It is part of the Little Rock\u2013North Little Rock\u2013Conway Metropolitan Statistical Area.",
        " Woodson and its accompanying Woodson Lake and Wood Hollow are the namesake for Ed Wood Sr., a prominent plantation owner, trader, and businessman at the turn of the 20th century.",
        " Woodson is adjacent to the Wood Plantation, the largest of the plantations own by Ed Wood Sr."
      ]
    ],
    [
      "Tyler Bates",
      [
        "Tyler Bates (born June 5, 1965) is an American musician, music producer, and composer for films, television, and video games.",
        " Much of his work is in the action and horror film genres, with films like \"Dawn of the Dead, 300, Sucker Punch,\" and \"John Wick.\"",
        " He has collaborated with directors like Zack Snyder, Rob Zombie, Neil Marshall, William Friedkin, Scott Derrickson, and James Gunn.",
        " With Gunn, he has scored every one of the director's films; including \"Guardians of the Galaxy\", which became one of the highest grossing domestic movies of 2014, and its 2017 sequel.",
        " In addition, he is also the lead guitarist of the American rock band Marilyn Manson, and produced its albums \"The Pale Emperor\" and \"Heaven Upside Down\"."
      ]
    ],
    [
      "Ed Wood",
      [
        "Edward Davis Wood Jr. (October 10, 1924 \u2013 December 10, 1978) was an American filmmaker, actor, writer, producer, and director."
      ]
    ],
    [
      "Deliver Us from Evil (2014 film)",
      [
        "Deliver Us from Evil is a 2014 American supernatural horror film directed by Scott Derrickson and produced by Jerry Bruckheimer.",
        " The film is officially based on a 2001 non-fiction book entitled \"Beware the Night\" by Ralph Sarchie and Lisa Collier Cool, and its marketing campaign highlighted that it was \"inspired by actual accounts\".",
        " The film stars Eric Bana, \u00c9dgar Ram\u00edrez, Sean Harris, Olivia Munn, and Joel McHale in the main roles and was released on July 2, 2014."
      ]
    ],
    [
      "Adam Collis",
      [
        "Adam Collis is an American filmmaker and actor.",
        " He attended the Duke University from 1986 to 1990 and the University of California, Los Angeles from 2007 to 2010.",
        " He also studied cinema at the University of Southern California from 1991 to 1997.",
        " Collis first work was the assistant director for the Scott Derrickson's short \"Love in the Ruins\" (1995).",
        " In 1998, he played \"Crankshaft\" in Eric Koyanagi's \"Hundred Percent\"."
      ]
    ],
    [
      "Sinister (film)",
      [
        "Sinister is a 2012 supernatural horror film directed by Scott Derrickson and written by Derrickson and C. Robert Cargill.",
        " It stars Ethan Hawke as fictional true-crime writer Ellison Oswalt who discovers a box of home movies in his attic that puts his family in danger."
      ]
    ],
    [
      "Conrad Brooks",
      [
        "Conrad Brooks (born Conrad Biedrzycki on January 3, 1931 in Baltimore, Maryland) is an American actor.",
        " He moved to Hollywood, California in 1948 to pursue a career in acting.",
        " He got his start in movies appearing in Ed Wood films such as \"Plan 9 from Outer Space\", \"Glen or Glenda\", and \"Jail Bait.\"",
        " He took a break from acting during the 1960s and 1970s but due to the ongoing interest in the films of Ed Wood, he reemerged in the 1980s and has become a prolific actor.",
        " He also has since gone on to write, produce and direct several films."
      ]
    ],
    [
      "Doctor Strange (2016 film)",
      [
        "Doctor Strange is a 2016 American superhero film based on the Marvel Comics character of the same name, produced by Marvel Studios and distributed by Walt Disney Studios Motion Pictures.",
        " It is the fourteenth film of the Marvel Cinematic Universe (MCU).",
        " The film was directed by Scott Derrickson, who wrote it with Jon Spaihts and C. Robert Cargill, and stars Benedict Cumberbatch as Stephen Strange, along with Chiwetel Ejiofor, Rachel McAdams, Benedict Wong, Michael Stuhlbarg, Benjamin Bratt, Scott Adkins, Mads Mikkelsen, and Tilda Swinton.",
        " In \"Doctor Strange\", surgeon Strange learns the mystic arts after a career-ending car accident."
      ]
    ]
  ],
  "type": "comparison",
  "level": "hard"
}
```

### 索引文档
让我们编写一个脚本加载 JSON 文件，索引其内容并查询它。创建一个名为 `rag_tutorial.py` 的文件：

```python
import os
import json

from dotenv import load_dotenv

from evoagentx.rag.schema import Query
from evoagentx.rag.rag import RAGEngine
from evoagentx.rag.rag_config import RAGConfig, ReaderConfig, ChunkerConfig, EmbeddingConfig, IndexConfig, RetrievalConfig
from evoagentx.storages.base import StorageHandler
from evoagentx.storages.storages_config import StoreConfig, VectorStoreConfig, DBConfig

# 加载环境变量
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 配置存储（SQLite 用于元数据，FAISS 用于向量）
store_config = StoreConfig(
    dbConfig=DBConfig(db_name="sqlite", path="./data/cache.db"),
    vectorConfig=VectorStoreConfig(vector_name="faiss", dimensions=1536, index_type="flat_l2"),
    graphConfig=None,
    path="./data/indexing"
)
storage_handler = StorageHandler(storageConfig=store_config)

# 配置 RAGEngine
rag_config = RAGConfig(
    reader=ReaderConfig(recursive=False, exclude_hidden=True),
    chunker=ChunkerConfig(strategy="simple", chunk_size=512, chunk_overlap=50),
    embedding=EmbeddingConfig(provider="openai", model_name="text-embedding-ada-002", api_key=OPENAI_API_KEY),
    index=IndexConfig(index_type="vector"),
    retrieval=RetrievalConfig(retrieval_type="vector", postprocessor_type="simple", top_k=3, similarity_cutoff=0.3)
)

# 初始化 RAGEngine
rag_engine = RAGEngine(config=rag_config, storage_handler=storage_handler)

print("RAGEngine 已准备就绪！")


# 步骤 1：加载并索引 HotPotQA 示例
with open("./data/hotpotqa_sample.json", "r", encoding="utf-8") as f:
    hotpotqa_data = json.load(f)

# 创建一个临时文本文件，包含上下文用于索引
context = hotpotqa_data["context"]
with open("./data/hotpotqa_context.txt", "w", encoding="utf-8") as f:
    for title, sentences in context:
        f.write(f"# {title}")
        for sentence in sentences:
            f.write(f"{sentence}\n")
        f.write(f"\n\n")

# 索引文本文件
corpus = rag_engine.read(
    file_paths="./data/hotpotqa_context.txt",
    filter_file_by_suffix=[".txt"],
    merge_by_file=True,
    show_progress=True,
    corpus_id="hotpotqa_corpus"
)
rag_engine.add(index_type="vector", nodes=corpus, corpus_id="hotpotqa_corpus")

print("文档索引成功！")

# 步骤 2：查询索引
query = Query(query_str="Were Scott Derrickson and Ed Wood of the same nationality?", top_k=3)
result = rag_engine.query(query, corpus_id="hotpotqa_corpus")

# 打印检索到的分块
print("\n检索到的答案：")
for i, chunk in enumerate(result.corpus.chunks, 1):
    print(f"{i}. {chunk.text}")

# 清理
rag_engine.clear(corpus_id="hotpotqa_corpus")
```

### 运行脚本
运行脚本：

```bash
python rag_tutorial.py
```

### 代码解析
- **加载数据**：我们读取 HotPotQA JSON 文件并提取其 `context` 字段，包含关于 Scott Derrickson 和 Ed Wood 的文本。
- **索引**：
  - 创建一个临时文本文件（`hotpotqa_context.txt`），格式化上下文以便阅读。
  - `read` 方法将文本文件加载到 `Corpus` 中，根据之前设置的 512 字符限制分割成块。
  - `add` 方法使用 OpenAI 嵌入将分块索引到向量索引中。
- **查询**：
  - 创建一个 `Query` 对象，包含数据集中的问题。
  - `query` 方法检索与问题最相似的 3 个分块。
  - 结果包括表明两人都是美国人的句子（例如，“Scott Derrickson… 是美国导演”和“爱德华·戴维斯·伍德二世… 是美国电影制片人”）。
- **清理**：清除索引以释放内存。

### 预期输出
您应该看到类似以下内容：

```
文档索引成功！

检索到的答案：
1. # Ed Wood (film)Ed Wood is a 1994 American biographical period comedy-drama film directed and produced by Tim Burton, and starring Johnny Depp as cult filmmaker Ed Wood.
 The film concerns the period in Wood's life when he made his best-known films as well as his relationship with actor Bela Lugosi, played by Martin Landau. 
 Sarah Jessica Parker, Patricia Arquette, Jeffrey Jones, Lisa Marie, and Bill Murray are among the supporting cast.


# Scott DerricksonScott Derrickson (born July 16, 1966) is an American director, screenwriter and producer.
 He lives in Los Angeles, California.
 He is best known for directing horror films such as "Sinister", "The Exorcism of Emily Rose", and "Deliver Us From Evil", as well as the 2016 Marvel Cinematic Universe installment, "Doctor Strange."


# Woodson, ArkansasWoodson is a census-designated place (CDP) in Pulaski County, Arkansas, in the United States.
 Its population was 403 at the 2010 census.
 It is part of the Little Rock–North Little Rock–Conway Metropolitan Statistical Area.
 Woodson and its accompanying Woodson Lake and Wood Hollow are the namesake for Ed Wood Sr., a prominent plantation owner, trader, and businessman at the turn of the 20th century.
 Woodson is adjacent to the Wood Plantation, the largest of the plantations own by Ed Wood Sr.


# Tyler BatesTyler Bates (born June 5, 1965) is an American musician, music producer, and composer for films, television, and video games.
 Much of his work is in the action and horror film genres, with films like "Dawn of the Dead, 300, Sucker Punch," and "John Wick."
 He has collaborated with directors like Zack Snyder, Rob Zombie, Neil Marshall, William Friedkin, Scott Derrickson, and James Gunn.
 With Gunn, he has scored every one of the director's films; including "Guardians of the Galaxy", which became one of the highest grossing domestic movies of 2014, and its 2017 sequel.
2. With Gunn, he has scored every one of the director's films; including "Guardians of the Galaxy", which became one of the highest grossing domestic movies of 2014, and its 2017 sequel.
 In addition, he is also the lead guitarist of the American rock band Marilyn Manson, and produced its albums "The Pale Emperor" and "Heaven Upside Down".  


# Ed WoodEdward Davis Wood Jr. (October 10, 1924 – December 10, 1978) was an American filmmaker, actor, writer, producer, and director.


# Deliver Us from Evil (2014 film)Deliver Us from Evil is a 2014 American supernatural horror film directed by Scott Derrickson and produced by Jerry Bruckheimer.
 The film is officially based on a 2001 non-fiction book entitled "Beware the Night" by Ralph Sarchie and Lisa Collier Cool, and its marketing campaign highlighted that it was "inspired by actual accounts".
 The film stars Eric Bana, Édgar Ramírez, Sean Harris, Olivia Munn, and Joel McHale in the main roles and was released on July 2, 2014.


# Adam CollisAdam Collis is an American filmmaker and actor.
 He attended the Duke University from 1986 to 1990 and the University of California, Los Angeles from 2007 to 2010.
 He also studied cinema at the University of Southern California from 1991 to 1997.
 Collis first work was the assistant director for the Scott Derrickson's short "Love in the Ruins" (1995).
 In 1998, he played "Crankshaft" in Eric Koyanagi's "Hundred Percent".


# Sinister (film)Sinister is a 2012 supernatural horror film directed by Scott Derrickson and written by Derrickson and C. Robert Cargill.
 It stars Ethan Hawke as fictional true-crime writer Ellison Oswalt who discovers a box of home movies in his attic that puts his family in danger.
3. It stars Ethan Hawke as fictional true-crime writer Ellison Oswalt who discovers a box of home movies in his attic that puts his family in danger.       


# Conrad BrooksConrad Brooks (born Conrad Biedrzycki on January 3, 1931 in Baltimore, Maryland) is an American actor.
 He moved to Hollywood, California in 1948 to pursue a career in acting.
 He got his start in movies appearing in Ed Wood films such as "Plan 9 from Outer Space", "Glen or Glenda", and "Jail Bait."
 He took a break from acting during the 1960s and 1970s but due to the ongoing interest in the films of Ed Wood, he reemerged in the 1980s and has become a prolific actor.
 He also has since gone on to write, produce and direct several films.


# Doctor Strange (2016 film)Doctor Strange is a 2016 American superhero film based on the Marvel Comics character of the same name, produced by Marvel Studios and distributed by Walt Disney Studios Motion Pictures.
 It is the fourteenth film of the Marvel Cinematic Universe (MCU).
 The film was directed by Scott Derrickson, who wrote it with Jon Spaihts and C. Robert Cargill, and stars Benedict Cumberbatch as Stephen Strange, along with Chiwetel Ejiofor, Rachel McAdams, Benedict Wong, Michael Stuhlbarg, Benjamin Bratt, Scott Adkins, Mads Mikkelsen, and Tilda Swinton.
 In "Doctor Strange", surgeon Strange learns the mystic arts after a career-ending car accident.
```

这些分块的top1可以确认 Scott Derrickson 和 Ed Wood 都是美国人。

有关索引和查询的更多详细信息，请参见 [RAGEngine 文档](../modules/rag.md)。

## 3. 保存和加载索引

索引文档后，您可以保存索引以便以后重复使用，而无需重新处理数据。这对于大型数据集或生产环境非常有用。

### 保存索引
修改 `rag_tutorial.py` 脚本，在索引后保存索引。在 `rag_engine.add` 后添加以下代码：

```python
# 将索引保存到磁盘
rag_engine.save(output_path="./data/indexing", corpus_id="hotpotqa_corpus", index_type="vector")
print("索引已保存到 ./data/indexing")
```

### 加载索引
要加载保存的索引并查询它，创建一个名为 `rag_load.py` 的新脚本：

```python
import os

from dotenv import load_dotenv

from evoagentx.rag.schema import Query
from evoagentx.rag.rag import RAGEngine
from evoagentx.rag.rag_config import RAGConfig, ReaderConfig, ChunkerConfig, EmbeddingConfig, IndexConfig, RetrievalConfig
from evoagentx.storages.base import StorageHandler
from evoagentx.storages.storages_config import StoreConfig, VectorStoreConfig, DBConfig

# 加载环境变量
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 配置存储（SQLite 用于元数据，FAISS 用于向量）
store_config = StoreConfig(
    dbConfig=DBConfig(db_name="sqlite", path="./data/cache.db"),
    vectorConfig=VectorStoreConfig(vector_name="faiss", dimensions=1536, index_type="flat_l2"),
    graphConfig=None,
    path="./data/indexing"
)
storage_handler = StorageHandler(storageConfig=store_config)

# 配置 RAGEngine
rag_config = RAGConfig(
    reader=ReaderConfig(recursive=False, exclude_hidden=True),
    chunker=ChunkerConfig(strategy="simple", chunk_size=512, chunk_overlap=50),
    embedding=EmbeddingConfig(provider="openai", model_name="text-embedding-ada-002", api_key=OPENAI_API_KEY),
    index=IndexConfig(index_type="vector"),
    retrieval=RetrievalConfig(retrieval_type="vector", postprocessor_type="simple", top_k=3, similarity_cutoff=0.3)
)

# 初始化 RAGEngine
rag_engine = RAGEngine(config=rag_config, storage_handler=storage_handler)

print("RAGEngine 已准备就绪！")

# 加载保存的索引
rag_engine.load(source="./data/indexing", corpus_id="hotpotqa_corpus", index_type="vector")
print("索引加载成功！")

# 查询加载的索引
query = Query(query_str="Were Scott Derrickson and Ed Wood of the same nationality?", top_k=3)
result = rag_engine.query(query, corpus_id="hotpotqa_corpus")

# 打印检索到的分块
print("\n检索到的答案：")
for i, chunk in enumerate(result.corpus.chunks, 1):
    print(f"{i}. {chunk.text}")

# 清理
rag_engine.clear(corpus_id="hotpotqa_corpus")
```

### 运行脚本
运行脚本：

```bash
python rag_load.py
```

### 代码解析
- **保存**：`save` 方法将向量索引和元数据存储到 `./data/indexing` 目录。
- **加载**：`load` 方法从保存的文件重建索引，无需重新索引即可查询。
- **查询**：我们运行与之前相同的查询，结果应一致。
- **清理**：清除索引以保持整洁。

### 预期输出
输出应与第 2 节相同，确认加载的索引正常工作。

有关保存和加载的更多详细信息，请参见 [RAGEngine 文档](../modules/rag.md)。

## 下一步
恭喜您！您已使用 `RAGEngine` 构建了第一个 RAG 系统。以下是一些进一步探索的建议：
- 尝试索引更大的数据集或不同文件类型（如 PDF）。
- 实验不同的分块大小或嵌入模型（如 Hugging Face, ollama）。
- 将 `RAGEngine` 与 EvoAgentX 代理结合，自动回答问题。

有关完整示例，请参见 [RAGEngine 示例](../../../examples/rag.py)。

祝您在 EvoAgentX 中愉快地构建！