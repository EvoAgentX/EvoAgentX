# This example shows how to use the workflow to build a RAG chatbot assistant with reference contents returned.
# By running this file, the final workflow will be saved in the examples/output/workflow_examples_output/rag_chatbot_with_ref_demo_{model_name}.json, where {model_name} is the model name used in the workflow.
# Before running this file, you need to add a file to be indexed in the examples/workflow_examples/rag_file.pdf.

import os 
from dotenv import load_dotenv 
import asyncio

from evoagentx.models import OpenAILLMConfig, OpenAILLM
from evoagentx.agents import CustomizeAgent, AgentManager
from evoagentx.prompts import StringTemplate
from evoagentx.rag.rag_config import RAGConfig, ReaderConfig, ChunkerConfig, EmbeddingConfig, IndexConfig, RetrievalConfig
from evoagentx.storages.storages_config import VectorStoreConfig, DBConfig, StoreConfig
from evoagentx.storages.base import StorageHandler
from evoagentx.tools import FaissToolkit
from evoagentx.workflow.workflow_graph import WorkFlowNode, WorkFlowEdge
from evoagentx.workflow import WorkFlow, WorkFlowGraph

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
model_name = "gpt-4o"
llm_config = OpenAILLMConfig(model=model_name, openai_key=OPENAI_API_KEY, stream=True, output_response=True)
llm = OpenAILLM(llm_config)
workflow_save_path = f"examples/workflow_examples/rag_chatbot_with_ref_demo_{model_name}.json"

def file_path_inputs_example() -> str:
    return {
        "file_path": "examples/workflow_examples/rag_file.pdf",  # make sure this file exists
        "query": "What are the three design choices for multi-modal generation and understanding models considered by the authors of this paper?"
    }


async def main():
    # Configure storage (SQLite for metadata, FAISS for vectors)
    store_config = StoreConfig(
        dbConfig=DBConfig(db_name="sqlite", path="./data/cache.db"),
        vectorConfig=VectorStoreConfig(vector_name="faiss", dimensions=1536, index_type="flat_l2"),
        graphConfig=None,
        path="./data/indexing"
    )
    storage_handler = StorageHandler(storageConfig=store_config)
    # Configure RAGEngine
    rag_config = RAGConfig(
        reader=ReaderConfig(recursive=False, exclude_hidden=True),
        chunker=ChunkerConfig(strategy="simple", chunk_size=512, chunk_overlap=50),
        embedding=EmbeddingConfig(provider="openai", model_name="text-embedding-ada-002", api_key=OPENAI_API_KEY),
        index=IndexConfig(index_type="vector"),
        retrieval=RetrievalConfig(retrieval_type="vector", postprocessor_type="simple", top_k=3, similarity_cutoff=0.3)
    )

    faiss_toolkit = FaissToolkit(storage_config=store_config, rag_config=rag_config)

    rag_agent_insert = CustomizeAgent(
        name="rag_agent_insert",
        description="An agent to insert documents into the vector database.",
        prompt_template=StringTemplate(
            instruction="You are a helpful assistant that can insert documents into the vector database. Based on the file path, you should insert the document into the vector database by using the tool that I provide",
        ),
        llm_config=llm_config,
        inputs=[
            {"name": "file_path", "type": "string", "description": "The file path to the document to be indexed."}
        ],
        outputs=[
            {"name":"insert_result", "type": "boolean", "description": "The result of the document insertion."}
        ],
        tools=[faiss_toolkit],
        parse_mode="str",
    )

    rag_agent_query = CustomizeAgent(
        name="rag_agent_query",
        description="An agent to query the vector database.",
        prompt_template=StringTemplate(
            instruction="You are a helpful assistant that can query the vector database. You should query the vector database by using the tool that I provide. Besides, provide the reference contents of the query in a list format. Each item in the list should be a string.",
        ),
        llm_config=llm_config,
        inputs=[
            {"name": "query", "type": "string", "description": "The query to the vector database."}
        ],
        outputs=[
            {"name":"query_result", "type": "string", "description": "The result of the query."},
            {"name": "reference_contents", "type": "list", "description": "The reference contents of the query. Must be presented in a list format. Each item in the list should be a string."}
        ],
        tools=[faiss_toolkit],
        parse_mode="str",
    )

    # workflow graph
    nodes = [
        WorkFlowNode(
            name="insert_node",
            description="insert the document into the vector database",
            agents=[rag_agent_insert],
            inputs=[
                {"name": "file_path", "type": "string", "description": "The file path to the document to be indexed."}
            ],
            outputs=[
                {"name":"insert_result", "type": "boolean", "description": "The result of the document insertion."}
            ]
        ),
        WorkFlowNode(
            name="query_node",
            description="query the vector database",
            agents=[rag_agent_query],
            inputs=[
                {"name": "query", "type": "string", "description": "The query to the vector database."}
            ],
            outputs=[
                {"name":"query_result", "type": "string", "description": "The result of the query."},
                {"name": "reference_contents", "type": "list", "description": "The reference contents of the query. Must be presented in a list format. Each item in the list should be a string."}
            ]
        )
    ]

    edges = [
        WorkFlowEdge(source="insert_node", target="query_node"),
    ]

    graph = WorkFlowGraph(
        goal="Firstly insert the document into the vector database and then answer the user's question based on the vector database",
        nodes=nodes,
        edges=edges
    )

    agents = [rag_agent_insert, rag_agent_query]
    workflow = WorkFlow(graph=graph, llm=llm, agent_manager=AgentManager(agents=agents))

    try:
        # save workflow
        graph.save_module(path=workflow_save_path)
        print(f"\n✅ workflow saved to {workflow_save_path}") 
        print("\n📋 start to execute the workflow")
        result = await workflow.async_execute(inputs=file_path_inputs_example())
        print(f"\n✅ workflow executed successfully!")
        print(f"result: {result}")
        
    except Exception as e:
        print(f"\n❌ workflow execution failed: {e}")

    # save json file
if __name__ == "__main__":
    asyncio.run(main()) 