import os 
from dotenv import load_dotenv
from evoagentx.models import OpenAILLMConfig, OpenAILLM
from evoagentx.workflow import WorkFlowGenerator, WorkFlowGraph, WorkFlow
from evoagentx.agents import AgentManager
from evoagentx.tools.file_tool import FileToolkit
from evoagentx.tools import ArxivToolkit   

load_dotenv()  
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def main():

    openai_config = OpenAILLMConfig(
        model="gpt-4o",
        openai_key=OPENAI_API_KEY,
        stream=True,
        output_response=True,
        max_tokens=16000
    )
    llm = OpenAILLM(config=openai_config)

    keywords = "medical, multiagent"
    max_results = 10
    date_from = "2024-01-01"
    categories = ["cs.AI", "cs.LG"]

    search_constraints = f"""
    Search constraints:
    - Query keywords: {keywords}
    - Max results: {max_results}
    - Date from: {date_from}
    - Categories: {', '.join(categories)}
    """

    goal = f"""Create a daily research paper recommendation assistant that takes user keywords and pushes new relevant papers with summaries.

    The assistant should:
    1. Use the ArxivToolkit to search for the latest papers using the given keywords.
    2. Apply the following search constraints:
    {search_constraints}
    3. Summarize the search results.
    4. Compile the summaries into a well-formatted Markdown digest.

    ### Output
    daily_paper_digest
    """

    target_directory = "EvoAgentX/examples/output/paper_push"
    module_save_path = os.path.join(target_directory, "paper_push_workflow.json")
    result_path = os.path.join(target_directory, "daily_paper_digest.md")
    os.makedirs(target_directory, exist_ok=True)

    arxiv_toolkit = ArxivToolkit()
    tools = [arxiv_toolkit, FileToolkit()]

    wf_generator = WorkFlowGenerator(llm=llm, tools=tools)
    workflow_graph: WorkFlowGraph = wf_generator.generate_workflow(goal=goal)

    workflow_graph.save_module(module_save_path)

    workflow_graph.display()

    agent_manager = AgentManager(tools=tools)
    agent_manager.add_agents_from_workflow(workflow_graph, llm_config=openai_config)

    workflow = WorkFlow(graph=workflow_graph, agent_manager=agent_manager, llm=llm)
    output = workflow.execute()

    with open(result_path, "w", encoding="utf-8") as f:
        f.write(output)

    print(f"✅ Your file has been saved to：{result_path}")
    print("📬 You can run this script everyday to obtain daily recommendation")


if __name__ == "__main__":
    main()
