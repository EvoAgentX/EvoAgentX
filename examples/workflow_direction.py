## This example shows how to use the workflow to recommend a PHD direction for a candidate based on their resume.
## It uses the arxiv-mcp-server to search the papers. You may find the project here: https://github.com/blazickjp/arxiv-mcp-server/tree/main

import os 
from dotenv import load_dotenv 
import sys

from evoagentx.models import OpenAILLMConfig, OpenAILLM
from evoagentx.workflow import WorkFlowGraph, WorkFlow
from evoagentx.workflow.workflow_generator import WorkFlowGenerator
from evoagentx.agents import AgentManager
from evoagentx.tools.mcp import MCPToolkit
from evoagentx.tools.file_tool import FileToolkit
load_dotenv() # Loads environment variables from .env file
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

output_file = "examples/output/direction/output.md"
mcp_config_path = "examples/output/direction/mcp_direction.config"
target_directory = "examples/output/direction/"
module_save_path = "examples/output/direction/direction_demo_4o_mini.json"

def main(goal=None):
    # LLM configuration
    openai_config = OpenAILLMConfig(model="gpt-4o-mini", openai_key=OPENAI_API_KEY, stream=True, output_response=True, max_tokens=16000)
    # Initialize the language model
    llm = OpenAILLM(config=openai_config)
    
    goal = """Read and analyze the candidate's pdf resume at examples/output/jobs/test_pdf.pdf, and recommend one future PHD directions based on the resume. You should provide a list of 5 review papers about the topic for the candidate to learn more about this direction as well."""
    # goal = making_goal(openai_config, goal)
    helper_prompt = """The input is one parameter called "goal", and the output is a markdown report. 
    You should firstly read the pdf resume and summarize the background and recommend one future PHD direction based on the resume.
    Then you should find 3 trending Review Papers about the topic by searching the keyword on arxiv (by searching web instead of using your out-dated training data) and provide the link of the papers.
    Lastly you should summarize all the information and provide a detailed markdown report.
    If you cannot find the papers, you should say "I cannot find the papers".
    """
    
    goal += helper_prompt
    
    ## Get tools
    mcp_Toolkit = MCPToolkit(config_path=mcp_config_path)
    tools = mcp_Toolkit.get_tools()
    tools.append(FileToolkit())
    
    
    ## _______________ Workflow Creation _______________
    wf_generator = WorkFlowGenerator(llm=llm, tools=tools)
    workflow_graph: WorkFlowGraph = wf_generator.generate_workflow(goal=goal)
    # [optional] save workflow 
    workflow_graph.save_module(module_save_path)
    
    
    ## _______________ Workflow Execution _______________
    #[optional] load saved workflow 
    workflow_graph: WorkFlowGraph = WorkFlowGraph.from_file(module_save_path)

    # [optional] display workflow
    workflow_graph.display()
    agent_manager = AgentManager(tools=tools)
    agent_manager.add_agents_from_workflow(workflow_graph, llm_config=openai_config)
    # from pdb import set_trace; set_trace()

    workflow = WorkFlow(graph=workflow_graph, agent_manager=agent_manager, llm=llm)
    workflow.init_module()
    output = workflow.execute()
    
    
    ## _______________ Save Output _______________
    try:
        # Write to file
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(output)
        print(f"Direction recommendations have been saved to {output_file}")
    except Exception as e:
        print(f"Error saving direction recommendations: {e}")
    
    # from pdb import set_trace; set_trace()
    print(output)
    
    # verfiy the code
    

if __name__ == "__main__": 
    # Get custom goal from positional argument if provided
    custom_goal = sys.argv[1] if len(sys.argv) > 1 else None
    
    # Run the main function with the provided goal
    main(custom_goal)
