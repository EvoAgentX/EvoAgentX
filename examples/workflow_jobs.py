import os 
from dotenv import load_dotenv 
import sys

from evoagentx.models import OpenAILLMConfig, OpenAILLM
from evoagentx.workflow import WorkFlowGraph, WorkFlow
from evoagentx.agents import AgentManager
from examples.pdf_test_prompt import formulate_goal
from evoagentx.tools.mcp import MCPToolkit
from evoagentx.tools.file_tool import FileToolKit
load_dotenv() # Loads environment variables from .env file
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

output_file = "examples/output/jobs/output.md"
mcp_config_path = "examples/output/jobs/mcp_jobs.config"
target_directory = "examples/output/jobs/"
module_save_path = "examples/output/jobs/jobs_demo_4o_mini.json"

def main(goal=None):

    # LLM configuration
    openai_config = OpenAILLMConfig(model="gpt-4o-mini", openai_key=OPENAI_API_KEY, stream=True, output_response=True, max_tokens=16000)
    # Initialize the language model
    llm = OpenAILLM(config=openai_config)
    
    if not goal:
            goal = """
                Read and analyze the pdf resume at examples/output/jobs/test_pdf.pdf, then find 5 real job opportunities based on the content of the resume by search the website.
                """
    goal = formulate_goal(goal)
    # goal = making_goal(openai_config, goal)
    
    ## Get tools
    mcp_toolkit = MCPToolkit(config_path=mcp_config_path)
    tools = mcp_toolkit.get_tools()
    tools.append(FileToolKit())
    
    
    # ## _______________ Workflow Creation _______________
    # wf_generator = WorkFlowGenerator(llm=llm, tools=tools)
    # workflow_graph: WorkFlowGraph = wf_generator.generate_workflow(goal=goal, suggestion=PDF_AGENT_SUGGESTION)
    # # [optional] display workflow
    # # [optional] save workflow 
    # workflow_graph.save_module(module_save_path)
    
    
    ## _______________ Workflow Execution _______________
    #[optional] load saved workflow 
    workflow_graph: WorkFlowGraph = WorkFlowGraph.from_file(module_save_path)

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
        print(f"Job recommendations have been saved to {output_file}")
    except Exception as e:
        print(f"Error saving job recommendations: {e}")
    
    # from pdb import set_trace; set_trace()
    print(output)
    
    # verfiy the code
    

if __name__ == "__main__": 
    # Get custom goal from positional argument if provided
    custom_goal = sys.argv[1] if len(sys.argv) > 1 else None
    
    # Run the main function with the provided goal
    main(custom_goal)
