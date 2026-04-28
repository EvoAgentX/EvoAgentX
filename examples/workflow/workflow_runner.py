## This example shows how to use the workflow to recommend a PHD direction for a candidate based on their resume.
## It uses the arxiv-mcp-server to search the papers. You may find the project here: https://github.com/blazickjp/arxiv-mcp-server/tree/main

import json
import os 
from dotenv import load_dotenv 
import sys

from evoagentx.models import OpenAILLMConfig, OpenAILLM
from evoagentx.workflow import WorkFlowGraph, WorkFlow
from evoagentx.workflow.workflow_generator import WorkFlowGenerator
from evoagentx.agents import AgentManager
from evoagentx.tools.mcp import MCPToolkit
from evoagentx.tools.file_tool import FileToolkit
from evoagentx.tools.storage_file import StorageToolkit
load_dotenv() # Loads environment variables from .env file
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

output_file = "examples/output/test_workflow_output.md"
module_save_path = "examples/workflow/graphs/json_workflow.json"

def main(goal=None):
    # LLM configuration
    openai_config = OpenAILLMConfig(model="gpt-4o-mini", openai_key=OPENAI_API_KEY, stream=True, output_response=True, max_tokens=16000)
    # Initialize the language model
    llm = OpenAILLM(config=openai_config)
    
    goal = """Write a workflow that generate script for a short video. The script must contain charactor design, background design etc."""

    
    ## Get tools
    tools = [FileToolkit(), StorageToolkit()]
    sample_inputs = {
        "story_theme": "courage",
        "story_context": "a moonlit glass forest where the trees sing softly before sunrise",
        "character_type": "young hero",
        "target_audience": "children",
        "basic_description": "a shy apprentice lantern-maker named Luma who is afraid of the dark",
        "plot_twist_element": "a cracked star hidden inside an old lantern"
    }
    
    
    # ## _______________ Workflow Creation _______________
    # wf_generator = WorkFlowGenerator(llm=llm, tools=tools)
    # workflow_graph: WorkFlowGraph = wf_generator.generate_workflow(goal=goal)
    # # [optional] save workflow 
    # # workflow_graph.save_module(module_save_path)
    
    
    ## _______________ Workflow Execution _______________
    #[optional] load saved workflow 
    workflow_graph: WorkFlowGraph = WorkFlowGraph.from_file(module_save_path, llm_config=openai_config, tools=tools)

    # [optional] display workflow
    # workflow_graph.display()
    agent_manager = AgentManager(tools=tools)
    agent_manager.add_agents_from_workflow(workflow_graph, llm_config=openai_config)
    # from pdb import set_trace; set_trace()

    workflow = WorkFlow(graph=workflow_graph, agent_manager=agent_manager, llm=llm)
    output = workflow.execute(inputs=sample_inputs, return_mode="json")
    
    
    ## _______________ Save Output _______________
    try:
        # Write to file
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(output, indent=4) if isinstance(output, dict) else output)
        print(f"Result have been saved to {output_file}")
    except Exception as e:
        print(f"Error saving result: {e}")
    
    # from pdb import set_trace; set_trace()
    print(output)
    
    # verfiy the code
    

if __name__ == "__main__": 
    # Get custom goal from positional argument if provided
    custom_goal = sys.argv[1] if len(sys.argv) > 1 else None
    
    # Run the main function with the provided goal
    main(custom_goal)
