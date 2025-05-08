import os 
from dotenv import load_dotenv

from evoagentx.core.registry import register_parse_function
from evoagentx.core.module_utils import extract_code_blocks
from evoagentx.workflow import SequentialWorkFlowGraph, WorkFlow 
from evoagentx.agents import AgentManager 
from evoagentx.models import OpenAILLMConfig, OpenAILLM


load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


@register_parse_function
def custom_parse_func(content: str) -> str:
    return {"code": extract_code_blocks(content)[0]} 


def build_sequential_workflow():
    
    # configure the LLM 
    llm_config = OpenAILLMConfig(model="gpt-4o-mini", openai_key=OPENAI_API_KEY, stream=True, output_response=True)
    llm = OpenAILLM(llm_config)
    
    # Define two sequential tasks: Planning and Coding
    tasks = [
        {
            "name": "Planning",
            "description": "Create a detailed plan for code generation",
            "inputs": [
                {"name": "problem", "type": "str", "required": True, "description": "Description of the problem to be solved"},
            ],
            "outputs": [
                {"name": "plan", "type": "str", "required": True, "description": "Detailed plan with steps, components, and architecture"}
            ],
            "prompt": "You are a software architect. Your task is to create a detailed implementation plan for the given problem.\n\nProblem: {problem}\n\nPlease provide a comprehensive implementation plan including:\n1. Problem breakdown\n2. Algorithm or approach selection\n3. Implementation steps\n4. Potential edge cases and solutions",
            "parse_mode": "str" , 
            # "llm_config": specific_llm_config # if you want to use a specific LLM for a task, you can add a key `llm_config` in the task dict. 
        },
        {
            "name": "Coding",
            "description": "Implement the code based on the implementation plan",
            "inputs": [
                {"name": "problem", "type": "str", "required": True, "description": "Description of the problem to be solved"},
                {"name": "plan", "type": "str", "required": True, "description": "Detailed implementation plan from the Planning phase"},
            ],
            "outputs": [
                {"name": "code", "type": "str", "required": True, "description": "Implemented code with explanations"}
            ],
            "prompt": "You are a software developer. Your task is to implement the code based on the provided problem and implementation plan.\n\nProblem: {problem}\nImplementation Plan: {plan}\n\nPlease provide the implementation code with appropriate comments.",
            "parse_mode": "custom", 
            "parse_func": custom_parse_func
        }
    ]
    
    # Create the sequential workflow
    graph = SequentialWorkFlowGraph(
        goal="Generate code to solve programming problems", # describe the goal of the workflow
        tasks=tasks, 
    )

    # [optional] save the workflow graph to a file 
    # graph.save_module("examples/output/saved_sequential_workflow.json")
    # [optional] load the workflow graph from a file 
    # graph = SequentialWorkFlowGraph.from_file("examples/output/saved_sequential_workflow.json")
    
    # create agent instance from the workflow graph 
    agent_manager = AgentManager()
    agent_manager.add_agents_from_workflow(
        graph, 
        llm_config=llm_config, # will be used for all tasks without `llm_config`. 
    )

    # create a workflow instance for execution 
    workflow = WorkFlow(graph=graph, agent_manager=agent_manager, llm=llm)

    output = workflow.execute(
        inputs = {
            "problem": "Write a function to find the longest palindromic substring in a given string."
        }
    )
    
    print("Workflow completed!")
    print("Workflow output:\n", output)



if __name__ == "__main__":
    build_sequential_workflow() 



