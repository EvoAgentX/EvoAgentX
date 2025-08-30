"""
Azure OpenAI Workflow Demo using LiteLLM

This demo shows how to use Azure OpenAI through the LiteLLM interface
for building and executing complex workflows.

## Setup

1. Install dependencies:
   uv venv
   source .venv/bin/activate
   uv pip install -r requirements.txt

2. Set environment variables in a `.env` file or directly in your shell:
   export AZURE_OPENAI_DEPLOYMENT_NAME="your-deployment-name"
   export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
   export AZURE_OPENAI_KEY="your-azure-openai-key"
   export AZURE_OPENAI_API_VERSION="2024-12-01-preview"  # Optional

3. Run the demo:
   python examples/workflow_demo_azure.py

## What this demo does:

1. Configures LiteLLM with Azure OpenAI
2. Generates a step-by-step workflow for creating a Tetris game
3. Executes the workflow to generate HTML/CSS/JS code
4. Verifies and extracts the code to files
5. Saves all intermediate results for inspection

The output will be saved to:
- examples/output/tetris_game/ (final game files)
- examples/output/workflow_intermediates/ (intermediate results)

"""

import os 
import json
from datetime import datetime
from dotenv import load_dotenv 
from evoagentx.models import LiteLLMConfig, LiteLLM
from evoagentx.workflow import WorkFlowGenerator, WorkFlowGraph, WorkFlow
from evoagentx.agents import AgentManager
from evoagentx.actions.code_extraction import CodeExtraction
from evoagentx.actions.code_verification import CodeVerification 
from evoagentx.core.module_utils import extract_code_blocks

load_dotenv()

def wait_for_user_confirmation(step_name: str):
    """Wait for user confirmation before proceeding"""
    while True:
        user_input = input(f"\nReady to proceed with {step_name}? (yes/no): ").strip().lower()
        if user_input == 'yes':
            return True
        elif user_input == 'no':
            print("Stopping execution.")
            exit(0)
        else:
            print("Please enter 'yes' or 'no'")

def configure_llm() -> LiteLLM:
    """1. LLM Configuration - Using LiteLLM with Azure OpenAI"""
    cfg = LiteLLMConfig(
        model="azure/" + os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        azure_key=os.getenv("AZURE_OPENAI_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
        stream=True,
        output_response=True,
        max_tokens=16000,
        temperature=0.7
    )
    return LiteLLM(config=cfg)

def save_intermediate_result(data, stage: str, output_dir: str):
    """Save intermediate results to JSON file"""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{stage}_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    
    # If data has to_dict method, use it; otherwise save directly
    if hasattr(data, 'to_dict'):
        data_dict = data.to_dict()
    elif hasattr(data, '__dict__'):
        data_dict = data.__dict__.copy()
        # Handle objects that cannot be serialized
        for key, value in data_dict.items():
            if hasattr(value, '__dict__') and not isinstance(value, (str, int, float, bool, list, dict, type(None))):
                try:
                    # Try serialization test
                    json.dumps(value)
                except (TypeError, ValueError):
                    # If cannot serialize, convert to string representation
                    data_dict[key] = str(value)
    else:
        data_dict = data
    
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data_dict, f, ensure_ascii=False, indent=2, default=str)
        print(f"Saved {stage} results to: {filepath}")
    except (TypeError, ValueError) as e:
        print(f"Warning: Cannot fully serialize {stage} results ({e}), saving string representation")
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(str(data))
    
    return filepath

def generate_plan(llm: LiteLLM, goal: str, output_dir: str):
    """2.1 Generate task planning"""
    wait_for_user_confirmation("task planning generation")
    print("Starting task planning generation...")
    wf = WorkFlowGenerator(llm=llm)
    plan = wf.generate_plan(goal=goal)
    
    # Save planning results
    save_intermediate_result(plan, "plan", output_dir)
    print(f"Task planning completed, containing {len(plan.sub_tasks)} sub-tasks")
    return plan

def build_workflow_from_plan(llm: LiteLLM, goal: str, plan, output_dir: str):
    """2.2 Build workflow from plan"""
    wait_for_user_confirmation("workflow graph construction")
    print("Starting workflow graph construction...")
    wf = WorkFlowGenerator(llm=llm)
    workflow = wf.build_workflow_from_plan(goal=goal, plan=plan)
    
    # Save workflow graph structure
    save_intermediate_result(workflow, "workflow_structure", output_dir)
    print(f"Workflow graph construction completed, containing {len(workflow.nodes)} nodes and {len(workflow.edges)} edges")
    return workflow

def generate_agents_for_workflow(llm: LiteLLM, goal: str, workflow: WorkFlowGraph, output_dir: str):
    """2.3 Generate agents for workflow"""
    wait_for_user_confirmation("agent generation for workflow")
    print("Starting agent generation for workflow...")
    wf = WorkFlowGenerator(llm=llm)
    workflow_with_agents = wf.generate_agents(goal=goal, workflow=workflow)
    
    # Save complete workflow with agents
    save_intermediate_result(workflow_with_agents, "workflow_with_agents", output_dir)
    print("Agent generation completed")
    return workflow_with_agents

def generate_workflow_step_by_step(llm: LiteLLM, goal: str, output_dir: str) -> WorkFlowGraph:
    """2. Generate and display workflow step by step"""
    print(f"Starting step-by-step workflow generation, goal: {goal}")
    print(f"Intermediate results will be saved to: {output_dir}")
    
    # Step 1: Generate task planning
    plan = generate_plan(llm, goal, output_dir)
    
    # Step 2: Build workflow graph
    workflow = build_workflow_from_plan(llm, goal, plan, output_dir)
    
    # Step 3: Generate agents
    workflow_with_agents = generate_agents_for_workflow(llm, goal, workflow, output_dir)
    
    # Display final workflow
    workflow_with_agents.display()
    
    # Save final complete workflow
    save_intermediate_result(workflow_with_agents, "final_workflow", output_dir)
    print("Workflow generation completed!")
    
    return workflow_with_agents

def execute_workflow(llm: LiteLLM, graph: WorkFlowGraph, goal: str, target_dir: str):
    """3. Register Agents and execute workflow"""
    wait_for_user_confirmation("workflow execution")
    print("Starting workflow execution...")
    
    # Create LiteLLMConfig for AgentManager
    cfg = llm.config
    mgr = AgentManager()
    mgr.add_agents_from_workflow(graph, llm_config=cfg)
    workflow = WorkFlow(graph=graph, agent_manager=mgr, llm=llm)
    output = workflow.execute()
    
    print("Workflow execution completed")
    return output

def verify_and_extract_code(llm: LiteLLM, goal: str, output: str, target_dir: str):
    """4. Verify code and extract to files"""
    wait_for_user_confirmation("code verification and extraction")
    print("Starting code verification and extraction...")
    
    verifier = CodeVerification()
    verified = verifier.execute(llm=llm, inputs={"requirements": goal, "code": output}).verified_code

    os.makedirs(target_dir, exist_ok=True)
    blocks = extract_code_blocks(verified)
    if len(blocks) == 1:
        path = os.path.join(target_dir, "index.html")
        with open(path, "w") as f:
            f.write(blocks[0])
        print(f"HTML file generated: {path}")
        return

    extractor = CodeExtraction()
    res = extractor.execute(llm=llm, inputs={"code_string": verified, "target_directory": target_dir})
    print(f"Extracted {len(res.extracted_files)} files:")
    for name, p in res.extracted_files.items():
        print(f" - {name}: {p}")
    if res.main_file:
        ext = os.path.splitext(res.main_file)[1].lower()
        tip = "can be opened in browser" if ext == ".html" else "main entry file"
        print(f"\nMain file: {res.main_file}, {tip}")

def main():
    goal = "Generate html code for the Tetris game that can be played in the browser."
    target_dir = "examples/output/tetris_game"
    output_dir = "examples/output/workflow_intermediates"

    # Configure LLM
    wait_for_user_confirmation("LLM configuration")
    llm = configure_llm()
    
    # Generate workflow
    graph = generate_workflow_step_by_step(llm, goal, output_dir)
    
    # Execute workflow
    output = execute_workflow(llm, graph, goal, target_dir)
    
    # Verify and extract code
    verify_and_extract_code(llm, goal, output, target_dir)
    
    print(f"\nComplete Tetris game has been generated to directory: {target_dir}")

if __name__ == "__main__":
    main()