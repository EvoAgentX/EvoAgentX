#!/usr/bin/env python3
"""
Workflow Demo with Tools - Demonstrating EvoAgentX workflow capabilities with CMDToolkit
This example shows how to use tools in workflows for file system operations.
"""

import os
import sys
from dotenv import load_dotenv

# Add the EvoAgentX project to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'EvoAgentX-clean_tools'))

from evoagentx.models import OpenAILLMConfig, OpenAILLM
from evoagentx.workflow import WorkFlowGenerator, WorkFlowGraph, WorkFlow
from evoagentx.agents import AgentManager
from evoagentx.tools import CMDToolkit

def load_api_key():
    """Load OpenAI API key from various sources"""
    load_dotenv()
    
    # Try to get from environment
    api_key = os.getenv("OPENAI_API_KEY")
    
    # Try to get from local file
    if not api_key and os.path.exists("openai_api_key.txt"):
        with open("openai_api_key.txt", "r") as f:
            content = f.read().strip()
            # Extract the key if it's in the format "OPENAI_API_KEY=..."
            if "=" in content:
                api_key = content.split("=", 1)[1].strip()
            else:
                api_key = content
    
    if not api_key:
        raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY environment variable or create openai_api_key.txt file")
    
    return api_key

def demo_basic_workflow():
    """Demonstrate basic workflow without tools"""
    print("Basic Workflow Demo - Creating a Python Calculator Application")
    print("=" * 70)
    
    # Setup LLM configuration
    api_key = load_api_key()
    openai_config = OpenAILLMConfig(
        model="gpt-4o-mini",
        openai_key=api_key,
        stream=True,
        output_response=True,
        max_tokens=8000
    )
    llm = OpenAILLM(config=openai_config)
    
    # Define the goal
    goal = "Create a simple Python calculator application"
    print(f"Goal: {goal}")
    
    # Generate workflow
    wf_generator = WorkFlowGenerator(llm=llm)
    workflow_graph: WorkFlowGraph = wf_generator.generate_workflow(goal=goal)
    
    # Display workflow structure
    print("\nGenerated Workflow Structure:")
    workflow_graph.display()
    
    # Create agent manager and add agents
    agent_manager = AgentManager()
    agent_manager.add_agents_from_workflow(workflow_graph, llm_config=openai_config)
    
    # Create and execute workflow
    workflow = WorkFlow(graph=workflow_graph, agent_manager=agent_manager, llm=llm)
    print("\nExecuting workflow...")
    output = workflow.execute()
    
    print("Basic workflow completed successfully")
    print(f"\nOutput (first 500 chars):\n{str(output)[:500]}...")
    
    return output

def demo_toolkit_workflow():
    """Demonstrate workflow with CMDToolkit for file system operations"""
    print("\nToolkit Workflow Demo - Creating Project Structure with CMDToolkit")
    print("=" * 70)
    
    # Setup LLM configuration
    api_key = load_api_key()
    openai_config = OpenAILLMConfig(
        model="gpt-4o-mini",
        openai_key=api_key,
        stream=True,
        output_response=True,
        max_tokens=8000
    )
    llm = OpenAILLM(config=openai_config)
    
    # Define the goal and tools
    goal = "Create a folder structure for a Python project and show the file tree"
    tools = [CMDToolkit()]  # Initialize the CMDToolkit tool
    print(f"Goal: {goal}")
    print(f"Tools: {[tool.__class__.__name__ for tool in tools]}")
    
    # Generate workflow with tools
    wf_generator = WorkFlowGenerator(llm=llm, tools=tools)
    workflow_graph: WorkFlowGraph = wf_generator.generate_workflow(goal=goal)
    
    # Display workflow structure
    print("\nGenerated Workflow Structure with Tools:")
    workflow_graph.display()
    
    # Create agent manager with tools and add agents
    agent_manager = AgentManager(tools=tools)
    agent_manager.add_agents_from_workflow(workflow_graph, llm_config=openai_config)
    
    # Create and execute workflow
    workflow = WorkFlow(graph=workflow_graph, agent_manager=agent_manager, llm=llm)
    print("\nExecuting workflow with CMDToolkit...")
    output = workflow.execute()
    
    print("Toolkit workflow completed successfully")
    print(f"\nOutput (first 800 chars):\n{str(output)[:800]}...")
    
    return output

def demo_workflow_save_load():
    """Demonstrate workflow save and load functionality"""
    print("\nWorkflow Save/Load Demo")
    print("=" * 70)
    
    # Setup LLM configuration
    api_key = load_api_key()
    openai_config = OpenAILLMConfig(
        model="gpt-4o-mini",
        openai_key=api_key,
        stream=True,
        output_response=True,
        max_tokens=8000
    )
    llm = OpenAILLM(config=openai_config)
    
    # Generate a simple workflow
    goal = "Create a simple Python calculator application"
    wf_generator = WorkFlowGenerator(llm=llm)
    workflow_graph: WorkFlowGraph = wf_generator.generate_workflow(goal=goal)
    
    # Save workflow
    save_path = "demo_workflow.json"
    workflow_graph.save_module(save_path)
    print(f"Workflow saved to: {save_path}")
    
    # Load workflow
    loaded_graph = WorkFlowGraph.from_file(save_path)
    print("Workflow loaded successfully")
    
    # Verify loaded workflow
    agent_manager = AgentManager()
    agent_manager.add_agents_from_workflow(loaded_graph, llm_config=openai_config)
    workflow = WorkFlow(graph=loaded_graph, agent_manager=agent_manager, llm=llm)
    
    print("Loaded workflow is executable")
    
    # Cleanup
    if os.path.exists(save_path):
        os.remove(save_path)
        print(f"Cleaned up temporary file: {save_path}")
    
    return loaded_graph

def main():
    """Main demonstration function"""
    print("EvoAgentX Workflow Demo with Tools")
    print("=" * 70)
    print("This demo showcases different workflow capabilities:")
    print("1. Basic workflow without tools")
    print("2. Workflow with CMDToolkit for file operations")
    print("3. Workflow save/load functionality")
    print("=" * 70)
    
    try:
        # Demo 1: Basic workflow
        print("\n" + "=" * 70)
        print("DEMO 1: Basic Workflow")
        print("=" * 70)
        basic_output = demo_basic_workflow()
        
        # Demo 2: Toolkit workflow
        print("\n" + "=" * 70)
        print("DEMO 2: Toolkit Workflow")
        print("=" * 70)
        toolkit_output = demo_toolkit_workflow()
        
        # Demo 3: Save/Load workflow
        print("\n" + "=" * 70)
        print("DEMO 3: Workflow Save/Load")
        print("=" * 70)
        loaded_workflow = demo_workflow_save_load()
        
        # Summary
        print("\n" + "=" * 70)
        print("DEMO SUMMARY")
        print("=" * 70)
        print("Basic Workflow Demo: PASSED")
        print("Toolkit Workflow Demo: PASSED")
        print("Workflow Save/Load Demo: PASSED")
        print("\nAll demos completed successfully!")
        print("\nKey Features Demonstrated:")
        print("- Workflow generation from natural language goals")
        print("- Tool integration (CMDToolkit for file operations)")
        print("- Workflow visualization and management")
        print("- Agent creation and management")
        print("- Workflow persistence (save/load)")
        
        return 0
        
    except Exception as e:
        print(f"\nDemo failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main()) 