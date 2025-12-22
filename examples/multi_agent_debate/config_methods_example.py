#!/usr/bin/env python3
"""
MultiAgentDebateActionGraph Configuration Methods Example
Demonstrates how to use save_module, load_module, from_dict, get_config methods
"""

import os
import tempfile
from dotenv import load_dotenv
from evoagentx.frameworks.multi_agent_debate.debate import MultiAgentDebateActionGraph
from evoagentx.models.model_configs import OpenAILLMConfig
from evoagentx.agents.customize_agent import CustomizeAgent

# Load environment variables
load_dotenv()

def create_sample_agents():
    """Create sample agents"""
    agents = []
    
    # Create first agent
    agent1 = CustomizeAgent(
        name="OptimistAgent",
        description="Optimistic debater who always sees the positive side of problems",
        prompt="You are an optimistic debater. Please analyze the problem from a positive perspective: {problem}",
        llm_config=OpenAILLMConfig(
            model="gpt-4o-mini",
            openai_key=os.getenv("OPENAI_API_KEY")
        ),
        inputs=[{"name": "problem", "type": "str", "description": "Problem"}],
        outputs=[{"name": "argument", "type": "str", "description": "Argument"}],
        parse_mode="title"
    )
    agents.append(agent1)
    
    # Create second agent
    agent2 = CustomizeAgent(
        name="PessimistAgent", 
        description="Pessimistic debater who always sees the negative side of problems",
        prompt="You are a pessimistic debater. Please analyze the problem from a negative perspective: {problem}",
        llm_config=OpenAILLMConfig(
            model="gpt-4o-mini",
            openai_key=os.getenv("OPENAI_API_KEY")
        ),
        inputs=[{"name": "problem", "type": "str", "description": "Problem"}],
        outputs=[{"name": "argument", "type": "str", "description": "Argument"}],
        parse_mode="title"
    )
    agents.append(agent2)
    
    return agents

def demo_save_and_load():
    """Demonstrate save and load functionality"""
    print("=== Demonstrate Save and Load Functionality ===")
    
    # 1. Create debate graph
    agents = create_sample_agents()
    graph = MultiAgentDebateActionGraph(
        name="Demo Debate",
        description="Demo debate graph",
        debater_agents=agents,
        llm_config=agents[0].llm_config if agents else None,
    )
    
    # 2. Get configuration
    print("\n1. Get current configuration...")
    config = graph.get_config()
    print(f"Configuration contains {len(config)} fields")
    print(f"Number of agents: {len(config.get('debater_agents', []))}")
    
    # 3. Save configuration - try temp file first, fallback to files directory
    print("\n2. Save configuration to file...")
    try:
        # Try to save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
            temp_path = temp_file.name
        save_path = graph.save_module(temp_path)
        print(f"Configuration saved to temporary file: {save_path}")
    except Exception as e:
        print(f"Failed to save to temp file: {e}")
        # Fallback: create files directory and save there
        files_dir = "examples/multi_agent_debate/files"
        os.makedirs(files_dir, exist_ok=True)
        save_path = graph.save_module(os.path.join(files_dir, "demo_debate_config.json"))
        print(f"Configuration saved to files directory: {save_path}")
    
    # 4. Create new instance from dictionary
    print("\n3. Create new instance from configuration dictionary...")
    new_graph_from_dict = MultiAgentDebateActionGraph.from_dict(config)
    print(f"New instance name: {new_graph_from_dict.name}")
    print(f"New instance agent count: {len(new_graph_from_dict.debater_agents or [])}")
    
    # 5. Load new instance from file
    print("\n4. Load new instance from file...")
    new_graph_from_file = MultiAgentDebateActionGraph.load_module(save_path)
    print(f"Loaded instance name: {new_graph_from_file.name}")
    print(f"Loaded instance agent count: {len(new_graph_from_file.debater_agents or [])}")
    
    # 6. Load to existing instance
    print("\n5. Load configuration to existing instance...")
    empty_graph = MultiAgentDebateActionGraph()
    empty_graph.load_module(save_path)
    print(f"Loaded instance name: {empty_graph.name}")
    print(f"Loaded agent count: {len(empty_graph.debater_agents or [])}")
    
    return save_path

def demo_error_handling():
    """Demonstrate error handling"""
    print("\n=== Demonstrate Error Handling ===")
    
    # 1. Try to load non-existent file
    print("\n1. Try to load non-existent file...")
    try:
        MultiAgentDebateActionGraph.load_module("nonexistent_file.json")
    except FileNotFoundError as e:
        print(f"Expected error: {e}")
    
    # 2. Try to create instance from invalid dictionary
    print("\n2. Try to create instance from invalid dictionary...")
    try:
        invalid_config = {"invalid_field": "invalid_value"}
        MultiAgentDebateActionGraph.from_dict(invalid_config)
        print("Successfully created instance (using default values)")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    print("MultiAgentDebateActionGraph Configuration Methods Demo")
    print("=" * 50)
    
    # Check environment variables
    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY environment variable not set")
        print("Please set environment variables before running this example")
    else:
        save_path = demo_save_and_load()
        demo_error_handling()
        
        print("\n=== Demo Complete ===")
        print("Generated files:")
        print(f"- {save_path} (main configuration file)")
        print(f"- {save_path.replace('.json', '_agents.json')} (agent pool file)")
