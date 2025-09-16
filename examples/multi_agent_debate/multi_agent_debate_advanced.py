#!/usr/bin/env python3
"""
MultiAgentDebate Advanced Example - Dynamic Role-Model Mapping

This example focuses on:
1. Dynamically selecting the most suitable model based on role characteristics
2. Optimizing debate strategies through role design
3. Demonstrating how intelligent matching improves debate quality

Differences from basic example:
- Basic example: Uses default configuration, demonstrates basic functionality
- Advanced example: Custom role-model mapping, optimizes debate effectiveness

Differences from group example:
- Group example: Demonstrates complex architectural design
- Advanced example: Demonstrates intelligent matching of roles and models
"""

import os
import random
from dotenv import load_dotenv

from evoagentx.frameworks.multi_agent_debate.debate import MultiAgentDebateActionGraph
from evoagentx.models.model_configs import OpenAILLMConfig, OpenRouterConfig
from evoagentx.agents.customize_agent import CustomizeAgent


def create_optimized_agent(role_name, role_description, model_config, temperature_adjustment=0.0):
    """Create optimized agent, adjust model parameters based on role characteristics"""
    
    # Adjust prompt based on role characteristics
    role_prompt = f"""
You are debater #{{agent_id}} (role: {{role}}). This is round {{round_index}} of {{total_rounds}}.

Problem:
{{problem}}

Conversation so far:
{{transcript_text}}

Instructions:
- You are a {role_name.upper()} who {role_description}
- Think briefly (<= 120 words), then present your {role_name.lower()} argument or rebuttal
- Focus on your unique perspective and expertise
- If confident, provide your current answer for this round
- Your output MUST follow this XML template:

<response>
  <thought>Your brief {role_name.lower()} reasoning</thought>
  <argument>Your {role_name.lower()} argument or rebuttal</argument>
  <answer>Optional current answer; leave empty if uncertain</answer>
</response>
"""
    
    # Adjust model configuration
    adjusted_config = model_config.model_copy()
    if hasattr(adjusted_config, 'temperature'):
        adjusted_config.temperature = max(0.0, min(1.0, adjusted_config.temperature + temperature_adjustment))
    
    inputs = [
        {"name": "problem", "type": "str", "description": "Problem statement"},
        {"name": "transcript_text", "type": "str", "description": "Formatted debate transcript so far"},
        {"name": "role", "type": "str", "description": "Debater role/persona"},
        {"name": "agent_id", "type": "str", "description": "Debater id (string)"},
        {"name": "round_index", "type": "str", "description": "1-based round index"},
        {"name": "total_rounds", "type": "str", "description": "Total rounds"},
    ]
    
    outputs = [
        {"name": "thought", "type": "str", "description": "Brief reasoning", "required": True},
        {"name": "argument", "type": "str", "description": "Argument or rebuttal", "required": True},
        {"name": "answer", "type": "str", "description": "Optional current answer", "required": False},
    ]
    
    return CustomizeAgent(
        name=role_name,
        description=f"{role_name} debater: {role_description}",
        prompt=role_prompt,
        llm_config=adjusted_config,
        inputs=inputs,
        outputs=outputs,
        parse_mode="xml",
    )


def create_role_model_mapping():
    """Create role-model mapping strategy"""
    load_dotenv()
    openai_key = os.getenv("OPENAI_API_KEY")
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    
    # Define roles and their characteristics
    roles = {
        "Optimist": "always sees the bright side and positive opportunities",
        "Pessimist": "focuses on risks, problems, and potential downsides", 
        "Analyst": "provides data-driven, balanced analysis",
        "Innovator": "thinks outside the box and suggests creative solutions",
        "Conservative": "values tradition, stability, and proven approaches",
        "Skeptic": "questions assumptions and demands evidence",
        "Advocate": "passionately defends a particular viewpoint",
        "Mediator": "seeks common ground and compromise",
        "Expert": "provides specialized knowledge and technical insights",
        "Critic": "identifies flaws and suggests improvements"
    }
    
    # Define model configurations
    models = {
        "gpt4o_mini": OpenAILLMConfig(model="gpt-4o-mini", openai_key=openai_key, temperature=0.3),
        "gpt4o": OpenAILLMConfig(model="gpt-4o", openai_key=openai_key, temperature=0.2),
        "llama": OpenRouterConfig(model="meta-llama/llama-3.1-70b-instruct", openrouter_key=openrouter_key, temperature=0.3),
    }
    
    # Role-model mapping strategy (select most suitable model based on role characteristics)
    role_model_mapping = {
        # Roles requiring creativity use GPT-4o
        "Innovator": ("gpt4o", 0.3),      # High temperature increases creativity
        "Advocate": ("gpt4o", 0.2),       # Medium temperature increases persuasiveness
        
        # Roles requiring precise analysis use Llama
        "Analyst": ("llama", -0.1),       # Low temperature increases precision
        "Expert": ("llama", 0.0),         # Standard temperature maintains professionalism
        "Skeptic": ("llama", 0.0),       # Standard temperature maintains criticality
        
        # Other roles use GPT-4o-mini (balances performance and cost)
        "Optimist": ("gpt4o_mini", 0.1),  # Slightly increase temperature
        "Pessimist": ("gpt4o_mini", 0.0), # Standard temperature
        "Conservative": ("gpt4o_mini", -0.1), # Lower temperature increases stability
        "Critic": ("gpt4o_mini", 0.0),    # Standard temperature
        "Mediator": ("gpt4o_mini", 0.1),  # Slightly increase temperature for flexibility
    }
    
    return roles, models, role_model_mapping


def run_optimized_debate():
    """Run optimized debate: select most suitable model based on role characteristics"""
    print("=== Optimized Debate: Intelligent Role-Model Matching ===")
    
    roles, models, mapping = create_role_model_mapping()
    
    # Select representative role combinations
    selected_roles = ["Analyst", "Innovator", "Skeptic", "Advocate", "Mediator"]
    
    # Create optimized agents for each role
    agents = []
    for role in selected_roles:
        model_name, temp_adjust = mapping[role]
        model_config = models[model_name]
        agent = create_optimized_agent(role, roles[role], model_config, temp_adjust)
        agents.append(agent)
    
    # Create debate graph
    graph = MultiAgentDebateActionGraph(
        debater_agents=agents,
        llm_config=agents[0].llm_config if agents else None,
    )
    
    # Execute debate
    result = graph.execute(
        problem="Should we invest heavily in AI research? Give a final Yes/No with reasons.",
        num_agents=5,
        num_rounds=3,
        judge_mode="llm_judge",
        return_transcript=True,
    )
    
    print("Final Answer:", result.get("final_answer"))
    print("Winner:", result.get("winner"))
    if result.get("winner_answer"):
        print("Winner Answer:", result.get("winner_answer"))
    
    # Display role-model matching information
    print("\nRole-Model Matching Strategy:")
    for i, agent in enumerate(agents):
        model_name = agent.llm_config.model if hasattr(agent.llm_config, 'model') else "Unknown"
        temp = agent.llm_config.temperature if hasattr(agent.llm_config, 'temperature') else "Unknown"
        print(f"  {agent.name}: {model_name} (Temperature: {temp}) - {roles[agent.name]}")





def main():
    """Main function"""
    print("MultiAgentDebate Advanced Example - Dynamic Role-Model Mapping")
    print("=" * 60)
    
    # Check environment variables
    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY environment variable not set")
    if not os.getenv("OPENROUTER_API_KEY"):
        print("Warning: OPENROUTER_API_KEY environment variable not set")
    
    # Run optimized debate
    run_optimized_debate()


if __name__ == "__main__":
    main()
