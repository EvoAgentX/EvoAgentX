
# Multi-Agent Debate Tutorial

This tutorial provides a practical guide to using EvoAgentX's Multi-Agent Debate (MAD) framework. You'll learn how to set up debates, configure agents, and solve real-world problems through collaborative AI reasoning.

## Quick Start

### Basic Debate Example

Here's the simplest way to run a multi-agent debate:

```python
from evoagentx.frameworks.multi_agent_debate.debate import MultiAgentDebateActionGraph
from evoagentx.models import OpenAILLMConfig

# Create a debate system
debate = MultiAgentDebateActionGraph(
    name="Simple Debate",
    description="Basic multi-agent debate example",
    llm_config=OpenAILLMConfig(model="gpt-4o-mini", temperature=0.5),
)

# Run the debate
result = debate.execute(
    problem="Should companies prioritize AI automation over human workers?",
    num_agents=3,
    num_rounds=3,
    judge_mode="llm_judge",
    return_transcript=True,
)

print(f"Final Answer: {result['final_answer']}")
print(f"Winning Agent: Agent #{result['winner']}")
```

### Self-Consistency Mode

For objective problems, you can use self-consistency mode where the final answer is determined by majority vote:

```python
result = debate.execute(
    problem="What is the optimal number of layers for a neural network with 10,000 parameters?",
    num_agents=5,
    num_rounds=3,
    judge_mode="self_consistency",  # No judge needed
    return_transcript=True,
)

print(f"Consensus Answer: {result['final_answer']}")
```

## Understanding the Parameters

### Core Parameters

- **`problem`**: The question or task for debate
- **`num_agents`**: Number of participating agents (recommended: 3-7)
- **`num_rounds`**: Number of debate rounds (recommended: 2-6)
- **`judge_mode`**: 
  - `"llm_judge"`: Uses an AI judge to evaluate and decide
  - `"self_consistency"`: Uses majority voting among final answers
- **`return_transcript`**: Include full debate history in results

### Advanced Parameters

- **`personas`**: Custom role definitions for agents
- **`transcript_mode`**: 
  - `"prev"`: Agents only see previous round (cost-effective)
  - `"all"`: Agents see full debate history (more context)
- **`enable_pruning`**: Remove low-quality candidates to reduce noise

## Customizing Agent Roles

### Using Default Personas

The framework automatically generates diverse personas if none are specified:

```python
# Uses default personas automatically
result = debate.execute(
    problem="Evaluate the pros and cons of remote work",
    num_agents=4,
    num_rounds=3,
)
```

### Creating Custom Personas

Define specific roles for your debate:

```python
custom_personas = [
    {
        "name": "Productivity Expert",
        "style": "Data-driven analysis of work efficiency",
        "goal": "Focus on measurable productivity outcomes"
    },
    {
        "name": "Employee Advocate", 
        "style": "Human-centered perspective on work-life balance",
        "goal": "Prioritize employee wellbeing and satisfaction"
    },
    {
        "name": "Business Strategist",
        "style": "Cost-benefit analysis and strategic planning",
        "goal": "Evaluate business impact and competitive advantages"
    },
    {
        "name": "Technology Specialist",
        "style": "Technical feasibility and implementation focus",
        "goal": "Assess technological requirements and challenges"
    }
]

result = debate.execute(
    problem="Should our company adopt a fully remote work policy?",
    personas=custom_personas,
    num_agents=4,
    num_rounds=4,
)
```

## Advanced Agent Configuration

### Custom Agent Creation

For more control, create custom agents with specific prompts and configurations:

```python
from evoagentx.agents.customize_agent import CustomizeAgent

# Create specialized debaters
optimist_agent = CustomizeAgent(
    name="Optimist",
    prompt="""You are an optimistic debater who focuses on positive outcomes and opportunities. 
    Always highlight potential benefits and constructive solutions.""",
    llm_config=OpenAILLMConfig(model="gpt-4o-mini", temperature=0.7)
)

pessimist_agent = CustomizeAgent(
    name="Pessimist", 
    prompt="""You are a cautious debater who identifies risks and potential problems.
    Focus on potential downsides and implementation challenges.""",
    llm_config=OpenAILLMConfig(model="gpt-4o-mini", temperature=0.3)
)

analyst_agent = CustomizeAgent(
    name="Data Analyst",
    prompt="""You are a data-driven analyst who evaluates evidence objectively.
    Focus on facts, statistics, and measurable outcomes.""",
    llm_config=OpenAILLMConfig(model="gpt-4o", temperature=0.1)
)

# Use custom agents
debate = MultiAgentDebateActionGraph(
    debater_agents=[optimist_agent, pessimist_agent, analyst_agent],
    llm_config=OpenAILLMConfig(model="gpt-4o-mini")
)

result = debate.execute(
    problem="Should we invest in renewable energy infrastructure?",
    num_agents=3,
    num_rounds=4,
)
```

### Role-Model Mapping

Match different roles to optimal models for better performance:

```python
from evoagentx.models import OpenAILLMConfig, OpenRouterConfig

# Define different model configurations
models = {
    "creative": OpenAILLMConfig(model="gpt-4o", temperature=0.7),
    "analytical": OpenAILLMConfig(model="gpt-4o", temperature=0.1), 
    "general": OpenAILLMConfig(model="gpt-4o-mini", temperature=0.3),
}

# Map roles to models
role_model_mapping = {
    "Innovator": ("creative", 0.0),      # Use creative model
    "Analyst": ("analytical", 0.0),     # Use analytical model  
    "Generalist": ("general", 0.0),     # Use general model
}

debate = MultiAgentDebateActionGraph(
    role_model_mapping=role_model_mapping,
    models=models
)
```

## Group Graphs for Complex Debates

When each debater position should be occupied by a sub-team, use group graphs:

```python
from evoagentx.workflow.action_graph import ActionGraph

class TeamGraph(ActionGraph):
    name: str = "TeamGraph"
    description: str = "A team of specialized agents"
    llm_config: OpenAILLMConfig
    team_size: int = 3
    
    def __init__(self, team_size: int = 3, **kwargs):
        super().__init__(**kwargs)
        self.team_size = team_size
        # Add specialized team members
        self.add_team_members()

# Create team graphs
team1 = TeamGraph(team_size=3, llm_config=llm_config)
team2 = TeamGraph(team_size=4, llm_config=llm_config)

# Enable group mode
debate = MultiAgentDebateActionGraph(
    group_graphs_enabled=True,
    group_graphs=[team1, team2],
    llm_config=llm_config
)

result = debate.execute(
    problem="Design a comprehensive AI ethics framework",
    num_agents=2,  # Two teams
    num_rounds=3,
)
```

## Cost Optimization Strategies

### Transcript Management

Control costs by managing how much context agents see:

```python
# Cost-effective: Only previous round visible
result = debate.execute(
    problem="Complex multi-faceted problem...",
    transcript_mode="prev",  # Lower cost
    num_rounds=5,
)

# Full context: All history visible  
result = debate.execute(
    problem="Complex multi-faceted problem...",
    transcript_mode="all",  # Higher cost, more context
    num_rounds=3,  # Fewer rounds to balance cost
)
```

### Pruning for Efficiency

Remove low-quality candidates to reduce noise and improve efficiency:

```python
result = debate.execute(
    problem="Evaluate multiple solution approaches",
    num_agents=7,  # Large number of agents
    num_rounds=4,
    enable_pruning=True,  # Remove poor candidates
    transcript_mode="prev",  # Combine with cost control
)
```

### Model Selection Strategy

Use different models for different roles to balance cost and quality:

```python
# Expensive model for critical roles
judge_agent = CustomizeAgent(
    name="Judge",
    prompt="You are an impartial judge...",
    llm_config=OpenAILLMConfig(model="gpt-4o", temperature=0.1)
)

# Cheaper models for general debaters
debater_agents = [
    CustomizeAgent(
        name="General Debater",
        llm_config=OpenAILLMConfig(model="gpt-4o-mini", temperature=0.3)
    )
]

debate = MultiAgentDebateActionGraph(
    debater_agents=debater_agents,
    judge_agent=judge_agent
)
```

## Configuration Management

### Saving and Loading Configurations

Save your debate setups for reuse:

```python
# Save configuration
debate.save_module("my_debate_config.json")

# Load configuration
loaded_debate = MultiAgentDebateActionGraph.load_module("my_debate_config.json")

# Create from dictionary
config_dict = debate.get_config()
new_debate = MultiAgentDebateActionGraph.from_dict(config_dict)
```

### Environment Setup

Ensure you have the required API keys:

```bash
export OPENAI_API_KEY="your_openai_key"
export OPENROUTER_API_KEY="your_openrouter_key"  # Optional
```

## Real-World Examples

### Business Decision Making

```python
business_personas = [
    {
        "name": "Financial Analyst",
        "style": "ROI and cost-benefit analysis",
        "goal": "Focus on financial implications and returns"
    },
    {
        "name": "Operations Manager", 
        "style": "Implementation feasibility and logistics",
        "goal": "Evaluate operational requirements and challenges"
    },
    {
        "name": "Customer Advocate",
        "style": "Customer impact and satisfaction",
        "goal": "Prioritize customer experience and needs"
    }
]

result = debate.execute(
    problem="Should we implement a new customer service chatbot system?",
    personas=business_personas,
    num_agents=3,
    num_rounds=4,
    judge_mode="llm_judge",
)
```

### Technical Architecture Decisions

```python
tech_personas = [
    {
        "name": "Scalability Expert",
        "style": "Performance and scalability focus",
        "goal": "Ensure system can handle growth"
    },
    {
        "name": "Security Specialist",
        "style": "Security and compliance focus", 
        "goal": "Identify and address security concerns"
    },
    {
        "name": "Developer Experience",
        "style": "Developer productivity and maintainability",
        "goal": "Focus on ease of development and maintenance"
    }
]

result = debate.execute(
    problem="Choose between microservices vs monolithic architecture for our new platform",
    personas=tech_personas,
    num_agents=3,
    num_rounds=3,
    judge_mode="self_consistency",
)
```

## Best Practices

### Problem Design
- Make problems specific and actionable
- Include clear success criteria
- Avoid overly subjective topics for self-consistency mode

### Agent Configuration  
- Balance diversity with coherence
- Match model capabilities to role requirements
- Use appropriate temperature settings (0.1-0.3 for analysis, 0.6-0.9 for creativity)

### Performance Tuning
- Start with 2-3 rounds for simple problems
- Increase rounds for complex, multi-faceted issues
- Monitor convergence patterns
- Use pruning for large agent counts

### Quality Assurance
- Enable transcript logging for analysis
- Use structured outputs for reliable parsing
- Implement validation for critical decisions

## Troubleshooting

### Common Issues

**Low Quality Debates:**
- Increase agent diversity
- Adjust temperature settings
- Improve prompt quality
- Add more rounds

**High Costs:**
- Use cheaper models for non-critical roles
- Enable pruning
- Reduce transcript visibility
- Limit number of rounds

**Inconsistent Results:**
- Use structured output formats
- Increase round count
- Improve judge prompts
- Enable self-consistency mode for objective problems

## Next Steps

1. **Run Basic Examples**: Start with the examples in `examples/multi_agent_debate/`
2. **Customize Agents**: Create specialized debaters for your domain
3. **Experiment with Configurations**: Try different model combinations and parameters
4. **Integrate with Workflows**: Use MAD as part of larger EvoAgentX workflows
5. **Scale Up**: Explore group graphs for complex multi-team scenarios

## Reference Examples

- **Basic Usage**: `examples/multi_agent_debate/multi_agent_debate.py`
- **Advanced Configurations**: `examples/multi_agent_debate/multi_agent_debate_advanced.py`  
- **Group Graphs**: `examples/multi_agent_debate/multi_agent_debate_group.py`
- **Configuration Management**: `examples/multi_agent_debate/config_methods_example.py`
