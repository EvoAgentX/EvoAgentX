# Multi-Agent Debate Framework

## Introduction

The Multi-Agent Debate (MAD) framework is a sophisticated system for orchestrating debates between multiple AI agents to solve complex problems through collaborative reasoning. Built on top of EvoAgentX's ActionGraph architecture, it implements Google's Multi-Agent Debate methodology with enhanced features for production use.

## Architecture Overview

The MAD framework consists of several key components:

### Core Classes

#### MultiAgentDebateActionGraph
The main orchestrator class that inherits from `ActionGraph` and manages the entire debate process.

```python
class MultiAgentDebateActionGraph(ActionGraph):
    name: str = "MultiAgentDebate"
    description: str = "多智能体辩论工作流框架"
    llm_config: LLMConfig
    debater_agents: Optional[List[CustomizeAgent]]
    judge_agent: Optional[CustomizeAgent]
    group_graphs_enabled: bool = False
    group_graphs: Optional[List[ActionGraph]]
```

#### DebateAgentOutput
Structured output format for individual debater responses:

```python
class DebateAgentOutput(LLMOutputParser):
    thought: str = Field(default="", description="思考过程")
    argument: str = Field(default="", description="该轮次给出的论据/反驳")
    answer: Optional[str] = Field(default=None, description="该轮次给出的当前答案（可选）")
```

#### DebateJudgeOutput
Structured output format for judge decisions:

```python
class DebateJudgeOutput(LLMOutputParser):
    rationale: str = Field(default="", description="评审理由")
    winning_agent_id: int = Field(default=0, description="优胜辩手ID（从0开始）")
    final_answer: str = Field(default="", description="最终答案")
```

## Key Features

### 1. Flexible Agent Configuration

The framework supports multiple ways to configure debaters:

#### Default Agent Generation
```python
debate = MultiAgentDebateActionGraph(
    llm_config=OpenAILLMConfig(model="gpt-4o-mini"),
    # Uses default personas if none specified
)
```

#### Custom Agent Pool
```python
debater_agents = [
    CustomizeAgent(
        name="Optimist",
        prompt="You are an optimistic debater...",
        llm_config=optimist_config
    ),
    CustomizeAgent(
        name="Pessimist", 
        prompt="You are a pessimistic debater...",
        llm_config=pessimist_config
    )
]

debate = MultiAgentDebateActionGraph(
    debater_agents=debater_agents
)
```

#### Role-Model Mapping
```python
role_model_mapping = {
    "Innovator": ("gpt-4o", 0.3),
    "Analyst": ("llama-3.1", 0.1),
    "Optimist": ("gpt-4o-mini", 0.1),
}

debate = MultiAgentDebateActionGraph(
    role_model_mapping=role_model_mapping
)
```

### 2. Judge Modes

#### LLM Judge Mode
Uses a dedicated judge agent to evaluate the debate:

```python
judge_agent = CustomizeAgent(
    name="Judge",
    prompt="You are an impartial judge...",
    llm_config=judge_config
)

debate = MultiAgentDebateActionGraph(
    judge_agent=judge_agent,
    judge_mode="llm_judge"
)
```

#### Self-Consistency Mode
Uses majority voting among final answers:

```python
debate = MultiAgentDebateActionGraph(
    judge_mode="self_consistency"
)
```

### 3. Group Graphs Support

For complex scenarios where each debater position should be occupied by a sub-team:

```python
class GroupOfManyGraph(ActionGraph):
    name: str = "GroupOfManyGraph"
    description: str = "Group with variable number of inner debaters"
    llm_config: OpenAILLMConfig
    num_inner: int = 3

group1 = GroupOfManyGraph(llm_config=llm_config, num_inner=3)
group2 = GroupOfManyGraph(llm_config=llm_config, num_inner=4)

debate = MultiAgentDebateActionGraph(
    group_graphs_enabled=True,
    group_graphs=[group1, group2]
)
```

### 4. Transcript Management

The framework provides flexible transcript handling:

#### Previous Round Only
```python
result = debate.execute(
    problem="...",
    transcript_mode="prev"  # Only previous round visible
)
```

#### All History
```python
result = debate.execute(
    problem="...",
    transcript_mode="all"  # All debate history visible
)
```

### 5. Pruning Pipeline

Advanced pruning capabilities to reduce noise and improve efficiency:

```python
from evoagentx.frameworks.multi_agent_debate.pruning import PruningPipeline

debate = MultiAgentDebateActionGraph(
    enable_pruning=True,
    pruning_pipeline=PruningPipeline(
        similarity_threshold=0.8,
        quality_threshold=0.7
    )
)
```

## Core Methods

### execute()
The main method for running a debate:

```python
def execute(
    self,
    problem: str,
    num_agents: int = 3,
    num_rounds: int = 3,
    judge_mode: str = "llm_judge",
    personas: Optional[List[Dict[str, str]]] = None,
    transcript_mode: str = "prev",
    enable_pruning: bool = False,
    return_transcript: bool = False,
    **kwargs
) -> Dict[str, Any]:
```

**Parameters:**
- `problem`: The debate question or task
- `num_agents`: Number of participating agents (default: 3)
- `num_rounds`: Number of debate rounds (default: 3)
- `judge_mode`: "llm_judge" or "self_consistency"
- `personas`: Custom role definitions
- `transcript_mode`: "prev" or "all"
- `enable_pruning`: Enable candidate pruning
- `return_transcript`: Include full debate transcript in results

**Returns:**
```python
{
    "final_answer": str,
    "winner": int,
    "transcript": List[Dict],  # if return_transcript=True
    "round_results": List[Dict],
    "judge_rationale": str  # if judge_mode="llm_judge"
}
```

### async_execute()
Asynchronous version of execute() for concurrent processing:

```python
async def async_execute(
    self,
    problem: str,
    **kwargs
) -> Dict[str, Any]:
```

## Advanced Configuration

### Custom Personas
Define specialized debate roles:

```python
personas = [
    {
        "name": "Data Analyst",
        "style": "Evidence-based, statistical reasoning",
        "goal": "Focus on quantitative analysis and data-driven conclusions"
    },
    {
        "name": "Ethics Expert", 
        "style": "Moral reasoning, ethical frameworks",
        "goal": "Evaluate ethical implications and moral considerations"
    },
    {
        "name": "Innovation Specialist",
        "style": "Creative thinking, future-oriented",
        "goal": "Explore novel solutions and innovative approaches"
    }
]

result = debate.execute(
    problem="...",
    personas=personas
)
```

### Dynamic Model Selection
Map different roles to optimal models:

```python
role_model_mapping = {
    "Creative": ("gpt-4o", 0.7),      # High creativity
    "Analytical": ("gpt-4o", 0.1),    # Low temperature for precision
    "General": ("gpt-4o-mini", 0.3),  # Cost-effective for general roles
}

debate = MultiAgentDebateActionGraph(
    role_model_mapping=role_model_mapping
)
```

### Structured Output Templates
Use XML/JSON templates for consistent parsing:

```python
debater_prompt = """
You are debater #{agent_id} (role: {role}). This is round {round_index} of {total_rounds}.

Previous arguments: {transcript}

<response>
  <thought>Your reasoning process</thought>
  <argument>Your argument or rebuttal</argument>
  <answer>Your current answer (if applicable)</answer>
</response>
"""

debater_agent = CustomizeAgent(
    prompt=debater_prompt,
    parse_mode="xml"
)
```

## Performance Optimization

### Cost Management
1. **Model Selection**: Use cheaper models for less critical roles
2. **Transcript Mode**: Use "prev" for long debates to reduce context
3. **Pruning**: Enable pruning to reduce redundant processing
4. **Round Limits**: Balance quality vs cost with appropriate round counts

### Quality Enhancement
1. **Role Diversity**: Ensure complementary perspectives
2. **Temperature Tuning**: Higher for creativity, lower for analysis
3. **Structured Outputs**: Use XML/JSON for reliable parsing
4. **Judge Quality**: Use strong models for critical judge decisions

## Error Handling

The framework includes comprehensive error handling:

```python
try:
    result = debate.execute(problem="...")
except DebateError as e:
    print(f"Debate failed: {e}")
except AgentError as e:
    print(f"Agent error: {e}")
except LLMError as e:
    print(f"LLM error: {e}")
```

## Integration with EvoAgentX

### Memory Integration
```python
debate = MultiAgentDebateActionGraph(
    enable_memory=True,
    memory_config=MemoryConfig(
        short_term=True,
        long_term=False
    )
)
```

### Tool Integration
```python
debater_agent = CustomizeAgent(
    name="Research Debater",
    tools=[WebSearchTool(), CalculatorTool()],
    prompt="Use available tools to research and support your arguments..."
)
```

### Workflow Integration
```python
# MAD as part of larger workflow
workflow = ActionGraph()
workflow.add_node("research", research_action)
workflow.add_node("debate", debate_graph)
workflow.add_node("synthesis", synthesis_action)

workflow.add_edge("research", "debate")
workflow.add_edge("debate", "synthesis")
```

## Best Practices

### 1. Problem Design
- Make problems specific and well-defined
- Include clear success criteria
- Avoid overly subjective topics for self-consistency mode

### 2. Agent Configuration
- Balance diversity with coherence
- Match model capabilities to role requirements
- Use appropriate temperature settings

### 3. Round Management
- Start with 2-3 rounds for simple problems
- Increase rounds for complex, multi-faceted issues
- Monitor convergence patterns

### 4. Quality Assurance
- Enable transcript logging for analysis
- Use structured outputs for reliable parsing
- Implement validation for critical decisions

## Troubleshooting

### Common Issues

**Low Quality Debates:**
- Increase agent diversity
- Adjust temperature settings
- Improve prompt quality

**High Costs:**
- Use cheaper models for non-critical roles
- Enable pruning
- Reduce transcript visibility

**Inconsistent Results:**
- Use structured output formats
- Increase round count
- Improve judge prompts

**Performance Issues:**
- Use async execution
- Enable pruning
- Optimize transcript management

## API Reference

### MultiAgentDebateActionGraph

#### Constructor Parameters
- `name`: Graph identifier
- `description`: Human-readable description
- `llm_config`: Default LLM configuration
- `debater_agents`: Custom debater agent pool
- `judge_agent`: Custom judge agent
- `group_graphs_enabled`: Enable group graph mode
- `group_graphs`: List of group graphs
- `role_model_mapping`: Role-to-model mapping
- `enable_pruning`: Enable pruning pipeline
- `pruning_pipeline`: Custom pruning configuration

#### Methods
- `execute()`: Run synchronous debate
- `async_execute()`: Run asynchronous debate
- `get_config()`: Export configuration
- `save_module()`: Save to file
- `load_module()`: Load from file
- `from_dict()`: Create from configuration dict

### Utility Functions

#### build_agent_prompt()
Constructs debater prompts with context and formatting.

#### build_judge_prompt()
Constructs judge prompts with debate transcript.

#### format_transcript()
Formats debate history for agent consumption.

#### collect_last_round_candidates()
Extracts final answers from last round.

#### collect_round_candidates()
Extracts answers from specific round.

## Examples

See the `examples/multi_agent_debate/` directory for comprehensive usage examples:

- `multi_agent_debate.py`: Basic usage patterns
- `multi_agent_debate_advanced.py`: Advanced configurations
- `multi_agent_debate_group.py`: Group graph implementations
- `config_methods_example.py`: Configuration management
