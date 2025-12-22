# Build Your First Workflow

In EvoAgentX, workflows allow multiple agents to collaborate sequentially on complex tasks. This tutorial will guide you through creating and using workflows:

1. **Understanding Sequential Workflows**: Learn how workflows connect multiple tasks together
2. **Building a Sequential Workflow**: Create a workflow with planning and coding steps
3. **Executing and Managing Workflows**: Run workflows with specific inputs

By the end of this tutorial, you'll be able to create sequential workflows that coordinate multiple agents to solve complex problems.

## 1. Understanding Sequential Workflows

A workflow in EvoAgentX represents a sequence of tasks that can be executed by different agents. The simplest workflow is a sequential workflow, where tasks are executed one after another with outputs from previous tasks feeding into subsequent ones.

Let's start by importing the necessary components:

```python
import os 
from dotenv import load_dotenv
from evoagentx.workflow import SequentialWorkFlowGraph, WorkFlow
from evoagentx.agents import AgentManager
from evoagentx.models import OpenAILLMConfig, OpenAILLM

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
```

## 2. Building a Sequential Workflow

A sequential workflow consists of a series of tasks where each task has:

- A name and description
- Input and output definitions
- A prompt template
- Parsing mode and function (optional) 

Here's how to build a sequential workflow with planning and coding tasks:

```python
# Configure the LLM 
llm_config = OpenAILLMConfig(model="gpt-4o-mini", openai_key=OPENAI_API_KEY, stream=True, output_response=True)
llm = OpenAILLM(llm_config)

# Define a custom parsing function (if needed)
from evoagentx.core.registry import register_parse_function
from evoagentx.core.module_utils import extract_code_blocks

# [optional] Define a custom parsing function (if needed)
# It is suggested to use the `@register_parse_function` decorator to register a custom parsing function, so the workflow can be saved and loaded correctly.  

@register_parse_function
def custom_parse_func(content: str) -> str:
    return {"code": extract_code_blocks(content)[0]}

# Define sequential tasks
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
        "parse_mode": "str",
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

# Create the sequential workflow graph
graph = SequentialWorkFlowGraph(
    goal="Generate code to solve programming problems",
    tasks=tasks
)
```

!!! note 
    When you create a `SequentialWorkFlowGraph` with a list of tasks, the framework will create a `CustomizeAgent` for each task. Each task in the workflow becomes a specialized agent configured with the specific prompt, input/output formats, and parsing mode you defined. These agents are connected in sequence, with outputs from one agent becoming inputs to the next. 

    The `parse_mode` controls how the output from an LLM is parsed into a structured format. Available options are: [`'str'` (default), `'json'`, `'title'`, `'xml'`, `'custom'`]. For detailed information about parsing modes and examples, please refer to the [CustomizeAgent documentation](../modules/customize_agent.md#parsing-modes).

## 3. Executing and Managing Workflows

Once you've created a workflow graph, you can create an instance of the workflow and execute it:

```python
# Create agent manager and add agents from the workflow. It will create a `CustomizeAgent` for each task in the workflow. 
agent_manager = AgentManager()
agent_manager.add_agents_from_workflow(
    graph, 
    llm_config=llm_config  # This config will be used for all tasks without `llm_config`. 
)

# Create workflow instance
workflow = WorkFlow(graph=graph, agent_manager=agent_manager, llm=llm)

# Execute the workflow with inputs
output = workflow.execute(
    inputs = {
        "problem": "Write a function to find the longest palindromic substring in a given string."
    }
)

print("Workflow completed!")
print("Workflow output:\n", output)
```

You should specify all the required inputs for the workflow in the `inputs` argument of the `execute` method. 

For a complete working example, please refer to the [Sequential Workflow example](https://github.com/EvoAgentX/EvoAgentX/blob/main/examples/sequential_workflow.py). 

## 4. Saving and Loading Workflows

You can save a workflow graph for future use:

```python
# Save the workflow graph to a file
graph.save_module("examples/output/saved_sequential_workflow.json")

# Load the workflow graph from a file
loaded_graph = SequentialWorkFlowGraph.from_file("examples/output/saved_sequential_workflow.json")

# Create a new workflow with the loaded graph
new_workflow = WorkFlow(graph=loaded_graph, agent_manager=agent_manager, llm=llm)
```

For more complex workflows or different types of workflow graphs, please refer to the [Workflow Graphs](../modules/workflow_graph.md) documentation and the [Action Graphs](../modules/action_graph.md) documentation. 
