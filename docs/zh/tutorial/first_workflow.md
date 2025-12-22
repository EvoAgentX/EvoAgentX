# 构建你的第一个工作流

在 EvoAgentX 中，工作流允许多个代理按顺序协作完成复杂任务。本教程将指导你创建和使用工作流：

1. **理解顺序工作流**：学习工作流如何将多个任务连接在一起
2. **构建顺序工作流**：创建一个包含规划和编码步骤的工作流
3. **执行和管理工作流**：使用特定输入运行工作流

通过本教程，你将能够创建顺序工作流，协调多个代理来解决复杂问题。

## 1. 理解顺序工作流

EvoAgentX 中的工作流代表一系列可以由不同代理执行的任务。最简单的工作流是顺序工作流，其中任务一个接一个地执行，前一个任务的输出作为后续任务的输入。

让我们从导入必要的组件开始：

```python
import os 
from dotenv import load_dotenv
from evoagentx.workflow import SequentialWorkFlowGraph, WorkFlow
from evoagentx.agents import AgentManager
from evoagentx.models import OpenAILLMConfig, OpenAILLM

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
```

## 2. 构建顺序工作流

顺序工作流由一系列任务组成，每个任务都有：

- 名称和描述
- 输入和输出定义
- 提示模板
- 解析模式和函数（可选）

以下是如何构建一个包含规划和编码任务的顺序工作流：

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
    当你使用任务列表创建 `SequentialWorkFlowGraph` 时，框架会为每个任务创建一个 `CustomizeAgent`。工作流中的每个任务都成为一个专门的代理，配置有你定义的特定提示、输入/输出格式和解析模式。这些代理按顺序连接，一个代理的输出成为下一个代理的输入。

    `parse_mode` 控制如何将 LLM 的输出解析为结构化格式。可用选项有：[`'str'`（默认）、`'json'`、`'title'`、`'xml'`、`'custom'`]。有关解析模式和示例的详细信息，请参阅 [CustomizeAgent 文档](../modules/customize_agent.md#parsing-modes)。

## 3. 执行和管理工作流

一旦你创建了工作流图，你就可以创建工作流实例并执行它：

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

你应该在 `execute` 方法的 `inputs` 参数中指定工作流所需的所有输入。

有关完整的工作示例，请参考 [顺序工作流示例](https://github.com/EvoAgentX/EvoAgentX/blob/main/examples/sequential_workflow.py)。

## 4. 保存和加载工作流

你可以保存工作流图以供将来使用：

```python
# Save the workflow graph to a file
graph.save_module("examples/output/saved_sequential_workflow.json")

# Load the workflow graph from a file
loaded_graph = SequentialWorkFlowGraph.from_file("examples/output/saved_sequential_workflow.json")

# Create a new workflow with the loaded graph
new_workflow = WorkFlow(graph=loaded_graph, agent_manager=agent_manager, llm=llm)
```

有关更复杂的工作流或不同类型的工作流图，请参阅 [工作流图](../modules/workflow_graph.md) 文档和 [动作图](../modules/action_graph.md) 文档。
