# Action Graph

## Introduction

The `ActionGraph` class is a fundamental component in the EvoAgentX framework for creating and executing sequences of operations (actions) within a single task. It provides a structured way to define, manage, and execute a series of operations that need to be performed in a specific order to complete a task.

An action graph represents a collection of operators (actions) that are executed in a predefined sequence to process inputs and produce outputs. Unlike the `WorkFlowGraph` which manages multiple tasks and their dependencies at a higher level, the `ActionGraph` focuses on the detailed execution steps within a single task.

## Architecture

### ActionGraph Architecture

An `ActionGraph` consists of several key components:

1. **Operators**: 
   
    Each operator represents a specific operation or action that can be performed as part of a task, with the following properties:

    - `name`: A unique identifier for the operator
    - `description`: Detailed description of what the operator does
    - `llm`: The LLM used to execute the operator
    - `outputs_format`: The structured format of the output of the operator
    - `interface`: The interface for calling the operator.
    - `prompt`: Template used to guide the LLM when executing this operator

2. **LLM**: 
   
    The ActionGraph uses a Language Learning Model (LLM) to execute the operators. It receives a `llm_config` as input and create an LLM instance, which will be passed to the operators for execution. The LLM provides the reasoning and generation capabilities needed to perform each action.

3. **Execution Flow**:
   
    The ActionGraph defines a specific execution sequence:

    - Actions are executed in a predetermined order (specified in the `execute` or `async_execute` method using code)
    - Each action can use the results from previous actions
    - The final output is produced after all actions have been executed

### Comparison with WorkFlowGraph

While both `ActionGraph` and `WorkFlowGraph` manage execution flows, they operate at different levels of abstraction:

| Feature | ActionGraph | WorkFlowGraph |
|---------|-------------|---------------|
| Scope | Single task execution | Multi-task workflow orchestration |
| Components | Operators (actions) | Nodes (tasks) and edges (dependencies) |
| Focus | Detailed steps within a task | Relationships between different tasks |
| Flexibility | Fixed execution sequence | Dynamic execution based on dependencies |
| Primary use | Define reusable task execution patterns | Orchestrate complex multi-step workflows |
| Granularity | Fine-grained operations | Coarse-grained tasks |

## Usage

### Basic ActionGraph Creation

```python
from evoagentx.workflow import ActionGraph
from evoagentx.workflow.operators import Custom
from evoagentx.models import OpenAILLMConfig 

# Create LLM configuration
llm_config = OpenAILLMConfig(model="gpt-4o-mini", openai_key="xxx")

# Create a custom ActionGraph
class MyActionGraph(ActionGraph):
    def __init__(self, llm_config, **kwargs):

        name = kwargs.pop("name") if "name" in kwargs else "Custom Action Graph"
        description = kwargs.pop("description") if "description" in kwargs else "A custom action graph for text processing"
        # create an LLM instance `self._llm` based on the `llm_config` and pass it to the operators
        super().__init__(name=name, description=description, llm_config=llm_config, **kwargs)
        # Define operators
        self.extract_entities = Custom(self._llm) # , prompt="Extract key entities from the following text: {input}")
        self.analyze_sentiment = Custom(self._llm) # , prompt="Analyze the sentiment of the following text: {input}")
        self.summarize = Custom(self._llm) # , prompt="Summarize the following text in one paragraph: {input}")

    def execute(self, text: str) -> dict:
        # Execute operators in sequence (specify the execution order of operators)
        entities = self.extract_entities(input=text, instruction="Extract key entities from the provided text")["response"]
        sentiment = self.analyze_sentiment(input=text, instruction="Analyze the sentiment of the provided text")["response"]
        summary = self.summarize(input=text, instruction="Summarize the provided text in one paragraph")["response"]

        # Return combined results
        return {
            "entities": entities,
            "sentiment": sentiment,
            "summary": summary
        }

# Create the action graph
action_graph = MyActionGraph(llm_config=llm_config)

# Execute the action graph
result = action_graph.execute(text="This is a test text")
print(result)
```

### Using ActionGraph in WorkFlowGraph

You can either use `ActionGraph` directly or use it in `WorkFlowGraph` as a node. 

```python
from evoagentx.workflow.workflow_graph import WorkFlowNode, WorkFlowGraph
from evoagentx.workflow.action_graph import QAActionGraph
from evoagentx.core.base_config import Parameter
from evoagentx.models import OpenAILLMConfig, OpenAILLM
from evoagentx.workflow import WorkFlow

# Create LLM configuration
llm_config = OpenAILLMConfig(model="gpt-4o-mini", openai_key="xxx", stream=True, output_response=True)
llm = OpenAILLM(llm_config)

# Create an action graph
qa_graph = QAActionGraph(llm_config=llm_config)

# Create a workflow node that uses the action graph
qa_node = WorkFlowNode(
    name="QATask",
    description="Answer questions using a QA system",
    # input names should match the parameters in the `execute` method of the action graph
    inputs=[Parameter(name="problem", type="string", description="The problem to answer")],
    outputs=[Parameter(name="answer", type="string", description="The answer to the problem")],
    action_graph=qa_graph  # Using action_graph instead of agents
)

# Create the workflow graph
workflow_graph = WorkFlowGraph(goal="Answer a question", nodes=[qa_node])

# define the workflow 
workflow = WorkFlow(graph=workflow_graph, llm=llm)

# Execute the workflow
result = workflow.execute(inputs={"problem": "What is the capital of France?"})
print(result)
```

!!! warning 
    When using `ActionGraph` in `WorkFlowNode`, the `inputs` parameter of the `WorkFlowNode` should match the required parameters in the `execute` method of the `ActionGraph`. The `execute` method is expected to return a **dictionary** or `LLMOutputParser` instance with keys matching the names of the `outputs` in the `WorkFlowNode`. 

### Saving and Loading an ActionGraph

```python
# Save action graph
action_graph.save_module("examples/output/my_action_graph.json")

# Load action graph
from evoagentx.workflow.action_graph import ActionGraph
loaded_graph = ActionGraph.from_file("examples/output/my_action_graph.json", llm_config=llm_config)
```

The `ActionGraph` class provides a powerful way to define complex sequences of operations within a single task, complementing the higher-level orchestration capabilities of the `WorkFlowGraph` in the EvoAgentX framework.
