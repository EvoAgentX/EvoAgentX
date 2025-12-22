# Evaluator

## Introduction

The `Evaluator` class is a fundamental component in the EvoAgentX framework for evaluating the performance of workflows and action graphs against benchmarks. It provides a structured way to measure how well AI agents perform on specific tasks by running evaluations on test data and computing metrics.


## Architecture

### Evaluator Architecture

An `Evaluator` consists of several key components:

1. **LLM Instance**: 
   
    The language model used for executing workflows during evaluation:

    - Provides the reasoning and generation capabilities needed for workflow execution
    - Can be any implementation that follows the `BaseLLM` interface

2. **Agent Manager**: 
   
    Manages the agents used by workflow graphs during evaluation:

    - Provides access to agents needed for workflow execution
    - Only required when evaluating `WorkFlowGraph` instances, and can be ignored when evaluating `ActionGraph` instances 

3. **Data Processing Functions**:
   
    Functions that prepare and process data during evaluation:

    - `collate_func`: Prepares benchmark examples for workflow execution
    - `output_postprocess_func`: Processes workflow outputs before evaluation


### Evaluation Process

The evaluation process follows these steps:

1. **Data Processing**: Obtain examples from the benchmark dataset and process them into the format expected by the workflow graph or action graph
2. **Workflow Execution**: Run each example through the workflow graph or action graph
3. **Output Processing**: Process the outputs into the format expected by the benchmark
4. **Metric Calculation**: Compute performance metrics by comparing outputs to ground truth
5. **Result Aggregation**: Aggregate individual metrics into overall performance scores

## Usage

### Basic Evaluator Creation & Execution

```python
from evoagentx.evaluators import Evaluator
from evoagentx.models import OpenAILLMConfig, OpenAILLM
from evoagentx.agents import AgentManager
from evoagentx.workflow.workflow_graph import WorkFlowGraph
from evoagentx.benchmark import SomeBenchmark
from evoagentx.core.callbacks import suppress_logger_info

# Initialize LLM
llm_config = OpenAILLMConfig(model="gpt-4o-mini", openai_key="xxx")
llm = OpenAILLM(llm_config)

# Initialize agent manager
agent_manager = AgentManager()

# Load your workflow graph
workflow_graph = WorkFlowGraph.from_file("path/to/workflow.json")

# Add agents to the agent manager
agent_manager.add_agents_from_workflow(workflow_graph, llm_config=llm_config)

# Create benchmark
benchmark = SomeBenchmark()

# Create evaluator
evaluator = Evaluator(
    llm=llm,
    agent_manager=agent_manager,
    num_workers=4,  # Use 4 parallel workers
    verbose=True    # Show progress bars
)

# Run evaluation with suppressed logging
with suppress_logger_info():
    results = evaluator.evaluate(
        graph=workflow_graph,
        benchmark=benchmark,
        eval_mode="test",    # Evaluate on test split (default)
        sample_k=100         # Use 100 random examples
    )

print(f"Evaluation results: {results}")
```

### Customizing Data Processing

```python
from evoagentx.evaluators import Evaluator
from evoagentx.models import OpenAILLMConfig, OpenAILLM
from evoagentx.agents import AgentManager
from evoagentx.core.callbacks import suppress_logger_info

# Custom collate function to prepare inputs. The keys should match the input parameters of the workflow graph or action graph. The return value will be directly passed to the `execute` method of the workflow graph or action graph. 
def custom_collate(example):
    return {
        "input_text": example["question"],
        "context": example.get("context", "")
    }

# Custom output processing, `output` is the output of the workflow and the return value will be passed to the `evaluate` method of the benchmark.  
def custom_postprocess(output):
    if isinstance(output, dict):
        return output.get("answer", "")
    return output

# Create evaluator with custom functions
evaluator = Evaluator(
    llm=llm,
    agent_manager=agent_manager,
    collate_func=custom_collate,
    output_postprocess_func=custom_postprocess,
    num_workers=4,  # Use 4 parallel workers
    verbose=True    # Show progress bars
)
```

### Evaluating an Action Graph

```python
from evoagentx.workflow.action_graph import ActionGraph
from evoagentx.evaluators import Evaluator
from evoagentx.models import OpenAILLMConfig, OpenAILLM
from evoagentx.core.callbacks import suppress_logger_info

# Initialize LLM
llm_config = OpenAILLMConfig(model="gpt-4o-mini", openai_key="xxx")
llm = OpenAILLM(llm_config)

# Load your action graph
action_graph = ActionGraph.from_file("path/to/action_graph.json", llm_config=llm_config)

# Create evaluator (no agent_manager needed for action graphs)
evaluator = Evaluator(llm=llm, num_workers=4, verbose=True)

# Run evaluation with suppressed logging
with suppress_logger_info():
    results = evaluator.evaluate(
        graph=action_graph,
        benchmark=benchmark
    )
```

### Asynchronous Evaluation

```python
import asyncio
from evoagentx.evaluators import Evaluator
from evoagentx.models import OpenAILLMConfig, OpenAILLM
from evoagentx.agents import AgentManager
from evoagentx.workflow.workflow_graph import WorkFlowGraph
from evoagentx.benchmark import SomeBenchmark
from evoagentx.core.callbacks import suppress_logger_info

# Initialize LLM and components
llm_config = OpenAILLMConfig(model="gpt-4o-mini", openai_key="xxx")
llm = OpenAILLM(llm_config)
agent_manager = AgentManager()
workflow_graph = WorkFlowGraph.from_file("path/to/workflow.json")
benchmark = SomeBenchmark()

# Create evaluator
evaluator = Evaluator(
    llm=llm,
    agent_manager=agent_manager,
    num_workers=4
)

# Run async evaluation
async def run_async_eval():
    with suppress_logger_info():
        results = await evaluator.async_evaluate(
            graph=workflow_graph,
            benchmark=benchmark
        )
    return results

# Execute async evaluation
results = asyncio.run(run_async_eval())
```

### Accessing Evaluation Records

```python
from evoagentx.evaluators import Evaluator
from evoagentx.models import OpenAILLMConfig, OpenAILLM
from evoagentx.benchmark import SomeBenchmark
from evoagentx.core.callbacks import suppress_logger_info

# Initialize components
llm_config = OpenAILLMConfig(model="gpt-4o-mini", openai_key="xxx")
llm = OpenAILLM(llm_config)
benchmark = SomeBenchmark()
evaluator = Evaluator(llm=llm)

# Run evaluation with suppressed logging
with suppress_logger_info():
    evaluator.evaluate(graph=graph, benchmark=benchmark)

# Get all evaluation records
all_records = evaluator.get_all_evaluation_records()

# Get record for a specific example
example = benchmark.get_test_data()[0]
record = evaluator.get_example_evaluation_record(benchmark, example)

# Get record by example ID
record_by_id = evaluator.get_evaluation_record_by_id(
    benchmark=benchmark,
    example_id="example-123",
    eval_mode="test"
)

# Access trajectory for workflow graph evaluations
if "trajectory" in record:
    for message in record["trajectory"]:
        print(f"{message.role}: {message.content}")
```

The `Evaluator` class provides a powerful way to assess the performance of workflows and action graphs, enabling quantitative comparison and improvement tracking in the EvoAgentX framework.
