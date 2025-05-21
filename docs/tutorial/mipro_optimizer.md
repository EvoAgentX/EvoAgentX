# MIPRO Optimizer Tutorial

This tutorial will guide you through setting up and running the [MIPRO](https://arxiv.org/abs/2406.11695) optimizer in EvoAgentX. We'll use the MATH benchmark as an example to demonstrate how to optimize a multi-agent workflow.

## 1. Overview

The MIPRO optimizer is a powerful tool in EvoAgentX that enables you to:

- Automatically optimize prompts within multi-agent workflows 
- Evaluate optimization results on benchmark datasets
- Support zero-shot and few-shot optimization
- Provide automated optimization strategies

## 2. Environment Setup

First, let's import the necessary modules for setting up the MIPRO optimizer:

```python
import os
from dotenv import load_dotenv
from evoagentx.agents import AgentManager
from evoagentx.models import OpenAILLM, OpenAILLMConfig
from evoagentx.benchmark import MATH
from evoagentx.workflow import SequentialWorkFlowGraph, WorkFlowGraph
from evoagentx.core.callbacks import suppress_logger_info
from evoagentx.optimizers import MiproOptimizer
from evoagentx.evaluators import Evaluator
import evoagentx
```

### Configure the LLM Model

Similar to other components in EvoAgentX, you'll need a valid OpenAI API key to initialize the LLM:

```python
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai_config = OpenAILLMConfig(model="gpt-4o-mini", openai_key=OPENAI_API_KEY)
evoagentx.configure(lm=openai_config)
```

## 3. Setting Up Components

### Step 1: Customize the Benchmark Dataset

For this tutorial, we'll create a custom class that splits the MATH benchmark data into training and test sets:

```python
class MathSplits(MATH):
    def _load_data(self):
        # load the original test data 
        super()._load_data()
        # split the data into dev and test
        import numpy as np 
        np.random.seed(42)
        permutation = np.random.permutation(len(self._test_data))
        full_test_data = self._test_data
        # randomly select 50 samples for dev and 100 samples for test
        self._train_data = [full_test_data[idx] for idx in permutation[:50]]
        self._test_data = [full_test_data[idx] for idx in permutation[50:150]]
```

### Step 2: Define the Workflow

We'll define a workflow for solving math problems:

```python
math_graph_data = {
    "goal": r"Answer the math question. The answer should be in box format, e.g., \boxed{{123}}",
    "tasks": [
        {
            "name": "answer_generate",
            "description": "Answer generation for Math.",
            "inputs": [
                {"name": "problem", "type": "str", "required": True, "description": "The problem to solve."}
            ],
            "outputs": [
                {"name": "answer", "type": "str", "required": True, "description": "The generated answer."}
            ],
            "prompt": "Answer the math question. The answer should be in box format, e.g., \\boxed{{123}}\n\nProblem: {problem}",
            "parse_mode": "str"
        }
    ] 
}

workflow_graph = SequentialWorkFlowGraph.from_dict(math_graph_data)
```

### Step 3: Prepare Dataset and Evaluation

```python
def collate_func(example: dict) -> dict:
    return {"problem": example["problem"]}

benchmark = MathSplits()
trainset = [{"problem": collate_func(example)["problem"], "solution": example["solution"]} 
           for example in benchmark._train_data]

def evaluate_metric(example, prediction, trace=None):
    example_ans = benchmark.extract_answer(example["solution"])
    prediction_ans = benchmark.extract_answer(prediction)
    return benchmark.math_equal(prediction_ans, example_ans)
```

## 4. Configuring and Running the MIPRO Optimizer

### Step 1: Set up the Agent Manager and Evaluator

```python
agent_manager = AgentManager()
agent_manager.add_agents_from_workflow(workflow_graph, llm_config=openai_config)

evaluate = Evaluator(
    llm = OpenAILLM(config=openai_config),
    agent_manager = agent_manager,
    collate_func = collate_func,
    num_workers = 32,
    verbose = True
)
```

### Step 2: Configure and Run the Optimizer

The MIPRO optimizer requires either a benchmark or a training dataset to be provided, but not both. This is because:
- If a benchmark is provided, it will use the benchmark's training data for optimization
- If a training dataset is provided, it will use that dataset directly for optimization

Here's how to configure the optimizer:

```python
# Option 1: Using a benchmark
optimizer = MiproOptimizer(
    graph = workflow_graph,
    metric = evaluate_metric,
    executor_llm = openai_config,
    benchmark = benchmark,  # Provide the benchmark
    max_bootstrapped_demos = 4,
    max_labeled_demos = 4,
    num_candidates = 5,
    auto = "light",
    num_threads = 32,
    save_path = "examples/mipro/output/logs",
    evaluator = evaluate
)

# OR Option 2: Using a training dataset
optimizer = MiproOptimizer(
    graph = workflow_graph,
    metric = evaluate_metric,
    executor_llm = openai_config,
    trainset = trainset,  # Provide the training dataset
    max_bootstrapped_demos = 4,
    max_labeled_demos = 4,
    num_candidates = 5,
    auto = "light",
    num_threads = 32,
    save_path = "examples/mipro/output/logs",
    evaluator = evaluate
)

# Run optimization with suppressed logging
with suppress_logger_info():
    best_program = optimizer.optimize(
        collate_func = collate_func,
    )

# Save the optimized program
output_path = "examples/mipro/output/best_program_math.json"
best_program.save_module(output_path)

# Evaluate the optimized program
# If using benchmark, you can evaluate directly on the benchmark
if benchmark:
    result = optimizer.evaluate(graph = best_program, benchmark = benchmark, eval_mode = "test")
else:
    # If using trainset, you'll need to provide your own evaluation dataset
    result = optimizer.evaluate(graph = best_program, eval_dataset = test_dataset)
print(result)
```

## 5. Important Parameter Notes

- `benchmark` or `trainset`: You must provide exactly one of these parameters. The optimizer will raise an error if both are provided or if neither is provided.
- `max_bootstrapped_demos`: Number of bootstrapped demonstrations for optimization (default: 4)
- `max_labeled_demos`: Number of labeled demonstrations for optimization (default: 4)
- `num_candidates`: Number of candidate programs to generate (default: 5)
- `auto`: Automation level, options include "light", "medium", "heavy"
- `num_threads`: Number of threads for parallel processing (default: 32)
- `save_path`: Directory to save optimization process logs

## 6. Notes

1. The tutorial uses a custom `MathSplits` class to split the MATH benchmark data into training and test sets
2. The workflow is defined using a dictionary format that gets converted to a `SequentialWorkFlowGraph`
3. The evaluation metric uses the benchmark's built-in answer extraction and comparison functions
4. The optimizer uses an `AgentManager` and `Evaluator` for managing agents and evaluating results
5. Make sure to adjust the number of workers and threads based on your hardware capabilities

Complete example code can be found in [mipro_math.py](../../examples/optimization/mipro/mipro_math.py). 