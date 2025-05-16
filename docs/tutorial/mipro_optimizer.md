# MIPRO Optimizer Tutorial

This tutorial will guide you through setting up and running the MIPRO (Multi-agent Interactive Prompt Optimization) optimizer in EvoAgentX. We'll use the MATH benchmark as an example to demonstrate how to optimize a multi-agent workflow.

## 1. Overview

The MIPRO optimizer is a powerful tool in EvoAgentX that enables you to:

- Automatically optimize multi-agent workflows (prompts and workflow structure)
- Evaluate optimization results on benchmark datasets
- Support zero-shot and few-shot optimization
- Provide automated optimization strategies

## 2. Environment Setup

First, let's import the necessary modules for setting up the MIPRO optimizer:

```python
from evoagentx.optimizers.mipro_optimizer import MiproOptimizer
from evoagentx.models import OpenAILLMConfig, OpenAILLM
from evoagentx.utils.mipro_utils.settings import settings
from evoagentx.evaluators.mipro_evaluator import Evaluate as mipro_evaluator
from evoagentx.workflow import SequentialWorkFlowGraph
import os
from dotenv import load_dotenv
```

Note: For this tutorial, we'll be using the MATH benchmark dataset. You can import it when needed:
```python
from evoagentx.benchmark.math import MATH
```

### Configure the LLM Model

Similar to other components in EvoAgentX, you'll need a valid OpenAI API key to initialize the LLM:

```python
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai_config = OpenAILLMConfig(model="gpt-4o-mini", openai_key=OPENAI_API_KEY)
executor_llm = OpenAILLM(config=openai_config)
settings.lm = openai_config
```

## 3. Setting Up Components

### Step 1: Initialize the Workflow

First, we need to define a task list to create a sequential workflow:

```python
tasks = [
    {
        "name": "answer",
        "description": "provide answer to the math problem",
        "inputs": [
            {"name": "problem", "type": "str", "required": True, "description": ""},
        ],
        "outputs": [
            {"name": "answer", "type": "str", "required": True, "description": ""}
        ],
        "prompt": "Problem: {problem} You must provide the answer in box format E.g. \\boxed{{123}}",
        "parse_mode": "str",
    }
]

# Create a sequential workflow
graph = SequentialWorkFlowGraph(
    goal="Generate answer to math question",
    tasks=tasks,
)
```

### Step 2: Prepare Evaluation Metric

MIPRO optimizer needs an evaluation metric function to evaluate optimization results. Here's an example for a mathematical problem evaluation:

```python
def evaluate_metric(example, prediction, trace=None):
    example_ans = math.extract_answer(example["solution"])
    prediction_ans = math.extract_answer(prediction)
    return math.math_equal(prediction_ans, example_ans)
```

### Step 3: Prepare Dataset

```python
math = MATH()
math._load_data()
trainset = math._test_data[:100]  # Training set
devset = math._test_data[100:200]  # Development set
```

## 4. Configuring and Running the MIPRO Optimizer

MIPRO optimizer can be configured with various parameters:

```python
optimizer = MiproOptimizer(
    graph=graph,                    # Workflow graph to optimize
    metric=evaluate_metric,         # Evaluation metric function
    prompt_model=executor_llm,      # Language model
    max_bootstrapped_demos=0,       # Maximum number of bootstrapped demos for zero-shot optimization
    max_labeled_demos=0,           # Maximum number of labeled demos for zero-shot optimization
    auto="medium",                 # Automation level, options include "low", "medium", "high"
    num_threads=32,                # Number of parallel threads
    log_dir="examples/output/logs"  # Directory to save optimization process logs
)
```

### Running Optimization

To start the optimization process:

```python
# Set up evaluator
evaluate = mipro_evaluator(
    devset=devset,
    metric=evaluate_metric,
    num_threads=32,
    display_progress=True,
    display_table=False,
)

# Evaluate results before optimization
prev_results = evaluate(program=graph, with_inputs={"problem":"problem"})

# Run optimization
best_program = optimizer.optimize(
    trainset=trainset, 
    with_inputs={"problem":"problem"}, 
    provide_traceback=True
)

# Save optimized workflow
output_path = "examples/mipro/output/best_program_math.json"
best_program.save_module(output_path)

# Evaluate results after optimization
post_results = evaluate(
    program=WorkFlowGraph.from_file(output_path), 
    with_inputs={"problem":"problem"}
)

print(f"Previous results: {prev_results}")
print(f"Optimized results: {post_results}")
```

## 5. Important Parameter Notes

- `max_bootstrapped_demos`: Maximum number of demos for bootstrapped optimization, set to 0 for zero-shot optimization
- `max_labeled_demos`: Maximum number of demos for labeled optimization, set to 0 for zero-shot optimization
- `auto`: Automation level, options include "low", "medium", "high"
- `num_threads`: Number of threads for parallel processing
- `log_dir`: Directory to save optimization process logs

## 6. Notes

1. Ensure your evaluation metric function can handle input and output formats correctly
2. Adjust task definitions in the workflow based on your specific task
3. Set appropriate dataset size to avoid excessive optimization time
4. Adjust `num_threads` parameter based on your hardware configuration

Complete example code can be found in [mipro_example.py](../../examples/mipro_example.py). 