# SEW Optimizer Tutorial

This tutorial will guide you through the process of setting up and running the SEW (Self-Evolving Workflow) optimizer in EvoAgentX. We'll use the HumanEval benchmark as an example to demonstrate how to optimize a multi-agent workflow.

## 1. Overview

The SEW optimizer is a powerful tool in EvoAgentX that enables you to:

- Automatically optimize multi-agent workflows (prompts and workflow structure)
- Evaluate optimization results on benchmark datasets
- Support different workflow representation scheme (Python, Yaml, BPMN, etc.)

## 2. Setting Up the Environment

First, let's import the necessary modules for setting up the SEW optimizer:

```python
from evoagentx.config import Config
from evoagentx.models import OpenAILLMConfig, OpenAILLM
from evoagentx.workflow import SEWWorkFlowGraph 
from evoagentx.agents import AgentManager
from evoagentx.benchmark import HumanEval 
from evoagentx.evaluators import Evaluator 
from evoagentx.optimizers import SEWOptimizer 
from evoagentx.core.callbacks import suppress_logger_info
```

### Configure the LLM Model

Similar to other components in EvoAgentX, you'll need a valid OpenAI API key to initialize the LLM. 

```python
llm_config = OpenAILLMConfig(model="gpt-4o-mini", openai_key=OPENAI_API_KEY)
llm = OpenAILLM(config=llm_config)
```

## 3. Setting Up the Components

### Step 1: Initialize the SEW Workflow

The SEW workflow is the core component that will be optimized. It represents a sequential workflow that aims to solve the code generation task. 

```python
sew_graph = SEWWorkFlowGraph(llm_config=llm_config)
agent_manager = AgentManager()
agent_manager.add_agents_from_workflow(sew_graph)
```

### Step 2: Prepare the Benchmark

For this tutorial, we'll use a modified version of the HumanEval benchmark that splits the test data into development and test sets:

```python
class HumanEvalSplits(HumanEval):
    def _load_data(self):
        # load the original test data 
        super()._load_data()
        # split the data into dev and test
        import numpy as np 
        np.random.seed(42)
        num_dev_samples = int(len(self._test_data) * 0.2)
        random_indices = np.random.permutation(len(self._test_data))
        self._dev_data = [self._test_data[i] for i in random_indices[:num_dev_samples]]
        self._test_data = [self._test_data[i] for i in random_indices[num_dev_samples:]]

# Initialize the benchmark
humaneval = HumanEvalSplits()
```

The SEWOptimizer will evaluate the performance on the development set by default. Please make sure the benchmark has a development set properly set up. You can either:
   - Use a benchmark that already provides a development set (like HotPotQA)
   - Split your dataset into development and test sets (like in the HumanEvalSplits example above)
   - Implement a custom benchmark with development set support

### Step 3: Set Up the Evaluator

The evaluator is responsible for assessing the performance of the workflow during optimization. For more detailed information about how to set up and use the evaluator, please refer to the [Benchmark and Evaluation Tutorial](./benchmark_and_evaluation.md).

```python
def collate_func(example: dict) -> dict:
    # convert raw example to the expected input for the SEW workflow
    return {"question": example["prompt"]}

evaluator = Evaluator(
    llm=llm, 
    agent_manager=agent_manager, 
    collate_func=collate_func, 
    num_workers=5, 
    verbose=True
)
```

## 4. Configuring and Running the SEW Optimizer

The SEW optimizer can be configured with various parameters to control the optimization process:

```python
optimizer = SEWOptimizer(
    graph=sew_graph,           # The workflow graph to optimize
    evaluator=evaluator,       # The evaluator for performance assessment
    llm=llm,                   # The language model
    max_steps=10,             # Maximum optimization steps
    eval_rounds=1,            # Number of evaluation rounds per step
    repr_scheme="python",     # Representation scheme for the workflow
    optimize_mode="prompt",   # What aspect to optimize (prompt/structure/all)
    order="zero-order"        # Optimization algorithm order (zero-order/first-order)
)
```

### Running the Optimization

To start the optimization process:

```python
# Optimize the SEW workflow
optimizer.optimize(dataset=humaneval)

# Evaluate the optimized workflow
with suppress_logger_info():
    metrics = optimizer.evaluate(dataset=humaneval, eval_mode="test")
print("Evaluation metrics: ", metrics)

# Save the optimized SEW workflow
optimizer.save("debug/optimized_sew_workflow.json")
```

For a complete working example, please refer to [sew_optimizer.py](https://github.com/EvoAgentX/EvoAgentX/blob/main/examples/sew_optimizer.py).
