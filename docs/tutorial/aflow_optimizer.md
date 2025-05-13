# AFlow Optimizer Tutorial

This tutorial will guide you through the process of setting up and running the [AFlow](https://arxiv.org/abs/2410.10762) optimizer in EvoAgentX. We'll use the HumanEval benchmark as an example to demonstrate how to optimize a multi-agent workflow for code generation tasks.

## 1. Overview

The AFlow optimizer in EvoAgentX enables you to:

- Automatically optimize multi-agent workflows for specific task types (code generation, QA, math, etc.)
- Support different types of operators (Custom, CustomCodeGenerate, Test, ScEnsemble, etc.)
- Evaluate optimization results on benchmark datasets
- Use different LLMs for optimization and execution

## 2. Setting Up the Environment

First, let's import the necessary modules for setting up the AFlow optimizer:

```python
import os
from dotenv import load_dotenv
from evoagentx.optimizers import AFlowOptimizer
from evoagentx.models import LiteLLMConfig, LiteLLM, OpenAILLMConfig, OpenAILLM
from evoagentx.benchmark import AFlowHumanEval

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
```

### Configure the LLM Models

Following the settings in the [original AFlow implementation](https://github.com/FoundationAgents/MetaGPT/tree/main/examples/aflow), the AFlow optimizer uses two different LLMs:
1. An optimizer LLM (e.g., Claude 3.5 Sonnet) for workflow optimization
2. An executor LLM (e.g., GPT-4o-mini) for task execution

```python
# Configure the optimizer LLM (Claude 3.5 Sonnet)
claude_config = LiteLLMConfig(
    model="anthropic/claude-3-5-sonnet-20240620", 
    anthropic_key=ANTHROPIC_API_KEY
)
optimizer_llm = LiteLLM(config=claude_config)

# Configure the executor LLM (GPT-4o-mini)
openai_config = OpenAILLMConfig(
    model="gpt-4o-mini", 
    openai_key=OPENAI_API_KEY
)
executor_llm = OpenAILLM(config=openai_config)
```

## 3. Setting Up the Components

### Step 1: Define Task Configuration

The AFlow optimizer requires a configuration that specifies the task type and available operators. Here's an example configuration for different task types:

```python
EXPERIMENTAL_CONFIG = {
    "humaneval": {
        "question_type": "code", 
        "operators": ["Custom", "CustomCodeGenerate", "Test", "ScEnsemble"] 
    }, 
    "mbpp": {
        "question_type": "code", 
        "operators": ["Custom", "CustomCodeGenerate", "Test", "ScEnsemble"] 
    },
    "hotpotqa": {
        "question_type": "qa", 
        "operators": ["Custom", "AnswerGenerate", "QAScEnsemble"]
    },
    "gsm8k": {
        "question_type": "math", 
        "operators": ["Custom", "ScEnsemble", "Programmer"]
    },
    "math": {
        "question_type": "math", 
        "operators": ["Custom", "ScEnsemble", "Programmer"]
    }
}
```

### Step 2: Define initial workflow 

The AFlow optimizer requires two files: 
- `graph.py`: which defines the initial workflow graph in python code. 
- `prompt.py`: which defines the prompts used in the workflow. 

Below is an example of the `graph.py` file for the HumanEval benchmark:

```python
import evoagentx.workflow.operators as operator
import examples.aflow.code_generation.prompt as prompt_custom # noqa: F401
from evoagentx.models.model_configs import LLMConfig
from evoagentx.benchmark.benchmark import Benchmark
from evoagentx.models.model_utils import create_llm_instance

class Workflow:
    
    def __init__(
        self,
        name: str,
        llm_config: LLMConfig,
        benchmark: Benchmark
    ):
        self.name = name
        self.llm = create_llm_instance(llm_config)
        self.benchmark = benchmark 
        self.custom = operator.Custom(self.llm)
        self.custom_code_generate = operator.CustomCodeGenerate(self.llm)

    async def __call__(self, problem: str, entry_point: str):
        """
        Implementation of the workflow
        Custom operator to generate anything you want.
        But when you want to get standard code, you should use custom_code_generate operator.
        """
        # await self.custom(input=, instruction="")
        solution = await self.custom_code_generate(problem=problem, entry_point=entry_point, instruction=prompt_custom.GENERATE_PYTHON_CODE_PROMPT) # But When you want to get standard code ,you should use customcodegenerator.
        return solution['response']
```

!!! note
    When defining your workflow, please pay attention to the following key points:

    1. **Prompt Import Path**: Ensure the import path for `prompt.py` is correctly specified (e.g., `examples.aflow.code_generation.prompt`). This path should match your project structure to enable proper prompt loading.

    2. **Operator Initialization**: In the `__init__` function, you must initialize all operators that will be used in the workflow. Each operator should be instantiated with the appropriate LLM instance. 

    3. **Workflow Execution**: The `__call__` function serves as the main entry point for workflow execution. It should define the complete execution logic of your workflow and return the final output that will be used for evaluation.


Below is an example of the `prompt.py` file for the HumanEval benchmark:

```python
GENERATE_PYTHON_CODE_PROMPT = """
Generate a functional and correct Python code for the given problem.

Problem: """
```

!!! note 
    If the workflow does not require any prompts, the `prompt.py` file can be empty. 

### Step 3: Prepare the Benchmark

For this tutorial, we'll use the AFlowHumanEval benchmark. It follows the exact same data split and format as used in the [original AFlow implementation](https://github.com/FoundationAgents/MetaGPT/tree/main/examples/aflow).

```python
# Initialize the benchmark
humaneval = AFlowHumanEval()
```

## 4. Configuring and Running the AFlow Optimizer

The AFlow optimizer can be configured with various parameters to control the optimization process:

```python
optimizer = AFlowOptimizer(
    graph_path="examples/aflow/code_generation",  # Path to the initial workflow graph
    optimized_path="examples/aflow/humaneval/optimized",  # Path to save optimized workflows
    optimizer_llm=optimizer_llm,  # LLM for optimization
    executor_llm=executor_llm,    # LLM for execution
    validation_rounds=3,          # Number of times to run validation on the development set during optimization
    eval_rounds=3,               # Number of times to run evaluation on the test set during testing
    max_rounds=20,               # Maximum optimization rounds
    **EXPERIMENTAL_CONFIG["humaneval"]  # Task-specific configuration, used to specify the task type and available operators
)
```

### Running the Optimization

To start the optimization process:

```python
# Optimize the workflow
optimizer.optimize(humaneval)
```

!!! note 
    During optimization, the workflow will be validated on the development set for `validation_rounds` times at each step. Make sure the benchmark `humaneval` contains a development set (i.e., `self._dev_data` is not empty).

### Test the Optimized Workflow

To test the optimized workflow:

```python
optimizer.test(humaneval)
```
By default, the optimizer will choose the workflow with the highest validation performance to test. You can also specify the test rounds using the `test_rounds: List[int]` parameter. For example, to evaluate the second round and the third round, you can use `optimizer.test(humaneval, test_rounds=[2, 3])`.

!!! note 
    During testing, the workflow will be evaluated on the test set for `eval_rounds` times. Make sure the benchmark `humaneval` contains a test set (i.e., `self._test_data` is not empty).

For a complete working example, please refer to [aflow_humaneval.py](https://github.com/EvoAgentX/EvoAgentX/blob/main/examples/optimization/aflow_humaneval.py).