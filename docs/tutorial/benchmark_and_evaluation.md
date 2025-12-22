# Benchmark and Evaluation Tutorial

This tutorial will guide you through the process of setting up and running benchmark evaluations using EvoAgentX. We'll use the HotpotQA dataset as an example to demonstrate how to set up and run the evaluation process.


## 1. Overview

EvoAgentX provides a flexible and modular evaluation framework that enables you to:

- Load and use predefined benchmark datasets
- Customize data loading, processing, and post-processing logic
- Evaluate the performance of your multi-agent workflows
- Process multiple evaluation tasks in parallel

## 2. Setting Up the Benchmark

To get started, you need to import the relevant modules and set up the language model (LLM) that your agent will use during evaluation.


```python
from evoagentx.config import Config
from evoagentx.models import OpenAIConfig, OpenAI 
from evoagentx.benchmark import HotpotQA
from evoagentx.workflow import QAActionGraph 
from evoagentx.evaluators import Evaluator 
from evoagentx.core.callbacks import suppress_logger_info
```

### Configure the LLM Model 
You'll need a valid OpenAI API key to initialize the LLM. It is recommended to save your API key in the `.env` file and load it using the `load_dotenv` function: 
```python 
import os 
from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
llm_config = OpenAILLMConfig(model="gpt-4o-mini", openai_key=OPENAI_API_KEY)
llm = OpenAILLM(config=llm_config)
```

## 3. Initialize the Benchmark 
EvoAgentX includes several predefined benchmarks for tasks like Question Answering, Math, and Coding. Please refer to the [Benchmark README](https://github.com/EvoAgentX/EvoAgentX/blob/main/evoagentx/benchmark/README.md) for more details about existing benchmarks. You can also define your own benchmark class by extending the base `Benchmark` interface, and we provide an example in the [Custom Benchmark](#custom-benchmark) section.

In this example, we will use the `HotpotQA` benchmark. 
```python 
benchmark = HotPotQA(mode="dev")
```
where `mode` parameter determines which split of the dataset is loaded. Options include:

* `"train"`: Training data
* `"dev"`: Development/validation data
* `"test"`: Test data
* `"all"` (default): Loads the entire dataset

The data will be automatically downloaded to a default cache folder, but you can change this location by specifying the `path` parameter.


## 4. Running the Evaluation 
Once you have your benchmark and LLM ready, the next step is to define your agent workflow and evaluation logic. EvoAgentX supports full customization of how benchmark examples are processed and how outputs are interpreted.

Here's how to run an evaluation using the `HotpotQA` benchmark and a QA workflow.

### Step 1: Define the Agent Workflow 
You can use one of the predefined workflows or implement your own. In this example, we use the [`QAActionGraph`](https://github.com/EvoAgentX/EvoAgentX/blob/main/evoagentx/workflow/action_graph.py#L99) designed for question answering, which simply use self-consistency to generate the final answer:

```python
workflow = QAActionGraph(
    llm_config=llm_config,
    description="This workflow aims to address multi-hop QA tasks."
)
``` 

### Step 2: Customize Data Preprocessing and Post-processing 

The next key aspect of evaluation is properly transforming data between your benchmark, workflow, and evaluation metrics.

### Why Preprocessing and Postprocessing Are Needed

In EvoAgentX, **preprocessing** and **postprocessing** are essential steps to ensure smooth interaction between benchmark data, workflows, and evaluation logic:

- **Preprocessing (`collate_func`)**:  

    The raw examples from a benchmark like HotpotQA typically consist of structured fields such as questions, answer, and context. However, your agent workflow usually expects a single prompt string or other structured input. The `collate_func` is used to convert each raw example into a format that can be consumed by your (custom) workflow.

- **Postprocessing (`output_postprocess_func`)**:

    The workflow output might include reasoning steps or additional formatting beyond just the final answer. Since the `Evaluator` internally calls the benchmark's `evaluate` method to compute metrics (e.g., exact match or F1), it's often necessary to extract the final answer in a clean format. The `output_postprocess_func` handles this and ensures the output is in the right form for evaluation.

In short, **preprocessing prepares benchmark examples for the workflow**, while **postprocessing prepares workflow outputs for evaluation**.

In the following example, we define a `collate_func` to format the raw examples into a prompt for the workflow, and a `output_postprocess_func` to extract the final answer from the workflow output.

Each example in the benchmark can be formatted using a `collate_func`, which transforms raw examples into a prompt or structured input for the agent.

```python
def collate_func(example: dict) -> dict:
    """
    Args:
        example (dict): A dictionary containing the raw example data.

    Returns: 
        The expected input for the (custom) workflow.
    """
    problem = "Question: {}\n\n".format(example["question"])
    context_list = []
    for item in example["context"]:
        context = "Title: {}\nText: {}".format(item[0], " ".join([t.strip() for t in item[1]]))
        context_list.append(context)
    context = "\n\n".join(context_list)
    problem += "Context: {}\n\n".format(context)
    problem += "Answer:" 
    return {"problem": problem}
```

After the agent generates an output, you can define how to extract the final answer using `output_postprocess_func`. 
```python
def output_postprocess_func(output: dict) -> dict:
    """
    Args:
        output (dict): The output from the workflow.

    Returns: 
        The processed output that can be used to compute the metrics. The output will be directly passed to the benchmark's `evaluate` method. 
    """
    return output["answer"]
```

### Step 3: Initialize the Evaluator 
The Evaluator ties everything together — it runs the workflow over the benchmark and calculates performance metrics.

```python
evaluator = Evaluator(
    llm=llm, 
    collate_func=collate_func,
    output_postprocess_func=output_postprocess_func,
    verbose=True, 
    num_workers=3 
)
``` 
If `num_workers` is greater than 1, the evaluation will be parallelized across multiple threads.  

### Step 4: Run the Evaluation 
You can now run the evaluation by providing the workflow and benchmark to the evaluator:

```python
with suppress_logger_info():
    results = evaluator.evaluate(
        graph=workflow, 
        benchmark=benchmark, 
        eval_mode="dev", # Evaluation split: train / dev / test 
        sample_k=10 # If set, randomly sample k examples from the benchmark for evaluation  
    )
    
print("Evaluation metrics: ", results)
```
where `suppress_logger_info` is used to suppress the logger info.

Please refer to the [benchmark and evaluation example](https://github.com/EvoAgentX/EvoAgentX/blob/main/examples/benchmark_and_evaluation.py) for a complete example.


## Custom Benchmark 

To define a custom benchmark, you need to extend the `Benchmark` class and implement the following methods:

- `_load_data(self)`: 
    
    Load the benchmark data, and set the `self._train_data`, `self._dev_data` and `self._test_data` attributes.


- `_get_id(self, example: Any) -> Any`: 

    Return the unique identifier of an example.

- `_get_label(self, example: Any) -> Any`:

    Return the label or ground truth associated with a given example.

    This is used to compare predictions against the correct answer during evaluation. The output will be directly passed to the `evaluate` method. 


- `evaluate(self, prediction: Any, label: Any) -> dict`: 

    Compute the evaluation metrics for a single example, based on its prediction and ground-truth label (obtained from `_get_label`).
    This method should return a dictionary of metric name(s) and value(s).

For a complete example of a benchmark implementation, please refer to the [HotPotQA](https://github.com/EvoAgentX/EvoAgentX/blob/main/evoagentx/benchmark/hotpotqa.py#L23) class.
