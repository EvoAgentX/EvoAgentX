# TextGrad Optimizer Tutorial

This tutorial will guide you through the process of setting up and running the TextGrad optimizer in EvoAgentX. We'll use the [MATH](https://www.modelscope.cn/datasets/opencompass/competition_math) dataset as an example to demonstrate how to optimize the prompts
and system prompts in a workflow.

## 1. TextGrad
TextGrad uses textual feedback from LLM to improve text variables. In EvoAgentX, we use TextGrad to optimize 
agents' prompts and system prompts. For more information on TextGrad, see their [paper](https://arxiv.org/abs/2406.07496) and [GitHub](https://github.com/zou-group/textgrad).

## 2. TextGrad Optimizer
The TextGrad optimizer in EvoAgentX enables you to:

- Automatically optimize multi-agent workflows (prompts and/or system prompts)
- Evaluate optimization results on datasets

## 3. Setting Up the Environment

First, let's import the necessary modules for setting up the TextGrad optimizer:

```python
from evoagentx.benchmark import MATH
from evoagentx.optimizers import TextGradOptimizer
from evoagentx.models import OpenAILLMConfig, OpenAILLM
from evoagentx.workflow import SequentialWorkFlowGraph
from evoagentx.core.callbacks import suppress_logger_info
from evoagentx.prompts import StringTemplate
```

### Configure the LLM Model
You'll need a valid API key to initialize the LLM. See [Quickstart](../quickstart.md) for more details on how to set up your API key.

`TextGradOptimizer` allows the use of different LLMs for workflow execution and optimization. For example, we can use GPT 4o-mini for workflow execution and GPT 4o for optimization.

```python
executor_config = OpenAILLMConfig(model="gpt-4o-mini", openai_key="your_openai_api_key")
executor_llm = OpenAILLM(config=executor_config)

optimizer_config = OpenAILLMConfig(model="gpt-4o", openai_key="your_openai_api_key")
optimizer_llm = OpenAILLM(config=optimizer_config)
```

## 3. Setting Up the Components

### Step 1: Initialize the Workflow
`TextGradOptimizer` only supports `SequentialWorkFlowGraph` and a specific variant of `WorkFlowGraph`. The workflow graph must have exactly one agent per node and each agent must only have one action. See [Workflow Graph](../modules/workflow_graph.md) for more information on `SequentialWorkFlowGraph` and `WorkFlowGraph`. For this example, let us create the
simplest workflow with only a single node.

```python
math_graph_data = {
    "goal": r"Answer the math question. The answer should be in box format, e.g., \boxed{123}",
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
            "prompt_template": StringTemplate(instruction="Answer the math question. The answer should be in box format, e.g., \\boxed{{123}}"),
            "parse_mode": "str"
        }
    ] 
}

workflow_graph = SequentialWorkFlowGraph.from_dict(math_graph_data)
```

`TextGradOptimizer` requires each agent be configured with a prompt template, rather than specifying the prompt using a string. This allows for a clear separation between the part of the prompt intended for optimization (i.e. instruction) and those that should remain unchanged (e.g. context, demonstrations). For more information on prompt templates, see [Prompt Template](../modules/prompt_template.md).


### Step 2: Prepare the dataset

For this tutorial, we will use the MATH dataset which consists of challenging competition mathematics problems,
spanning various difficulty levels and subject areas. The dataset is split into 7.5K training problems and 5K test problems. For demonstration purpose, let's take a smaller subset of the dataset to speed up the validation and evaluation process.

```python
class MathSplits(MATH):
    def _load_data(self):
        super()._load_data()
        import numpy as np 
        np.random.seed(42)
        permutation = np.random.permutation(len(self._test_data))
        full_test_data = self._test_data
        # randomly select 10 samples for train, 40 for dev and 100 for test
        self._train_data = [full_test_data[idx] for idx in permutation[:10]]
        self._dev_data = [full_test_data[idx] for idx in permutation[10:50]]
        self._test_data = [full_test_data[idx] for idx in permutation[50:150]]

math_splits = MathSplits()
```

During optimization, the `TextGradOptimizer` will evaluate the performance on the development set by default. Please make sure the dataset has a development set properly set up (i.e., `benchmark._dev_data` is not None). You can either:
   - Use a dataset that already provides a development set
   - Split your dataset to create a development set (like in the example above)
   - Implement a custom dataset (inherits from `evoagentx.benchmark.Benchmark`) that properly sets up the development set. 


### Step 3: Set Up the Evaluator

The evaluator is responsible for assessing the performance of the workflow during optimization. For more detailed information about how to set up and use the evaluator, please refer to the [Benchmark and Evaluation Tutorial](./benchmark_and_evaluation.md).


```python
def collate_func(example: dict) -> dict:
    return {"problem": example["problem"]}

evaluator = Evaluator(
    llm=llm, 
    agent_manager=agent_manager, 
    collate_func=collate_func, 
    num_workers=5, 
    verbose=True
)
```

## 4. Configuring and Running the TextGrad Optimizer

The TextGradOptimizer can be configured with various parameters to control the optimization process:

- `graph`: The workflow graph to optimize
- `optimize_mode`: The mode of optimization:
    * "all": optimize both instruction prompts and system prompts
    * "instruction": optimize only the instruction prompts
    * "system_prompt": optimize only the system prompts
- `executor_llm`: The LLM used to execute the workflow
- `optimizer_llm`: The LLM used to optimize the workflow
- `batch_size`: The batch size for optimization
- `max_steps`: The maximum number of optimization steps
- `evaluator`: The evaluator to perform evaluation during optimization.
- `eval_interval`: The number of steps between evaluations
- `eval_rounds`: The number of evaluation rounds
- `eval_config`: The evaluation configuration during optimization (passed to `TextGradOptimizer.evaluate()`). For example, if we don't want to evaluate on the entire development set, we can set  `eval_config = {"sample_k": 100}` to only evaluate on 100 random samples from the development set.
- `save_interval`: The number of steps between saving the workflow graph
- `save_path`: The path to save the workflow graph
- `rollback`: Whether to rollback to the best workflow graph during optimization
- `constraints`: An optional list of constraints for optimization. For example, "The system prompt must not exceed 100 words".


```python
textgrad_optimizer = TextGradOptimizer(
    graph=workflow_graph, 
    optimize_mode="all",
    executor_llm=executor_llm, 
    optimizer_llm=optimizer_llm,
    batch_size=3,
    max_steps=20,
    evaluator=evaluator,
    eval_every_n_steps=1,
    eval_rounds=1,
    save_interval=None,
    save_path="./",
    rollback=True,
    constraints=[]
)
```

### Running the Optimization

To start the optimization process:
```python
textgrad_optimizer.optimize(dataset=math_splits, seed=8)
```
The `seed` is used for shuffling the training data. The training data is automatically re-shuffled every epoch. If `seed` is
provided, the effective seed for shuffling the training data is `seed + epoch`.

The final graph at the end of the optimization is not necessarily the best graph. If you wish to restore the graph that performed best on the development set, simply call
```python
textgrad_optimizer.restore_best_graph()
```

We can evaluate the workflow again to see the improvement after optimization.
```python
with suppress_logger_info():
    result = textgrad_optimizer.evaluate(dataset=math_splits, eval_mode="test")
print(f"Evaluation result (after optimization):\n{result}")
```

`TextGradOptimizer` always saves the final workflow graph and the best workflow graph to `save_path`. It also saves graphs during optimization if `save_interval` is not `None`. You can also save the workflow graph manually by calling `textgrad_optimizer.save()`.


Note that `TextGradOptimizer` does not change the workflow structure but saving the workflow graph also saves the prompts and system prompts which will be different after optimization.
Below is an example of a saved workflow graph after optimization using `TextGradOptimizer`.


```json
{
    "class_name": "SequentialWorkFlowGraph",
    "goal": "Answer the math question. The answer should be in box format, e.g., \\boxed{123}",
    "tasks": [
        {
            "name": "answer_generate",
            "description": "Answer generation for Math.",
            "inputs": [
                {
                    "name": "problem",
                    "type": "str",
                    "description": "The problem to solve.",
                    "required": true
                }
            ],
            "outputs": [
                {
                    "name": "answer",
                    "type": "str",
                    "description": "The generated answer.",
                    "required": true
                }
            ],
            "prompt": null,
            "prompt_template": {
                "class_name": "StringTemplate",
                "instruction": "To solve the math problem, follow these steps:\n\n1. **Contextual Overview**: Begin with a brief overview of the problem-solving strategy, using logical reasoning and mathematical principles to derive the solution. Include any relevant geometric or algebraic insights.\n\n2. **Key Steps Identification**: Break down the problem-solving process into distinct parts:\n   - Identify the relevant mathematical operations and properties, such as symmetry, roots of unity, or trigonometric identities.\n   - Perform the necessary calculations, ensuring each step logically follows from the previous one.\n   - Present the final answer.\n\n3. **Conciseness and Clarity**: Provide a clear and concise explanation of your solution, avoiding unnecessary repetition. Use consistent formatting and notation throughout.\n\n4. **Mathematical Justification**: Explain the reasoning behind each step to ensure the solution is well-justified. Include explanations of reference angles, geometric interpretations, and any special conditions or edge cases.\n\n5. **Verification Step**: Include a quick verification step to confirm the accuracy of your calculations. Consider recalculating key values if initial assumptions were incorrect.\n\n6. **Visual Aids**: Where applicable, include diagrams or sketches to visually represent the problem and solution, enhancing understanding.\n\n7. **Final Answer Presentation**: Present the final answer clearly and ensure it is boxed, reflecting the correct solution. Verify that it aligns with the problem's requirements and any known correct solutions."
            },
            "system_prompt": "You are a math-focused assistant dedicated to providing clear, concise, and educational solutions to mathematical problems. Your goal is to deliver structured and pedagogically sound explanations, ensuring mathematical accuracy and logical reasoning. Begin with a brief overview of the problem-solving approach, followed by detailed calculations, and conclude with a verification step. Use precise mathematical notation and consider potential edge cases. Present the final answer clearly, using the specified format, and incorporate visual aids or analogies where appropriate to enhance understanding and engagement. \n\nExplicitly include geometric explanations when applicable, describing the geometric context and relationships. Emphasize the importance of visual aids, such as diagrams or sketches, to enhance understanding. Ensure consistency in formatting and mathematical notation. Provide a brief explanation of the reference angle concept and its significance. Include contextual explanations of trigonometric identities and their applications. Critically evaluate initial assumptions and verify geometric properties before proceeding. Highlight the use of symmetry and conjugate pairs in complex numbers. Encourage re-evaluation and verification of steps, ensuring logical flow and clarity. Focus on deriving the correct answer and consider problem-specific strategies or known techniques.",
            "parse_mode": "str",
            "parse_func": null,
            "parse_title": null
        }
    ]
}
```

For a complete working example, please refer to [examples/textgrad/math_textgrad.py](https://github.com/EvoAgentX/EvoAgentX/blob/main/examples/optimization/textgrad/math_textgrad.py). Additional TextGrad optimization scripts for other datasets (e.g., [`hotpotqa_textgrad.py`](https://github.com/EvoAgentX/EvoAgentX/blob/main/examples/optimization/textgrad/hotpotqa_textgrad.py) and [`mbqq_textgrad.py`](https://github.com/EvoAgentX/EvoAgentX/blob/main/examples/optimization/textgrad/mbpp_textgrad.py)) are available in the [examples/optimization/textgrad](https://github.com/EvoAgentX/EvoAgentX/tree/main/examples/optimization/textgrad) directory.