# TextGrad 优化器教程

本教程将指导您设置和运行 EvoAgentX 中的 TextGrad 优化器的过程。我们将使用 [MATH](https://www.modelscope.cn/datasets/opencompass/competition_math) 数据集作为示例，演示如何优化工作流中的提示词和系统提示词。

## 1. TextGrad
TextGrad 使用来自 LLM 的文本反馈来改进文本变量。在 EvoAgentX 中，我们使用 TextGrad 来优化代理的提示词和系统提示词。有关 TextGrad 的更多信息，请参阅他们的[论文](https://arxiv.org/abs/2406.07496)和 [GitHub](https://github.com/zou-group/textgrad)。

## 2. TextGrad 优化器
EvoAgentX 中的 TextGrad 优化器使您能够：

- 自动优化多代理工作流（提示词和/或系统提示词）
- 在数据集上评估优化结果

## 3. 设置环境

首先，让我们导入设置 TextGrad 优化器所需的模块：

```python
from evoagentx.benchmark import MATH
from evoagentx.optimizers import TextGradOptimizer
from evoagentx.models import OpenAILLMConfig, OpenAILLM
from evoagentx.workflow import SequentialWorkFlowGraph
from evoagentx.core.callbacks import suppress_logger_info
```

### 配置 LLM 模型
您需要有效的 API 密钥来初始化 LLM。有关如何设置 API 密钥的更多详细信息，请参阅[快速入门](../quickstart.md)。

`TextGradOptimizer` 允许在工作流执行和优化中使用不同的 LLM。例如，我们可以使用 GPT 4o-mini 进行工作流执行，使用 GPT 4o 进行优化。

```python
executor_config = OpenAILLMConfig(model="gpt-4o-mini", openai_key="your_openai_api_key")
executor_llm = OpenAILLM(config=executor_config)

optimizer_config = OpenAILLMConfig(model="gpt-4o", openai_key="your_openai_api_key")
optimizer_llm = OpenAILLM(config=optimizer_config)
```

## 3. 设置组件

### 步骤 1：初始化工作流
目前，`TextGradOptimizer` 仅支持 `SequentialWorkFlowGraph`。有关 `SequentialWorkFlowGraph` 的更多信息，请参阅[工作流图](../modules/workflow_graph.md)。对于此示例，让我们创建只有单个节点的最简单工作流。

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
            "prompt": "Answer the math question. The answer should be in box format, e.g., \\boxed{{123}}\n\nProblem: {problem}",
            "parse_mode": "str"
        }
    ] 
}

workflow_graph = SequentialWorkFlowGraph.from_dict(math_graph_data)
```

### 步骤 2：准备数据集

对于本教程，我们将使用 MATH 数据集，该数据集包含各种难度级别和主题领域的具有挑战性的竞赛数学问题。该数据集分为 7.5K 训练问题和 5K 测试问题。出于演示目的，让我们取数据集的一个较小子集，以加快验证和评估过程。

```python
class MathSplits(MATH):
    def _load_data(self):
        super()._load_data()
        import numpy as np 
        np.random.seed(42)
        permutation = np.random.permutation(len(self._test_data))
        full_test_data = self._test_data
        # 随机选择10个样本用于训练，40个用于开发，100个用于测试
        self._train_data = [full_test_data[idx] for idx in permutation[:10]]
        self._dev_data = [full_test_data[idx] for idx in permutation[10:50]]
        self._test_data = [full_test_data[idx] for idx in permutation[50:150]]

math_splits = MathSplits()
```

在优化过程中，`TextGradOptimizer` 默认会在开发集上评估性能。请确保数据集有一个正确设置的开发集（即 `benchmark._dev_data` 不为 None）。您可以：
   - 使用已经提供开发集的数据集
   - 拆分您的数据集以创建一个开发集（如上例所示）
   - 实现一个自定义数据集（继承自 `evoagentx.benchmark.Benchmark`），正确设置开发集

### 步骤 3：设置评估器

评估器负责在优化过程中评估工作流的性能。有关如何设置和使用评估器的更详细信息，请参考[基准测试和评估教程](./benchmark_and_evaluation.md)。

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

## 4. 配置和运行 TextGrad 优化器

TextGradOptimizer 可以通过各种参数配置来控制优化过程：

- `graph`：要优化的工作流图
- `optimize_mode`：优化模式：
    * "all"：优化提示词和系统提示词
    * "prompt"：仅优化提示词
    * "system_prompt"：仅优化系统提示词
- `executor_llm`：用于执行工作流的 LLM
- `optimizer_llm`：用于优化工作流的 LLM
- `batch_size`：优化的批量大小
- `max_steps`：最大优化步数
- `evaluator`：在优化期间执行评估的评估器
- `eval_interval`：评估之间的步数
- `eval_rounds`：评估轮数
- `eval_config`：优化期间的评估配置（传递给 `TextGradOptimizer.evaluate()`）。例如，如果我们不想在整个开发集上进行评估，可以设置 `eval_config = {"sample_k": 100}` 以仅在开发集中的 100 个随机样本上进行评估。
- `save_interval`：保存工作流图之间的步数
- `save_path`：保存工作流图的路径
- `rollback`：是否在优化过程中回滚到最佳工作流图

```python
textgrad_optimizer = TextGradOptimizer(
    graph=workflow_graph, 
    optimize_mode="all",
    executor_llm=executor_llm, 
    optimizer_llm=optimizer_llm,
    batch_size=3,
    max_steps=20,
    evaluator=evaluator,
    eval_interval=1,
    eval_rounds=1,
    save_interval=None,
    save_path="./",
    rollback=True
)
```

### 运行优化

要开始优化过程：
```python
textgrad_optimizer.optimize(dataset=math_splits, seed=8)
```
`seed` 用于随机打乱训练数据。训练数据每个轮次会自动重新打乱。如果提供了 `seed`，则用于打乱训练数据的有效种子为 `seed + epoch`。

优化结束时的最终图不一定是最好的图。如果您希望恢复在开发集上表现最好的图，只需调用
```python
textgrad_optimizer.restore_best_graph()
```

我们可以再次评估工作流，看看优化后的改进情况。
```python
with suppress_logger_info():
    result = textgrad_optimizer.evaluate(dataset=math_splits, eval_mode="test")
print(f"Evaluation result (after optimization):\n{result}")
```

`TextGradOptimizer` 始终将最终工作流图和最佳工作流图保存到 `save_path`。如果 `save_interval` 不为 `None`，它也会在优化过程中保存图。您也可以通过调用 `textgrad_optimizer.save()` 手动保存工作流图。

请注意，`TextGradOptimizer` 不会更改工作流结构，但保存工作流图也会保存在优化后会有所不同的提示词和系统提示词。
以下是使用 `TextGradOptimizer` 优化后保存的工作流图示例。

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
            "prompt": "To solve the math problem, follow these steps:\n\n1. **Contextual Overview**: Begin with a brief overview of the problem-solving strategy, using logical reasoning and mathematical principles to derive the solution. Include any relevant geometric or algebraic insights.\n\n2. **Key Steps Identification**: Break down the problem-solving process into distinct parts:\n   - Identify the relevant mathematical operations and properties, such as symmetry, roots of unity, or trigonometric identities.\n   - Perform the necessary calculations, ensuring each step logically follows from the previous one.\n   - Present the final answer.\n\n3. **Conciseness and Clarity**: Provide a clear and concise explanation of your solution, avoiding unnecessary repetition. Use consistent formatting and notation throughout.\n\n4. **Mathematical Justification**: Explain the reasoning behind each step to ensure the solution is well-justified. Include explanations of reference angles, geometric interpretations, and any special conditions or edge cases.\n\n5. **Verification Step**: Include a quick verification step to confirm the accuracy of your calculations. Consider recalculating key values if initial assumptions were incorrect.\n\n6. **Visual Aids**: Where applicable, include diagrams or sketches to visually represent the problem and solution, enhancing understanding.\n\n7. **Final Answer Presentation**: Present the final answer clearly and ensure it is boxed, reflecting the correct solution. Verify that it aligns with the problem's requirements and any known correct solutions.\n\nProblem: <input>{problem}</input>",
            "system_prompt": "You are a math-focused assistant dedicated to providing clear, concise, and educational solutions to mathematical problems. Your goal is to deliver structured and pedagogically sound explanations, ensuring mathematical accuracy and logical reasoning. Begin with a brief overview of the problem-solving approach, followed by detailed calculations, and conclude with a verification step. Use precise mathematical notation and consider potential edge cases. Present the final answer clearly, using the specified format, and incorporate visual aids or analogies where appropriate to enhance understanding and engagement. \n\nExplicitly include geometric explanations when applicable, describing the geometric context and relationships. Emphasize the importance of visual aids, such as diagrams or sketches, to enhance understanding. Ensure consistency in formatting and mathematical notation. Provide a brief explanation of the reference angle concept and its significance. Include contextual explanations of trigonometric identities and their applications. Critically evaluate initial assumptions and verify geometric properties before proceeding. Highlight the use of symmetry and conjugate pairs in complex numbers. Encourage re-evaluation and verification of steps, ensuring logical flow and clarity. Focus on deriving the correct answer and consider problem-specific strategies or known techniques.",
            "parse_mode": "str",
            "parse_func": null,
            "parse_title": null
        }
    ]
}
```

完整的工作示例，请参阅 [examples/textgrad/math_textgrad.py](https://github.com/EvoAgentX/EvoAgentX/blob/main/examples/optimization/textgrad/math_textgrad.py)。其他数据集的 TextGrad 优化脚本（例如，[`hotpotqa_textgrad.py`](https://github.com/EvoAgentX/EvoAgentX/blob/main/examples/optimization/textgrad/hotpotqa_textgrad.py) 和 [`mbqq_textgrad.py`](https://github.com/EvoAgentX/EvoAgentX/blob/main/examples/optimization/textgrad/mbpp_textgrad.py)）可以在 [examples/optimization/textgrad](https://github.com/EvoAgentX/EvoAgentX/tree/main/examples/optimization/textgrad) 目录中找到。