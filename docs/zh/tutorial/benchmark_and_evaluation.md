# 基准测试和评估教程

本教程将指导你使用 EvoAgentX 设置和运行基准测试评估。我们将使用 HotpotQA 数据集作为示例，演示如何设置和运行评估过程。

## 1. 概述

EvoAgentX 提供了一个灵活且模块化的评估框架，使你能够：

- 加载和使用预定义的基准数据集
- 自定义数据加载、处理和后期处理逻辑
- 评估多代理工作流的性能
- 并行处理多个评估任务

## 2. 设置基准测试

首先，你需要导入相关模块并设置评估过程中代理将使用的语言模型（LLM）。

```python
from evoagentx.config import Config
from evoagentx.models import OpenAIConfig, OpenAI 
from evoagentx.benchmark import HotpotQA
from evoagentx.workflow import QAActionGraph 
from evoagentx.evaluators import Evaluator 
from evoagentx.core.callbacks import suppress_logger_info
```

### 配置 LLM 模型
你需要一个有效的 OpenAI API 密钥来初始化 LLM。建议将 API 密钥保存在 `.env` 文件中，并使用 `load_dotenv` 函数加载它：
```python 
import os 
from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
llm_config = OpenAILLMConfig(model="gpt-4o-mini", openai_key=OPENAI_API_KEY)
llm = OpenAILLM(config=llm_config)
```

## 3. 初始化基准测试
EvoAgentX 包含多个预定义的基准测试，用于问答、数学和编码等任务。有关现有基准测试的更多详细信息，请参阅 [Benchmark README](https://github.com/EvoAgentX/EvoAgentX/blob/main/evoagentx/benchmark/README.md)。你还可以通过扩展基础 `Benchmark` 接口来定义自己的基准测试类，我们在 [自定义基准测试](#custom-benchmark) 部分提供了一个示例。

在这个示例中，我们将使用 `HotpotQA` 基准测试。
```python 
benchmark = HotPotQA(mode="dev")
```
其中 `mode` 参数决定加载数据集的哪个部分。选项包括：

* `"train"`：训练数据
* `"dev"`：开发/验证数据
* `"test"`：测试数据
* `"all"`（默认）：加载整个数据集

数据将自动下载到默认缓存文件夹，但你可以通过指定 `path` 参数来更改此位置。

## 4. 运行评估
一旦你准备好了基准测试和 LLM，下一步就是定义你的代理工作流和评估逻辑。EvoAgentX 支持完全自定义基准测试示例的处理方式和输出的解释方式。

以下是如何使用 `HotpotQA` 基准测试和 QA 工作流运行评估。

### 步骤 1：定义代理工作流
你可以使用预定义的工作流之一或实现自己的工作流。在这个示例中，我们使用为问答设计的 [`QAActionGraph`](https://github.com/EvoAgentX/EvoAgentX/blob/main/evoagentx/workflow/action_graph.py#L99)，它简单地使用自一致性来生成最终答案：

```python
workflow = QAActionGraph(
    llm_config=llm_config,
    description="This workflow aims to address multi-hop QA tasks."
)
``` 

### 步骤 2：自定义数据预处理和后处理

评估的下一个关键方面是正确地在基准测试、工作流和评估指标之间转换数据。

### 为什么需要预处理和后处理

在 EvoAgentX 中，**预处理**和**后处理**是确保基准测试数据、工作流和评估逻辑之间顺畅交互的重要步骤：

- **预处理（`collate_func`）**：  

    来自 HotpotQA 等基准测试的原始示例通常包含结构化字段，如问题、答案和上下文。但是，你的代理工作流通常期望一个单一的提示字符串或其他结构化输入。`collate_func` 用于将每个原始示例转换为你的（自定义）工作流可以使用的格式。

- **后处理（`output_postprocess_func`）**：

    工作流输出可能包括推理步骤或超出最终答案的额外格式。由于 `Evaluator` 内部调用基准测试的 `evaluate` 方法来计算指标（例如，精确匹配或 F1），通常需要以干净的格式提取最终答案。`output_postprocess_func` 处理这一点，确保输出适合评估。

简而言之，**预处理为工作流准备基准测试示例**，而**后处理为评估准备工作流输出**。

在以下示例中，我们定义了一个 `collate_func` 来将原始示例格式化为工作流的提示，以及一个 `output_postprocess_func` 来从工作流输出中提取最终答案。

可以使用 `collate_func` 格式化基准测试中的每个示例，它将原始示例转换为代理的提示或结构化输入。

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

在代理生成输出后，你可以定义如何使用 `output_postprocess_func` 提取最终答案。
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

### 步骤 3：初始化评估器
评估器将所有内容联系在一起——它在基准测试上运行工作流并计算性能指标。

```python
evaluator = Evaluator(
    llm=llm, 
    collate_func=collate_func,
    output_postprocess_func=output_postprocess_func,
    verbose=True, 
    num_workers=3 
)
``` 
如果 `num_workers` 大于 1，评估将在多个线程上并行进行。

### 步骤 4：运行评估
现在，你可以通过向评估器提供工作流和基准测试来运行评估：

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
其中 `suppress_logger_info` 用于抑制日志信息。

有关完整示例，请参考 [基准测试和评估示例](https://github.com/EvoAgentX/EvoAgentX/blob/main/examples/benchmark_and_evaluation.py)。


## 自定义基准测试

要定义自定义基准测试，你需要扩展 `Benchmark` 类并实现以下方法：

- `_load_data(self)`： 
    
    加载基准测试数据，并设置 `self._train_data`、`self._dev_data` 和 `self._test_data` 属性。

- `_get_id(self, example: Any) -> Any`： 

    返回示例的唯一标识符。

- `_get_label(self, example: Any) -> Any`：

    返回与给定示例关联的标签或真实值。

    这用于在评估期间将预测与正确答案进行比较。输出将直接传递给 `evaluate` 方法。

- `evaluate(self, prediction: Any, label: Any) -> dict`： 

    基于预测和真实标签（从 `_get_label` 获取）计算单个示例的评估指标。
    此方法应返回指标名称和值的字典。

- `evaluate(self, prediction: Any, label: Any) -> dict`: 

    Compute the evaluation metrics for a single example, based on its prediction and ground-truth label (obtained from `_get_label`).
    This method should return a dictionary of metric name(s) and value(s).

For a complete example of a benchmark implementation, please refer to the [HotPotQA](https://github.com/EvoAgentX/EvoAgentX/blob/main/evoagentx/benchmark/hotpotqa.py#L23) class.
