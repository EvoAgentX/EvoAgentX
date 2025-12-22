# 基准测试

## 基准测试概述

EvoAgentX 提供了一套基准测试工具，用于评估不同的基于代理的系统。以下是当前包含的基准测试及其基本数据集统计信息：

| 任务                      | 数据集名称      | 训练集数量 | 验证集数量 | 测试集数量 |
| ------------------------- | --------------- | --------- | ------- | ------ |
| 问答                      | NQ              | 79,168    | 8,757   | 3,610  |
| 多跳问答                  | HotPotQA        | 90,447    | 7,405   | /      |
| 数学                      | GSM8K           | 7,473     | /       | 1,319  |
| 数学                      | MATH            | 7,500     | /       | 5,000  |
| 代码生成                  | HumanEval       | /         | /       | 164    |
| 代码生成                  | MBPP            | /         | /       | 427    |
| 代码生成                  | LiveCodeBench(v1~v5) | /         | /       | 400~880    |
| 代码执行                  | LiveCodeBench   | /         | /       | 479      |
| 测试输出预测              | LiveCodeBench   | /         | /       | 442      |

我们的框架提供了自动数据集下载功能，所有基准测试都内置了评估方法。该框架设计允许用户轻松加载、使用和评估各种任务的数据集，而无需手动处理数据下载和评估逻辑。
所有数据集在首次使用时都会自动下载到默认路径（~/.evoagentx/data/），或者用户可以通过 `path` 参数指定自定义路径。每个基准测试类都实现了标准化接口，包括数据加载、标签检索和预测评估的方法。

下面，我们介绍每个基准测试的预处理步骤和评估指标。

- [问答](#question-answering)
    - [NQ](#nq)
    - [HotPotQA](#hotpotqa)
- [数学](#math)
    - [GSM8K](#gsm8k)
    - [MATH](#math)
- [代码生成](#code-generation)
    - [HumanEval](#humaneval)
    - [MBPP](#mbpp)
    - [LiveCodeBench](#livecodebench)

## 预处理和评估指标

### 问答

对于问答数据集，我们默认使用精确匹配（EM）、F1 分数和准确率（ACC）作为评估指标。EM 要求预测答案与真实答案完全一致。ACC 要求预测答案包含真实答案，这在 LLM 用于生成答案时特别有用。

#### NQ
[Natural Questions (NQ)](https://github.com/google-research-datasets/natural-questions) 包含来自谷歌搜索引擎的问题，答案由人工标注者标注，是维基百科前 5 个搜索结果页面中的段落或实体。我们使用 [DPR](https://github.com/facebookresearch/DPR) 仓库提供的数据集划分，包含 79,168 个训练样本、8,757 个开发样本和 3,610 个测试样本。

您可以使用以下代码加载数据集：
```python
from evoagentx.benchmark import NQ
nq_dataset = NQ() # 可选：path="/path/to/save_data"
test_data = nq_dataset.get_test_data()
```
数据集中的每个样本格式如下：
```json
{
    "id": "test-1", 
    "question": "问题", 
    "answers": ["可能的答案"]
}
```

#### HotPotQA
[HotPotQA](https://hotpotqa.github.io/) 是一个多跳问答数据集，需要多步推理来回答问题。我们使用数据集的干扰项设置。每个样本包含一个问题、一个答案、一些包含支持信息和干扰信息的上下文，以及支持事实。我们只包含训练集和开发集，因为测试集不公开。

您可以使用以下代码加载数据集：
```python
from evoagentx.benchmark import HotPotQA
hotpotqa_dataset = HotPotQA() # 可选：path="/path/to/save_data"
test_data = hotpotqa_dataset.get_test_data()
```
数据集中的每个样本格式如下，其中 supporting_fact 的第二个元素（整数）是支持答案的上下文句子索引：
```json
{
        "_id": "样本ID", 
        "question": "问题", 
        "answer": "答案", 
        "context": [["上下文标题", ["上下文句子", "另一个句子"]]],
        "supporting_facts": [["支持标题", 0]]
    }
```

### 数学

对于数学数据集，我们使用解决率作为评估指标。解决率是正确解决的样本数量与总样本数量的比率。

#### GSM8K
[GSM8K](https://github.com/openai/grade-school-math) 由人工问题编写者创建的高质量小学数学问题组成。这些问题需要多步数学推理来解决。我们使用原始仓库提供的数据集划分，包含 7.5K 个训练问题和 1K 个测试问题。

您可以使用以下代码加载数据集：
```python
from evoagentx.benchmark import GSM8K
gsm8k_dataset = GSM8K() # 可选：path="/path/to/save_data"
test_data = gsm8k_dataset.get_test_data()
```
数据集中的每个样本格式如下：
```json
{
    "id": "test-1", 
    "question": "问题", 
    "answer": "答案"
}
```

#### MATH
[Mathematics Aptitude Test of Heuristics (MATH)](https://github.com/hendrycks/math) 数据集包含来自数学竞赛的问题，包括 AMC 10、AMC 12、AIME 等。MATH 中的每个问题都有逐步解决方案。我们使用原始仓库提供的数据集划分，包含 7.5K 个训练问题和 5K 个测试问题。

您可以使用以下代码加载数据集：
```python
from evoagentx.benchmark import MATH
math_dataset = MATH() # 可选：path="/path/to/save_data"
test_data = math_dataset.get_test_data()
```
数据集中的每个样本格式如下。对于 `level` 字段，有效值为："Level 1"、"Level 2"、"Level 3"、"Level 4"、"Level 5" 和 "Level ?"。`type` 字段可以是以下之一："Geometry"、"Algebra"、"Intermediate Algebra"、"Counting & Probability"、"Precalculus"、"Number Theory" 或 "Prealgebra"。

```json
{
    "id": "test-1", 
    "problem": "问题", 
    "solution": "解决方案",
    "level": "Level 1",
    "type": "Algebra"
}
```

### 代码生成
对于代码生成基准测试，我们使用 pass@k 作为评估指标，其中 k 是每个问题的解决方案数量。默认情况下，k 设置为 1。

#### HumanEval
[HumanEval](https://github.com/openai/human-eval) 是一个包含 164 个编程问题的数据集，来自 HumanEval 基准测试。每个问题包含一个函数签名、一个规范解决方案和一组单元测试。

您可以使用以下代码加载数据集：
```python
from evoagentx.benchmark import HumanEval
humaneval_dataset = HumanEval() # 可选：path="/path/to/save_data"
test_data = humaneval_dataset.get_test_data()
```
数据集中的每个样本格式如下：
```json
{
    "task_id": "HumanEval/0", 
    "prompt": "问题提示", 
    "entry_point": "要测试的函数名称",
    "canonical_solution": "问题的规范解决方案", 
    "test": "问题的单元测试"
}
```

#### MBPP
[Mostly Basic Python Problems (MBPP)](https://github.com/google-research/google-research/tree/master/mbpp) 包含数百个入门级 Python 编程问题。每个问题包含任务描述、代码解决方案和 3 个自动化测试用例。我们使用 MBPP 数据集的[清理子集](https://github.com/google-research/google-research/blob/master/mbpp/sanitized-mbpp.json)，包含 427 个经过作者手动验证的问题。为了便于评估，我们将 MBPP 数据集转换为 HumanEval 格式。

您可以使用以下代码加载数据集：
```python
from evoagentx.benchmark import MBPP
mbpp_dataset = MBPP() # 可选：path="/path/to/save_data"
test_data = mbpp_dataset.get_test_data()
```
数据集中的每个样本格式如下，我们保留原始的 MBPP `task_id`：
```json
{
    "task_id": 2, 
    "prompt": "问题提示", 
    "entry_point": "要测试的函数名称",
    "canonical_solution": "问题的规范解决方案", 
    "test": "问题的单元测试"
}
```
您还可以通过使用 `example["code"]` 访问原始 MBPP 属性，如 "code"、"test_list" 等。

#### LiveCodeBench
[LiveCodeBench](https://livecodebench.github.io/) 是一个无污染的 LLM 代码评估基准测试，它随时间不断收集新问题。特别是，LiveCodeBench 还关注更广泛的代码相关能力，如代码执行和测试输出预测，而不仅仅是代码生成。目前，LiveCodeBench 托管了 300 多个高质量编程问题，发布时间在 2023 年 5 月至 2024 年 2 月之间。

您可以使用以下代码加载数据集，其中 `scenario` 可以是 [`code_generation`、`test_output_prediction`、`code_execution`] 之一，表示不同的任务。`version` 表示代码生成数据集的不同版本，仅适用于 `code_generation` 场景，可以是 `["release_v1"、"release_v2"、"release_v3"、"release_v4"、"release_v5"、"release_latest"]` 之一。有关更多详细信息，请参阅 [LiveCodeBench](https://livecodebench.github.io/) 仓库。

```python
from evoagentx.benchmark import LiveCodeBench
livecodebench_dataset = LiveCodeBench(scenario="code_generation", version="release_v1") # 可选：path="/path/to/save_data"
test_data = livecodebench_dataset.get_test_data()
```
