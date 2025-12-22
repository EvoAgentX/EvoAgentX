# AFlow 优化器教程

本教程将指导你如何使用 EvoAgentX 的 AFlow 优化器来优化你的工作流。AFlow 优化器是一个强大的工具，可以帮助你自动优化工作流的性能。

## 1. 概述

AFlow 优化器是 EvoAgentX 框架中的一个重要组件，它提供了以下功能：

- 自动优化工作流的性能
- 支持多种优化策略
- 提供详细的优化报告
- 支持自定义优化目标

## 2. 设置 AFlow 优化器

首先，你需要导入相关模块并设置 AFlow 优化器。

```python
from evoagentx.optimizers import AFlowOptimizer
from evoagentx.config import Config
from evoagentx.models import OpenAIConfig, OpenAI
```

### 配置 LLM 模型
你需要一个有效的 OpenAI API 密钥来初始化 LLM。建议将 API 密钥保存在 `.env` 文件中，并使用 `load_dotenv` 函数加载它：
```python
import os
from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
llm_config = OpenAIConfig(model="gpt-4", openai_key=OPENAI_API_KEY)
llm = OpenAI(config=llm_config)
```

## 3. 设置组件

### 第一步：定义任务配置

AFlow 优化器需要一个配置来指定任务类型和可用的操作符。以下是不同任务类型的示例配置：

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

### 第二步：定义初始工作流

AFlow 优化器需要两个文件：
- `graph.py`：该文件用 Python 代码定义初始工作流图。
- `prompt.py`：该文件定义工作流中使用的提示。

以下是 HumanEval 基准的 `graph.py` 文件示例：

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
        工作流的实现
        Custom 操作符可以生成任何你想要的内容。
        但当你想获取标准代码时，应该使用 custom_code_generate 操作符。
        """
        # await self.custom(input=, instruction="")
        solution = await self.custom_code_generate(problem=problem, entry_point=entry_point, instruction=prompt_custom.GENERATE_PYTHON_CODE_PROMPT) # 但当你想获取标准代码时，应该使用 customcodegenerator
        return solution['response']
```

!!! 注意
    在定义工作流时，请注意以下关键点：

    1. **提示导入路径**：确保正确指定 `prompt.py` 的导入路径（例如 `examples.aflow.code_generation.prompt`）。此路径应该与你的项目结构匹配，以便正确加载提示。

    2. **操作符初始化**：在 `__init__` 函数中，你必须初始化工作流中将使用的所有操作符。每个操作符都应该使用适当的 LLM 实例进行实例化。

    3. **工作流执行**：`__call__` 函数是工作流执行的主要入口点。它应该定义工作流的完整执行逻辑，并返回将用于评估的最终输出。


以下是 HumanEval 基准的 `prompt.py` 文件示例：

```python
GENERATE_PYTHON_CODE_PROMPT = """
Generate a functional and correct Python code for the given problem.

Problem: """
```

!!! 注意 
    如果工作流不需要任何提示，则 `prompt.py` 文件可以为空。

### 第三步：准备基准

在本教程中，我们将使用 AFlowHumanEval 基准。它遵循与 [原始 AFlow 实现](https://github.com/FoundationAgents/MetaGPT/tree/main/examples/aflow) 中相同的数据划分和格式。

```python
# Initialize the benchmark
humaneval = AFlowHumanEval()
```

## 4. 配置和运行 AFlow 优化器

AFlow 优化器可以通过各种参数进行配置，以控制优化过程：

```python
optimizer = AFlowOptimizer(
    graph_path="examples/aflow/code_generation",  # 初始工作流图的路径
    optimized_path="examples/aflow/humaneval/optimized",  # 保存优化工作流的路径
    optimizer_llm=optimizer_llm,  # 用于优化的 LLM
    executor_llm=executor_llm,    # 用于执行的 LLM
    validation_rounds=3,          # 优化期间在开发集上运行验证的次数
    eval_rounds=3,               # 测试期间在测试集上运行评估的次数
    max_rounds=20,               # 最大优化轮数
    **EXPERIMENTAL_CONFIG["humaneval"]  # 特定任务的配置，用于指定任务类型和可用操作符
)
```

### 运行优化

要开始优化过程：

```python
# 优化工作流
optimizer.optimize(humaneval)
```

!!! 注意 
    在优化过程中，工作流将在每一步骤上对开发集进行 `validation_rounds` 次验证。确保基准 `humaneval` 包含开发集（即 `self._dev_data` 不为空）。

### 测试优化后的工作流

要测试优化后的工作流：

```python
# 测试优化后的工作流
optimizer.test(humaneval)
```
默认情况下，优化器将选择验证性能最高的工作流进行测试。你还可以使用 `test_rounds: List[int]` 参数指定测试轮次。例如，要评估第二轮和第三轮，可以使用 `optimizer.test(humaneval, test_rounds=[2, 3])`。

!!! 注意 
    在测试期间，工作流将在测试集上进行 `eval_rounds` 次评估。确保基准 `humaneval` 包含测试集（即 `self._test_data` 不为空）。

有关完整的工作示例，请参阅 [aflow_humaneval.py](https://github.com/EvoAgentX/EvoAgentX/blob/main/examples/optimization/aflow/aflow_humaneval.py)。