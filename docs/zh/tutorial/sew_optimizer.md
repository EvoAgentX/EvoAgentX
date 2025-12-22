# SEW优化器教程

本教程将指导您设置和运行EvoAgentX中的SEW（Self-Evolving Workflow，自进化工作流）优化器。我们将使用HumanEval基准作为示例，演示如何优化多智能体工作流。

## 1. 概述

SEW优化器是EvoAgentX中的强大工具，它使您能够：

- 自动优化多智能体工作流（提示词和工作流结构）
- 在基准数据集上评估优化结果
- 支持不同的工作流表示方案（Python、Yaml、BPMN等）

## 2. 设置环境

首先，让我们导入设置SEW优化器所需的必要模块：

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

### 配置LLM模型

与EvoAgentX中的其他组件类似，您需要一个有效的OpenAI API密钥来初始化LLM。

```python
llm_config = OpenAILLMConfig(model="gpt-4o-mini", openai_key=OPENAI_API_KEY)
llm = OpenAILLM(config=llm_config)
```

## 3. 设置组件

### 步骤1：初始化SEW工作流

SEW工作流是将被优化的核心组件。它代表一个顺序工作流，旨在解决代码生成任务。

```python
sew_graph = SEWWorkFlowGraph(llm_config=llm_config)
agent_manager = AgentManager()
agent_manager.add_agents_from_workflow(sew_graph)
```

### 步骤2：准备基准测试

对于本教程，我们将使用HumanEval基准的修改版本，该版本将测试数据分为开发集和测试集：

```python
class HumanEvalSplits(HumanEval):
    def _load_data(self):
        # 加载原始测试数据
        super()._load_data()
        # 将数据分为开发集和测试集
        import numpy as np 
        np.random.seed(42)
        num_dev_samples = int(len(self._test_data) * 0.2)
        random_indices = np.random.permutation(len(self._test_data))
        self._dev_data = [self._test_data[i] for i in random_indices[:num_dev_samples]]
        self._test_data = [self._test_data[i] for i in random_indices[num_dev_samples:]]

# 初始化基准
humaneval = HumanEvalSplits()
```

SEWOptimizer默认会在开发集上评估性能。请确保基准测试正确设置了开发集。您可以：
   - 使用已经提供开发集的基准（如HotPotQA）
   - 将数据集分为开发集和测试集（如上面HumanEvalSplits示例所示）
   - 实现带有开发集支持的自定义基准

### 步骤3：设置评估器

评估器负责在优化过程中评估工作流的性能。有关如何设置和使用评估器的更详细信息，请参阅[基准和评估教程](./benchmark_and_evaluation.md)。

```python
def collate_func(example: dict) -> dict:
    # 将原始示例转换为SEW工作流的预期输入
    return {"question": example["prompt"]}

evaluator = Evaluator(
    llm=llm, 
    agent_manager=agent_manager, 
    collate_func=collate_func, 
    num_workers=5, 
    verbose=True
)
```

## 4. 配置和运行SEW优化器

SEW优化器可以通过各种参数配置，以控制优化过程：

```python
optimizer = SEWOptimizer(
    graph=sew_graph,           # 要优化的工作流图
    evaluator=evaluator,       # 用于性能评估的评估器
    llm=llm,                   # 语言模型
    max_steps=10,              # 最大优化步骤数
    eval_rounds=1,             # 每步评估轮数
    repr_scheme="python",      # 工作流的表示方案
    optimize_mode="prompt",    # 要优化的方面（提示/结构/全部）
    order="zero-order"         # 优化算法顺序（零阶/一阶）
)
```

### 运行优化

要启动优化过程：

```python
# 优化SEW工作流
optimizer.optimize(dataset=humaneval)

# 评估优化后的工作流
with suppress_logger_info():
    metrics = optimizer.evaluate(dataset=humaneval, eval_mode="test")
print("Evaluation metrics: ", metrics)

# 保存优化后的SEW工作流
optimizer.save("debug/optimized_sew_workflow.json")
```

有关完整的工作示例，请参阅[sew_optimizer.py](https://github.com/EvoAgentX/EvoAgentX/blob/main/examples/sew_optimizer.py)。
