# 评估器

## 简介

`Evaluator` 类是 EvoAgentX 框架中的一个基础组件，用于评估工作流和动作图在基准测试上的性能。它提供了一种结构化的方式来衡量 AI 代理在特定任务上的表现，通过运行测试数据并计算指标。

## 架构

### 评估器架构

`Evaluator` 由几个关键组件组成：

1. **LLM 实例**：
   
    用于在评估期间执行工作流的语言模型：

    - 提供工作流执行所需的推理和生成能力
    - 可以是任何遵循 `BaseLLM` 接口的实现

2. **代理管理器**：
   
    管理评估期间工作流图使用的代理：

    - 提供工作流执行所需的代理访问
    - 仅在评估 `WorkFlowGraph` 实例时需要，评估 `ActionGraph` 实例时可以忽略

3. **数据处理函数**：
   
    在评估期间准备和处理数据的函数：

    - `collate_func`：为工作流执行准备基准测试示例
    - `output_postprocess_func`：在评估前处理工作流输出

### 评估流程

评估流程遵循以下步骤：

1. **数据处理**：从基准测试数据集中获取示例，并将其处理成工作流图或动作图期望的格式
2. **工作流执行**：通过工作流图或动作图运行每个示例
3. **输出处理**：将输出处理成基准测试期望的格式
4. **指标计算**：通过比较输出与真实值来计算性能指标
5. **结果聚合**：将单个指标聚合成整体性能分数

## 使用方法

### 基本评估器创建与执行

```python
from evoagentx.evaluators import Evaluator
from evoagentx.models import OpenAILLMConfig, OpenAILLM
from evoagentx.agents import AgentManager
from evoagentx.workflow.workflow_graph import WorkFlowGraph
from evoagentx.benchmark import SomeBenchmark
from evoagentx.core.callbacks import suppress_logger_info

# 初始化 LLM
llm_config = OpenAILLMConfig(model="gpt-4o-mini", openai_key="xxx")
llm = OpenAILLM(llm_config)

# 初始化代理管理器
agent_manager = AgentManager()

# 加载工作流图
workflow_graph = WorkFlowGraph.from_file("path/to/workflow.json")

# 将代理添加到代理管理器
agent_manager.add_agents_from_workflow(workflow_graph, llm_config=llm_config)

# 创建基准测试
benchmark = SomeBenchmark()

# 创建评估器
evaluator = Evaluator(
    llm=llm,
    agent_manager=agent_manager,
    num_workers=4,  # 使用 4 个并行工作器
    verbose=True    # 显示进度条
)

# 运行评估并抑制日志
with suppress_logger_info():
    results = evaluator.evaluate(
        graph=workflow_graph,
        benchmark=benchmark,
        eval_mode="test",    # 在测试集上评估（默认）
        sample_k=100         # 使用 100 个随机示例
    )

print(f"评估结果: {results}")
```

### 自定义数据处理

```python
from evoagentx.evaluators import Evaluator
from evoagentx.models import OpenAILLMConfig, OpenAILLM
from evoagentx.agents import AgentManager
from evoagentx.core.callbacks import suppress_logger_info

# 自定义整理函数来准备输入。键应该匹配工作流图或动作图的输入参数。返回值将直接传递给工作流图或动作图的 `execute` 方法。
def custom_collate(example):
    return {
        "input_text": example["question"],
        "context": example.get("context", "")
    }

# 自定义输出处理，`output` 是工作流的输出，返回值将传递给基准测试的 `evaluate` 方法。
def custom_postprocess(output):
    if isinstance(output, dict):
        return output.get("answer", "")
    return output

# 使用自定义函数创建评估器
evaluator = Evaluator(
    llm=llm,
    agent_manager=agent_manager,
    collate_func=custom_collate,
    output_postprocess_func=custom_postprocess,
    num_workers=4,  # 使用 4 个并行工作器
    verbose=True    # 显示进度条
)
```

### 评估动作图

```python
from evoagentx.workflow.action_graph import ActionGraph
from evoagentx.evaluators import Evaluator
from evoagentx.models import OpenAILLMConfig, OpenAILLM
from evoagentx.core.callbacks import suppress_logger_info

# 初始化 LLM
llm_config = OpenAILLMConfig(model="gpt-4o-mini", openai_key="xxx")
llm = OpenAILLM(llm_config)

# 加载动作图
action_graph = ActionGraph.from_file("path/to/action_graph.json", llm_config=llm_config)

# 创建评估器（动作图不需要 agent_manager）
evaluator = Evaluator(llm=llm, num_workers=4, verbose=True)

# 运行评估并抑制日志
with suppress_logger_info():
    results = evaluator.evaluate(
        graph=action_graph,
        benchmark=benchmark
    )
```

### 异步评估

```python
import asyncio
from evoagentx.evaluators import Evaluator
from evoagentx.models import OpenAILLMConfig, OpenAILLM
from evoagentx.agents import AgentManager
from evoagentx.workflow.workflow_graph import WorkFlowGraph
from evoagentx.benchmark import SomeBenchmark
from evoagentx.core.callbacks import suppress_logger_info

# 初始化 LLM 和组件
llm_config = OpenAILLMConfig(model="gpt-4o-mini", openai_key="xxx")
llm = OpenAILLM(llm_config)
agent_manager = AgentManager()
workflow_graph = WorkFlowGraph.from_file("path/to/workflow.json")
benchmark = SomeBenchmark()

# 创建评估器
evaluator = Evaluator(
    llm=llm,
    agent_manager=agent_manager,
    num_workers=4
)

# 运行异步评估
async def run_async_eval():
    with suppress_logger_info():
        results = await evaluator.async_evaluate(
            graph=workflow_graph,
            benchmark=benchmark
        )
    return results

# 执行异步评估
results = asyncio.run(run_async_eval())
```

### 访问评估记录

```python
from evoagentx.evaluators import Evaluator
from evoagentx.models import OpenAILLMConfig, OpenAILLM
from evoagentx.benchmark import SomeBenchmark
from evoagentx.core.callbacks import suppress_logger_info

# 初始化组件
llm_config = OpenAILLMConfig(model="gpt-4o-mini", openai_key="xxx")
llm = OpenAILLM(llm_config)
benchmark = SomeBenchmark()
evaluator = Evaluator(llm=llm)

# 运行评估并抑制日志
with suppress_logger_info():
    evaluator.evaluate(graph=graph, benchmark=benchmark)

# 获取所有评估记录
all_records = evaluator.get_all_evaluation_records()

# 获取特定示例的记录
example = benchmark.get_test_data()[0]
record = evaluator.get_example_evaluation_record(benchmark, example)

# 通过示例 ID 获取记录
record_by_id = evaluator.get_evaluation_record_by_id(
    benchmark=benchmark,
    example_id="example-123",
    eval_mode="test"
)

# 访问工作流图评估的轨迹
if "trajectory" in record:
    for message in record["trajectory"]:
        print(f"{message.role}: {message.content}")
```

`Evaluator` 类提供了一种强大的方式来评估工作流和动作图的性能，使 EvoAgentX 框架中的定量比较和改进跟踪成为可能。
