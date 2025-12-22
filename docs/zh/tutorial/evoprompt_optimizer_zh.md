# EvoPrompt 优化器教程

本教程将指导您在 EvoAgentX 中设置和运行 EvoPrompt 优化器。我们将使用 BIG-Bench Hard 基准测试作为示例，展示如何通过遗传算法 (GA) 和差分进化 (DE) 算法来优化多代理工作流的提示。

## 1. 概述

EvoAgentX 中的 EvoPrompt 优化器使您能够：

- 使用进化算法自动优化多代理工作流中的提示
- 支持遗传算法 (GA) 和差分进化 (DE) 两种优化算法
- 在基准数据集上评估优化结果
- 支持多节点提示并行进化和组合优化
- 提供详细的训练过程可视化和日志记录

## 2. 性能表现

基于我们在 BIG-Bench Hard 数据集上的实验结果，EvoPrompt 优化器在多个任务上展现了显著的性能提升：

### 性能对比表

| 任务 | COT基线 | GA最佳 | DE最佳 | GA提升 | DE提升 |
|------|---------|---------|---------|---------|---------|
| **snarks** | 0.7109 | 0.8281 | 0.8281 | +16.5% | +16.5% |
| **geometric_shapes** | 0.3950 | 0.3700 | 0.4250 | -6.3% | +7.6% |
| **multistep_arithmetic_two** | 0.9450 | 0.9850 | 0.9750 | +4.2% | +3.2% |
| **ruin_names** | 0.5150 | 0.6850 | 0.7400 | +33.0% | +43.7% |

### 关键发现
- **ruin_names** 任务获得了最高的性能提升，DE算法提升达43.7%
- **snarks** 任务在两种算法下都获得了相同且优异的表现
- **multistep_arithmetic_two** 虽然基线最高，但仍取得了稳定的改进
- 训练过程图表可在相应的 `performance_summary_OVERALL.png` 文件中查看

## 3. 环境设置

首先，让我们导入设置 EvoPrompt 优化器所需的模块：

```python
import asyncio
import os
import re
from collections import Counter
from dotenv import load_dotenv

from evoagentx.optimizers.evoprompt_optimizer import DEOptimizer, GAOptimizer
from evoagentx.benchmark.bigbenchhard import BIGBenchHard
from evoagentx.models import OpenAILLM, OpenAILLMConfig
from evoagentx.optimizers.engine.registry import ParamRegistry
from evoagentx.core.logging import logger
```

### 配置 LLM 模型

您需要有效的 API 密钥来初始化 LLM。有关如何设置 API 密钥的详细信息，请参见 [快速开始](../quickstart.md)。

```python
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("在环境变量中未找到 OPENAI_API_KEY。")

# 进化用 LLM 配置（用于生成变异的提示）
evo_llm_config = OpenAILLMConfig(
    model="gpt-4.1-nano",
    openai_key=OPENAI_API_KEY,
    stream=False,
    top_p=0.95,
    temperature=0.5  # 较高的温度用于生成多样化的提示
)

# 评估用 LLM 配置（用于任务执行）
eval_llm_config = OpenAILLMConfig(
    model="gpt-4.1-nano",
    openai_key=OPENAI_API_KEY,
    stream=False,
    temperature=0  # 确定性评估
)
llm = OpenAILLM(config=eval_llm_config)
```

## 4. 设置组件

### 第1步：定义程序工作流

EvoPrompt 优化器需要一个程序类来定义工作流逻辑。以下是一个使用多提示投票机制的示例：

```python
class SarcasmClassifierProgram:
    """
    一个使用三提示多数投票集成来分类讽刺的程序。
    每个提示都是一个独立的"投票者"，可以独立进化。
    """
    def __init__(self, model: OpenAILLM):
        self.model = model
        # 三个不同的通用提示节点，用于多样化的任务处理
        self.prompt_direct = "As a straightforward responder, follow the task instruction exactly and provide the final answer."
        self.prompt_expert = "As an expert assistant, interpret the task instruction carefully and provide the final answer."
        self.prompt_cot = "As a thoughtful assistant, think step-by-step, then follow the task instruction and provide the final answer."
        self.task_instruction = "Respond with your final answer wrapped like this: FINAL_ANSWER(ANSWER)"

    def __call__(self, input: str) -> tuple[str, dict]:
        answers = []
        prompts = [self.prompt_direct, self.prompt_expert, self.prompt_cot]
        pattern = r"FINAL_ANSWER\((.*?)\)"

        for prompt in prompts:
            full_prompt = f"{prompt}\n\n{self.task_instruction}\n\nText:\n{input}"
            response = self.model.generate(prompt=full_prompt)
            prediction = response.content.strip()
            
            match = re.search(pattern, prediction, re.IGNORECASE)
            if match:
                answers.append(match.group(1))

        if not answers:
            return "N/A", {"votes": []}

        vote_counts = Counter(answers)
        most_common_answer = vote_counts.most_common(1)[0][0]
        
        return most_common_answer, {"votes": answers}

    def save(self, path: str):
        # 此处可添加保存逻辑
        pass

    def load(self, path: str):
        # 此处可添加加载逻辑
        pass
```

!!! note
    定义工作流时，请注意以下关键点：
    
    1. **多提示节点**：程序中定义的每个提示变量（如 `prompt_direct`、`prompt_expert`、`prompt_cot`）都可以作为独立的优化节点。
    
    2. **投票机制**：使用多提示投票可以提高结果的鲁棒性，每个提示节点独立进化。
    
    3. **结果解析**：确保 `__call__` 方法返回的格式与基准测试评估要求一致。

### 第2步：注册提示参数

使用 `ParamRegistry` 来跟踪需要优化的提示节点：

```python
# 设置基准测试和程序
benchmark = BIGBenchHard("snarks", dev_sample_num=15, seed=10)
program = SarcasmClassifierProgram(model=llm)

# 注册提示节点
registry = ParamRegistry()
registry.track(program, "prompt_direct", name="direct_prompt_node")
registry.track(program, "prompt_expert", name="expert_prompt_node")
registry.track(program, "prompt_cot", name="cot_prompt_node")
```

### 第3步：准备基准测试

我们使用 BIG-Bench Hard 基准测试，它包含各种具有挑战性的任务：

```python
# 可用的任务包括
tasks = [
    "snarks",                    # 讽刺识别
    "geometric_shapes",          # 几何形状识别
    "multistep_arithmetic_two",  # 多步算术
    "ruin_names",               # 损坏名称识别
    "sports_understanding",      # 体育理解
    "logical_deduction_three_objects",  # 逻辑推理
    # ... 更多任务
]

# 初始化基准测试
benchmark = BIGBenchHard(
    task_name="snarks", 
    dev_sample_num=15,  # 开发集样本数
    seed=10             # 随机种子确保结果可重现
)
```

## 5. 配置和运行 EvoPrompt 优化器

### 差分进化 (DE) 优化器

DE 算法通过差分变异和交叉操作来优化提示：

```python
# 配置参数
POPULATION_SIZE = 4           # 种群大小
ITERATIONS = 10              # 迭代次数
CONCURRENCY_LIMIT = 100      # 并发限制
COMBINATION_SAMPLE_SIZE = 3  # 每次组合的样本大小

# DE 优化器
optimizer_DE = DEOptimizer(
    registry=registry,
    program=program,
    population_size=POPULATION_SIZE,
    iterations=ITERATIONS,
    llm_config=evo_llm_config,
    concurrency_limit=CONCURRENCY_LIMIT,
    combination_sample_size=COMBINATION_SAMPLE_SIZE,
    enable_logging=True,         # 启用日志记录
    enable_early_stopping=True,  # 启用早停机制
    early_stopping_patience=3    # 早停容忍轮数
)

# 运行优化
logger.info("使用 DE 优化...")
await optimizer_DE.optimize(benchmark=benchmark)

# 评估结果
logger.info("使用 DE 评估...")
de_metrics = await optimizer_DE.evaluate(benchmark=benchmark, eval_mode="test")
logger.info(f"DE 结果: {de_metrics['accuracy']}")
```

### 遗传算法 (GA) 优化器

GA 算法通过选择、交叉和变异操作来优化提示：

```python
# GA 优化器
optimizer_GA = GAOptimizer(
    registry=registry,
    program=program,
    population_size=POPULATION_SIZE,
    iterations=ITERATIONS,
    llm_config=evo_llm_config,
    concurrency_limit=CONCURRENCY_LIMIT,
    combination_sample_size=COMBINATION_SAMPLE_SIZE,
    enable_logging=True,
    enable_early_stopping=True,
    early_stopping_patience=3
)

# 运行优化
logger.info("使用 GA 优化...")
await optimizer_GA.optimize(benchmark=benchmark)

# 评估结果
logger.info("使用 GA 评估...")
ga_metrics = await optimizer_GA.evaluate(benchmark=benchmark, eval_mode="test")
logger.info(f"GA 结果: {ga_metrics['accuracy']}")
```

## 6. 优化参数详解

### 关键参数说明

- `population_size`: 种群大小，决定每一代的个体数量
- `iterations`: 进化迭代次数
- `combination_sample_size`: 组合评估时的样本数量，用于减少计算开销
- `concurrency_limit`: 并发请求限制，控制对LLM API的并发调用
- `enable_early_stopping`: 是否启用早停机制，在性能不再提升时提前终止
- `early_stopping_patience`: 早停容忍轮数

### 算法特点对比

| 算法 | 特点 | 适用场景 |
|------|------|----------|
| **DE** | 基于差分变异，探索能力强 | 复杂优化景观，需要全局搜索 |
| **GA** | 基于选择和交叉，收敛稳定 | 已知较好解的局部优化 |

## 7. 日志和可视化

优化器会自动生成详细的日志文件和可视化图表：

### 日志文件结构
```
node_evolution_logs_{ALGO}_{MODEL}_{TASK}_{TIMESTAMP}/
├── combo_generation_XX_log.csv          # 每代组合评估日志
├── evaluation_testset_test_*.csv         # 测试集评估结果
├── optimization_summary_{algo}.csv       # 优化摘要
├── best_config.json                      # 机器可读的最优配置（自动保存）
├── performance_summary_OVERALL.png      # 训练过程可视化
└── individual_plots/                     # 单个节点性能图表
    └── performance_plot_*.png
```

### 可视化图表

优化器会生成训练过程的可视化图表，展示：
- 每代最佳性能和平均性能
- 各个提示节点的进化轨迹
- 组合性能的收敛曲线

## 8. 完整示例

以下是一个完整的运行示例：

```python
import asyncio
import os
from dotenv import load_dotenv
from evoagentx.optimizers.evoprompt_optimizer import DEOptimizer, GAOptimizer
from evoagentx.benchmark.bigbenchhard import BIGBenchHard
from evoagentx.models import OpenAILLM, OpenAILLMConfig
from evoagentx.optimizers.engine.registry import ParamRegistry

async def main():
    # 环境设置
    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    # 配置
    POPULATION_SIZE = 4
    ITERATIONS = 10
    CONCURRENCY_LIMIT = 100
    COMBINATION_SAMPLE_SIZE = 3
    DEV_SAMPLE_NUM = 15
    
    # LLM 配置
    evo_llm_config = OpenAILLMConfig(
        model="gpt-4.1-nano",
        openai_key=OPENAI_API_KEY,
        stream=False,
        top_p=0.95,
        temperature=0.5
    )
    
    eval_llm_config = OpenAILLMConfig(
        model="gpt-4.1-nano",
        openai_key=OPENAI_API_KEY,
        stream=False,
        temperature=0
    )
    llm = OpenAILLM(config=eval_llm_config)
    
    # 设置基准测试和程序
    benchmark = BIGBenchHard("snarks", dev_sample_num=DEV_SAMPLE_NUM, seed=10)
    program = SarcasmClassifierProgram(model=llm)
    
    # 注册提示节点
    registry = ParamRegistry()
    registry.track(program, "prompt_direct", name="direct_prompt_node")
    registry.track(program, "prompt_expert", name="expert_prompt_node")  
    registry.track(program, "prompt_cot", name="cot_prompt_node")
    
    # DE 优化
    optimizer_DE = DEOptimizer(
        registry=registry,
        program=program,
        population_size=POPULATION_SIZE,
        iterations=ITERATIONS,
        llm_config=evo_llm_config,
        concurrency_limit=CONCURRENCY_LIMIT,
        combination_sample_size=COMBINATION_SAMPLE_SIZE,
        enable_logging=True
    )
    
    await optimizer_DE.optimize(benchmark=benchmark)
    de_metrics = await optimizer_DE.evaluate(benchmark=benchmark, eval_mode="test")
    print(f"DE 结果: {de_metrics['accuracy']}")

if __name__ == "__main__":
    asyncio.run(main())
```

完整的工作示例请参考 [evoprompt_workflow.py](https://github.com/EvoAgentX/EvoAgentX/blob/main/examples/optimization/evoprompt/evoprompt_workflow.py)。

## 9. 对象使用与 JSON 持久化

- 直接使用优化后的对象
  - `optimize()` 完成后，优化器会自动将最优提示写回已注册的节点。传入的 `program` 实例即为“可直接使用”的最终工作流对象。

- 自动保存 JSON
  - 日志目录内会同时保存 `best_config.json`，其内容为 `{ 节点名: 最优提示 }` 的映射，便于后续复用或审计。

- 在新实例中复用（两种方式）
  - 方式一（助手方法）：`optimizer.load_and_apply_config("/path/to/best_config.json")`
  - 方式二（注册表）：读取 JSON 后，逐项 `registry.set(name, value)` 应用到新建的 program

最小示例：

```python
# 将 best_config.json 应用到新 program（基于 ParamRegistry）
with open(json_path, "r", encoding="utf-8") as f:
    best_cfg = json.load(f)
for k, v in best_cfg.items():
    registry.set(k, v)
```

参考示例：
- examples/optimization/evoprompt/evoprompt_bestconfig_json.py
- examples/optimization/evoprompt/evoprompt_save_load_json_min.py

## 10. 最佳实践

1. **合理设置种群大小**：建议4-8个个体，平衡探索和计算开销
2. **使用早停机制**：避免过度训练，节省计算资源
3. **调整温度参数**：进化时使用较高温度(0.5-0.8)，评估时使用低温度(0-0.2)
4. **监控日志**：关注收敛趋势，及时调整参数
5. **多任务测试**：在多个任务上验证优化器的通用性

## 11. 故障排除

### 常见问题

**Q: 优化过程中性能不提升？**
A: 检查基线性能、调整温度参数、增加种群大小或迭代次数。

**Q: 内存使用过高？**
A: 减小 `concurrency_limit` 和 `combination_sample_size`。

**Q: API 调用频率限制？**
A: 降低 `concurrency_limit`，添加适当的延迟。

有关更多优化器示例和高级配置，请查看 [examples/optimization/evoprompt](https://github.com/EvoAgentX/EvoAgentX/tree/main/examples/optimization/evoprompt) 目录中的其他脚本。
