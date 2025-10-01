# MASS Optimizer with CustomizeAgent Blocks

这个目录包含了基于新的 `CustomizeAgent` blocks 重新实现的 MASS 优化器，**完全匹配原始优化流程**。

## 🎯 优化流程

新的实现严格按照原始 MASS 算法的优化流程：

### **Step 0: 优化单独的 Predictor**
```python
predictor_score, optimized_predictor = optimize_predictor(predictor, optimizer_llm, benchmark)
```

### **Step 1: 基于优化后的 Predictor 逐个优化每个 Block**
```python
# 逐个优化每个 block，计算影响力分数
optimized_summarizer = optimize_summarizer(optimized_predictor, executor_llm, optimizer_llm, benchmark, predictor_score)
optimized_aggregator = optimize_aggregator(optimized_predictor, optimizer_llm, benchmark, predictor_score)
optimized_reflector = optimize_reflector(optimized_predictor, executor_llm, optimizer_llm, benchmark, predictor_score)
optimized_debater = optimize_debater(optimized_predictor, executor_llm, optimizer_llm, benchmark, predictor_score)
optimized_executer = optimize_executer(optimized_predictor, executor_llm, optimizer_llm, benchmark, predictor_score)
```

### **Step 2: 优化完整的工作流（包括 n 和 prompt）**
```python
mass = MassOptimiser(workflow=block_workflow, optimizer_llm=optimizer_llm, ...)
best_program = mass.optimize(benchmark=benchmark, softmax_temperature=1.0, agent_budget=10)
```

## 📁 目录结构

```
examples/mass/
├── blocks/                 # 新的blocks模块
│   ├── predictor_agent.py  # Predictor block
│   ├── aggregate.py       # Aggregate block
│   ├── debate.py          # Debate block
│   ├── reflect.py         # Reflect block
│   ├── summarize.py       # Summarize block
│   ├── execute.py         # Execute block
│   ├── utils.py           # 工具函数
│   ├── test_all_blocks.py # 测试脚本
│   └── __init__.py        # 包初始化
├── mass_optimizer.py      # 新的MASS优化器（匹配原始流程）
├── example_usage.py       # 使用示例
├── mass.py               # 原始实现（保留）
└── README.md             # 本文档
```

## 🚀 主要特性

### 1. **完全匹配原始优化流程**
- **Step 0**: 先优化单独的 predictor
- **Step 1**: 基于优化后的 predictor 逐个优化每个 block
- **Step 2**: 优化完整的工作流（包括 n 和 prompt）

### 2. **基于 CustomizeAgent 的 Blocks**
- 所有 blocks 都使用 `CustomizeAgent` 实现
- 统一的接口：`__call__`, `execute`, `save`, `load`
- 支持多种解析模式：`title`, `json`, `xml`, `str`
- 完整的序列化支持

### 3. **匹配原始工作流逻辑**
- 工作流执行逻辑完全匹配原始 `WorkFlow` 类
- 相同的 block 顺序：summarizer, aggregater, reflector, debater, executer
- 相同的保存/加载逻辑

## 📖 使用方法

### 方法1: 使用完整的优化流程（推荐）

```python
from mass_optimizer import run_full_optimization
from evoagentx.models import OpenAILLMConfig, OpenAILLM

# 创建LLM
executor_llm = OpenAILLM(config=OpenAILLMConfig(...))
optimizer_llm = OpenAILLM(config=OpenAILLMConfig(...))

# 创建数据集
benchmark = MathSplits()

# 运行完整的优化流程
best_workflow = run_full_optimization(executor_llm, optimizer_llm, benchmark)
```

### 方法2: 手动执行每个步骤

```python
from mass_optimizer import (
    optimize_predictor, optimize_summarizer, optimize_aggregator,
    optimize_reflector, optimize_debater, optimize_executer,
    MassWorkflow, MassOptimiser
)

# Step 0: 优化 Predictor
predictor = create_predictor_agent(executor_llm)
predictor_score, optimized_predictor = optimize_predictor(predictor, optimizer_llm, benchmark)

# Step 1: 逐个优化每个block
optimized_summarizer = optimize_summarizer(optimized_predictor, executor_llm, optimizer_llm, benchmark, predictor_score)
optimized_aggregator = optimize_aggregator(optimized_predictor, optimizer_llm, benchmark, predictor_score)
# ... 其他 blocks

# Step 2: 构建工作流
block_workflow = MassWorkflow([optimized_summarizer, optimized_aggregator, ...])

# Step 3: 优化完整工作流
mass = MassOptimiser(workflow=block_workflow, optimizer_llm=optimizer_llm, ...)
best_workflow = mass.optimize(benchmark=benchmark)
```

### 测试

```bash
# 测试所有 blocks
cd examples/mass/blocks
python test_all_blocks.py

# 运行完整优化示例
cd examples/mass
python example_usage.py
```

## 🔧 核心函数

### 优化函数

| 函数 | 描述 | 对应原始实现 |
|------|------|-------------|
| `optimize_predictor()` | Step 0: 优化单独的 predictor | `optimize_predictor()` |
| `optimize_summarizer()` | Step 1: 优化 summarizer block | `optimize_summarizer()` |
| `optimize_aggregator()` | Step 1: 优化 aggregator block | `optimize_aggregator()` |
| `optimize_reflector()` | Step 1: 优化 reflector block | `optimize_reflector()` |
| `optimize_debater()` | Step 1: 优化 debater block | `optimize_debater()` |
| `optimize_executer()` | Step 1: 优化 executer block | `optimize_executer()` |
| `run_full_optimization()` | 完整优化流程 | `main()` |

### 工作流类

| 类 | 描述 | 对应原始实现 |
|------|------|-------------|
| `MassBlock` | Block 包装器 | 原始 block 类 |
| `MassWorkflow` | 工作流管理 | `WorkFlow` 类 |
| `MassOptimiser` | 优化器 | `MassOptimiser` 类 |

## 📊 与原始实现的对比

| 方面 | 原始实现 | 新实现 |
|------|----------|--------|
| **优化流程** | ✅ 先优化 predictor | ✅ 完全匹配 |
| **Block 优化** | ✅ 逐个优化 | ✅ 完全匹配 |
| **工作流优化** | ✅ 优化 n 和 prompt | ✅ 完全匹配 |
| **Blocks** | 旧 operators | CustomizeAgent blocks |
| **序列化** | 硬编码字段 | 通用序列化 |
| **扩展性** | 难以扩展 | 易于添加新 blocks |
| **测试** | 难以测试 | 模块化测试 |

## 🎯 关键改进

1. **保持原始优化流程**: 完全匹配原始 MASS 算法的优化策略
2. **改进的模块化**: 每个 block 独立，易于测试和维护
3. **统一的接口**: 所有 blocks 使用相同的接口
4. **更好的序列化**: 支持完整的配置保存和加载
5. **CustomizeAgent 集成**: 充分利用 CustomizeAgent 的特性
6. **保持兼容性**: 与原始实现完全兼容

## ⚠️ 注意事项

1. 确保设置了正确的 `OPENAI_API_KEY` 环境变量
2. 优化流程严格按照原始 MASS 算法的顺序执行
3. 每个 step 都会保存中间结果到指定路径
4. 最终工作流配置会保存到 `examples/mass/best_workflow_config.json`
5. 在生产环境中建议使用更强大的模型（如 gpt-4）

## 🔄 迁移指南

如果你正在使用原始的 `mass.py`，迁移到新实现非常简单：

```python
# 原始代码
from mass import main
main()

# 新代码
from mass_optimizer import run_full_optimization
best_workflow = run_full_optimization(executor_llm, optimizer_llm, benchmark)
```

新实现提供了相同的功能和结果，但具有更好的模块化、可维护性和扩展性。
