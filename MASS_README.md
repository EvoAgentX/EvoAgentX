# MASS 框架详细说明

MASS (Multi-Agent Search System) 是一个基于多智能体的搜索优化框架，通过组合不同的工作流块(blocks)来解决复杂问题。本文档详细介绍MASS框架中的各个核心组件。

## 📁 项目结构与组件位置

### 1. Workflow Operators (工作流操作器)
**位置**: `evoagentx/workflow/operators.py`

这是MASS框架的核心操作器库，提供了各种可复用的智能体操作原语：

- **Predictor**: 基础预测器，用于问题推理和答案生成
- **Reflector**: 反思器，用于评估和批评现有解决方案
- **Refiner**: 精炼器，基于反思结果改进解决方案
- **Summarizer**: 总结器，提取和压缩上下文信息
- **Debater**: 辩论器，从多个候选方案中选择最佳答案
- **CodeReflector**: 代码反思器，专门用于代码执行结果的分析和改进

### 2. Prompt Operators (提示词操作器)
**位置**: `evoagentx/prompts/operators.py`

包含所有操作器使用的标准化提示词模板：

- `PREDICTOR_PROMPT`: 预测器提示词模板
- `REFLECTOR_PROMPT`: 反思器提示词模板  
- `REFINER_PROMPT`: 精炼器提示词模板
- `SUMMARIZER_PROMPT`: 总结器提示词模板
- `DEBATER_PROMPT`: 辩论器提示词模板
- `CODE_REFLECTOR_PROMPT`: 代码反思器提示词模板

### 3. MASS Optimizer (MASS优化器)
**位置**: `evoagentx/optimizers/mass_optimizer.py`

MASS优化器实现了智能的工作流优化策略：

**核心功能**:
- 使用softmax概率分布选择最有影响力的blocks
- 通过influence_score动态调整不同组件的权重
- 基于预算约束优化智能体配置
- 结合MiproOptimizer进行深度优化

**关键方法**:
- `optimize()`: 主优化循环，搜索最佳工作流配置
- `_softmax_with_temperature()`: 基于温度的概率选择策略

### 4. MASS 示例实现
**位置**: `examples/mass/mass.py`

这是MASS框架的完整使用示例，展示了如何：

**核心组件**:
- `MathSplits`: 数学问题基准测试数据集
- `WorkFlow`: 完整的MASS工作流实现
- `optimize_block()`: 单个block的优化函数
- `optimize_blocks_batch()`: 批量优化多个blocks

**工作流程**:
1. 优化基础Predictor
2. 批量优化所有workflow blocks
3. 构建最终的优化工作流

### 5. Workflow Blocks (工作流块)
**位置**: `evoagentx/workflow/blocks/`

MASS框架的模块化组件，每个block封装了特定的功能：

#### 5.1 Summarize Block
**文件**: `evoagentx/workflow/blocks/summarize.py`

**作用**: 从长文本中提取关键信息，为后续处理提供精炼的上下文

**核心功能**:
- 基于问题相关性过滤上下文信息
- 支持多轮总结以进一步压缩信息
- 可作为独立模块或工作流组件使用

#### 5.2 Aggregate Block  
**文件**: `evoagentx/workflow/blocks/aggregate.py`

**作用**: 聚合多个预测结果，通过自一致性选择最佳答案

**核心功能**:
- 生成多个独立预测
- 文本标准化和频次统计
- 返回最常见的预测结果

#### 5.3 Reflect Block
**文件**: `evoagentx/workflow/blocks/reflect.py`

**作用**: 对解决方案进行反思和改进

**核心功能**:
- 评估当前解决方案的正确性
- 提供改进建议和反馈
- 基于反思结果精炼答案

#### 5.4 Debate Block
**文件**: `evoagentx/workflow/blocks/debate.py`

**作用**: 通过多智能体辩论选择最佳解决方案

**核心功能**:
- 生成多个候选解决方案
- 智能体间的辩论和协商
- 迭代改进候选方案质量

#### 5.5 Execute Block
**文件**: `evoagentx/workflow/blocks/execute.py`

**作用**: 专门用于代码生成和执行验证

**核心功能**:
- 代码生成和测试用例执行
- 基于执行结果的错误分析
- 代码错误修复和优化

## 🔄 MASS工作流程

1. **上下文总结** (Summarize): 提取问题相关的关键信息
2. **方案聚合** (Aggregate): 生成多个候选解决方案
3. **反思优化** (Reflect): 对候选方案进行反思和改进
4. **辩论选择** (Debate): 通过辩论选出最佳方案
5. **执行验证** (Execute): 对代码类问题进行执行验证

## 🎯 优化策略

MASS采用了多层次的优化策略：

1. **Block级优化**: 使用MiproOptimizer优化每个block的内部参数
2. **工作流级优化**: 通过MassOptimizer优化block的组合和配置
3. **影响力评分**: 基于每个block对最终性能的贡献计算influence_score
4. **预算约束**: 在计算资源限制下寻找最优配置

## 🚀 使用示例

```python
# 1. 创建基准测试数据集
benchmark = MathSplits()

# 2. 优化基础Predictor
predictor = Predictor(llm=executor_llm)
optimized_predictor = mipro_optimize(...)

# 3. 批量优化所有blocks
optimized_blocks = optimize_blocks_batch(...)

# 4. 构建最终工作流
workflow = WorkFlow(
    summarizer=optimized_blocks['summarizer'],
    aggregater=optimized_blocks['aggregator'],
    reflector=optimized_blocks['reflector'],
    debater=optimized_blocks['debater'],
    executer=optimized_blocks['executer']
)
```

通过这种模块化和优化的设计，MASS框架能够自适应地组合不同的智能体组件，在各种问题类型上实现优异的性能表现。 