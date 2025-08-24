# Workflow Evaluation Pipeline

## 评估架构

### 第一层：工作流程结构评估 (Structure Evaluation)
**目标**: 评估生成的工作流程JSON文件的结构完整性和合理性

**流程**:
1. 调用`WorkFlowGenerator.generate_workflow()`方法生成工作流程对象
2. 将工作流程对象序列化为JSON文件
3. 使用LLM模型对JSON文件进行结构和逻辑评估

**评估指标**:
- generate_workflow() 执行成功率
- 工作流程结构完整性（开放式指标）
- 输入输出参数匹配度（生成过程应该已经检测过）
- 任务分解的逻辑性（开放式指标）

### 第二层：工作流程执行评估 (Execution Evaluation)
**目标**: 测试工作流程的实际执行能力和稳定性

**流程**:
1. 根据提供的测试数据集，为每个工作流程的输入字段提供测试值
2. 调用工作流程的`async_execute()`方法执行
3. 统计执行成功率和失败原因

**统计指标**:
- 执行成功率 (Success Rate)
- 错误类型分布:
  - 预期异常 (Expected Exceptions)
  - 未知错误 (Unknown Errors)
- 平均执行时间（后续再考虑）
- 资源使用情况（后续再考虑）

**错误分类**:
- **预期异常**: 代码中预先考虑并抛出的异常
- **未知错误**: 未预料到的系统错误或运行时错误

### 第三层：工作流程输出质量评估 (Output Quality Evaluation)
**目标**: 评估成功执行的工作流程输出质量

**流程**:
1. 收集成功执行的工作流程输出结果
2. 结合原始用户查询（workflow generation query）
3. 使用LLM模型评估输出质量

**评估维度**:
- 任务目标的完成度（半开放式指标）：是否完成用户query的所有要求
- 输出内容具体质量（开放式指标）：目前考虑一致性、多样性、有用性这三点，一致性评估输出内容自身的逻辑协调，多样性评估输出是否"白开水"，过于单一，有用性评估是否对具体query有效，有多大效果。

## 评估数据流

```
输入参数 → generate_workflow() → WorkFlowGraph对象
                                       ↓
JSON序列化 → 第一层评估 (LLM结构评估)
                                       ↓
测试输入数据 → async_execute() → 第二层评估 (执行统计)
                                       ↓ (成功案例)
工作流程输出 + 原始查询 → 第三层评估 (LLM输出质量评估)
```

## 实现要点

- workflow验证数据规模较大时考虑多进程并行
- 支持pipeline的断点resume
- 根据系统资源动态调整进程数和批次大小
- 记录异常类型、模块、消息和完整traceback
- 最后提供eval报告

详细实现指南请参考 [implementation_guide.md](./implementation_guide.md)
