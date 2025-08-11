# EvoAgentX Optimization Comparison - Configuration Summary

## 统一配置参数

两个示例文件现在使用完全一致的配置参数，确保公平对比：

### 📊 核心配置
| 参数 | 节点独立进化 | Mega-Prompt进化 | 说明 |
|------|-------------|----------------|------|
| **种群大小** | 4 (每个节点) | 4 (mega-prompt) | 统一种群大小 |
| **迭代次数** | 3 | 3 | 相同迭代次数 |
| **并发限制** | 50 | 50 | 相同并发控制 |
| **LLM模型** | gpt-4o-mini | gpt-4o-mini | 相同模型 |

### 🔢 数据集配置
| 参数 | 节点独立进化 | Mega-Prompt进化 | 说明 |
|------|-------------|----------------|------|
| **任务** | snarks | snarks | 相同任务 |
| **开发集大小** | 15 | 15 | 用于进化的数据 |
| **测试集大小** | 5 | 5 | 用于最终评估 |

### ⚖️ 评估工作量平衡
| 方法 | 种群配置 | 评估策略 | 每代评估量 |
|------|----------|----------|-----------|
| **节点独立进化** | 3节点 × 4提示词 = 12个体 | 组合采样 (4个组合) | 4 × 15 = 60次评估 |
| **Mega-Prompt进化** | 4个mega-prompt | 完整评估 | 4 × 15 = 60次评估 |

### 🎯 关键差异
虽然配置参数统一，但两种方法的核心理念不同：

#### 节点独立进化 (evo2_optimizer.py)
- ✅ 每个提示词节点独立进化
- ✅ 组合采样控制成本
- ✅ 细粒度性能追踪
- ✅ 更好的可扩展性

#### Mega-Prompt进化 (evoprompt_optimizer.py)  
- ✅ 整体mega-prompt进化
- ✅ 传统EvoPrompt策略
- ✅ XML结构处理
- ✅ 简单直接的进化

### 🧪 测试命令
```bash
# 测试节点独立进化
python examples/optimization/evoprompt/bbh_snarks_node_evolution_example.py

# 测试Mega-Prompt进化
python examples/optimization/evoprompt/bbh_snarks_mega_prompt_example.py
```

两个示例现在配置完全一致，可以进行公平的性能对比！
