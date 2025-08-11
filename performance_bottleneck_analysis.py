"""
性能瓶颈详细分析：组合爆炸与缓存缺失

1. 组合爆炸问题 (Combinatorial Explosion)
=====================================

问题描述：
在evo2_optimizer中，算法需要评估所有可能的prompt组合。
这些组合通过笛卡尔积生成，数量随节点数和种群大小指数增长。

数学表示：
总组合数 = P1 × P2 × P3 × ... × Pn
其中 Pi 是第i个节点的种群大小

实际影响：
节点数=3, 种群大小=5  → 组合数=125     (可接受)
节点数=4, 种群大小=10 → 组合数=10,000  (开始变慢)
节点数=5, 种群大小=15 → 组合数=759,375 (严重性能问题)

代码中的问题：
```python
# 原始代码 - 无限制生成所有组合
for combination in itertools.product(*node_prompts):
    combo_dict = {node_names[i]: combination[i] for i in range(len(node_names))}
    all_combinations.append(combo_dict)
```

优化后的代码：
```python
# 优化后 - 限制组合数量
total_possible = 1
for prompts in node_prompts:
    total_possible *= len(prompts)

if total_possible > 10000:
    logger.warning(f"组合数量过大: {total_possible}")

count = 0
for combination in itertools.product(*node_prompts):
    if count >= 5000:  # 硬性限制
        break
    # ... 处理组合
    count += 1
```

2. 缓存机制缺失 (Missing Cache Mechanism)
=======================================

问题描述：
进化算法中经常会重新生成相同或相似的prompt组合，
但原始代码没有缓存机制，导致重复评估相同的组合。

重复评估的来源：
a) 跨代重复：不同代数可能产生相同的prompt组合
b) 代内重复：同一代内可能有重复的组合
c) 局部搜索重复：相似的prompt可能产生相同的结果

性能影响量化：
假设场景：100个组合，每个在50个样本上评估

无缓存：
- 第1代：100 × 50 = 5,000次LLM调用
- 第2代：假设30%重复，仍需 100 × 50 = 5,000次调用
- 第3代：假设50%重复，仍需 100 × 50 = 5,000次调用
- 总计：15,000次LLM调用

有缓存：
- 第1代：100 × 50 = 5,000次LLM调用 (首次)
- 第2代：70 × 50 = 3,500次LLM调用 (30%缓存命中)
- 第3代：50 × 50 = 2,500次LLM调用 (50%缓存命中)
- 总计：11,000次LLM调用 (节省26.7%)

代码实现对比：

原始代码 - 无缓存：
```python
async def evaluate_combination(self, combination, example):
    # 每次都重新计算，即使是相同的组合
    original_config = self.get_current_cfg()
    self.apply_cfg(combination)
    prediction = self.program(**example)
    score = benchmark.evaluate(prediction, label)
    self.apply_cfg(original_config)
    return score
```

优化后 - 有缓存：
```python
async def evaluate_combination(self, combination, example):
    # 生成缓存键
    combo_key = tuple(sorted(combination.items()))
    example_key = str(hash(str(example)))
    cache_key = hash((combo_key, example_key))
    
    # 检查缓存
    if cache_key in self._eval_cache:
        return self._eval_cache[cache_key]  # 缓存命中，瞬间返回
    
    # 缓存未命中，进行计算
    original_config = self.get_current_cfg()
    self.apply_cfg(combination)
    prediction = self.program(**example)
    score = benchmark.evaluate(prediction, label)
    self.apply_cfg(original_config)
    
    # 存储到缓存
    self._eval_cache[cache_key] = score
    return score
```

3. 配置切换开销 (Configuration Switching Overhead)
===============================================

问题描述：
每次评估都需要：
1. 获取当前配置 (get_current_cfg)
2. 应用新配置 (apply_cfg)
3. 执行评估
4. 恢复原配置 (apply_cfg)

这个过程在大量评估中累积成为显著开销。

优化策略：
- 批量处理相同配置的评估
- 减少配置切换频率
- 缓存配置状态

4. 内存管理问题
==============

随着运行时间增长，缓存可能无限增长导致内存问题。

解决方案：
```python
# 限制缓存大小
if len(self._eval_cache) > 5000:
    # 删除最旧的20%条目
    to_remove = list(self._eval_cache.keys())[:1000]
    for key in to_remove:
        del self._eval_cache[key]
```

5. 实际性能提升估算
=================

优化前的时间复杂度：
O(G × C × E × T)
其中：
G = 代数
C = 组合数量
E = 评估样本数
T = 单次LLM调用时间

优化后的时间复杂度：
O(G × C' × E' × T × (1 - cache_hit_rate))
其中：
C' = 限制后的组合数量 (C' << C)
E' = 限制后的样本数量 (E' << E)
cache_hit_rate = 缓存命中率 (0.3-0.7)

预期性能提升：
- 组合限制：减少80-90%的组合数量
- 样本限制：减少50-80%的评估样本
- 缓存机制：减少30-70%的重复计算
- 总体提升：5-20倍的性能提升
"""
