"""
缓存机制缺失的性能影响示例

问题场景：
假设我们有以下情况：
- 3个节点，每个节点有5个prompt变体
- 总共有125种组合
- 每种组合需要在100个测试样本上评估

没有缓存的情况下：
1. 第1代：评估125个组合 × 100个样本 = 12,500次LLM调用
2. 第2代：由于进化产生了一些重复的prompt组合，但我们不知道已经评估过
   - 假设有30%的组合是重复的，但仍然重新评估
   - 又是12,500次LLM调用，其中3,750次是不必要的重复计算

累积浪费：
- 10代演化后，重复计算可能占到总计算量的40-60%
- 原本需要125,000次LLM调用的任务，实际可能执行了200,000次

具体的重复情况：
"""

# 示例：没有缓存时的重复计算
def evaluate_without_cache():
    """模拟没有缓存的评估过程"""
    evaluations = {}  # 假设这是真实的评估记录
    
    # 第1代
    combination_1 = {"node1": "prompt_A", "node2": "prompt_X", "node3": "prompt_1"}
    result_1 = expensive_llm_evaluation(combination_1)  # 耗时操作
    evaluations[str(combination_1)] = result_1
    
    # 第2代 - 进化后可能产生相同组合
    combination_2 = {"node1": "prompt_A", "node2": "prompt_X", "node3": "prompt_1"}  # 与combination_1相同!
    result_2 = expensive_llm_evaluation(combination_2)  # 重复计算！浪费时间
    
    # 第3代 - 又出现了第1代的组合
    combination_3 = {"node1": "prompt_A", "node2": "prompt_X", "node3": "prompt_1"}  # 又是相同的!
    result_3 = expensive_llm_evaluation(combination_3)  # 再次重复计算！

def evaluate_with_cache():
    """模拟有缓存的评估过程"""
    cache = {}  # 缓存已评估的结果
    
    # 第1代
    combination_1 = {"node1": "prompt_A", "node2": "prompt_X", "node3": "prompt_1"}
    cache_key = hash(str(sorted(combination_1.items())))
    
    if cache_key not in cache:
        result_1 = expensive_llm_evaluation(combination_1)  # 只计算一次
        cache[cache_key] = result_1
    else:
        result_1 = cache[cache_key]  # 直接使用缓存，几乎瞬间完成
    
    # 第2代 - 相同组合
    combination_2 = {"node1": "prompt_A", "node2": "prompt_X", "node3": "prompt_1"}
    cache_key = hash(str(sorted(combination_2.items())))
    
    if cache_key not in cache:
        result_2 = expensive_llm_evaluation(combination_2)
        cache[cache_key] = result_2
    else:
        result_2 = cache[cache_key]  # 缓存命中！节省大量时间
    
    return cache

def expensive_llm_evaluation(combination):
    """模拟耗时的LLM评估过程"""
    import time
    time.sleep(1)  # 模拟LLM调用延迟
    return 0.85  # 模拟评估分数
