"""
重新分析：evo2_optimizer 真正的性能瓶颈

用户指出的关键点：
1. 每轮都会淘汰一半，维持种群大小
2. 种群大小不会无限增长
3. 但仍然会"运行一段时间后变得很慢很慢"

那么真正的性能瓶颈是什么？

可能的原因分析：
===============

1. 缓存无效化 (Cache Invalidation)
   问题：虽然有缓存，但如果缓存键设计不当，可能导致缓存命中率下降
   现象：初期缓存命中率高，后期由于prompt变异过大，缓存失效
   
2. 内存泄漏 (Memory Leaks)  
   问题：虽然种群大小固定，但其他数据结构可能在增长
   可能位置：
   - 日志记录累积
   - 中间计算结果未释放
   - LLM调用的内存未清理
   
3. 垃圾回收压力 (GC Pressure)
   问题：频繁创建和销毁对象导致GC压力增大
   现象：随着运行时间增长，GC暂停时间变长

4. 网络/API限流 (Rate Limiting)
   问题：随着运行时间增长，API调用可能触发限流
   现象：LLM调用延迟逐渐增加

5. 日志文件增长 (Log File Growth)
   问题：详细日志文件随代数增长而变大
   现象：文件I/O操作越来越慢

6. 临时文件累积 (Temporary File Accumulation)
   问题：某些操作可能产生临时文件未清理
   现象：磁盘空间不足或文件系统性能下降

具体代码分析：
=============

让我检查代码中可能导致性能逐渐下降的部分：

1. 缓存实现检查：
```python
# 当前缓存实现
if not hasattr(self, '_eval_cache'):
    self._eval_cache = {}

# 潜在问题：缓存只增不减（除非手动清理）
if cache_key in self._eval_cache:
    return self._eval_cache[cache_key]

# 缓存清理机制（现在有了）
if len(self._eval_cache) > 5000:
    to_remove = list(self._eval_cache.keys())[:1000]
    for key in to_remove:
        del self._eval_cache[key]
```

2. 日志累积检查：
```python
# 每代都写入新的CSV文件
filename = f"detailed_evaluation_gen_{generation:02d}.csv"
filepath = os.path.join(self.log_dir, filename)

# 问题：随着代数增加，文件数量和总大小增长
# 如果日志过于详细，会影响I/O性能
```

3. 内存管理检查：
```python
# 每次评估都创建新的任务列表
tasks = [self._evaluate_combination_on_example(combination, benchmark, ex) 
         for ex in eval_dev_set]
results = await asyncio.gather(*tasks)

# 潜在问题：大量异步任务可能导致内存压力
```

可能的真正性能瓶颈：
==================

基于代码分析，最可能的性能瓶颈是：

1. **日志I/O累积效应** ⭐⭐⭐
   - 每代产生新的日志文件
   - 日志目录中文件数量线性增长
   - 文件系统性能可能下降

2. **缓存命中率下降** ⭐⭐
   - 随着prompt进化，新组合越来越少能命中缓存
   - 缓存效果逐渐降低

3. **异步任务调度开销** ⭐⭐  
   - 大量异步任务可能导致调度开销
   - 内存分配/释放压力

4. **API调用累积延迟** ⭐
   - 长时间运行可能触发API限流
   - 网络连接池可能出现问题

建议的诊断方法：
==============

1. 添加性能监控：
```python
import time
import psutil

# 在关键位置添加时间和内存监控
start_time = time.time()
start_memory = psutil.Process().memory_info().rss

# ... 执行代码 ...

end_time = time.time()
end_memory = psutil.Process().memory_info().rss
print(f"操作耗时: {end_time - start_time:.2f}秒")
print(f"内存变化: {(end_memory - start_memory) / 1024 / 1024:.2f}MB")
```

2. 缓存命中率监控：
```python
self._cache_hits = 0
self._cache_misses = 0

# 在缓存检查处
if cache_key in self._eval_cache:
    self._cache_hits += 1
    return self._eval_cache[cache_key]
else:
    self._cache_misses += 1
    
print(f"缓存命中率: {self._cache_hits / (self._cache_hits + self._cache_misses):.2%}")
```

3. 文件I/O监控：
```python
import os

# 监控日志目录大小
log_dir_size = sum(os.path.getsize(os.path.join(self.log_dir, f)) 
                   for f in os.listdir(self.log_dir))
print(f"日志目录大小: {log_dir_size / 1024 / 1024:.2f}MB")
```
"""
