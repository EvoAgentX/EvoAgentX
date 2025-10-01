#!/usr/bin/env python3
"""
测试所有 benchmark 的 get_label_test_case 方法
"""

from evoagentx.benchmark import MATH, GSM8K, HotPotQA, NQ, HumanEval, MBPP, LiveCodeBench

def test_benchmark_methods():
    """测试所有 benchmark 类都有 get_label_test_case 方法"""
    
    benchmark_classes = [
        MATH,
        GSM8K, 
        HotPotQA,
        NQ,
        HumanEval,
        MBPP,
        LiveCodeBench
    ]
    
    print("=== 测试所有 benchmark 的 get_label_test_case 方法 ===")
    
    for benchmark_class in benchmark_classes:
        print(f"\n测试 {benchmark_class.__name__}:")
        
        # 检查方法是否存在
        if hasattr(benchmark_class, 'get_label_test_case'):
            print(f"  ✓ {benchmark_class.__name__} 有 get_label_test_case 方法")
        else:
            print(f"  ✗ {benchmark_class.__name__} 缺少 get_label_test_case 方法")
        
        # 检查方法是否可调用
        if callable(getattr(benchmark_class, 'get_label_test_case', None)):
            print(f"  ✓ {benchmark_class.__name__} 的 get_label_test_case 是可调用的")
        else:
            print(f"  ✗ {benchmark_class.__name__} 的 get_label_test_case 不可调用")
    
    print("\n=== 测试完成 ===")

if __name__ == "__main__":
    test_benchmark_methods() 