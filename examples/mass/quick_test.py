#!/usr/bin/env python3
"""
快速测试脚本 - 验证 mass_optimizer 的基本功能
参考原始 mass.py 的结构，但使用新的 mass_optimizer
"""

import os 
import sys
from dotenv import load_dotenv

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from evoagentx.benchmark import MATH
from evoagentx.models import OpenAILLMConfig, OpenAILLM

# 导入新的 mass_optimizer
from examples.mass.mass_optimizer import (
    run_full_optimization,
    create_mass_workflow,
    MassOptimiser,
    MassWorkflow
)

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

class MathSplits(MATH):
    """简化的 MathSplits 类，用于快速测试"""
    
    def _load_data(self):
        super()._load_data()
        import numpy as np 
        np.random.seed(42)
        permutation = np.random.permutation(len(self._test_data))
        full_test_data = self._test_data
        # 只使用少量样本进行快速测试
        self._train_data = [full_test_data[idx] for idx in permutation[:20]]  # 减少到20个
        self._test_data = [full_test_data[idx] for idx in permutation[20:40]]  # 减少到20个

    def get_input_keys(self):
        return ["problem"]
    
    def evaluate(self, prediction, label):
        return super().evaluate(prediction, label)


def test_basic_workflow():
    """测试基本工作流功能"""
    print("🔧 测试基本工作流功能...")
    
    if not OPENAI_API_KEY:
        print("❌ 错误: 未设置 OPENAI_API_KEY")
        return False
    
    try:
        # 初始化 LLM
        config = OpenAILLMConfig(
            model="gpt-4o", 
            openai_key=OPENAI_API_KEY, 
            stream=False,  # 关闭流式输出以加快测试
            output_response=False
        )
        llm = OpenAILLM(config=config)
        
        # 创建工作流
        workflow = create_mass_workflow(llm)
        
        # 测试简单问题
        test_problem = "What is 2 + 2?"
        result, context = workflow(test_problem)
        
        print(f"✅ 工作流测试通过")
        print(f"   问题: {test_problem}")
        print(f"   结果: {result}")
        return True
        
    except Exception as e:
        print(f"❌ 工作流测试失败: {e}")
        return False


def test_optimization():
    """测试优化功能（简化版本）"""
    print("🔧 测试优化功能...")
    
    if not OPENAI_API_KEY:
        print("❌ 错误: 未设置 OPENAI_API_KEY")
        return False
    
    try:
        # 初始化 LLM
        config = OpenAILLMConfig(
            model="gpt-4o", 
            openai_key=OPENAI_API_KEY, 
            stream=False,
            output_response=False
        )
        executor_llm = OpenAILLM(config=config)
        optimizer_llm = OpenAILLM(config=config)
        
        # 创建数据集
        benchmark = MathSplits()
        
        # 创建工作流
        workflow = create_mass_workflow(executor_llm)
        
        # 设置激活状态
        workflow.blocks[0].n = 1  # summarizer
        workflow.blocks[1].n = 3  # aggregater
        workflow.blocks[2].n = 0  # reflector
        workflow.blocks[3].n = 1  # debater
        workflow.blocks[4].n = 0  # executer
        
        # 创建优化器（使用最小参数）
        mass = MassOptimiser(
            workflow=workflow,
            optimizer_llm=optimizer_llm,
            max_bootstrapped_demos=1,
            max_labeled_demos=0,
            auto="light",
            eval_rounds=1,
            num_threads=2,
            save_path="examples/mass/quick_test_optimization",
            max_steps=1  # 只运行1步
        )
        
        # 运行优化
        best_program = mass.optimize(benchmark=benchmark)
        
        print("✅ 优化测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 优化测试失败: {e}")
        return False


def test_serialization():
    """测试序列化功能"""
    print("🔧 测试序列化功能...")
    
    if not OPENAI_API_KEY:
        print("❌ 错误: 未设置 OPENAI_API_KEY")
        return False
    
    try:
        # 初始化 LLM
        config = OpenAILLMConfig(
            model="gpt-4o", 
            openai_key=OPENAI_API_KEY, 
            stream=False,
            output_response=False
        )
        llm = OpenAILLM(config=config)
        
        # 创建工作流
        workflow = create_mass_workflow(llm)
        
        # 设置状态
        workflow.blocks[0].n = 1
        workflow.blocks[1].n = 3
        
        # 测试状态保存和恢复
        state = workflow.get_state()
        workflow.blocks[0].n = 5
        workflow.set_state(state)
        
        assert workflow.blocks[0].n == 1
        assert workflow.blocks[1].n == 3
        
        # 测试文件保存和加载
        save_path = "examples/mass/quick_test_config.json"
        workflow.save(save_path)
        
        new_workflow = create_mass_workflow(llm)
        new_workflow.load(save_path)
        
        print("✅ 序列化测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 序列化测试失败: {e}")
        return False


def main():
    """主函数"""
    print("🚀 开始快速测试 mass_optimizer...")
    print("=" * 50)
    
    # 创建必要的目录
    os.makedirs("examples/mass", exist_ok=True)
    
    # 运行测试
    tests = [
        ("基本工作流", test_basic_workflow),
        ("优化功能", test_optimization),
        ("序列化功能", test_serialization),
    ]
    
    results = {}
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results[test_name] = result
        except Exception as e:
            print(f"❌ {test_name} 发生异常: {e}")
            results[test_name] = False
    
    # 输出结果
    print(f"\n{'='*50}")
    print("测试结果:")
    print(f"{'='*50}")
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name}: {status}")
    
    print(f"\n总计: {passed}/{total} 个测试通过")
    
    if passed == total:
        print("🎉 所有测试通过！mass_optimizer 工作正常")
    else:
        print("⚠️  部分测试失败，请检查错误信息")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
