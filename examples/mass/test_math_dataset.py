import os 
import json
from dotenv import load_dotenv
from typing import Any

from evoagentx.benchmark import MATH
from evoagentx.models import OpenAILLMConfig, OpenAILLM
from evoagentx.optimizers import MiproOptimizer
from evoagentx.utils.mipro_utils.register_utils import MiproRegistry

# 导入新的 mass_optimizer
from .mass_optimizer import (
    run_full_optimization,
    create_mass_workflow,
    MassOptimiser,
    MassWorkflow,
    MassBlock
)

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

MAX_BOOTSTRAPPED_DEMOS = 1
MAX_LABELED_DEMOS = 0
AUTO = "light"
NUM_THREADS = 16
EVALUATION_ROUNDS = 1

class MathSplits(MATH):
    """参考原始 mass.py 的 MathSplits 类"""

    def _load_data(self):
        # load the original test data 
        super()._load_data()
        # split the data into dev and test
        import numpy as np 
        np.random.seed(42)
        permutation = np.random.permutation(len(self._test_data))
        full_test_data = self._test_data
        # randomly select 100 samples for training and 100 samples for test
        self._train_data = [full_test_data[idx] for idx in permutation[:100]]
        self._test_data = [full_test_data[idx] for idx in permutation[100:200]]

    # define the input keys. 
    # If defined, the corresponding input key and value will be passed to the __call__ method of the program, 
    # i.e., program.__call__(**{k: v for k, v in example.items() if k in self.get_input_keys()})
    # If not defined, the program will be executed with the entire input example, i.e., program.__call__(**example)
    def get_input_keys(self):
        return ["problem"]
    
    # the benchmark must have a `evaluate` method that receives the program's `prediction` (output from the program's __call__ method) 
    # and the `label` (obtained using the `self.get_label` method) and return a dictionary of metrics. 
    def evaluate(self, prediction: Any, label: Any) -> dict:
        return super().evaluate(prediction, label)


def get_save_path(program):
    """获取保存路径"""
    return f"examples/mass/{program}"


def test_simple_workflow():
    """测试简单的工作流执行"""
    print("=== 测试简单工作流执行 ===")
    
    # 初始化 LLM
    openai_config = OpenAILLMConfig(
        model="gpt-4o", 
        openai_key=OPENAI_API_KEY, 
        stream=True, 
        output_response=False
    )
    executor_llm = OpenAILLM(config=openai_config)
    
    # 创建测试数据集
    benchmark = MathSplits()
    
    # 创建简单的工作流
    workflow = create_mass_workflow(executor_llm)
    
    # 测试单个问题
    test_problem = "What is 2 + 2?"
    print(f"测试问题: {test_problem}")
    
    try:
        result, context = workflow(test_problem)
        print(f"结果: {result}")
        print(f"上下文: {context}")
        print("✅ 简单工作流测试通过")
    except Exception as e:
        print(f"❌ 简单工作流测试失败: {e}")
        return False
    
    return True


def test_individual_blocks():
    """测试各个独立的 blocks"""
    print("\n=== 测试各个独立的 blocks ===")
    
    # 初始化 LLM
    openai_config = OpenAILLMConfig(
        model="gpt-4o", 
        openai_key=OPENAI_API_KEY, 
        stream=True, 
        output_response=False
    )
    executor_llm = OpenAILLM(config=openai_config)
    
    # 创建测试数据集
    benchmark = MathSplits()
    
    # 测试 Predictor
    from .blocks import create_predictor_agent
    predictor = create_predictor_agent(executor_llm)
    
    test_problem = "Solve: 3x + 5 = 20"
    print(f"测试 Predictor: {test_problem}")
    
    try:
        result = predictor(test_problem)
        print(f"Predictor 结果: {result}")
        print("✅ Predictor 测试通过")
    except Exception as e:
        print(f"❌ Predictor 测试失败: {e}")
        return False
    
    # 测试 Aggregate
    from .blocks import create_aggregate
    aggregate = create_aggregate(predictor, n=3)
    
    print(f"测试 Aggregate: {test_problem}")
    
    try:
        result = aggregate(test_problem)
        print(f"Aggregate 结果: {result}")
        print("✅ Aggregate 测试通过")
    except Exception as e:
        print(f"❌ Aggregate 测试失败: {e}")
        return False
    
    return True


def test_optimization_workflow():
    """测试优化工作流（简化版本）"""
    print("\n=== 测试优化工作流（简化版本）===")
    
    # 初始化 LLM
    openai_config = OpenAILLMConfig(
        model="gpt-4o", 
        openai_key=OPENAI_API_KEY, 
        stream=True, 
        output_response=False
    )
    executor_llm = OpenAILLM(config=openai_config)
    optimizer_llm = OpenAILLM(config=openai_config)
    
    # 创建测试数据集
    benchmark = MathSplits()
    
    # 创建工作流
    workflow = create_mass_workflow(executor_llm)
    
    # 设置一些 blocks 为激活状态
    workflow.blocks[0].n = 1  # summarizer
    workflow.blocks[1].n = 3  # aggregater
    workflow.blocks[2].n = 0  # reflector (不激活)
    workflow.blocks[3].n = 1  # debater
    workflow.blocks[4].n = 0  # executer (不激活)
    
    # 创建优化器
    mass = MassOptimiser(
        workflow=workflow,
        optimizer_llm=optimizer_llm,
        max_bootstrapped_demos=1,  # 减少以加快测试
        max_labeled_demos=0,
        auto="light",
        eval_rounds=1,
        num_threads=4,  # 减少线程数
        save_path="examples/mass/test_optimization",
        max_steps=2  # 减少步数以加快测试
    )
    
    try:
        print("开始优化...")
        best_program = mass.optimize(benchmark=benchmark)
        print("✅ 优化工作流测试通过")
        return True
    except Exception as e:
        print(f"❌ 优化工作流测试失败: {e}")
        return False


def test_full_optimization():
    """测试完整优化流程（参考原始 main 函数）"""
    print("\n=== 测试完整优化流程 ===")
    
    # 初始化 LLM
    openai_config = OpenAILLMConfig(
        model="gpt-4o", 
        openai_key=OPENAI_API_KEY, 
        stream=True, 
        output_response=False
    )
    executor_llm = OpenAILLM(config=openai_config)
    optimizer_llm = OpenAILLM(config=openai_config)
    
    # 创建测试数据集
    benchmark = MathSplits()
    
    try:
        print("开始完整优化流程...")
        best_program = run_full_optimization(executor_llm, optimizer_llm, benchmark)
        print("✅ 完整优化流程测试通过")
        return True
    except Exception as e:
        print(f"❌ 完整优化流程测试失败: {e}")
        return False


def test_workflow_serialization():
    """测试工作流序列化"""
    print("\n=== 测试工作流序列化 ===")
    
    # 初始化 LLM
    openai_config = OpenAILLMConfig(
        model="gpt-4o", 
        openai_key=OPENAI_API_KEY, 
        stream=True, 
        output_response=False
    )
    executor_llm = OpenAILLM(config=openai_config)
    
    # 创建工作流
    workflow = create_mass_workflow(executor_llm)
    
    # 设置一些状态
    workflow.blocks[0].n = 1
    workflow.blocks[1].n = 3
    workflow.blocks[2].n = 0
    workflow.blocks[3].n = 1
    workflow.blocks[4].n = 0
    
    try:
        # 保存状态
        state = workflow.get_state()
        print("✅ 状态保存成功")
        
        # 修改状态
        workflow.blocks[0].n = 5
        workflow.blocks[1].n = 7
        
        # 恢复状态
        workflow.set_state(state)
        
        # 验证恢复
        assert workflow.blocks[0].n == 1
        assert workflow.blocks[1].n == 3
        print("✅ 状态恢复成功")
        
        # 测试文件保存和加载
        save_path = "examples/mass/test_workflow_config.json"
        workflow.save(save_path)
        print("✅ 工作流配置保存成功")
        
        # 创建新的工作流并加载配置
        new_workflow = create_mass_workflow(executor_llm)
        new_workflow.load(save_path)
        print("✅ 工作流配置加载成功")
        
        return True
    except Exception as e:
        print(f"❌ 工作流序列化测试失败: {e}")
        return False


def main():
    """主测试函数"""
    print("开始测试 MATH 数据集...")
    
    # 检查环境变量
    if not OPENAI_API_KEY:
        print("❌ 错误: 未设置 OPENAI_API_KEY 环境变量")
        return
    
    # 创建必要的目录
    os.makedirs("examples/mass", exist_ok=True)
    
    # 运行各个测试
    tests = [
        ("简单工作流", test_simple_workflow),
        ("独立 blocks", test_individual_blocks),
        ("优化工作流", test_optimization_workflow),
        ("工作流序列化", test_workflow_serialization),
        ("完整优化流程", test_full_optimization),
    ]
    
    results = {}
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"运行测试: {test_name}")
        print(f"{'='*50}")
        
        try:
            result = test_func()
            results[test_name] = result
        except Exception as e:
            print(f"❌ 测试 {test_name} 发生异常: {e}")
            results[test_name] = False
    
    # 输出测试结果
    print(f"\n{'='*50}")
    print("测试结果汇总:")
    print(f"{'='*50}")
    
    passed = 0
    total = len(tests)
    
    for test_name, result in results.items():
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n总计: {passed}/{total} 个测试通过")
    
    if passed == total:
        print("🎉 所有测试都通过了！")
    else:
        print("⚠️  部分测试失败，请检查错误信息")


if __name__ == "__main__":
    main()
