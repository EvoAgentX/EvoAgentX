import os
from dotenv import load_dotenv
from evoagentx.benchmark import MATH
from evoagentx.models import OpenAILLMConfig, OpenAILLM
from mass_optimizer import (
    MassOptimiser, create_mass_workflow, 
    run_full_optimization, optimize_predictor,
    optimize_summarizer, optimize_aggregator, 
    optimize_reflector, optimize_debater, optimize_executer
)

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


class MathSplits(MATH):
    """数学数据集分割类，匹配原始实现"""
    
    def _load_data(self):
        # 加载原始测试数据
        super()._load_data()
        # 分割数据为dev和test
        import numpy as np 
        np.random.seed(42)
        permutation = np.random.permutation(len(self._test_data))
        full_test_data = self._test_data
        # 随机选择100个样本用于训练，100个样本用于测试（匹配原始实现）
        self._train_data = [full_test_data[idx] for idx in permutation[:100]]
        self._test_data = [full_test_data[idx] for idx in permutation[100:200]]

    def get_input_keys(self):
        return ["problem"]
    
    def evaluate(self, prediction, label):
        return super().evaluate(prediction, label)


def main():
    """主函数：演示完整的MASS优化流程"""
    
    # 创建LLM配置（匹配原始实现）
    executor_config = OpenAILLMConfig(
        model="gpt-4o-mini", 
        openai_key=OPENAI_API_KEY, 
        stream=False, 
        output_response=False
    )
    executor_llm = OpenAILLM(config=executor_config)
    
    optimizer_config = OpenAILLMConfig(
        model="gpt-4o-mini", 
        openai_key=OPENAI_API_KEY, 
        stream=False, 
        output_response=False
    )
    optimizer_llm = OpenAILLM(config=optimizer_config)
    
    # 创建数据集
    benchmark = MathSplits()
    
    print("=== 开始完整的MASS优化流程 ===")
    
    # 方法1: 使用完整的优化流程（推荐）
    print("\n方法1: 使用完整的优化流程")
    best_workflow = run_full_optimization(executor_llm, optimizer_llm, benchmark)
    
    # 测试最终工作流
    print("\n=== 测试最终工作流 ===")
    test_problem = "What is 15 + 27?"
    result, metadata = best_workflow(test_problem)
    print(f"问题: {test_problem}")
    print(f"结果: {result}")
    print(f"元数据: {metadata}")
    
    print("\n=== 优化完成 ===")


def manual_optimization_example():
    """手动优化示例，展示每个步骤"""
    
    # 创建LLM配置
    executor_config = OpenAILLMConfig(
        model="gpt-4o-mini", 
        openai_key=OPENAI_API_KEY, 
        stream=False, 
        output_response=False
    )
    executor_llm = OpenAILLM(config=executor_config)
    
    optimizer_config = OpenAILLMConfig(
        model="gpt-4o-mini", 
        openai_key=OPENAI_API_KEY, 
        stream=False, 
        output_response=False
    )
    optimizer_llm = OpenAILLM(config=optimizer_config)
    
    # 创建数据集
    benchmark = MathSplits()
    
    print("=== 手动优化示例 ===")
    
    # Step 0: 优化 Predictor
    print("\nStep 0: 优化 Predictor...")
    from mass_optimizer import create_predictor_agent
    predictor = create_predictor_agent(executor_llm)
    predictor_score, optimized_predictor = optimize_predictor(predictor, optimizer_llm, benchmark)
    print(f"Predictor优化完成，分数: {predictor_score}")

    # Step 1: 逐个优化每个block
    print("\nStep 1: 逐个优化每个block...")
    
    print("优化 summarizer...")
    optimized_summarizer = optimize_summarizer(optimized_predictor, executor_llm, optimizer_llm, benchmark, predictor_score)
    
    print("优化 aggregator...")
    optimized_aggregator = optimize_aggregator(optimized_predictor, optimizer_llm, benchmark, predictor_score)
    
    print("优化 reflector...")
    optimized_reflector = optimize_reflector(optimized_predictor, executor_llm, optimizer_llm, benchmark, predictor_score)
    
    print("优化 debater...")
    optimized_debater = optimize_debater(optimized_predictor, executor_llm, optimizer_llm, benchmark, predictor_score)
    
    print("优化 executer...")
    optimized_executer = optimize_executer(optimized_predictor, executor_llm, optimizer_llm, benchmark, predictor_score)

    # Step 2: 构建工作流
    print("\nStep 2: 构建工作流...")
    from mass_optimizer import MassWorkflow, MassBlock
    block_workflow = MassWorkflow([
        optimized_summarizer,
        optimized_aggregator,
        optimized_reflector,
        optimized_debater,
        optimized_executer
    ])

    # Step 3: 优化完整工作流（包括n和prompt）
    print("\nStep 3: 优化完整工作流...")
    mass = MassOptimiser(
        workflow=block_workflow,
        optimizer_llm=optimizer_llm,
        max_bootstrapped_demos=3,  # 减少demo数量以加快速度
        max_labeled_demos=8,
        auto="light",
        eval_rounds=1,
        num_threads=2,  # 减少线程数
        save_path="examples/mass/optimization_logs",
        max_steps=3  # 减少优化步数
    )

    best_workflow = mass.optimize(
        benchmark=benchmark,
        softmax_temperature=1.0,
        agent_budget=10
    )
    
    # 保存最佳工作流
    print("\n=== 保存最佳工作流 ===")
    best_workflow.save("examples/mass/best_workflow.json")
    print("最佳工作流已保存到 examples/mass/best_workflow.json")
    
    # 测试最佳工作流
    print("\n=== 测试最佳工作流 ===")
    test_problem = "What is 15 + 27?"
    final_result, final_metadata = best_workflow(test_problem)
    print(f"最终结果: {final_result}")
    print(f"最终元数据: {final_metadata}")
    
    print("\n=== 手动优化完成 ===")


if __name__ == "__main__":
    # 运行完整的优化流程
    main()
    
    # 可选：运行手动优化示例
    # manual_optimization_example()

