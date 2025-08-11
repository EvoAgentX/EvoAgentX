import asyncio
import os
from typing import Dict, List
import re
from collections import Counter

from dotenv import load_dotenv
from tqdm.asyncio import tqdm as aio_tqdm

# 假设 evo2_optimizer.py 文件与此脚本在同一目录或在 Python 路径中
# 该文件应包含我们之前讨论过的 EvopromptOptimizer, GAOptimizer, DEOptimizer 类
from evoagentx.optimizers.evo2_optimizer import DEOptimizer, GAOptimizer
from evoagentx.benchmark.bigbenchhard import BIGBenchHard
from evoagentx.models import OpenAILLM, OpenAILLMConfig
from evoagentx.optimizers.engine.registry import ParamRegistry

class SinglePromptSarcasmClassifier:
    """
    一个使用单一、可进化的“思维链”前缀来分类讽刺的程序。
    """
    def __init__(self, model: OpenAILLM):
        self.model = model

        # 核心任务指令是固定的
        self.task_instruction = "From the two sentences provided, (A) and (B), determine which one is sarcastic. Respond with your final choice wrapped like this: FINAL_ANSWER((A))"
        
        # 这是一个可进化的“节点”，它的初始种群由下面的列表提供。
        # 优化器将在这个种子库的基础上进行演化。
        self.chain_of_thought_prefix = [
            "Let's think step by step.",
            "Let’s work this out in a step by step way to be sure we have the right answer.",
            "First,",
            "Let’s think about this logically.",
            "Let’s solve this problem by splitting it into steps.",
            "Let’s be realistic and think step by step.",
            "Let’s think like a detective step by step.",
            "Let’s think",
            "Before we dive into the answer,",
            "The answer is after the proof.",
            "Let's break this problem down step by step.",
            "We'll tackle this math task one piece at a time.",
            "Let's approach this logically, step by step.",
            "We'll solve this by analyzing each part of the problem.",
            "Let's unravel this mathematical challenge gradually.",
            "We'll methodically work through this problem together.",
            "Let's systematically dissect this math task.",
            "We'll take this mathematical reasoning challenge one step at a time.",
            "Let's meticulously examine each aspect of this problem.",
            "We'll thoughtfully progress through this task step by step."
        ]

    def __call__(self, input: str) -> tuple[str, dict]:
        # 将可进化的前缀和固定的任务指令组合成一个完整的提示
        # 在优化过程中，optimizer会不断更新 self.chain_of_thought_prefix 的值
        full_prompt = f"{self.chain_of_thought_prefix}\n\n{self.task_instruction}\n\nText:\n{input}"
        
        response = self.model.generate(prompt=full_prompt)
        prediction = response.content.strip()
        
        pattern = r'FINAL_ANSWER\((\([^)]*\))\)'
        match = re.search(pattern, prediction)
        
        if match:
            answer = match.group(1)
        else:
            answer = "N/A"
            
        return answer, {"full_prompt": full_prompt}

    def save(self, path: str):
        pass

    def load(self, path: str):
        pass

async def run_node_evolution_example(algorithm="DE", combination_sample_size=10):
    """
    运行单节点演化示例，该节点的初始种群由一个列表提供。
    """
    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not found in environment variables.")

    # --- 配置 ---
    # 由于您提供了20个种子，我们可以将种群大小设置为20以便直接使用
    POPULATION_SIZE = 10 
    ITERATIONS = 10 # 增加迭代次数以看到更明显的进化效果
    CONCURRENCY_LIMIT = 100
    BENCHMARK_DEV_SAMPLES = 50 # 增加评估样本数以获得更可靠的分数

    # --- 设置 LLM ---
    llm_config_optimizer = OpenAILLMConfig(
        model="gpt-3.5-turbo-0125", # 使用更新、性价比更高的模型
        openai_key=OPENAI_API_KEY,
        stream=False,
        temperature=0.5,  # 设置温度以增加生成的多样性
        top_p=0.95  # 使用 top-p 采样来控制生成的多样性
    )
    llm_config_inference = OpenAILLMConfig(
        model="gpt-3.5-turbo-0125", # 使用更新、性价比更高的模型
        openai_key=OPENAI_API_KEY,
        stream=False,
        temperature=0,  # 设置温度以增加生成的多样性
    )
    llm = OpenAILLM(config=llm_config_inference)

    # --- 设置基准测试 ---
    benchmark = BIGBenchHard("snarks", dev_sample_num=BENCHMARK_DEV_SAMPLES)
    benchmark._load_data()

    # --- 创建程序 ---
    program = SinglePromptSarcasmClassifier(model=llm)

    # --- 将提示注册为独立节点以进行独立演化 ---
    # 🔥核心修改：现在我们只追踪一个节点，即包含种子库的 "chain_of_thought_prefix"
    registry = ParamRegistry()
    registry.track(program, "chain_of_thought_prefix", name="cot_prefix_node")

    # --- 选择优化器 ---
    optimizer_class = DEOptimizer if algorithm == "DE" else GAOptimizer
    optimizer = optimizer_class(
        registry=registry,
        program=program,
        population_size=POPULATION_SIZE,
        iterations=ITERATIONS,
        llm_config=llm_config_optimizer,
        concurrency_limit=CONCURRENCY_LIMIT,
        combination_sample_size=combination_sample_size
    )

    # --- 运行优化 ---
    print(f"\n--- Running {algorithm} Optimization ---")
    best_config, _, _ = await optimizer.optimize(benchmark=benchmark)
    
    # 优化器已将 program.chain_of_thought_prefix 设置为找到的最佳提示
    print(f"\n✅ Optimization Complete! Best prompt found: '{program.chain_of_thought_prefix}'")


    # --- 在测试集上评估优化后的程序 ---
    print("\n--- Evaluating on Test Set ---")
    test_data = benchmark.get_test_data()

    async def evaluate_example_concurrently(example: Dict) -> float:
        prediction, _ = await asyncio.to_thread(
            program,
            input=example["input"]
        )
        score_dict = benchmark.evaluate(prediction, benchmark.get_label(example))
        return score_dict.get("em", 0.0)

    if test_data:
        tasks = [evaluate_example_concurrently(ex) for ex in test_data]
        results = await aio_tqdm.gather(*tasks, desc="Evaluating on Test Set")
        correct_count = sum(results)
        test_accuracy = correct_count / len(test_data)
        print(f"Test Accuracy: {test_accuracy:.4f}")
    else:
        test_accuracy = 0.0
    
    # 将最终找到的最佳配置（尽管只有一个）和测试准确率返回
    return best_config, test_accuracy

async def main():
    """
    主函数，演示对单个节点的演化，该节点从一个种子库初始化。
    """
    # 依次运行 DE 和 GA 算法
    results = {}
    for algo in ["DE", "GA"]:
        # 由于只有一个节点，combination_sample_size 不再有意义，可以设为 None
        config, accuracy = await run_node_evolution_example(
            algorithm=algo, 
            combination_sample_size=None
        )
        results[algo] = {"config": config, "accuracy": accuracy}
    
    # --- 输出结果到CSV日志文件 ---
    import csv
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = f"single_node_evolution_log_{timestamp}.csv"
    with open(log_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Algorithm", "Best Prompt", "Test Accuracy"])
        for algo, res in results.items():
            # 从配置字典中提取出提示文本
            prompt_text = list(res["config"].values())[0]
            writer.writerow([algo, prompt_text, f"{res['accuracy']:.4f}"])
            
    print(f"\n📊 Final results saved to {log_path}")

if __name__ == "__main__":
    # 在运行前，请确保您已创建 .env 文件并填入了 OPENAI_API_KEY
    # 同时，确保 evo2_optimizer.py 文件存在且包含了我们之前修改过的代码
    asyncio.run(main())