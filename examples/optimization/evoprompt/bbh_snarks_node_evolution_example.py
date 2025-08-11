import asyncio
import os
from typing import Dict, List
import re
from collections import Counter

from dotenv import load_dotenv
from tqdm.asyncio import tqdm as aio_tqdm

from evoagentx.optimizers.evo2_optimizer import DEOptimizer, GAOptimizer
from evoagentx.benchmark.bigbenchhard import BIGBenchHard
from evoagentx.models import OpenAILLM, OpenAILLMConfig
from evoagentx.optimizers.engine.registry import ParamRegistry

class SarcasmClassifierProgram:
    """
    一个使用三提示多数投票集成来分类讽刺的程序。
    每个提示都是一个独立的“投票者”，可以独立进化。
    """
    def __init__(self, model: OpenAILLM):
        self.model = model
        # 三个不同的提示，充当三个独立的投票者。
        # 每个提示代表一个可以独立进化的不同节点。
        self.prompt_direct = "From the two sentences provided, (A) and (B), determine which one is sarcastic. Respond with your final choice wrapped like this: FINAL_ANSWER((A))"
        self.prompt_expert = "You are an expert in linguistics and humor. Analyze the following two sentences, (A) and (B), and identify the sarcastic one. Your answer must be wrapped like this: FINAL_ANSWER((B))"
        self.prompt_cot = "Consider the context and potential double meanings in sentences (A) and (B). Which one uses sarcasm? Think step-by-step and conclude with your final choice, wrapped like this: FINAL_ANSWER((A))"

    def __call__(self, input: str) -> tuple[str, dict]:
        answers = []
        prompts = [self.prompt_direct, self.prompt_expert, self.prompt_cot]
        pattern = r'FINAL_ANSWER\((\([^)]*\))\)'

        for prompt in prompts:
            full_prompt = f"{prompt}\n\nText:\n{input}"
            response = self.model.generate(prompt=full_prompt)
            prediction = response.content.strip()
            
            match = re.search(pattern, prediction)
            if match:
                answers.append(match.group(1))

        if not answers:
            return "N/A", {"votes": []}

        vote_counts = Counter(answers)
        most_common_answer = vote_counts.most_common(1)[0][0]
        
        return most_common_answer, {"votes": answers}

    def save(self, path: str):
        # 此处可添加保存逻辑
        pass

    def load(self, path: str):
        # 此处可添加加载逻辑
        pass

async def run_node_evolution_example(algorithm="DE", combination_sample_size=None):
    """
    运行节点级演化示例。
    
    Args:
        algorithm: "DE" 代表差分进化, "GA" 代表遗传算法
        combination_sample_size: 用于评估的组合样本数量
    """
    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not found in environment variables.")

    # 配置
    POPULATION_SIZE = 4
    ITERATIONS = 3
    CONCURRENCY_LIMIT = 50
    


    # 设置 LLM
    llm_config = OpenAILLMConfig(
        model="gpt-3.5-turbo-0125",
        openai_key=OPENAI_API_KEY,
        stream=False
    )
    llm = OpenAILLM(config=llm_config)

    # 设置基准测试
    benchmark = BIGBenchHard("snarks", dev_sample_num=5)
    benchmark._load_data()

    # 创建程序
    program = SarcasmClassifierProgram(model=llm)

    # 将提示注册为独立节点以进行独立演化
    registry = ParamRegistry()
    registry.track(program, "prompt_direct", name="direct_prompt_node")
    registry.track(program, "prompt_expert", name="expert_prompt_node")  
    registry.track(program, "prompt_cot", name="cot_prompt_node")

    # 选择优化器
    if algorithm == "DE":
        optimizer = DEOptimizer(
            registry=registry,
            program=program,
            population_size=POPULATION_SIZE,
            iterations=ITERATIONS,
            llm_config=llm_config,
            concurrency_limit=CONCURRENCY_LIMIT,
            combination_sample_size=combination_sample_size
        )
    elif algorithm == "GA":
        optimizer = GAOptimizer(
            registry=registry,
            program=program,
            population_size=POPULATION_SIZE,
            iterations=ITERATIONS,
            llm_config=llm_config,
            concurrency_limit=CONCURRENCY_LIMIT,
            combination_sample_size=combination_sample_size
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    # 运行优化
    best_config, _, _ = await optimizer.optimize(benchmark=benchmark)

    # 在测试集上评估优化后的程序
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
        results = await aio_tqdm.gather(*tasks)
        correct_count = sum(results)
        test_accuracy = correct_count / len(test_data)
    else:
        test_accuracy = 0.0
    
    return best_config, test_accuracy

async def main():
    """
    主函数，演示 GA 和 DE 的节点级演化。
    """
    # 运行 DE 示例
    de_config, de_accuracy = await run_node_evolution_example(
        algorithm="DE", 
        combination_sample_size=4
    )
    
    # 运行 GA 示例
    ga_config, ga_accuracy = await run_node_evolution_example(
        algorithm="GA",
        combination_sample_size=4
    )
    
    # 输出结果到CSV日志文件
    import csv
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = f"node_evolution_log_{timestamp}.csv"
    with open(log_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Algorithm", "Config", "Test Accuracy"])
        writer.writerow(["DE", str(de_config), f"{de_accuracy:.4f}"])
        writer.writerow(["GA", str(ga_config), f"{ga_accuracy:.4f}"])
    print(f"结果已保存到 {log_path}")

if __name__ == "__main__":
    asyncio.run(main())