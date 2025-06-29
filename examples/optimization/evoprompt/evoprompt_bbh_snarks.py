# -*- coding: utf-8 -*-
"""
EvoPrompt (GA) Implementation. (Object-Oriented Version)

This script refactors the original procedural code into a class-based structure.
The EvoPromptGA class encapsulates the entire logic for evolving prompts,
making the process more organized, reusable, and maintainable.
"""

import os
import asyncio
from dotenv import load_dotenv

from evoagentx.benchmark.bigbenchhard import BIGBenchHard
from evoagentx.models import OpenAILLMConfig
from evoagentx.optimizers.evoprompt_optimizer import EvoPromptGA



async def main():
    """
    主函数，用于设置和运行 EvoPrompt (GA) 实验。
    """
    # --- 加载配置和定义常量 ---
    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    # 算法参数
    POPULATION_SIZE = 7
    ITERATIONS = 4
    CONCURRENCY_LIMIT = 25

    # 所有智能体共享的 LLM 配置
    OPENAI_LLM_CONFIG = OpenAILLMConfig(
        model="gpt-4o-mini",
        openai_key=OPENAI_API_KEY,
        stream=False
    )
    
    # 实验设置
    benchmark = BIGBenchHard("snarks", sample_num=50)
    INITIAL_PROMPTS = [
            # 中等性能 1: 指令清晰，但完全没有格式说明
            "From the two sentences provided, (A) and (B), determine which one is sarcastic.",

            # 中等性能 2: 使用了较温和的格式建议
            "Your job is to spot the sarcasm in one of the following sentences. Please respond with the correct label, (A) or (B).",

            # 中等性能 3: 角色扮演很好，但缺少严格的格式限制
            "As a specialist in linguistics, identify the sarcastic statement between option (A) and option (B).",

            # 中等性能 4: 提到了“预期输出”，但不如强制命令有效
            "Given sentence (A) and sentence (B), which one is sarcastic? The expected output is the corresponding label.",
            
            # 中等性能 5: 直接提问，并要求提供标签，但语气不强制
            "Which of the two sentences, labeled (A) and (B), uses sarcasm? Provide the label of the sentence.",

            # 中等性能 6: 任务描述稍微复杂，但核心指令明确
            "Evaluate the pair of sentences. One is literal, one is sarcastic. Return the label, (A) or (B), for the sarcastic one.",

            # 中等性能 7: 比较好的一个提示，但仍有优化的空间
            "Analyze the tone of sentences (A) and (B). Which one has a sarcastic tone? Answer with '(A)' or '(B)'.",
        ]

    print("Starting EvoPrompt (GA) algorithm (Object-Oriented Version)...\n")
    
    # 1. 实例化 EvoPromptGA 类
    evolver = EvoPromptGA(
        population_size=POPULATION_SIZE,
        iterations=ITERATIONS,
        benchmark=benchmark,
        llm_config=OPENAI_LLM_CONFIG,
        concurrency_limit=CONCURRENCY_LIMIT
    )

    # 2. 运行进化过程
    optimal_prompt, bestscore, averagescore = await evolver.optimize(initial_prompts=INITIAL_PROMPTS)

    print(f"\nOptimal prompt found (p*):\n'{optimal_prompt}'")
    print("Bestscore record\n",bestscore)
    print("Averagescore record\n",averagescore)
    print("\nEvaluating the optimal prompt on the test set...")
    test_score = await evolver.evaluate_prompt(optimal_prompt, benchmark._test_data)
    print(f"Final test set score: {test_score:.2f}")

if __name__ == "__main__":
    asyncio.run(main())
    