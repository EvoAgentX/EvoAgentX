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

    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    POPULATION_SIZE = 7
    ITERATIONS = 4
    CONCURRENCY_LIMIT = 25

    OPENAI_LLM_CONFIG = OpenAILLMConfig(
        model="gpt-4o-mini",
        openai_key=OPENAI_API_KEY,
        stream=False
    )
    
    benchmark = BIGBenchHard("snarks", sample_num=50)
    INITIAL_PROMPTS = [
            "From the two sentences provided, (A) and (B), determine which one is sarcastic.",

            "Your job is to spot the sarcasm in one of the following sentences. Please respond with the correct label, (A) or (B).",

            "As a specialist in linguistics, identify the sarcastic statement between option (A) and option (B).",

            "Given sentence (A) and sentence (B), which one is sarcastic? The expected output is the corresponding label.",
            
            "Which of the two sentences, labeled (A) and (B), uses sarcasm? Provide the label of the sentence.",

            "Evaluate the pair of sentences. One is literal, one is sarcastic. Return the label, (A) or (B), for the sarcastic one.",

            "Analyze the tone of sentences (A) and (B). Which one has a sarcastic tone? Answer with '(A)' or '(B)'.",
        ]

    print("Starting EvoPrompt (GA) algorithm (Object-Oriented Version)...\n")
    
    evolver = EvoPromptGA(
        population_size=POPULATION_SIZE,
        iterations=ITERATIONS,
        benchmark=benchmark,
        llm_config=OPENAI_LLM_CONFIG,
        concurrency_limit=CONCURRENCY_LIMIT
    )

    optimal_prompt, bestscore, averagescore = await evolver.optimize(initial_prompts=INITIAL_PROMPTS)

    print(f"\nOptimal prompt found (p*):\n'{optimal_prompt}'")
    print("Bestscore record\n",bestscore)
    print("Averagescore record\n",averagescore)
    print("\nEvaluating the optimal prompt on the test set...")
    test_score = await evolver.evaluate_prompt(optimal_prompt, benchmark._test_data)
    print(f"Final test set score: {test_score:.2f}")

if __name__ == "__main__":
    asyncio.run(main())
    