import asyncio
import os
from typing import Dict
import re
from tqdm.asyncio import tqdm as aio_tqdm
from dotenv import load_dotenv

from evoagentx.optimizers.evoprompt_optimizer import DEOptimizer, _split_mega_prompt_to_config
from evoagentx.benchmark.bigbenchhard import BIGBenchHard
from evoagentx.models import OpenAILLM, OpenAILLMConfig
from evoagentx.optimizers.engine.registry import ParamRegistry
from evoagentx.core.logging import logger

class SarcasmClassifierProgram:
    def __init__(self, model: OpenAILLM):
        self.model = model
        self.instruction = "From the two sentences provided, (A) and (B), determine which one is sarcastic."

    def __call__(self, input: str) -> tuple[str, dict]:
        full_prompt = f"{self.instruction}\n\nText:\n{input}"
        response = self.model.generate(prompt=full_prompt)
        prediction = response.content.strip()
        pattern = r'\([^)]+\)'
        match = re.search(pattern, prediction)
        return (match.group(0) if match else "N/A"), {}

    def save(self, path: str):
        logger.info(f"DUMMY: Saving program state to {path}")
        pass

    def load(self, path: str):
        logger.info(f"DUMMY: Loading program state from {path}")
        pass

async def main():
    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not found in environment variables.")

    POPULATION_SIZE = 5
    ITERATIONS = 5
    CONCURRENCY_LIMIT = 100

    llm_config = OpenAILLMConfig(
        model="gpt-4o-mini",
        openai_key=OPENAI_API_KEY,
        stream=False
    )
    llm = OpenAILLM(config=llm_config)

    benchmark = BIGBenchHard("snarks", sample_num=25)
    benchmark._load_data()

    program = SarcasmClassifierProgram(model=llm)

    registry = ParamRegistry()
    registry.track(program, "instruction", name="sarcasm_instruction")

    logger.info("Starting EvoPrompt Optimizer...")

    optimizer = DEOptimizer(
        registry=registry,
        program=program,
        population_size=POPULATION_SIZE,
        iterations=ITERATIONS,
        llm_config=llm_config,
        concurrency_limit=CONCURRENCY_LIMIT
    )

    best_mega_prompt, best_scores, avg_scores = await optimizer.optimize(benchmark=benchmark)

    logger.info("\n--- Evolution Complete ---")

    final_config = _split_mega_prompt_to_config(best_mega_prompt)
    best_instruction = final_config.get('sarcasm_instruction', 'Error: prompt not found.')

    logger.info(f"Initial instruction was: '{SarcasmClassifierProgram(None).instruction}'")
    logger.info(f"Final optimized instruction is: '{best_instruction}'")

    logger.info("\n--- Performance Record ---")
    for gen, score in best_scores.items():
        logger.info(f"{gen}: Best Score = {score:.4f}, Average Score = {avg_scores[gen]:.4f}")

    logger.info("\nEvaluating the final optimized program on the unseen test set (concurrently)...")
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
        results = await aio_tqdm.gather(*tasks, desc="Final Concurrent Test Evaluation")
        correct_count = sum(results)
        test_accuracy = correct_count / len(test_data)
    else:
        test_accuracy = 0.0

    logger.info(f"Final test set accuracy with optimized prompt: {test_accuracy:.4f}")

if __name__ == "__main__":
    asyncio.run(main())