"""
Best-Config JSON Demonstration

This example shows how EvoPrompt optimizers now:
- auto-save a machine-readable best_config.json into the log directory, and
- load and apply that JSON to a fresh program instance for immediate reuse.

Flow:
1) Run a short DE optimization on BIG-Bench Hard ("snarks") to get optimized prompts.
2) Load best_config.json into a fresh program/registry.
3) Use the optimized workflow to answer a sample question (hard-coded in the script).

Note: This example requires OPENAI_API_KEY in your environment.
"""

import asyncio
import os
import re
from collections import Counter

from dotenv import load_dotenv
from evoagentx.core.logging import logger

from evoagentx.optimizers.evoprompt_optimizer import DEOptimizer
from evoagentx.benchmark.bigbenchhard import BIGBenchHard
from evoagentx.models import OpenAILLM, OpenAILLMConfig
from evoagentx.optimizers.engine.registry import ParamRegistry


class SarcasmClassifierProgram:
    """
    Multi-prompt ensemble classifier using majority voting strategy.

    Three independent prompt "voters" that can evolve independently.
    """

    def __init__(self, model: OpenAILLM):
        self.model = model

        # Three distinct generic prompt nodes for diverse task processing
        self.prompt_direct = (
            "As a straightforward responder follow the task instruction exactly and provide the final answer."
        )
        self.prompt_expert = (
            "As an expert assistant interpret the task instruction carefully and provide the final answer."
        )
        self.prompt_cot = (
            "As a thoughtful assistant think step-by-step, then follow the task instruction and provide the final answer."
        )
        self.task_instruction = (
            "Respond with your final answer wrapped like this: FINAL_ANSWER(ANSWER)"
        )

    def __call__(self, input: str) -> tuple[str, dict]:
        answers = []
        prompts = [self.prompt_direct, self.prompt_expert, self.prompt_cot]
        pattern = r"FINAL_ANSWER\((.*?)\)"

        for prompt in prompts:
            full_prompt = f"{prompt}\n\n{self.task_instruction}\n\nText:\n{input}"
            response = self.model.generate(prompt=full_prompt)
            prediction = response.content.strip()

            match = re.search(pattern, prediction, re.IGNORECASE)
            if match:
                answers.append(match.group(1))

        if not answers:
            return "N/A", {"votes": []}

        vote_counts = Counter(answers)
        most_common_answer = vote_counts.most_common(1)[0][0]

        return most_common_answer, {"votes": answers}

    def save(self, path: str):
        pass

    def load(self, path: str):
        pass


async def main():
    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not found in environment variables.")

    # Keep the run small for a quick demo
    POPULATION_SIZE = 3
    ITERATIONS = 2
    CONCURRENCY_LIMIT = 2
    COMBINATION_SAMPLE_SIZE = 2
    DEV_SAMPLE_NUM = 4

    # LLM configuration
    evo_llm_config = OpenAILLMConfig(
        model="gpt-4.1-nano",
        openai_key=OPENAI_API_KEY,
        stream=False,
        top_p=0.95,
        temperature=0.5,
    )

    eval_llm_config = OpenAILLMConfig(
        model="gpt-4.1-nano",
        openai_key=OPENAI_API_KEY,
        stream=False,
        temperature=0,
    )
    llm = OpenAILLM(config=eval_llm_config)

    task_name = "snarks"
    benchmark = BIGBenchHard(task_name, dev_sample_num=DEV_SAMPLE_NUM, seed=10)

    # Phase A: run DE optimization (this will auto-save best_config.json)
    program_a = SarcasmClassifierProgram(model=llm)
    registry_a = ParamRegistry()
    registry_a.track(program_a, "prompt_direct", name="direct_prompt_node")
    registry_a.track(program_a, "prompt_expert", name="expert_prompt_node")
    registry_a.track(program_a, "prompt_cot", name="cot_prompt_node")

    optimizer_a = DEOptimizer(
        registry=registry_a,
        program=program_a,
        population_size=POPULATION_SIZE,
        iterations=ITERATIONS,
        llm_config=evo_llm_config,
        concurrency_limit=CONCURRENCY_LIMIT,
        combination_sample_size=COMBINATION_SAMPLE_SIZE,
        enable_logging=True,
    )

    logger.info("Starting DE optimization (Phase A)...")
    await optimizer_a.optimize(benchmark=benchmark)
    logger.info("Optimization complete.")

    # Path to the saved JSON
    json_path = os.path.join(optimizer_a.log_dir, "best_config.json")
    logger.info(f"best_config.json saved at: {json_path}")

    # Phase B: answer a sample question using the optimized workflow
    question = "Oh great, another Monday morning meeting. Can't wait."
    logger.info("Sample question: %s", question)
    # Note: optimize() has already applied the best configuration to program_a
    answer, _ = program_a(question)
    logger.info("Answer (optimized workflow): %s", answer)


if __name__ == "__main__":
    asyncio.run(main())
