"""
Minimal Save & Load JSON Demo

This script shows how to:
1) Run a tiny DE optimization (auto-saves best_config.json in the log dir)
2) Load best_config.json into a fresh program via ParamRegistry
3) Use the loaded program to answer a sample question

Note: Requires OPENAI_API_KEY in your environment.
"""

import asyncio
import json
import os
import re

from dotenv import load_dotenv
from evoagentx.core.logging import logger

from evoagentx.optimizers.evoprompt_optimizer import DEOptimizer
from evoagentx.benchmark.bigbenchhard import BIGBenchHard
from evoagentx.models import OpenAILLM, OpenAILLMConfig
from evoagentx.optimizers.engine.registry import ParamRegistry


class SarcasmClassifierProgram:
    """Three-prompt ensemble with majority voting (kept minimal)."""

    def __init__(self, model: OpenAILLM):
        self.model = model
        self.prompt_direct = "As a straightforward responder, follow the task instruction exactly and provide the final answer."
        self.prompt_expert = "As an expert assistant, interpret the task instruction carefully and provide the final answer."
        self.prompt_cot = "As a thoughtful assistant, think step-by-step, then follow the task instruction and provide the final answer."
        self.task_instruction = "Respond with your final answer wrapped like this: FINAL_ANSWER(ANSWER)"

    def __call__(self, input: str) -> tuple[str, dict]:
        answers = []
        prompts = [self.prompt_direct, self.prompt_expert, self.prompt_cot]
        pattern = r"the answer is\s*(.*)"
        for prompt in prompts:
            full_prompt = f"{prompt}\n\n{self.task_instruction}\n\nText:\n{input}"
            response = self.model.generate(prompt=full_prompt)
            text = response.content.strip()
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                answers.append(m.group(1))
        if not answers:
            return "N/A", {"votes": []}
        # majority
        from collections import Counter as _Counter
        vote = _Counter(answers).most_common(1)[0][0]
        return vote, {"votes": answers}


async def main():
    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not found in environment variables.")

    # Tiny run; fast to finish
    POPULATION_SIZE = 2
    ITERATIONS = 1
    DEV_SAMPLE_NUM = 6

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

    # A) Optimize and auto-save best_config.json
    program = SarcasmClassifierProgram(model=llm)
    registry = ParamRegistry()
    registry.track(program, "prompt_direct", name="direct_prompt_node")
    registry.track(program, "prompt_expert", name="expert_prompt_node")
    registry.track(program, "prompt_cot", name="cot_prompt_node")

    optimizer = DEOptimizer(
        registry=registry,
        program=program,
        population_size=POPULATION_SIZE,
        iterations=ITERATIONS,
        llm_config=evo_llm_config,
        concurrency_limit=10,
        combination_sample_size=2,
        enable_logging=True,
    )
    await optimizer.optimize(benchmark=benchmark)
    json_path = os.path.join(optimizer.log_dir, "best_config.json")
    logger.info("best_config.json at: %s", json_path)

    # B) Load JSON into a fresh program and use it
    program2 = SarcasmClassifierProgram(model=llm)
    registry2 = ParamRegistry()
    registry2.track(program2, "prompt_direct", name="direct_prompt_node")
    registry2.track(program2, "prompt_expert", name="expert_prompt_node")
    registry2.track(program2, "prompt_cot", name="cot_prompt_node")

    with open(json_path, "r", encoding="utf-8") as f:
        best_cfg = json.load(f)
    for k, v in best_cfg.items():
        registry2.set(k, v)

    # C) Answer a sample question using the loaded program
    question = "Oh fantastic, my computer crashed again. Such a joy."
    answer, meta = program2(question)
    logger.info("Sample question: %s", question)
    logger.info("Answer (loaded best config): %s", answer)


if __name__ == "__main__":
    asyncio.run(main())
