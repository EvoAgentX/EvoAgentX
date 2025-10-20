"""
Multi-Agent EvoPrompt Workflow Example

This script demonstrates multi-prompt evolution using ensemble voting strategies.
It optimizes multiple prompts simultaneously to improve task performance through
collaborative evolutionary optimization.
"""

import asyncio
import os
import re
from collections import Counter

from dotenv import load_dotenv
from evoagentx.core.logging import logger

from evoagentx.optimizers.evoprompt_optimizer import DEOptimizer, GAOptimizer
from evoagentx.benchmark.bigbenchhard import BIGBenchHard
from evoagentx.models import OpenAILLM, OpenAILLMConfig
from evoagentx.optimizers.engine.registry import ParamRegistry


class SarcasmClassifierProgram:
    """
    Multi-prompt ensemble classifier using majority voting strategy.
    
    This program employs three independent prompt "voters" that can evolve
    independently to achieve better collective performance through diversity.
    """
    
    def __init__(self, model: OpenAILLM):
        """
        Initialize the multi-prompt ensemble classifier.
        
        Args:
            model: The language model to use for inference
        """
        self.model = model
        
        # Three distinct generic prompt nodes for diverse task processing
        self.prompt_direct = "As a straightforward responder, follow the task instruction exactly and provide the final answer."
        self.prompt_expert = "As an expert assistant, interpret the task instruction carefully and provide the final answer."
        self.prompt_cot = "As a thoughtful assistant, think step-by-step, then follow the task instruction and provide the final answer."
        self.task_instruction = "Respond with your final answer wrapped like this: FINAL_ANSWER(ANSWER)"

    def __call__(self, input: str) -> tuple[str, dict]:
        """
        Execute ensemble prediction using majority voting.
        
        Args:
            input: The input text to process
            
        Returns:
            Tuple of (final_answer, metadata)
        """
        answers = []
        prompts = [self.prompt_direct, self.prompt_expert, self.prompt_cot]
        pattern = r"the answer is\s*(.*)"

        # Query each prompt voter independently
        for prompt in prompts:
            full_prompt = f"{prompt}\n\n{self.task_instruction}\n\nText:\n{input}"
            response = self.model.generate(prompt=full_prompt)
            prediction = response.content.strip()
            
            # Extract answer using regex pattern
            match = re.search(pattern, prediction, re.IGNORECASE)
            if match:
                answers.append(match.group(1))

        # Handle case where no valid answers are found
        if not answers:
            return "N/A", {"votes": []}

        # Apply majority voting strategy
        vote_counts = Counter(answers)
        most_common_answer = vote_counts.most_common(1)[0][0]
        
        return most_common_answer, {"votes": answers}

    def save(self, path: str):
        """Save program state (placeholder for future implementation)."""
        pass

    def load(self, path: str):
        """Load program state (placeholder for future implementation)."""
        pass

async def main():
    """Main execution function for multi-agent EvoPrompt optimization."""
    
    # Load environment configuration
    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not found in environment variables.")

    # Configuration parameters
    POPULATION_SIZE = 4
    ITERATIONS = 10
    CONCURRENCY_LIMIT = 100
    COMBINATION_SAMPLE_SIZE = 3  # Sample size per combination
    DEV_SAMPLE_NUM = 15  # Development set sample count

    # Configure LLM for evolution
    evo_llm_config = OpenAILLMConfig(
        model="gpt-4.1-nano",
        openai_key=OPENAI_API_KEY,
        stream=False,
        top_p=0.95,
        temperature=0.5
    )

    # Configure LLM for evaluation
    eval_llm_config = OpenAILLMConfig(
        model="gpt-4.1-nano",
        openai_key=OPENAI_API_KEY,
        stream=False,
        temperature=0
    )
    llm = OpenAILLM(config=eval_llm_config)

    # Tasks to optimize with both DE and GA algorithms
    tasks = [
        "snarks",
        "sports_understanding",
        "logical_deduction_three_objects",
        "dyck_languages",
        "multistep_arithmetic_two",
    ]
    
    # Run optimization for each task
    for task_name in tasks:
        logger.info(f"=== Task: {task_name} ===")
        
        # Set up benchmark and program
        benchmark = BIGBenchHard(task_name, dev_sample_num=DEV_SAMPLE_NUM, seed=10)
        program = SarcasmClassifierProgram(model=llm)
        
        # Register prompt nodes for optimization
        registry = ParamRegistry()
        registry.track(program, "prompt_direct", name="direct_prompt_node")
        registry.track(program, "prompt_expert", name="expert_prompt_node")
        registry.track(program, "prompt_cot", name="cot_prompt_node")

        # Differential Evolution optimizer
        optimizer_DE = DEOptimizer(
            registry=registry,
            program=program,
            population_size=POPULATION_SIZE,
            iterations=ITERATIONS,
            llm_config=evo_llm_config,
            concurrency_limit=CONCURRENCY_LIMIT,
            combination_sample_size=COMBINATION_SAMPLE_SIZE,
            enable_logging=True
        )
        logger.info("Starting DE optimization...")
        await optimizer_DE.optimize(benchmark=benchmark)
        logger.info("DE optimization completed. Starting evaluation...")
        de_metrics = await optimizer_DE.evaluate(benchmark=benchmark, eval_mode="test")
        logger.info(f"DE results for {task_name}: {de_metrics['accuracy']}")

        # Genetic Algorithm optimizer
        optimizer_GA = GAOptimizer(
            registry=registry,
            program=program,
            population_size=POPULATION_SIZE,
            iterations=ITERATIONS,
            llm_config=evo_llm_config,
            concurrency_limit=CONCURRENCY_LIMIT,
            combination_sample_size=COMBINATION_SAMPLE_SIZE,
            enable_logging=True
        )
        logger.info("Starting GA optimization...")
        await optimizer_GA.optimize(benchmark=benchmark)
        logger.info("GA optimization completed. Starting evaluation...")
        ga_metrics = await optimizer_GA.evaluate(benchmark=benchmark, eval_mode="test")
        logger.info(f"GA results for {task_name}: {ga_metrics['accuracy']}")


if __name__ == "__main__":
    asyncio.run(main())
