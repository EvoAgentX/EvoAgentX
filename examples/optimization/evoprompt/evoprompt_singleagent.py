"""
Single-Agent EvoPrompt Optimization Example

This script demonstrates single-prompt evolution using both GA and DE optimizers.
It optimizes a single chain-of-thought prefix prompt for better task performance.
"""

import asyncio
import os
import re

from dotenv import load_dotenv
from evoagentx.core.logging import logger

from evoagentx.optimizers.evoprompt_optimizer import DEOptimizer, GAOptimizer
from evoagentx.benchmark.bigbenchhard import BIGBenchHard
from evoagentx.models import OpenAILLM, OpenAILLMConfig
from evoagentx.optimizers.engine.registry import ParamRegistry


class SinglePromptProgram:
    """
    A program that uses a single, evolvable prompt to process tasks.
    
    This program uses few-shot learning combined with an evolvable chain-of-thought
    prefix to improve task performance through evolutionary optimization.
    """
    
    def __init__(self, model: OpenAILLM, task_name: str):
        """
        Initialize the single prompt program.
        
        Args:
            model: The language model to use for inference
            task_name: Name of the task for loading few-shot examples
        """
        self.model = model
        self.task_name = task_name
        
        # Load task-specific few-shot prompt examples
        lib_path = os.path.join(os.path.dirname(__file__), 'lib_prompt', f'{task_name}.txt')
        try:
            with open(lib_path, 'r', encoding='utf-8') as f:
                examples = [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            examples = []
        self.fewshot_prompt = '\n'.join(examples)
        
        # Evolvable chain-of-thought prefixes - optimizer will evolve these
        self.chain_of_thought_prefix = [
            "Let's think step by step.",
            "Let's work this out in a step by step way to be sure we have the right answer.",
            "First,",
            "Let's think about this logically.",
            "Let's solve this problem by splitting it into steps.",
            "Let's be realistic and think step by step.",
            "Let's think like a detective step by step.",
            "Let's think",
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
        self.task_prompt = "Please provide the answer in the format: 'the answer is ."
    
    def __call__(self, input: str) -> tuple[str, dict]:
        """
        Execute the program with the given input.
        
        Args:
            input: The input text to process
            
        Returns:
            Tuple of (answer, metadata)
        """
        # Select current prompt prefix (after optimization, may be string instead of list)
        prefix = (self.chain_of_thought_prefix[0] 
                 if isinstance(self.chain_of_thought_prefix, list) 
                 else self.chain_of_thought_prefix)
        
        # Build few-shot prompt
        prompt_body = []
        if self.fewshot_prompt:
            # Replace all '<prompt>' placeholders with current prefix
            prompt_body.append(self.fewshot_prompt.replace("<prompt>", prefix))
        prompt_body.append(f"Q: {input}")
        prompt_body.append(f"A: {prefix}")
        full_prompt = f'\n'.join(prompt_body) + f"{self.task_prompt}"
        
        # Call model and extract answer
        response = self.model.generate(prompt=full_prompt)
        text = response.content.strip()
        
        # Match 'the answer is (B)' and extract content after 'is'
        match = re.search(r"the answer is\s*(.*)", text, re.IGNORECASE)
        answer = match.group(1).strip().rstrip('.') if match else "N/A"
        
        return answer, {"full_prompt": full_prompt}

    def save(self, path: str):
        """Save the program state (placeholder for future implementation)."""
        pass

    def load(self, path: str):
        """Load the program state (placeholder for future implementation)."""
        pass


async def main():
    """Main execution function for single-agent EvoPrompt optimization."""
    
    # Load environment variables
    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not found in environment variables.")

    # Configuration parameters
    POPULATION_SIZE = 10
    ITERATIONS = 10
    CONCURRENCY_LIMIT = 7
    DEV_SAMPLE_NUM = 50

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

    # Tasks to optimize
    tasks = [
        "geometric_shapes",
        "multistep_arithmetic_two"
    ]
    
    # Run optimization for each task
    for task_name in tasks:
        logger.info(f"=== Task: {task_name} ===")
        
        # Set up benchmark and program
        benchmark = BIGBenchHard(task_name, dev_sample_num=DEV_SAMPLE_NUM, seed=10)
        program = SinglePromptProgram(model=llm, task_name=task_name)
        
        # Register single prompt node for optimization
        registry = ParamRegistry()
        registry.track(program, "chain_of_thought_prefix", name="cot_prefix_node")

        # Differential Evolution optimizer
        logger.info(f"Creating DE optimizer with concurrency_limit={CONCURRENCY_LIMIT}")
        optimizer_DE = DEOptimizer(
            registry=registry,
            program=program,
            population_size=POPULATION_SIZE,
            iterations=ITERATIONS,
            llm_config=evo_llm_config,
            concurrency_limit=CONCURRENCY_LIMIT,
            enable_logging=True,
            enable_early_stopping=True,
            early_stopping_patience=10
        )
        
        logger.info("Starting DE optimization...")
        await optimizer_DE.optimize(benchmark=benchmark)
        logger.info("DE optimization completed. Starting evaluation...")
        de_metrics = await optimizer_DE.evaluate(benchmark=benchmark, eval_mode="test")
        logger.info("DE evaluation completed.")
        logger.info(f"DE results for {task_name}: {de_metrics['accuracy']}")

        # Genetic Algorithm optimizer
        logger.info(f"Creating GA optimizer with concurrency_limit={CONCURRENCY_LIMIT}")
        optimizer_GA = GAOptimizer(
            registry=registry,
            program=program,
            population_size=POPULATION_SIZE,
            iterations=ITERATIONS,
            llm_config=evo_llm_config,
            concurrency_limit=CONCURRENCY_LIMIT,
            enable_logging=True,
            enable_early_stopping=True,
            early_stopping_patience=10
        )
        
        logger.info("Starting GA optimization...")
        await optimizer_GA.optimize(benchmark=benchmark)
        logger.info("GA optimization completed. Starting evaluation...")
        ga_metrics = await optimizer_GA.evaluate(benchmark=benchmark, eval_mode="test")
        logger.info(f"GA results for {task_name}: {ga_metrics['accuracy']}")


if __name__ == "__main__":
    asyncio.run(main())