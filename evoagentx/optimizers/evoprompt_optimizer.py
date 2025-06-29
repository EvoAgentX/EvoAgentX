import os
import random
import re
import asyncio
import numpy as np
from tqdm.asyncio import tqdm as aio_tqdm

from evoagentx.agents import CustomizeAgent
from evoagentx.benchmark.bigbenchhard import BIGBenchHard
from evoagentx.models import OpenAILLMConfig

class EvoPromptGA:
    """
    A class that encapsulates EvoPrompt optimization using a Genetic Algorithm (GA).

    This class manages the entire process of agents, population, evaluation, and the evolutionary loop.
    """

    def __init__(self, population_size: int, iterations: int, benchmark: BIGBenchHard, llm_config: OpenAILLMConfig, concurrency_limit: int = 5):
        """
        Initializes the EvoPromptGA evolver.

        Args:
            population_size (int): The number of individuals (prompts) in the population.
            iterations (int): The number of evolutionary iterations (generations).
            benchmark (BIGBenchHard): The benchmark instance used to evaluate prompt performance.
            llm_config (OpenAILLMConfig): The language model configuration for all agents.
            concurrency_limit (int): The maximum number of concurrent API calls.
        """
        self.population_size = population_size
        self.iterations = iterations
        self.benchmark = benchmark
        self.llm_config = llm_config
        self.semaphore = asyncio.Semaphore(concurrency_limit)
        self.bestscore = {}
        self.averagescore = {}
        self.population = []
        self.scores = []

        # --- Define agents within the class ---
        self.eval_agent = CustomizeAgent(
            name="EvalAgent",
            description="An agent to evaluate prompts based on a given question.",
            prompt="""Instruction:
                        {instruction}

                        Question:
                        {question}

                        Answer:""",
            llm_config=self.llm_config,
            inputs=[
                {"name": "instruction", "type": "string", "description": "The instruction for the agent."},
                {"name": "question", "type": "string", "description": "The question to answer."}
            ]
        )

        self.paraphrase_agent = CustomizeAgent(
            name="ParaphraseAgent",
            description="An agent that paraphrases a given instruction.",
            prompt="Generate a variation of the following instruction while keeping the semantic meaning. Input: {instruction} Output:",
            llm_config=self.llm_config,
            inputs=[
                {"name": "instruction", "type": "string", "description": "The instruction to paraphrase."},
            ]
        )

        self.evo_agent = CustomizeAgent(
            name="EvoAgent",
            description="An agent that evolves a new prompt from two parent prompts.",
            prompt="""Please follow the instruction step-by-step to generate a better prompt.
            1. Crossover the following prompts to generate a new prompt:
            Prompt 1: Your task is to classify the comment as one of the following categories: terrible, bad, okay, good, great.
            Prompt 2: In this task, you are given sentences from movie reviews. The task is to classify a sentence as one of the following categories: terrible, bad, okay, good, great.
            2. Mutate the prompt generated in Step 1 and generate a final prompt bracketed with <prompt> and </prompt>.

            1. Crossover Prompt: In this task, you are given comments from movie reviews. Your task is to classify each comment as one of the following categories: terrible, bad, okay, good, great.
            2. <FinalEvoPrompt>Given a sentence from a movie review, classify it into one of the following categories: terrible, bad, okay, good, or great.</FinalEvoPrompt>

            Please follow the instruction step-by-step to generate a better prompt.
            1. Crossover the following prompts and generate a new prompt:
            Prompt 1: {parent1}
            Prompt 2: {parent2}
            2. Mutate the prompt generated in Step 1 and generate a final prompt bracketed with <FinalEvoPrompt> and </FinalEvoPrompt>.""",
            llm_config=self.llm_config,
            inputs=[
                {"name": "parent1", "type": "string", "description": "The first parent prompt."},
                {"name": "parent2", "type": "string", "description": "The second parent prompt."}
            ]
        )

    async def _evaluate_single_example(self, prompt: str, example: dict) -> float:
        """Internal helper: Evaluates a prompt on a single example, protected by a concurrency lock."""
        async with self.semaphore:
            answer = await asyncio.to_thread(self.eval_agent, inputs={"instruction": prompt, "question": example["input"]})
            label = self.benchmark.get_label(example)
            return self.benchmark.evaluate(answer.content.content, label)["em"]

    async def evaluate_prompt(self, prompt: str, dev_set: list) -> float:
        """
        Concurrently evaluates the performance of a single prompt on the development set.
        """
        if not dev_set:
            return 0.0
        tasks = [self._evaluate_single_example(prompt, example) for example in dev_set]
        results = await asyncio.gather(*tasks)
        return sum(results) / len(dev_set)

    async def _perform_paraphrase(self, instruction: str) -> str:
        """Performs paraphrasing concurrently using an LLM."""
        async with self.semaphore:
            output = await asyncio.to_thread(self.paraphrase_agent, inputs={"instruction": instruction})
            return output.content.content

    async def _perform_ga_evolution(self, parent1: str, parent2: str) -> str:
        """Concurrently evolves a new prompt from two parents via crossover and mutation."""
        async with self.semaphore:
            output = await asyncio.to_thread(self.evo_agent, inputs={"parent1": parent1, "parent2": parent2})
            child_prompt_raw = output.content.content.strip()
            match = re.search(r"<FinalEvoPrompt>(.*?)</FinalEvoPrompt>", child_prompt_raw, re.DOTALL)
            if match:
                return match.group(1).strip()
            print("Warning: No <FinalEvoPrompt> tag found. Using raw LLM output.")
            return child_prompt_raw

    async def _initialize_population(self, initial_prompts: list[str]):
        """Initializes the population, generating more individuals by paraphrasing if needed."""
        self.population = list(initial_prompts)
        if len(self.population) < self.population_size:
            print(f"Initial prompts ({len(self.population)}) less than population size ({self.population_size}). Populating...")
            num_to_generate = self.population_size - len(self.population)
            paraphrase_tasks = [self._perform_paraphrase(random.choice(self.population)) for _ in range(num_to_generate)]
            new_prompts = await aio_tqdm.gather(*paraphrase_tasks, desc="Paraphrasing")
            self.population.extend(new_prompts)

        self.population = self.population[:self.population_size]

    async def optimize(self, initial_prompts: list[str]) -> str:
        """
        The main flow for running the EvoPrompt (GA) algorithm.

        Args:
            initial_prompts (list[str]): The list of initial prompts to start the evolution.

        Returns:
            str: The best prompt found after evolution.
        """
        # 1. Initialization
        await self._initialize_population(initial_prompts)

        # Initial fitness evaluation (concurrent)
        print("--- Step 1: Evaluating initial population... ---")
        eval_tasks = [self.evaluate_prompt(p, self.benchmark._dev_data) for p in self.population]
        self.scores = await aio_tqdm.gather(*eval_tasks, desc="Initial Evaluation")
        print(f"Initial best score: {max(self.scores):.2f}\n")

        # 2. Main evolutionary loop
        for t in range(self.iterations):
            print(f"--- Generation {t + 1}/{self.iterations} ---")

            total_fitness = sum(self.scores)
            probabilities = [s / total_fitness for s in self.scores] if total_fitness > 0 else None

            # 3. Concurrently generate N new children
            evolution_tasks = [self._perform_ga_evolution(
                *random.choices(self.population, weights=probabilities, k=2)
            ) for _ in range(self.population_size)]

            new_children = await aio_tqdm.gather(*evolution_tasks, desc=f"Generating Gen {t+1}")

            # 4. Concurrently evaluate new children
            new_eval_tasks = [self.evaluate_prompt(p, self.benchmark._dev_data) for p in new_children]
            new_scores = await aio_tqdm.gather(*new_eval_tasks, desc=f"Evaluating Gen {t+1}")

            # 5. Update population (elitism)
            combined_population = self.population + new_children
            combined_scores = self.scores + new_scores

            scored_candidates = sorted(zip(combined_scores, combined_population), key=lambda x: x[0], reverse=True)
            top_n = scored_candidates[:self.population_size]
            self.scores, self.population = [list(t) for t in zip(*top_n)]
            self.bestscore[f"--- Generation {t + 1}"] = max(self.scores)
            self.averagescore[f"--- Generation {t + 1}"] = np.mean(self.scores)
            print(f"Generation best score: {max(self.scores):.2f}")
            print(f"Generation average score: {np.mean(self.scores):.2f}\n")

        # 6. Return the best prompt found
        print("--- Evolution complete ---")
        best_score = max(self.scores)
        best_prompt = self.population[self.scores.index(best_score)]

        return best_prompt, self.bestscore, self.averagescore