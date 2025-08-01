"""
EvoPrompt Optimizer - A prompt optimizer based on evolutionary algorithms.

This module implements an optimizer that uses genetic algorithms to enhance prompts.
It can optimize multiple prompts simultaneously, treating them as a single entity 
for evolution to achieve better overall performance.
"""

# Standard Library
import asyncio
import random
import re
from typing import Callable, Dict, List

# Third Party
import numpy as np
from tqdm.asyncio import tqdm as aio_tqdm

# Local
from evoagentx.agents import CustomizeAgent
from evoagentx.benchmark.bigbenchhard import BIGBenchHard
from evoagentx.core.logging import logger
from evoagentx.models import OpenAILLMConfig
from evoagentx.optimizers.engine.base import BaseOptimizer
from evoagentx.optimizers.engine.registry import ParamRegistry

# ----------------- Helper Functions -----------------

def _build_mega_prompt_from_config(config: dict) -> str:
    """
    Concatenates a configuration dictionary into a single string with dynamic tags.

    Args:
        config: A dictionary containing the prompt configuration, where keys are
                tag names and values are the prompt content.

    Returns:
        A single string with tagged prompt parts.
    """
    parts = []
    for name, content in config.items():
        parts.append(f"<{name}>{content}</{name}>")
    return "\n".join(parts)


def _split_mega_prompt_to_config(mega_prompt: str) -> Dict[str, str]:
    """
    Parses a single string with XML-style tags into a configuration dictionary.

    Args:
        mega_prompt: A prompt string with <tag>content</tag> style tags.

    Returns:
        A dictionary parsed from the mega prompt.
    """
    pattern = re.compile(r"<(\w+)>(.*?)</\1>", re.DOTALL)
    matches = pattern.findall(mega_prompt)
    return {name: content.strip() for name, content in matches}

# ----------------- Main Optimizer Class -----------------

class EvopromptOptimizer(BaseOptimizer):
    """
    An optimizer that uses a genetic algorithm to evolve multiple prompts registered
    in the registry. The evaluation logic is internalized, directly using 
    benchmark.evaluate().
    """

    def __init__(self,
                 registry: ParamRegistry,
                 program: Callable,
                 population_size: int,
                 iterations: int,
                 llm_config: OpenAILLMConfig,
                 concurrency_limit: int = 10):
        """
        Initializes the optimizer. The evaluator parameter from the parent class
        is no longer needed as evaluation is handled internally.
        """
        super().__init__(registry=registry, program=program)
        
        self.population_size = population_size
        self.iterations = iterations
        self.llm_config = llm_config
        self.semaphore = asyncio.Semaphore(concurrency_limit)
        
        # Initialize state tracking variables
        self.population: List[str] = []
        self.scores: List[float] = []
        self.best_scores_per_gen: Dict[str, float] = {}
        self.avg_scores_per_gen: Dict[str, float] = {}

        # --- Evolutionary Agents ---
        self.paraphrase_agent = CustomizeAgent(
            name="ParaphraseAgent",
            description="An agent that paraphrases a given instruction.",
            prompt="""Task: Generate a semantically equivalent but differently worded version of the user-provided instruction, while strictly preserving the original XML tags.
                        Here is an example of the correct format:
                        Input: "<example_tag>This is a test.</example_tag>"
                        Output: "<example_tag>Here is a test case.</example_tag>"
                        Now, please process the following instruction exactly according to the example above:
                        Input: {instruction}
                        Output:""",
            llm_config=self.llm_config,
            inputs=[
                {"name": "instruction", "type": "string", "description": "The instruction to paraphrase."},
            ]
        )
        
    async def _evaluate_program_on_example(self, program: Callable, benchmark: BIGBenchHard, example: Dict) -> float:
        """Internal helper to evaluate the program on a single example concurrently."""
        async with self.semaphore:
            try:
                # 1. Prepare inputs
                inputs = {k: v for k, v in example.items() if k in benchmark.get_input_keys()}
                
                # 2. Run the program to get a prediction
                prediction, _ = await asyncio.to_thread(program, **inputs)
                
                # 3. Get the ground truth label
                label = benchmark.get_label(example)
                
                # 4. Call the benchmark's evaluate method to get a score
                #    We assume it returns a dictionary and we use the "em" key.
                score_dict = benchmark.evaluate(prediction, label)
                return score_dict.get("em", 0.0)
            except Exception as e:
                logger.error(f"Error evaluating single example: {e}")
                return 0.0
            
    async def evaluate_mega_prompt_fitness(self, mega_prompt: str, benchmark: BIGBenchHard, dev_set: list) -> float:
        """
        Calculates the fitness of a mega_prompt by evaluating it on the dev set.
        This method internalizes the evaluation loop instead of using an external evaluator.
        """
        original_config = self.get_current_cfg()
        try:
            new_config = _split_mega_prompt_to_config(mega_prompt)
            if not new_config or new_config.keys() != original_config.keys():
                logger.warning("Evolved mega_prompt has corrupted structure. Assigning fitness of 0.")
                return 0.0

            self.apply_cfg(new_config)
            
            # Create a list of concurrent evaluation tasks
            tasks = [self._evaluate_program_on_example(self.program, benchmark, ex) for ex in dev_set]
            results = await asyncio.gather(*tasks)
            
            # Calculate the average score
            return sum(results) / len(results) if results else 0.0

        except Exception as e:
            logger.error(f"Error during fitness evaluation: {e}")
            return 0.0
        finally:
            # Restore the original configuration to avoid side effects
            self.apply_cfg(original_config)

    async def _perform_paraphrase(self, mega_prompt: str) -> str:
        """Paraphrases a random sub-prompt within the mega_prompt."""
        config = _split_mega_prompt_to_config(mega_prompt)
        if not config: return mega_prompt
                
        async with self.semaphore:
            output = await asyncio.to_thread(
                self.paraphrase_agent,
                inputs={"instruction": mega_prompt}
            )
            return output.content.content.strip()

    async def _perform_evolution(self, 
                                agent: Callable,
                                inputs: Dict[str, str],
                                parsing_tag: str = "FinalEvoPrompt"
                                ) -> str:
        """
        A generic asynchronous evolution function to invoke any evolutionary agent.

        Args:
            agent: The callable agent to use for evolution (e.g., self.ga_agent).
            inputs: A dictionary containing all required inputs for the agent.
            parsing_tag: The XML tag name used to parse the agent's output.

        Returns:
            The newly evolved prompt string.
        """
        async with self.semaphore:
            # Execute the agent in a separate thread
            output = await asyncio.to_thread(agent, inputs=inputs)
            
            # Use the specified parsing_tag for generic parsing
            pattern = re.compile(f"<{parsing_tag}>(.*?)</{parsing_tag}>", re.DOTALL)
            match = pattern.search(output.content.content.strip())
            
            if match:
                return match.group(1).strip()
            
            logger.warning(f"Evolution agent output parsing failed for tag '<{parsing_tag}>'. Raw output: {output.content.content[:200]}...")
            return "" # Return empty or a fallback if parsing fails

    async def _initialize_population(self, initial_seed: str):
        """Initializes the entire population from a single mega_prompt seed."""
        self.population = [initial_seed]
        if self.population_size > 1:
            num_to_generate = self.population_size - 1
            paraphrase_tasks = [self._perform_paraphrase(initial_seed) for _ in range(num_to_generate)]
            new_prompts = await aio_tqdm.gather(*paraphrase_tasks, desc="Paraphrasing to create initial population")
            self.population.extend(new_prompts)

class GAOptimizer(EvopromptOptimizer):
    """
    Implements a Genetic Algorithm (GA) to optimize prompts.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ga_agent = CustomizeAgent(
            name="ga_agent",
            description="An agent that evolves a new prompt from two parent prompts.",
            prompt="""Please follow the instructions step-by-step to generate a better prompt.
            1. Crossover the following prompts to generate a new prompt:
            Prompt 1: <tag>Your task is to classify the comment as one of the following categories: terrible, bad, okay, good, great.</tag>
            Prompt 2: <tag>In this task, you are given sentences from movie reviews. The task is to classify a sentence as one of the following categories: terrible, bad, okay, good, great.</tag>
            2. Mutate the prompt generated in Step 1 and generate a final prompt bracketed with <FinalEvoPrompt> and </FinalEvoPrompt>. While strictly preserving the original XML tags inside the <FinalEvoPrompt> and </FinalEvoPrompt>

            1. Crossover Prompt: In this task, you are given comments from movie reviews. Your task is to classify each comment as one of the following categories: terrible, bad, okay, good, great.
            2. <FinalEvoPrompt><tag>Given a sentence from a movie review, classify it into one of the following categories: terrible, bad, okay, good, or great.</tag></FinalEvoPrompt>

            Please follow the instructions step-by-step to generate a better prompt.
            1. Crossover the following prompts and generate a new prompt:
            Prompt 1: {parent1}
            Prompt 2: {parent2}
            2. Mutate the prompt generated in Step 1 and generate a final prompt bracketed with <FinalEvoPrompt> and </FinalEvoPrompt>. While strictly preserving the original XML tags inside the <FinalEvoPrompt> and </FinalEvoPrompt>""",
            llm_config=self.llm_config,
            inputs=[
                {"name": "parent1", "type": "string", "description": "The first parent prompt."},
                {"name": "parent2", "type": "string", "description": "The second parent prompt."}
            ]
        )

    async def optimize(self, benchmark: BIGBenchHard) -> tuple[str, dict, dict]:
        """
        The main optimization loop, which now accepts a benchmark object as a parameter.
        
        Args:
            benchmark: A benchmark instance containing training/validation data.
        """
        # 1. Extract initial config from registry and build the seed prompt
        initial_config = self.get_current_cfg()
        if not initial_config:
            raise ValueError("Registry is empty. Please track prompts using registry.track().")
        
        initial_mega_prompt_seed = _build_mega_prompt_from_config(initial_config)
        logger.info(f"Optimizing {len(initial_config)} registered prompts together.")

        # 2. Initialize the population
        await self._initialize_population(initial_mega_prompt_seed)

        # 3. Evaluate the initial population's fitness
        dev_set = benchmark.get_dev_data()
        if not dev_set:
            raise ValueError("The provided benchmark does not have a development set for evaluation.")

        print("--- Step 1: Evaluating initial population... ---")
        eval_tasks = [self.evaluate_mega_prompt_fitness(p, benchmark, dev_set) for p in self.population]
        self.scores = await aio_tqdm.gather(*eval_tasks, desc="Initial Evaluation")
        print(f"Initial best score: {max(self.scores):.4f}\n")

        # 4. Main evolution loop
        for t in range(self.iterations):
            print(f"--- Generation {t + 1}/{self.iterations} ---")
            total_fitness = sum(self.scores)
            probabilities = [s / total_fitness for s in self.scores] if total_fitness > 0 else None

            # --- Generate offspring using GA ---
            evolution_tasks = []
            for _ in range(self.population_size):
                # Select parents
                parent1, parent2 = random.choices(self.population, weights=probabilities, k=2)
                
                # Create a task to call the generic evolution function with the GA agent
                task = self._perform_evolution(
                    agent=self.ga_agent,
                    inputs={
                        "parent1": parent1,
                        "parent2": parent2
                    },
                    parsing_tag="FinalEvoPrompt" # Specify the parsing tag
                )
                evolution_tasks.append(task)

            # Concurrently execute all evolution tasks
            new_children = await aio_tqdm.gather(*evolution_tasks, desc=f"Generating Gen {t+1}")

            # Evaluate the new children
            new_eval_tasks = [self.evaluate_mega_prompt_fitness(p, benchmark, dev_set) for p in new_children]
            new_scores = await aio_tqdm.gather(*new_eval_tasks, desc=f"Evaluating Gen {t+1}")

            # Survival of the fittest (Elitism)
            combined = sorted(zip(self.scores + new_scores, self.population + new_children), key=lambda x: x[0], reverse=True)
            self.scores, self.population = [list(t) for t in zip(*combined[:self.population_size])]
            
            gen_name = f"Generation {t + 1}"
            self.best_scores_per_gen[gen_name] = max(self.scores)
            self.avg_scores_per_gen[gen_name] = np.mean(self.scores)
            print(f"Generation best score: {self.best_scores_per_gen[gen_name]:.4f}")
            print(f"Generation average score: {self.avg_scores_per_gen[gen_name]:.4f}\n")
        
        # 5. Return results and update the program
        print("--- Evolution complete ---")
        best_mega_prompt = self.population[self.scores.index(max(self.scores))]
        final_best_config = _split_mega_prompt_to_config(best_mega_prompt)
        
        self.apply_cfg(final_best_config)
        logger.info("Optimization finished! The best set of prompts has been applied to the program.")

        return best_mega_prompt, self.best_scores_per_gen, self.avg_scores_per_gen

class DEOptimizer(EvopromptOptimizer):
    """
    An optimizer that enhances prompts using the Differential Evolution (DE) algorithm.
    It inherits the evaluation and initialization capabilities of EvopromptOptimizer
    but implements the core DE loop.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Define the agent specifically designed for the DE algorithm.
        self.de_agent = CustomizeAgent(
            name="DE_Agent",
            description="Generates a new trial prompt using the Differential Evolution strategy.",
            prompt=(
            "Please follow the instructions step-by-step to generate a better prompt."
            "1. Identify the different parts between Prompt 1 and Prompt 2:"
            "Prompt 1: {donor1}"
            "Prompt 2: {donor2}"
            "2. Randomly mutate the different parts."
            "3. Combine the mutated different parts with Prompt 3, selectively replacing its content to generate a new prompt."
            "Prompt 3: {best_prompt}"
            "4. Crossover the prompt from Step 3 with the following basic prompt and generate a final prompt bracketed with <FinalEvoPrompt> and </FinalEvoPrompt>. Strictly preserve the original XML tags inside the final prompt."
            "Basic Prompt: {current_prompt}"
            ),
            llm_config=self.llm_config,
            inputs=[
                {"name": "current_prompt", "type": "string", "description": "The base prompt to be mutated, p_i."},
                {"name": "donor1", "type": "string", "description": "The first donor prompt, p_r1."},
                {"name": "donor2", "type": "string", "description": "The second donor prompt, p_r2."},
                {"name": "best_prompt", "type": "string", "description": "The best prompt found so far in the population, p_best."},
            ]
        )

    async def optimize(self, benchmark: BIGBenchHard) -> tuple[str, dict, dict]:
        """
        Overrides the optimize method to implement the main Differential Evolution (DE) flow.
        This implementation follows a parallel/batch model for high performance.
        """
        # --- 1. Initialization Phase (Identical to GA version) ---
        initial_config = self.get_current_cfg()
        if not initial_config:
            raise ValueError("Registry is empty. Please track prompts using registry.track().")
        
        initial_mega_prompt_seed = _build_mega_prompt_from_config(initial_config)
        logger.info(f"Optimizing {len(initial_config)} registered prompts together using Differential Evolution (DE).")

        # Initialize population using the parent class method (via paraphrasing)
        await self._initialize_population(initial_mega_prompt_seed)

        # Initial fitness evaluation
        dev_set = benchmark.get_dev_data()
        if not dev_set:
            raise ValueError("The provided benchmark does not have a development set for evaluation.")

        print("--- Step 1: Evaluating initial population... ---")
        self.scores = await aio_tqdm.gather(
            *[self.evaluate_mega_prompt_fitness(p, benchmark, dev_set) for p in self.population],
            desc="Initial Evaluation"
        )
        print(f"Initial best score: {max(self.scores):.4f}\n")

        # --- 2. Main Evolution Loop (DE Core Logic) ---
        for t in range(self.iterations):
            print(f"--- Generation {t + 1}/{self.iterations} ---")
            
            # Find the global best p_best in the current generation
            current_best_score = max(self.scores)
            current_best_prompt = self.population[self.scores.index(current_best_score)]

            # --- Generate trial individuals using DE ---
            evolution_tasks = []
            population_indices = list(range(self.population_size))

            for i in range(self.population_size):
                # Select donors
                donor_indices = random.sample([idx for idx in population_indices if idx != i], 2)
                p_i = self.population[i]
                p_r1 = self.population[donor_indices[0]]
                p_r2 = self.population[donor_indices[1]]
                
                # Create a task to call the generic evolution function with the DE agent
                task = self._perform_evolution(
                    agent=self.de_agent,
                    inputs={
                        "current_prompt": p_i,
                        "donor1": p_r1,
                        "donor2": p_r2,
                        "best_prompt": current_best_prompt
                    },
                    parsing_tag="FinalEvoPrompt" # DE Agent must also follow this output format
                )
                evolution_tasks.append(task)

            # Concurrently execute all DE evolution tasks
            trial_prompts = await aio_tqdm.gather(*evolution_tasks, desc=f"DE Evolution Gen {t+1}")

            # Concurrently evaluate all trial prompts
            trial_scores = await aio_tqdm.gather(
                *[self.evaluate_mega_prompt_fitness(p, benchmark, dev_set) for p in trial_prompts],
                desc=f"Evaluating Trial Prompts Gen {t+1}"
            )
            
            # Perform selection to determine the next generation's population
            next_population = []
            next_scores = []
            for i in range(self.population_size):
                # One-to-one comparison: original individual vs. trial individual
                if trial_scores[i] > self.scores[i]:
                    # If the trial individual is better, it enters the next generation
                    next_population.append(trial_prompts[i])
                    next_scores.append(trial_scores[i])
                else:
                    # Otherwise, the original individual survives
                    next_population.append(self.population[i])
                    next_scores.append(self.scores[i])
            
            # Update population and scores for the next iteration
            self.population = next_population
            self.scores = next_scores

            # Log and print generation stats
            gen_name = f"Generation {t + 1}"
            best_score_gen = max(self.scores)
            avg_score_gen = sum(self.scores) / len(self.scores)
            self.best_scores_per_gen[gen_name] = best_score_gen
            self.avg_scores_per_gen[gen_name] = avg_score_gen
            print(f"Generation best score: {best_score_gen:.4f}")
            print(f"Generation average score: {avg_score_gen:.4f}\n")

        # --- 3. Return Results (Identical to GA version) ---
        print("--- DE Evolution complete ---")
        best_mega_prompt = self.population[self.scores.index(max(self.scores))]
        final_best_config = _split_mega_prompt_to_config(best_mega_prompt)
        
        self.apply_cfg(final_best_config)
        logger.info("Optimization finished! The best set of prompts has been applied to the program.")

        return best_mega_prompt, self.best_scores_per_gen, self.avg_scores_per_gen