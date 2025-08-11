import asyncio
import random
import re
import os
import csv
import time
import itertools
from typing import Callable, Dict, List
from datetime import datetime

import numpy as np
from tqdm.asyncio import tqdm as aio_tqdm
import matplotlib.pyplot as plt

from evoagentx.agents import CustomizeAgent
from evoagentx.benchmark.bigbenchhard import BIGBenchHard
from evoagentx.core.logging import logger
from evoagentx.models import OpenAILLMConfig
from evoagentx.optimizers.engine.base import BaseOptimizer
from evoagentx.optimizers.engine.registry import ParamRegistry


class EvopromptOptimizer(BaseOptimizer):
    def __init__(self,
                 registry: ParamRegistry,
                 program: Callable,
                 population_size: int,
                 iterations: int,
                 llm_config: OpenAILLMConfig,
                 concurrency_limit: int = 10,
                 combination_sample_size: int = None,
                 enable_logging: bool = True,
                 log_dir: str = None):
        super().__init__(registry=registry, program=program)

        self.population_size = population_size
        self.iterations = iterations
        self.llm_config = llm_config
        self.semaphore = asyncio.Semaphore(concurrency_limit)
        self.combination_sample_size = combination_sample_size
        self.enable_logging = enable_logging
        self.log_dir_base = log_dir
        self.log_dir = None

        self._eval_cache = {}
        self.node_populations: Dict[str, List[str]] = {}
        self.node_scores: Dict[str, List[float]] = {}
        self.best_scores_per_gen: Dict[str, Dict[str, float]] = {}
        self.avg_scores_per_gen: Dict[str, Dict[str, float]] = {}
        self.best_combo_scores_per_gen: Dict[str, float] = {}
        self.avg_combo_scores_per_gen: Dict[str, float] = {}
        
        self.paraphrase_agent = CustomizeAgent(
            name="ParaphraseAgent",
            description="An agent that paraphrases a given instruction.",
            prompt="""Task: Generate a semantically equivalent but differently worded version of the user-provided instruction.
                    
Now, please process the following instruction:
Input: {instruction}

Please provide the paraphrased version in the following format:

## paraphrased_instruction
[Your paraphrased version here]""",
            llm_config=self.llm_config,
            inputs=[
                {"name": "instruction", "type": "string", "description": "The instruction to paraphrase."},
            ],
            outputs=[
                {"name": "paraphrased_instruction", "type": "string", "description": "The paraphrased instruction."}
            ],
            parse_mode="title"
        )

    def _setup_logging_directory(self, benchmark: BIGBenchHard):
        if not self.enable_logging or self.log_dir:
            return

        task_name = benchmark.task if hasattr(benchmark, 'task') else 'unknown_task'

        if self.log_dir_base is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            algo_name = self.__class__.__name__.replace("Optimizer", "")
            self.log_dir = f"node_evolution_logs_{algo_name}_{self.llm_config.model}_{task_name}_{timestamp}"
        else:
            self.log_dir = self.log_dir_base

        os.makedirs(self.log_dir, exist_ok=True)
        logger.info(f"Logging enabled. Log files will be saved to: {self.log_dir}")

    def _log_generation_summary(self, generation: int, operation: str = "Evolution"):
        if not self.enable_logging:
            return

        filename = f"generation_{generation:02d}_{operation.lower()}.csv"
        filepath = os.path.join(self.log_dir, filename)

        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Node_Name', 'Individual_ID', 'Prompt_Text', 'Fitness_Score', 'Status', 'Rank_in_Node', 'Generation', 'Timestamp'])
            timestamp = datetime.now().isoformat()

            for node_name in self.node_populations.keys():
                node_pop = self.node_populations[node_name]
                node_scores = self.node_scores[node_name]

                if not node_pop: continue

                sorted_indices = sorted(range(len(node_scores)), key=lambda i: node_scores[i], reverse=True)

                for rank, idx in enumerate(sorted_indices, 1):
                    prompt = node_pop[idx]
                    score = node_scores[idx]
                    status = "Best" if rank == 1 else "Survivor" if rank <= self.population_size else "Eliminated"

                    writer.writerow([
                        node_name, f"{node_name}_{idx}", prompt[:200] + "..." if len(prompt) > 200 else prompt,
                        f"{score:.6f}", status, rank, generation, timestamp
                    ])

    def _log_detailed_evaluation(self, generation: int, combinations: List[Dict[str, str]],
                                 combination_scores: List[float]):
        if not self.enable_logging:
            return

        filename = f"detailed_evaluation_gen_{generation:02d}.csv"
        filepath = os.path.join(self.log_dir, filename)

        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            node_names = list(combinations[0].keys()) if combinations else []
            header = ['Combination_ID', 'Average_Score']
            for node_name in node_names:
                header.append(f'{node_name}_Prompt_Preview')
            header.extend(['Generation', 'Timestamp'])
            writer.writerow(header)

            timestamp = datetime.now().isoformat()

            for combo_id, (combination, avg_score) in enumerate(zip(combinations, combination_scores)):
                try:
                    row = [f"combo_{combo_id}", f"{avg_score:.6f}"]
                    for node_name in node_names:
                        prompt = combination[node_name]
                        row.append(prompt[:50] + "..." if len(prompt) > 50 else prompt)
                    row.extend([generation, timestamp])
                    writer.writerow(row)
                except Exception as e:
                    logger.error(f"Error logging evaluation for combination {combo_id}: {e}")

    def _create_single_metric_plot(self, metric_name: str, generations: List[int],
                                   best_scores: List[float], avg_scores: List[float],
                                   algorithm_name: str, plot_dir: str):
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.plot(generations, best_scores, marker='o', linestyle='-', linewidth=2, markersize=8, label='Best Score')
        ax.plot(generations, avg_scores, marker='x', linestyle='--', linewidth=2, markersize=8, label='Average Score')

        title = f"Performance for '{metric_name}' ({algorithm_name})"
        ax.set_title(title, fontsize=16, weight='bold')
        ax.set_xlabel('Generation', fontsize=12)
        ax.set_ylabel('Fitness Score', fontsize=12)
        ax.set_xticks(generations)
        ax.set_xticklabels([f"Gen {g}" for g in generations], rotation=45, ha="right")
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

        plt.tight_layout()

        safe_metric_name = re.sub(r'[^a-zA-Z0-9_-]', '_', metric_name)
        filename = f"performance_plot_{safe_metric_name}.png"
        filepath = os.path.join(plot_dir, filename)

        try:
            plt.savefig(filepath, dpi=200, bbox_inches='tight')
        except Exception as e:
            logger.error(f"Failed to save individual plot for {metric_name}: {e}")
        finally:
            plt.close(fig)

    def _plot_and_save_performance_graph(self, algorithm_name: str):
        if not self.enable_logging or plt is None:
            if plt is None: logger.warning("Matplotlib not found, skipping plot generation.")
            return
        if not self.best_scores_per_gen and not self.best_combo_scores_per_gen:
            logger.warning("No performance data to plot.")
            return

        plt.style.use('seaborn-v0_8-whitegrid')
        generations = sorted([int(re.search(r'\d+', gen).group()) for gen in
                              (self.best_scores_per_gen.keys() or self.best_combo_scores_per_gen.keys())])

        fig_combined, ax_combined = plt.subplots(figsize=(16, 9))

        if self.best_combo_scores_per_gen:
            combo_best = [self.best_combo_scores_per_gen.get(f"Gen_{g}") for g in generations]
            combo_avg = [self.avg_combo_scores_per_gen.get(f"Gen_{g}") for g in generations]
            ax_combined.plot(generations, combo_best, marker='*', linestyle='-', linewidth=2.5, markersize=10, label='Best Combination Score (Overall)')
            ax_combined.plot(generations, combo_avg, marker='D', linestyle='--', linewidth=2.5, markersize=8, label='Average Combination Score (Overall)')

        all_node_metrics = set()
        for gen_data in self.best_scores_per_gen.values():
            all_node_metrics.update(gen_data.keys())

        for metric in sorted(list(all_node_metrics)):
            best_scores = [self.best_scores_per_gen.get(f"Gen_{g}", {}).get(metric) for g in generations]
            avg_scores = [self.avg_scores_per_gen.get(f"Gen_{g}", {}).get(metric) for g in generations]
            ax_combined.plot(generations, best_scores, marker='o', linestyle='-', alpha=0.7, label=f'Best Score ({metric})')
            ax_combined.plot(generations, avg_scores, marker='x', linestyle='--', alpha=0.7, label=f'Average Score ({metric})')

        ax_combined.set_title(f'Overall Performance Evolution ({algorithm_name})', fontsize=18, weight='bold')
        ax_combined.set_xlabel('Generation', fontsize=14)
        ax_combined.set_ylabel('Fitness Score', fontsize=14)
        ax_combined.set_xticks(generations)
        ax_combined.set_xticklabels([f"Gen {g}" for g in generations], rotation=45, ha="right")
        handles, labels = ax_combined.get_legend_handles_labels()
        combo_indices = [i for i, label in enumerate(labels) if 'Combination' in label]
        node_indices = [i for i, label in enumerate(labels) if 'Combination' not in label]
        ax_combined.legend([handles[i] for i in combo_indices + node_indices],
                           [labels[i] for i in combo_indices + node_indices],
                           loc='best', fontsize=10)
        ax_combined.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()

        combined_filepath = os.path.join(self.log_dir, f"performance_summary_OVERALL.png")
        try:
            plt.savefig(combined_filepath, dpi=300, bbox_inches='tight')
            logger.info(f"Overall performance plot saved to: {combined_filepath}")
        except Exception as e:
            logger.error(f"Failed to save overall performance plot: {e}")
        finally:
            plt.close(fig_combined)

        individual_plot_dir = os.path.join(self.log_dir, 'individual_plots')
        os.makedirs(individual_plot_dir, exist_ok=True)

        for metric in sorted(list(all_node_metrics)):
            best_scores = [self.best_scores_per_gen.get(f"Gen_{g}", {}).get(metric) for g in generations]
            avg_scores = [self.avg_scores_per_gen.get(f"Gen_{g}", {}).get(metric) for g in generations]
            self._create_single_metric_plot(metric, generations, best_scores, avg_scores, algorithm_name, individual_plot_dir)

        if self.best_combo_scores_per_gen:
            combo_best = [self.best_combo_scores_per_gen.get(f"Gen_{g}") for g in generations]
            combo_avg = [self.avg_combo_scores_per_gen.get(f"Gen_{g}") for g in generations]
            self._create_single_metric_plot("Combination", generations, combo_best, combo_avg, algorithm_name, individual_plot_dir)

        logger.info(f"Individual performance plots saved to: {individual_plot_dir}")

    def _log_optimization_summary(self, algorithm_name: str, best_config: Dict[str, str],
                                  test_accuracy: float = None):
        if not self.enable_logging:
            return

        filename = f"optimization_summary_{algorithm_name.lower()}.csv"
        filepath = os.path.join(self.log_dir, filename)

        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Metric', 'Value', 'Timestamp'])
            timestamp = datetime.now().isoformat()

            writer.writerow(['Algorithm', algorithm_name, timestamp])
            writer.writerow(['Population_Size', self.population_size, timestamp])
            writer.writerow(['Iterations', self.iterations, timestamp])
            writer.writerow(['Combination_Sample_Size', self.combination_sample_size, timestamp])

            if test_accuracy is not None:
                writer.writerow(['Final_Test_Accuracy', f"{test_accuracy:.6f}", timestamp])

            for node_name, prompt in best_config.items():
                writer.writerow([f'Best_{node_name}', prompt, timestamp])

            for gen_name in self.best_scores_per_gen.keys():
                for metric_name, best_score in self.best_scores_per_gen[gen_name].items():
                    writer.writerow([f'{gen_name}_{metric_name}_Best', f"{best_score:.6f}", timestamp])

                if gen_name in self.avg_scores_per_gen:
                    for metric_name, avg_score in self.avg_scores_per_gen[gen_name].items():
                        writer.writerow([f'{gen_name}_{metric_name}_Avg', f"{avg_score:.6f}", timestamp])

        self._plot_and_save_performance_graph(algorithm_name)

    def _generate_combinations(self, node_populations: Dict[str, List[str]]) -> List[Dict[str, str]]:
        node_names = list(node_populations.keys())
        node_prompts = [node_populations[node] for node in node_names]
        total_possible = np.prod([len(p) for p in node_prompts])

        if self.combination_sample_size is None:
            target_size = min(self.population_size, int(total_possible), 200)
        else:
            target_size = min(self.combination_sample_size, int(total_possible))

        logger.info(f"Total possible combinations: {total_possible}, sampling: {target_size}")

        if target_size >= total_possible:
            all_combinations = []
            for combination in itertools.product(*node_prompts):
                combo_dict = {node_names[i]: combination[i] for i in range(len(node_names))}
                all_combinations.append(combo_dict)
            return all_combinations

        sampled_combinations = []
        sampled_keys = set()
        max_attempts = target_size * 5
        attempts = 0

        while len(sampled_combinations) < target_size and attempts < max_attempts:
            combination = {name: random.choice(prompts) for name, prompts in node_populations.items()}
            combo_key = tuple(sorted(combination.items()))
            if combo_key not in sampled_keys:
                sampled_combinations.append(combination)
                sampled_keys.add(combo_key)
            attempts += 1

        logger.info(f"Generated {len(sampled_combinations)} unique combinations")
        return sampled_combinations

    async def _evaluate_combination_on_example(self, combination: Dict[str, str],
                                             benchmark: BIGBenchHard, example: Dict) -> float:
        combo_key = tuple(sorted(combination.items()))
        example_key = str(hash(str(example)))
        cache_key = hash((combo_key, example_key))

        if not hasattr(self, '_eval_cache'):
            self._eval_cache = {}

        if cache_key in self._eval_cache:
            return self._eval_cache[cache_key]

        async with self.semaphore:
            try:
                original_config = self.get_current_cfg()
                self.apply_cfg(combination)
                inputs = {k: v for k, v in example.items() if k in benchmark.get_input_keys()}
                prediction, _ = await asyncio.to_thread(self.program, **inputs)
                label = benchmark.get_label(example)
                score_dict = benchmark.evaluate(prediction, label)
                score = score_dict.get("em", 0.0)
                self.apply_cfg(original_config)
                self._eval_cache[cache_key] = score
                if len(self._eval_cache) > 5000:
                    keys_to_del = list(self._eval_cache.keys())[:1000]
                    for key in keys_to_del:
                        del self._eval_cache[key]
                return score
            except Exception as e:
                logger.error(f"Error evaluating combination: {e}")
                return 0.0

    async def _evaluate_combinations_and_update_node_scores(self, combinations: List[Dict[str, str]],
                                                          benchmark: BIGBenchHard, dev_set: list) -> List[float]:
        eval_dev_set = dev_set[:50] if len(dev_set) > 50 else dev_set
        combination_scores = []
        print(f"Evaluating {len(combinations)} combinations on {len(eval_dev_set)} examples...")
        combo_pbar = aio_tqdm(total=len(combinations), desc="Evaluating Combinations")
        for combination in combinations:
            tasks = [self._evaluate_combination_on_example(combination, benchmark, ex) for ex in eval_dev_set]
            example_scores = await asyncio.gather(*tasks)
            avg_score = sum(example_scores) / len(example_scores) if example_scores else 0.0
            combination_scores.append(avg_score)
            combo_pbar.update(1)
        combo_pbar.close()

        for node_name in self.node_populations.keys():
            self.node_scores[node_name] = [0.0] * len(self.node_populations[node_name])
            for prompt_idx, prompt in enumerate(self.node_populations[node_name]):
                participating_scores = [
                    combo_score for combo_idx, combo_score in enumerate(combination_scores)
                    if combinations[combo_idx].get(node_name) == prompt
                ]
                if participating_scores:
                    self.node_scores[node_name][prompt_idx] = sum(participating_scores) / len(participating_scores)
                else:
                    self.node_scores[node_name][prompt_idx] = 0.0
        return combination_scores

    async def _perform_paraphrase(self, prompt: str) -> str:
        async with self.semaphore:
            output = await asyncio.to_thread(
                self.paraphrase_agent,
                inputs={"instruction": prompt}
            )
            return output.content.paraphrased_instruction.strip()

    async def _perform_evolution(self, agent: Callable, inputs: Dict[str, str]) -> str:
        async with self.semaphore:
            output = await asyncio.to_thread(agent, inputs=inputs)
            if hasattr(output.content, 'evolved_prompt'):
                return output.content.evolved_prompt.strip()
            return str(output.content).strip()

    async def _initialize_node_populations(self, initial_config: Dict[str, any]):
        for node_name, initial_value in initial_config.items():
            node_population = []
            if isinstance(initial_value, list):
                provided_size = len(initial_value)
                if self.population_size < provided_size:
                    logger.info(f"Node '{node_name}': Provided population ({provided_size}) is larger than target size ({self.population_size}). Randomly sampling.")
                    node_population = random.sample(initial_value, self.population_size)
                elif self.population_size == provided_size:
                    logger.info(f"Node '{node_name}': Provided population size ({provided_size}) matches target size. Using directly.")
                    node_population = list(initial_value)
                else:
                    logger.info(f"Node '{node_name}': Target population size ({self.population_size}) is larger than provided ({provided_size}). Expanding.")
                    node_population = list(initial_value)
                    num_to_generate = self.population_size - provided_size
                    source_prompts_for_generation = random.choices(initial_value, k=num_to_generate)
                    paraphrase_tasks = [self._perform_paraphrase(prompt) for prompt in source_prompts_for_generation]
                    new_prompts = await aio_tqdm.gather(
                        *paraphrase_tasks, desc=f"Expanding population for {node_name}"
                    )
                    node_population.extend(new_prompts)
            elif isinstance(initial_value, str):
                logger.info(f"Node '{node_name}': Generating population from a single initial prompt.")
                node_population = [initial_value]
                if self.population_size > 1:
                    num_to_generate = self.population_size - 1
                    paraphrase_tasks = [self._perform_paraphrase(initial_value) for _ in range(num_to_generate)]
                    new_prompts = await aio_tqdm.gather(
                        *paraphrase_tasks, desc=f"Generating initial population for {node_name}"
                    )
                    node_population.extend(new_prompts)
            else:
                raise TypeError(f"Unsupported type for tracked parameter '{node_name}': {type(initial_value)}. Must be str or list.")
            self.node_populations[node_name] = node_population
            self.node_scores[node_name] = [0.0] * self.population_size


class GAOptimizer(EvopromptOptimizer):
    def __init__(self, *args, full_evaluation: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.full_evaluation = full_evaluation
        if self.full_evaluation:
            logger.info("GAOptimizer initialized with 'full_evaluation' mode.")
        else:
            logger.info("GAOptimizer initialized with 'fast_evaluation' mode.")
        self.ga_agent = CustomizeAgent(
            name="ga_agent",
            description="An agent that evolves a new prompt from two parent prompts.",
            prompt="""Please follow the instructions step-by-step to generate a better prompt.

1. Crossover the following prompts to generate a new prompt:
Prompt 1: {parent1}
Prompt 2: {parent2}

2. Mutate the prompt generated in Step 1 and generate a final evolved prompt. Strictly preserve the original XML tags structure.

Now process the given prompts and provide your output in the following format:

## evolved_prompt
[Your evolved prompt here]""",
            llm_config=self.llm_config,
            inputs=[
                {"name": "parent1", "type": "string", "description": "The first parent prompt."},
                {"name": "parent2", "type": "string", "description": "The second parent prompt."}
            ],
            outputs=[
                {"name": "evolved_prompt", "type": "string", "description": "The evolved prompt with XML tags preserved."}
            ],
            parse_mode="title"
        )

    async def _perform_node_evolution(self, node_name: str, node_population: List[str],
                                      node_scores: List[float], evolution_agent: Callable) -> List[str]:
        total_fitness = sum(node_scores)
        probabilities = [s / total_fitness for s in node_scores] if total_fitness > 0 else None
        num_children_to_create = self.population_size
        evolution_tasks = []
        for _ in range(num_children_to_create):
            parents = random.choices(node_population, weights=probabilities, k=2) if probabilities else random.choices(node_population, k=2)
            task = self._perform_evolution(agent=evolution_agent, inputs={"parent1": parents[0], "parent2": parents[1]})
            evolution_tasks.append(task)
        new_children = await aio_tqdm.gather(*evolution_tasks, desc=f"Evolving {node_name}")
        return new_children

    async def optimize(self, benchmark: BIGBenchHard) -> tuple[Dict[str, str], dict, dict]:
        self._setup_logging_directory(benchmark)
        initial_config = self.get_current_cfg()
        if not initial_config: raise ValueError("Registry is empty.")
        logger.info(f"Optimizing {len(initial_config)} nodes independently with GA.")
        await self._initialize_node_populations(initial_config)
        dev_set = benchmark.get_dev_data()
        if not dev_set: raise ValueError("Benchmark has no development set.")
        print("--- Step 1: Initial evaluation of node combinations... ---")
        combinations = self._generate_combinations(self.node_populations)
        combination_scores = await self._evaluate_combinations_and_update_node_scores(combinations, benchmark, dev_set)
        self._log_generation_summary(0, "Initial")
        self._log_detailed_evaluation(0, combinations, combination_scores)
        self.best_scores_per_gen["Gen_0"] = {name: max(scores) if scores else 0 for name, scores in self.node_scores.items()}
        self.avg_scores_per_gen["Gen_0"] = {name: np.mean(scores) if scores else 0 for name, scores in self.node_scores.items()}
        for node_name, scores in self.node_scores.items():
            if scores:
                print(f"Node {node_name} - Initial best: {max(scores):.4f}, avg: {np.mean(scores):.4f}")
        for t in range(self.iterations):
            generation_start_time = time.time()
            print(f"\n--- Generation {t + 1}/{self.iterations} ---")
            children_populations = {}
            for node_name in self.node_populations.keys():
                children = await self._perform_node_evolution(
                    node_name, self.node_populations[node_name], self.node_scores[node_name], self.ga_agent
                )
                children_populations[node_name] = children
            if self.full_evaluation:
                print("Mode: Full Evaluation. Evaluating parents and children together...")
                combined_populations = {
                    name: self.node_populations[name] + children_populations[name]
                    for name in self.node_populations.keys()
                }
                self.node_populations = combined_populations
                combinations = self._generate_combinations(self.node_populations)
                combination_scores = await self._evaluate_combinations_and_update_node_scores(combinations, benchmark, dev_set)
                for node_name in self.node_populations.keys():
                    population = self.node_populations[node_name]
                    scores = self.node_scores[node_name]
                    sorted_pairs = sorted(zip(scores, population), key=lambda x: x[0], reverse=True)
                    selected_pairs = sorted_pairs[:self.population_size]
                    if selected_pairs:
                        selected_scores, selected_population = zip(*selected_pairs)
                        self.node_scores[node_name] = list(selected_scores)
                        self.node_populations[node_name] = list(selected_population)
                    else:
                        self.node_scores[node_name], self.node_populations[node_name] = [], []
                    print(f"Node {node_name}: Selected top {len(self.node_populations[node_name])} from {len(population)} individuals")
            else:
                print("Mode: Fast Evaluation. Evaluating only new children...")
                parent_populations = {name: pop.copy() for name, pop in self.node_populations.items()}
                parent_scores = {name: scores.copy() for name, scores in self.node_scores.items()}
                self.node_populations = children_populations
                combinations = self._generate_combinations(self.node_populations)
                combination_scores = await self._evaluate_combinations_and_update_node_scores(combinations, benchmark, dev_set)
                children_scores = {name: scores.copy() for name, scores in self.node_scores.items()}
                for node_name in parent_populations.keys():
                    parent_pairs = list(zip(parent_scores[node_name], parent_populations[node_name]))
                    children_pairs = list(zip(children_scores.get(node_name, []), children_populations.get(node_name, [])))
                    combined_pairs = parent_pairs + children_pairs
                    sorted_pairs = sorted(combined_pairs, key=lambda x: x[0], reverse=True)
                    selected_pairs = sorted_pairs[:self.population_size]
                    if selected_pairs:
                        selected_scores, selected_population = zip(*selected_pairs)
                        self.node_scores[node_name] = list(selected_scores)
                        self.node_populations[node_name] = list(selected_population)
                    else:
                        self.node_scores[node_name], self.node_populations[node_name] = [], []
                    print(f"Node {node_name}: Selected top {len(self.node_populations[node_name])} from {len(combined_pairs)} individuals (Parents + Children)")
            self._log_generation_summary(t + 1, "Evolution")
            self._log_detailed_evaluation(t + 1, combinations, combination_scores)
            generation_time = time.time() - generation_start_time
            print(f"Generation {t + 1} completed in {generation_time:.2f}s")
            gen_name = f"Gen_{t + 1}"
            self.best_scores_per_gen[gen_name] = {}
            self.avg_scores_per_gen[gen_name] = {}
            for node_name in self.node_populations.keys():
                if self.node_scores[node_name]:
                    best_score = max(self.node_scores[node_name])
                    avg_score = np.mean(self.node_scores[node_name])
                    self.best_scores_per_gen[gen_name][node_name] = best_score
                    self.avg_scores_per_gen[gen_name][node_name] = avg_score
                    print(f"Node {node_name} - Best: {best_score:.4f}, Avg: {avg_score:.4f}")
        print("\n--- Evolution complete ---")
        best_config = {
            name: self.node_populations[name][np.argmax(self.node_scores[name])]
            for name in self.node_populations.keys() if self.node_populations.get(name)
        }
        self._log_optimization_summary("GA", best_config)
        self.apply_cfg(best_config)
        logger.info("Optimization finished! The best configuration has been applied to the program.")
        return best_config, self.best_scores_per_gen, self.avg_scores_per_gen


class DEOptimizer(EvopromptOptimizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.de_agent = CustomizeAgent(
            name="DE_Agent",
            description="Generates a new trial prompt using the Differential Evolution strategy.",
            prompt="""Please follow the instructions step-by-step to generate a better prompt using Differential Evolution strategy.

1. Identify the different parts between these two donor prompts:
Donor Prompt 1: {donor1}
Donor Prompt 2: {donor2}

2. Randomly mutate the different parts identified above.

3. Combine the mutated parts with the best prompt, selectively replacing its content:
Best Prompt: {best_prompt}

4. Crossover the result from Step 3 with the current prompt to generate the final evolved prompt. Strictly preserve the original XML tags structure:
Current Prompt: {current_prompt}

Please provide the final evolved prompt in the following format:

## evolved_prompt
[Your evolved prompt here]""",
            llm_config=self.llm_config,
            inputs=[
                {"name": "current_prompt", "type": "string", "description": "The base prompt to be mutated, p_i."},
                {"name": "donor1", "type": "string", "description": "The first donor prompt, p_r1."},
                {"name": "donor2", "type": "string", "description": "The second donor prompt, p_r2."},
                {"name": "best_prompt", "type": "string", "description": "The best prompt found so far in the population, p_best."},
            ],
            outputs=[
                {"name": "evolved_prompt", "type": "string", "description": "The evolved prompt with XML tags preserved."}
            ],
            parse_mode="title"
        )

    def _log_de_generation(self, generation: int, combos_with_scores: List[tuple]):
        if not self.enable_logging:
            return
        filename = f"de_generation_{generation:02d}_log.csv"
        filepath = os.path.join(self.log_dir, filename)
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            header = ['Combination_ID', 'Combination_Score', 'Node_Name', 'Prompt_Text', 'Generation', 'Timestamp']
            writer.writerow(header)
            timestamp = datetime.now().isoformat()
            sorted_combos = sorted(combos_with_scores, key=lambda x: x[1], reverse=True)
            for combo_rank, (combination, avg_score) in enumerate(sorted_combos):
                combo_id = f"combo_rank_{combo_rank}"
                for node_name, prompt_text in combination.items():
                    writer.writerow([
                        combo_id,
                        f"{avg_score:.6f}",
                        node_name,
                        prompt_text[:200] + "..." if len(prompt_text) > 200 else prompt_text,
                        generation,
                        timestamp
                    ])

    async def _evaluate_combination_list(self, combinations: List[Dict], benchmark: BIGBenchHard, dev_set: list) -> List[float]:
        eval_dev_set = dev_set[:50] if len(dev_set) > 50 else dev_set
        all_scores = []
        pbar = aio_tqdm(total=len(combinations), desc="Evaluating batch", leave=False)
        for combo in combinations:
            tasks = [self._evaluate_combination_on_example(combo, benchmark, ex) for ex in eval_dev_set]
            example_scores = await asyncio.gather(*tasks)
            avg_score = sum(example_scores) / len(example_scores) if example_scores else 0.0
            all_scores.append(avg_score)
            pbar.update(1)
        pbar.close()
        return all_scores

    async def _evolve_and_select_one(
        self,
        target_combo_with_score: tuple,
        full_pop_with_scores: List[tuple],
        benchmark: BIGBenchHard,
        dev_set: list
    ) -> tuple:
        target_combo, target_score = target_combo_with_score
        best_combo, _ = max(full_pop_with_scores, key=lambda x: x[1])
        donor_pool = [c for c in full_pop_with_scores if c[0] != target_combo]
        if len(donor_pool) < 2:
            donors = random.choices(full_pop_with_scores, k=2)
        else:
            donors = random.sample(donor_pool, 2)
        donor1_combo, _ = donors[0]
        donor2_combo, _ = donors[1]
        evolution_tasks = []
        node_names = list(target_combo.keys())
        for node_name in node_names:
            task = self._perform_evolution(
                agent=self.de_agent,
                inputs={
                    "current_prompt": target_combo[node_name],
                    "donor1": donor1_combo[node_name],
                    "donor2": donor2_combo[node_name],
                    "best_prompt": best_combo[node_name]
                }
            )
            evolution_tasks.append(task)
        evolved_components = await asyncio.gather(*evolution_tasks)
        trial_combo = {name: comp for name, comp in zip(node_names, evolved_components)}
        trial_scores = await self._evaluate_combination_list([trial_combo], benchmark, dev_set)
        trial_score = trial_scores[0]
        if trial_score > target_score:
            return (trial_combo, trial_score)
        else:
            return (target_combo, target_score)

    async def optimize(self, benchmark: BIGBenchHard) -> tuple[Dict[str, str], dict, dict]:
        self._setup_logging_directory(benchmark)
        initial_config = self.get_current_cfg()
        if not initial_config: raise ValueError("Registry is empty.")
        logger.info(f"Optimizing with DEOptimizer (Pipelined Combination Evolution).")
        await self._initialize_node_populations(initial_config)
        dev_set = benchmark.get_dev_data()
        if not dev_set: raise ValueError("Benchmark has no development set.")
        print("--- Step 1: Creating and evaluating initial combination population... ---")
        initial_combinations = self._generate_combinations(self.node_populations)
        initial_scores = await self._evaluate_combination_list(initial_combinations, benchmark, dev_set)
        combo_pop_with_scores = list(zip(initial_combinations, initial_scores))
        self._log_de_generation(0, combo_pop_with_scores)
        initial_best = max(initial_scores) if initial_scores else 0
        initial_avg = np.mean(initial_scores) if initial_scores else 0
        self.best_scores_per_gen["Gen_0"] = {"combination": initial_best}
        self.avg_scores_per_gen["Gen_0"] = {"combination": initial_avg}
        print(f"Initial population - Best score: {initial_best:.4f}, Avg score: {initial_avg:.4f}")
        for t in range(self.iterations):
            print(f"\n--- Generation {t + 1}/{self.iterations} ---")
            evolution_pipeline_tasks = [
                self._evolve_and_select_one(combo_with_score, combo_pop_with_scores, benchmark, dev_set)
                for combo_with_score in combo_pop_with_scores
            ]
            pbar = aio_tqdm(total=len(evolution_pipeline_tasks), desc=f"Pipelined Evolution Gen {t+1}")
            next_gen_pop_with_scores = []
            for future in asyncio.as_completed(evolution_pipeline_tasks):
                result = await future
                next_gen_pop_with_scores.append(result)
                pbar.update(1)
            pbar.close()
            combo_pop_with_scores = next_gen_pop_with_scores
            self._log_de_generation(t + 1, combo_pop_with_scores)
            current_scores = [score for _, score in combo_pop_with_scores]
            best_gen_score = max(current_scores) if current_scores else 0
            avg_gen_score = np.mean(current_scores) if current_scores else 0
            gen_name = f"Gen_{t + 1}"
            self.best_scores_per_gen[gen_name] = {"combination": best_gen_score}
            self.avg_scores_per_gen[gen_name] = {"combination": avg_gen_score}
            print(f"Generation {t + 1} complete. Best score: {best_gen_score:.4f}, Avg score: {avg_gen_score:.4f}")
        print("\n--- Combination-Level Evolution complete ---")
        best_combination, best_score = max(combo_pop_with_scores, key=lambda x: x[1]) if combo_pop_with_scores else ({}, 0)
        logger.info(f"Optimization finished! Best combination found with score {best_score:.4f}.")
        self._log_optimization_summary("DE_Combination", best_combination)
        self.apply_cfg(best_combination)
        return best_combination, self.best_scores_per_gen, self.avg_scores_per_gen