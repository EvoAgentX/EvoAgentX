from collections import defaultdict
import logging
import random
import optuna
from optuna.distributions import CategoricalDistribution
import numpy as np
from pydantic import Field
from typing import Any, Callable, Dict, Literal, Optional, Union, List, Tuple

from ..evaluators.mipro_evaluator import Evaluate as mipro_evaluator
from ..workflow.workflow import WorkFlowGraph
from ..workflow.action_graph import ActionGraph
from ..core.module import BaseModule
from ..models.base_model import BaseLLM
from ..utils.mipro_utils.settings import settings
from ..utils.mipro_utils.grounded_proposer import GroundedProposer
from ..utils.mipro_utils.utils import (create_minibatch,
                                      create_n_fewshot_demo_sets,
                                      eval_candidate_program, get_program_with_highest_avg_score,
                                      )

logger = logging.getLogger("MIPRO")
logging.basicConfig(level=logging.INFO)

# Constants
BOOTSTRAPPED_FEWSHOT_EXAMPLES_IN_CONTEXT = 3
LABELED_FEWSHOT_EXAMPLES_IN_CONTEXT = 0
MIN_MINIBATCH_SIZE = 50

AUTO_RUN_SETTINGS = {
    "light": {"num_trials": 6, "val_size": 100},
    "medium": {"num_trials": 12, "val_size": 300},
    "heavy": {"num_trials": 18, "val_size": 1000},
}

# ANSI escape codes for colors
YELLOW = "\033[93m"
GREEN = "\033[92m"
BLUE = "\033[94m"
BOLD = "\033[1m"
ENDC = "\033[0m"  # Resets the color to default


class MiproOptimizer(BaseModule):

    graph: Optional[Union[WorkFlowGraph, ActionGraph]] = Field(default=None, description="The workflow to optimize.")
    grpah_path: Optional[str] = Field(default=None, description="Path to the workflow to optimize.")
    metric: Optional[Callable] = Field(default=None, description="Evaluation function used to assess the quality of generated results")
    prompt_model: BaseLLM = Field(default=settings.lm, description="Language model used for prompt generation")
    task_model: BaseLLM = Field(default=settings.lm, description="Language model used for task execution")
    teacher_settings: Dict = Field(default_factory=dict, description="Configuration parameters for the teacher model")
    max_bootstrapped_demos: int = Field(default=4, description="Maximum number of bootstrapped examples")
    max_labeled_demos: int = Field(default=4, description="Maximum number of labeled examples")
    auto: Optional[Literal["light", "medium", "heavy"]] = Field(default="medium", description="Automation level, can be light/medium/heavy")
    num_candidates: int = Field(default=10, description="Number of candidates generated per round")
    num_threads: Optional[int] = Field(default=1, description="Number of parallel threads, None means using all available threads")
    max_errors: int = Field(default=10, description="Maximum number of allowed errors")
    seed: int = Field(default=9, description="Random seed for reproducibility")
    init_temperature: float = Field(default=0.5, description="Initial sampling temperature")
    verbose: bool = Field(default=False, description="Whether to output detailed logs")
    track_stats: bool = Field(default=True, description="Whether to track statistics")
    log_dir: Optional[str] = Field(default=None, description="Directory for saving logs, None means no saving")
    metric_threshold: Optional[float] = Field(default=None, description="Metric threshold to stop optimization when reached")


    def init_module(self):
        self.prompt_model_total_calls = 0
        self.total_calls = 0
        self.rng = None
        if not self.graph and not self.graph_path:
            raise ValueError("Either graph or graph_path must be provided")
        self.graph = self.graph or WorkFlowGraph.from_file(self.graph_path)
        
    def optimize(
        self,
        *,
        trainset: List,
        with_inputs: List[str],
        teacher: Any = None,
        valset: Optional[List] = None,
        num_trials: int = 30,
        max_bootstrapped_demos: Optional[int] = None,
        max_labeled_demos: Optional[int] = None,
        seed: Optional[int] = None,
        minibatch: bool = True,
        minibatch_size: int = 35,
        minibatch_full_eval_steps: int = 5,
        program_aware_proposer: bool = True,
        data_aware_proposer: bool = True,
        view_data_batch_size: int = 10,
        tip_aware_proposer: bool = True,
        fewshot_aware_proposer: bool = True,
        requires_permission_to_run: bool = True,
        provide_traceback: Optional[bool] = None,
    ) -> Any:
        self.with_inputs = with_inputs
        
        
        seed = seed or self.seed
        self._set_random_seeds(seed)
        
        if max_bootstrapped_demos is not None:
            self.max_bootstrapped_demos = max_bootstrapped_demos
        
        if max_labeled_demos is not None:
            self.max_labeled_demos = max_labeled_demos
        
        
        trainset, valset = self._set_and_validate_datasets(trainset, valset)
        
        
        zeroshot_opt = (self.max_bootstrapped_demos == 0) and (self.max_labeled_demos == 0)
        num_trials, valset, minibatch = self._set_hyperparams_from_run_mode(
            self.graph, num_trials, minibatch, zeroshot_opt, valset
        )
        
        if self.auto:
            self._print_auto_run_settings(num_trials, minibatch, valset)

        if minibatch and minibatch_size > len(valset):
            raise ValueError(f"Minibatch size cannot exceed the size of the valset. Valset size: {len(valset)}.")

        program = self.graph.deep_copy()
        # TODO: 敲定evaluator的细节
        evaluate = mipro_evaluator(
            devset=valset,
            metric=self.metric,
            num_threads=self.num_threads,
            display_progress=True,
            display_table=False,
            max_errors=self.max_errors,
            provide_traceback = provide_traceback,
        )
        

        # Step 1: Bootstrap few-shot examples
        demo_candidates = self._bootstrap_fewshot_examples(program, trainset, seed, teacher)

        
        # Step 2: Propose instruction candidates
        instruction_candidates = self._propose_instructions(
            program,
            trainset,
            demo_candidates,
            view_data_batch_size,
            program_aware_proposer,
            data_aware_proposer,
            tip_aware_proposer,
            fewshot_aware_proposer,
        )


        # If zero-shot, discard demos
        if zeroshot_opt:
            demo_candidates = None



        # Step 3: Find optimal prompt parameters
        best_program = self._optimize_prompt_parameters(
            program,
            instruction_candidates,
            demo_candidates,
            evaluate,
            valset,
            num_trials,
            minibatch,
            minibatch_size,
            minibatch_full_eval_steps,
            seed,
        )

        return best_program

            
    def _set_and_validate_datasets(self, trainset: List, valset: Optional[List]):
        if not trainset:
            raise ValueError("Trainset cannot be empty.")

        if valset is None:
            if len(trainset) < 2:
                raise ValueError("Trainset must have at least 2 examples if no valset specified.")
            valset_size = min(1000, max(1, int(len(trainset) * 0.80)))
            cutoff = len(trainset) - valset_size
            valset = trainset[cutoff:]
            trainset = trainset[:cutoff]
        else:
            if len(valset) < 1:
                raise ValueError("Validation set must have at least 1 example.")

        return trainset, valset

    def _set_random_seeds(self, seed: int):
        self.rng = random.Random(seed)
        np.random.seed(seed)
    
    def _set_hyperparams_from_run_mode(
        self,
        program: Any,
        num_trials: int,
        minibatch: bool,
        zeroshot_opt: bool,
        valset: List,
    ) -> Tuple[int, List, bool]:
        if self.auto is None:
            return num_trials, valset, minibatch


        auto_settings = AUTO_RUN_SETTINGS[self.auto]
        
        valset = create_minibatch(valset, batch_size=auto_settings["val_size"], rng=self.rng)
        minibatch = len(valset) > MIN_MINIBATCH_SIZE
        
        self.num_instruct_candidates = auto_settings["num_trials"] if zeroshot_opt else int(auto_settings["num_trials"] * 0.5)
        self.num_fewshot_candidates = auto_settings['num_trials']
        
        
        num_vars = len(program.agents())
        if not zeroshot_opt:
            num_vars *= 2

        num_trials = int(max(2 * num_vars * np.log2(auto_settings['num_trials']), 1.5 * auto_settings['num_trials']))

        return num_trials, valset, minibatch

    def _print_auto_run_settings(self, num_trials: int, minibatch: bool, valset: List):
        logger.info(
            f"\nRUNNING WITH THE FOLLOWING {self.auto.upper()} AUTO RUN SETTINGS:"
            f"\nnum_trials: {num_trials}"
            f"\nminibatch: {minibatch}"
            f"\nnum_candidates: {self.num_candidates}"
            f"\nvalset size: {len(valset)}\n"
        )

    def _bootstrap_fewshot_examples(self, program: Any, trainset: List, seed: int, teacher: Any) -> Optional[List]:
        logger.info("\n==> STEP 1: BOOTSTRAP FEWSHOT EXAMPLES <==")
        if self.max_bootstrapped_demos > 0:
            logger.info(
                "These will be used as few-shot example candidates for our program and for creating instructions.\n"
            )
        else:
            logger.info("These will be used for informing instruction proposal.\n")

        logger.info(f"Bootstrapping N={self.num_candidates} sets of demonstrations...")

        zeroshot = self.max_bootstrapped_demos == 0 and self.max_labeled_demos == 0
        
        try:
            demo_candidates = create_n_fewshot_demo_sets(
                student=program,
                num_candidate_sets=self.num_candidates,
                trainset=trainset,
                trainset_inputs=self.with_inputs,
                max_labeled_demos=(LABELED_FEWSHOT_EXAMPLES_IN_CONTEXT if zeroshot else self.max_labeled_demos),
                max_bootstrapped_demos=(
                    BOOTSTRAPPED_FEWSHOT_EXAMPLES_IN_CONTEXT if zeroshot else self.max_bootstrapped_demos
                ),
                metric=self.metric,
                max_errors=self.max_errors,
                teacher=teacher,
                teacher_settings=self.teacher_settings,
                seed=seed,
                metric_threshold=self.metric_threshold,
                rng=self.rng,
            )
        except Exception as e:
            logger.info(f"Error generating few-shot examples: {e}")
            logger.info("Running without few-shot examples.")
            demo_candidates = None

        return demo_candidates
    
    def _propose_instructions(
        self,
        program: Any,
        trainset: List,
        demo_candidates: Optional[List],
        view_data_batch_size: int,
        program_aware_proposer: bool,
        data_aware_proposer: bool,
        tip_aware_proposer: bool,
        fewshot_aware_proposer: bool,
    ) -> Dict[int, List[str]]:
        logger.info("\n==> STEP 2: PROPOSE INSTRUCTION CANDIDATES <==")
        logger.info(
            "We will use the few-shot examples from the previous step, a generated dataset summary, a summary of the program code, and a randomly selected prompting tip to propose instructions."
        )

        proposer = GroundedProposer(
            program=program,
            trainset=trainset,
            prompt_model=self.prompt_model,
            view_data_batch_size=view_data_batch_size,
            program_aware=program_aware_proposer,
            use_dataset_summary=data_aware_proposer,
            use_task_demos=fewshot_aware_proposer,
            num_demos_in_context=BOOTSTRAPPED_FEWSHOT_EXAMPLES_IN_CONTEXT,
            use_tip=tip_aware_proposer,
            set_tip_randomly=tip_aware_proposer,
            use_instruct_history=False,
            set_history_randomly=False,
            verbose=self.verbose,
            rng=self.rng,
        )

        logger.info("\nProposing instructions...\n")
        instruction_candidates = proposer.propose_instructions_for_program(
            trainset=trainset,
            program=program,
            demo_candidates=demo_candidates,
            N=self.num_candidates,
            T=self.init_temperature,
            trial_logs={},
        )

        for i, agent in enumerate(program.agents()):
            logger.info(f"Proposed Instructions for Predictor {i}:\n")
            instruction_candidates[i][0] = agent['prompt']
            for j, instruction in enumerate(instruction_candidates[i]):
                logger.info(f"{j}: {instruction}\n")
            logger.info("\n")

        return instruction_candidates

    def _optimize_prompt_parameters(
        self,
        program: Any,
        instruction_candidates: Dict[int, List[str]],
        demo_candidates: Optional[List],
        evaluate: mipro_evaluator,
        valset: List,
        num_trials: int,
        minibatch: bool,
        minibatch_size: int,
        minibatch_full_eval_steps: int,
        seed: int,
    ) -> Optional[Any]:
        
        # Run optimization
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        logger.info("==> STEP 3: FINDING OPTIMAL PROMPT PARAMETERS <==")
        logger.info(
            "We will evaluate the program over a series of trials with different combinations of instructions and few-shot examples to find the optimal combination using Bayesian Optimization.\n"
        )
        
        # Compute the adjusted total trials that we will run (including full evals)
        run_additional_full_eval_at_end = 1 if num_trials % minibatch_full_eval_steps != 0 else 0
        adjusted_num_trials = int((num_trials + num_trials // minibatch_full_eval_steps + 1 + run_additional_full_eval_at_end) if minibatch else num_trials)
        logger.info(f"== Trial {1} / {adjusted_num_trials} - Full Evaluation of Default Program ==")
        
        default_score, _ = eval_candidate_program(
            len(valset), valset, self.with_inputs, program, evaluate, self.rng, return_all_scores=True
        )
        
        logger.info(f"Default Program Score: {default_score}")
        
        trial_logs = {}
        trial_logs[1] = {}
        trial_logs[1]["full_eval_program_path"] = program.save_module(self.log_dir, trial_num=-1)
        trial_logs[1]["full_eval_score"] = default_score
        trial_logs[1]["total_eval_calls_so_far"] = len(valset)
        trial_logs[1]["full_eval_program"] = program.deep_copy()
        
        # Initialize optimization variables 
        best_score = default_score
        best_program = program.deep_copy()
        total_eval_calls = len(valset)
        score_data = [{"score": best_score, "program": program.deep_copy(), "full_eval": True}]
        param_score_dict = defaultdict(list)
        fully_evaled_param_combos = {}   
        
        # Define the objective function
        def objective(trial):
            nonlocal program, best_program, best_score, trial_logs, total_eval_calls, score_data

            trial_num = trial.number + 1
            if minibatch:
                logger.info(f"== Trial {trial_num} / {adjusted_num_trials} - Minibatch ==")
            else:
                logger.info(f"===== Trial {trial_num} / {num_trials} =====")

            trial_logs[trial_num] = {}
            
            # Create a new candidate program
            candidate_program = program.deep_copy()

            # Choose instructions and demos, insert them into the program
            chosen_params, raw_chosen_params = self._select_and_insert_instructions_and_demos(
                candidate_program,
                instruction_candidates,
                demo_candidates,
                trial,
                trial_logs,
                trial_num,
            )
            
            # Log assembled program
            if self.verbose:
                logger.info("Evaluating the following candidate program...\n")
                candidate_program.display()

            # Evaluate the candidate program (on minibatch if minibatch=True)
            batch_size = minibatch_size if minibatch else len(valset)
            score = eval_candidate_program(batch_size, valset, self.with_inputs, candidate_program, evaluate, self.rng)
            total_eval_calls += batch_size

            # Update best score and program
            if not minibatch and score > best_score:
                best_score = score
                best_program = candidate_program.deep_copy()
                logger.info(f"{GREEN}Best full score so far!{ENDC} Score: {score}")

            # Log evaluation results
            score_data.append(
                {"score": score, "program": candidate_program, "full_eval": batch_size >= len(valset)}
            )  # score, prog, full_eval
            if minibatch:
                self._log_minibatch_eval(
                    score,
                    best_score,
                    batch_size,
                    chosen_params,
                    score_data,
                    trial,
                    adjusted_num_trials,
                    trial_logs,
                    trial_num,
                    candidate_program,
                    total_eval_calls,
                )
            else:
                self._log_normal_eval(
                    score,
                    best_score,
                    chosen_params,
                    score_data,
                    trial,
                    num_trials,
                    trial_logs,
                    trial_num,
                    valset,
                    batch_size,
                    candidate_program,
                    total_eval_calls,
                )
            categorical_key = ",".join(map(str, chosen_params))
            param_score_dict[categorical_key].append(
                (score, candidate_program, raw_chosen_params),
            )

            # If minibatch, perform full evaluation at intervals (and at the very end)
            if minibatch and ((trial_num % (minibatch_full_eval_steps+1) == 0) or (trial_num == (adjusted_num_trials-1))):
                best_score, best_program, total_eval_calls = self._perform_full_evaluation(
                    trial_num,
                    adjusted_num_trials,
                    param_score_dict,
                    fully_evaled_param_combos,
                    evaluate,
                    valset,
                    trial_logs,
                    total_eval_calls,
                    score_data,
                    best_score,
                    best_program,
                    study,
                    instruction_candidates,
                    demo_candidates,
                )

            return score

        sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True)
        study = optuna.create_study(direction="maximize", sampler=sampler)

        default_params = {f"{i}_predictor_instruction": 0 for i in range(len(program.agents()))}
        if demo_candidates:
            default_params.update({f"{i}_predictor_demos": 0 for i in range(len(program.agents()))})

        # Add default run as a baseline in optuna (TODO: figure out how to weight this by # of samples evaluated on)
        trial = optuna.trial.create_trial(
            params=default_params,
            distributions=self._get_param_distributions(program, instruction_candidates, demo_candidates),
            value=default_score,
        )
        study.add_trial(trial)
        study.optimize(objective, n_trials=num_trials)

        # Attach logs to best program
        if best_program is not None and self.track_stats:
            best_program.trial_logs = trial_logs
            best_program.score = best_score
            best_program.prompt_model_total_calls = self.prompt_model_total_calls
            best_program.total_calls = self.total_calls
            sorted_candidate_programs = sorted(score_data, key=lambda x: x["score"], reverse=True)
            # Attach all minibatch programs
            best_program.mb_candidate_programs = [
                score_data for score_data in sorted_candidate_programs if not score_data["full_eval"]
            ]
            # Attach all programs that were evaluated on the full trainset, in descending order of score
            best_program.candidate_programs = [
                score_data for score_data in sorted_candidate_programs if score_data["full_eval"]
            ]

        logger.info(f"Returning best identified program with score {best_score}!")

        return best_program
    
    def _log_minibatch_eval(
        self,
        score,
        best_score,
        batch_size,
        chosen_params,
        score_data,
        trial,
        adjusted_num_trials,
        trial_logs,
        trial_num,
        candidate_program,
        total_eval_calls,
    ):
        trial_logs[trial_num]["mb_program_path"] = candidate_program.save_module(self.log_dir, trial_num=trial_num)
        trial_logs[trial_num]["mb_score"] = score
        trial_logs[trial_num]["total_eval_calls_so_far"] = total_eval_calls
        trial_logs[trial_num]["mb_program"] = candidate_program.deep_copy()

        logger.info(f"Score: {score} on minibatch of size {batch_size} with parameters {chosen_params}.")
        minibatch_scores = ", ".join([f"{s['score']}" for s in score_data if not s["full_eval"]])
        logger.info(f"Minibatch scores so far: {'[' + minibatch_scores + ']'}")
        full_eval_scores = ", ".join([f"{s['score']}" for s in score_data if s["full_eval"]])
        trajectory = "[" + full_eval_scores + "]"
        logger.info(f"Full eval scores so far: {trajectory}")
        logger.info(f"Best full score so far: {best_score}")
        logger.info(
            f"{'=' * len(f'== Trial {trial.number + 1} / {adjusted_num_trials} - Minibatch Evaluation ==')}\n\n"
        )

    def _log_normal_eval(
        self,
        score,
        best_score,
        chosen_params,
        score_data,
        trial,
        num_trials,
        trial_logs,
        trial_num,
        valset,
        batch_size,
        candidate_program,
        total_eval_calls,
    ):
        trial_logs[trial_num]["full_eval_program_path"] = candidate_program.save_module(self.log_dir, trial_num=trial_num)
        trial_logs[trial_num]["full_eval_score"] = score
        trial_logs[trial_num]["total_eval_calls_so_far"] = total_eval_calls
        trial_logs[trial_num]["full_eval_program"] = candidate_program.deep_copy()

        logger.info(f"Score: {score} with parameters {chosen_params}.")
        full_eval_scores = ", ".join([f"{s['score']}" for s in score_data if s["full_eval"]])
        logger.info(f"Scores so far: {'[' + full_eval_scores + ']'}")
        logger.info(f"Best score so far: {best_score}")
        logger.info(f"{'=' * len(f'===== Trial {trial.number + 1} / {num_trials} =====')}\n\n")

    def _select_and_insert_instructions_and_demos(
        self,
        candidate_program: Any,
        instruction_candidates: Dict[int, List[str]],
        demo_candidates: Optional[List],
        trial: optuna.trial.Trial,
        trial_logs: Dict,
        trial_num: int,
    ) -> List[str]:
        chosen_params = []
        raw_chosen_params = {}

        
        for i, agent in enumerate(candidate_program.agents()):
            # Select instruction
            instruction_idx = trial.suggest_categorical(
                f"{i}_predictor_instruction", range(len(instruction_candidates[i]))
            )
            selected_instruction = instruction_candidates[i][instruction_idx]
            candidate_program.agents()[i]['prompt'] = selected_instruction

            trial_logs[trial_num][f"{i}_predictor_instruction"] = instruction_idx
            chosen_params.append(f"Predictor {i}: Instruction {instruction_idx}")
            raw_chosen_params[f"{i}_predictor_instruction"] = instruction_idx
            # Select demos if available
            if demo_candidates:
                agent_name = agent['name']
                demos_idx = trial.suggest_categorical(f"{i}_predictor_demos", range(len(demo_candidates[agent_name])))
                agent['demos'] = demo_candidates[agent_name][demos_idx]
                trial_logs[trial_num][f"{i}_predictor_demos"] = demos_idx
                chosen_params.append(f"Predictor {i}: Few-Shot Set {demos_idx}")
                raw_chosen_params[f"{i}_predictor_demos"] = instruction_idx

        return chosen_params, raw_chosen_params

    def _get_param_distributions(self, program, instruction_candidates, demo_candidates):
        param_distributions = {}

        for i in range(len(instruction_candidates)):
            param_distributions[f"{i}_predictor_instruction"] = CategoricalDistribution(
                range(len(instruction_candidates[i]))
            )
            if demo_candidates:
                agent_name = program.agents()[i]['name']
                param_distributions[f"{i}_predictor_demos"] = CategoricalDistribution(range(len(demo_candidates[agent_name])))

        return param_distributions

    def _perform_full_evaluation(
        self,
        trial_num: int,
        adjusted_num_trials: int,
        param_score_dict: Dict,
        fully_evaled_param_combos: Dict,
        evaluate: mipro_evaluator,
        valset: List,
        trial_logs: Dict,
        total_eval_calls: int,
        score_data,
        best_score: float,
        best_program: Any,
        study: optuna.Study,
        instruction_candidates: List,
        demo_candidates: List,
    ):
        logger.info(f"===== Trial {trial_num + 1} / {adjusted_num_trials} - Full Evaluation =====")

        # Identify best program to evaluate fully
        highest_mean_program, mean_score, combo_key, params = get_program_with_highest_avg_score(
            param_score_dict, fully_evaled_param_combos
        )
        logger.info(f"Doing full eval on next top averaging program (Avg Score: {mean_score}) from minibatch trials...")
        full_eval_score = eval_candidate_program(len(valset), valset, self.with_inputs, highest_mean_program, evaluate, self.rng)
        score_data.append({"score": full_eval_score, "program": highest_mean_program, "full_eval": True})

        # Log full eval as a trial so that optuna can learn from the new results
        trial = optuna.trial.create_trial(
            params=params,
            distributions=self._get_param_distributions(best_program, instruction_candidates, demo_candidates),
            value=full_eval_score,
        )
        study.add_trial(trial)

        # Log full evaluation results
        fully_evaled_param_combos[combo_key] = {
            "program": highest_mean_program,
            "score": full_eval_score,
        }
        total_eval_calls += len(valset)
        trial_logs[trial_num + 1] = {}
        trial_logs[trial_num + 1]["total_eval_calls_so_far"] = total_eval_calls
        trial_logs[trial_num + 1]["full_eval_program_path"] = highest_mean_program.save_module(self.log_dir, trial_num=trial_num + 1)
        trial_logs[trial_num + 1]["full_eval_program"] = highest_mean_program
        trial_logs[trial_num + 1]["full_eval_score"] = full_eval_score

        # Update best score and program if necessary
        if full_eval_score > best_score:
            logger.info(f"{GREEN}New best full eval score!{ENDC} Score: {full_eval_score}")
            best_score = full_eval_score
            best_program = highest_mean_program.deep_copy()
        full_eval_scores = ", ".join([f"{s['score']}" for s in score_data if s["full_eval"]])
        trajectory = "[" + full_eval_scores + "]"
        logger.info(f"Full eval scores so far: {trajectory}")
        logger.info(f"Best full score so far: {best_score}")
        logger.info(len(f"===== Full Eval {len(fully_evaled_param_combos) + 1} =====") * "=")
        logger.info("\n")

        return best_score, best_program, total_eval_calls
