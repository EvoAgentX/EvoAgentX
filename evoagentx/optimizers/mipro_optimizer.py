import os  
import tqdm
# import types 
import inspect
import threading
from copy import deepcopy 
from functools import wraps
from collections import defaultdict

import optuna
from typing import Optional, Callable, Literal, List, Any, Dict, Union, Tuple, Set 

import dspy 
from dspy import MIPROv2
from dspy.clients import LM, Provider
from dspy.utils.callback import BaseCallback 
from dspy.utils.parallelizer import ParallelExecutor 
from dspy.propose.grounded_proposer import GroundedProposer 
from dspy.teleprompt.utils import (
    create_n_fewshot_demo_sets, 
    get_signature, 
    create_minibatch, 
    print_full_program, 
    save_candidate_program, 
    get_program_with_highest_avg_score
) 

from ..core.logging import logger 
from ..core.callbacks import suppress_cost_logging, suppress_logger_info
from ..models.base_model import BaseLLM 
from ..benchmark.benchmark import Benchmark 
from .engine.base import BaseOptimizer
from .engine.registry import ParamRegistry
from ..utils.mipro_utils.register_utils import MiproRegistry
from ..agents.agent_manager import AgentManager 
from ..workflow.workflow_graph import WorkFlowGraph
from ..workflow.workflow import WorkFlow 
from ..evaluators.evaluator import Evaluator 
from ..prompts.template import PromptTemplate, MiproPromptTemplate
# from ..utils.mipro_utils.signature_utils import signature_from_registry
from ..utils.mipro_utils.module_utils import PromptTuningModule

# Constants
BOOTSTRAPPED_FEWSHOT_EXAMPLES_IN_CONTEXT = 3
LABELED_FEWSHOT_EXAMPLES_IN_CONTEXT = 0
MIN_MINIBATCH_SIZE = 50

AUTO_RUN_SETTINGS = {
    "light": {"n": 6, "val_size": 100},
    "medium": {"n": 12, "val_size": 300},
    "heavy": {"n": 18, "val_size": 1000},
}

# ANSI escape codes for colors
YELLOW = "\033[93m"
GREEN = "\033[92m"
BLUE = "\033[94m"
BOLD = "\033[1m"
ENDC = "\033[0m"  # Resets the color to default


class MiproLMWrapper(LM):

    """
    A wrapper class for the LLM model. It converts the BaseLLM model in EvoAgentX to a dspy.LM object. 
    """

    def __init__(
        self, 
        model: BaseLLM, 
        model_type: Literal["chat", "text"] = "chat", 
        temperature: float = 0.0, 
        max_tokens: int = 4000, 
        cache: bool = True,
        cache_in_memory: bool = True,
        callbacks: Optional[List[BaseCallback]] = None,
        num_retries: int = 3,
        provider=None,
        finetuning_model: Optional[str] = None,
        launch_kwargs: Optional[dict[str, Any]] = None,
        train_kwargs: Optional[dict[str, Any]] = None,
        **kwargs,
    ):
        self.model = model 
        self.model_type = model_type
        self.cache = cache
        self.cache_in_memory = cache_in_memory
        self.callbacks = callbacks or []
        self.history = []
        self.provider = provider or Provider()
        self.num_retries = num_retries
        self.finetuning_model = finetuning_model
        self.launch_kwargs = launch_kwargs or {}
        self.train_kwargs = train_kwargs or {}
        self.kwargs = dict(temperature=temperature, max_tokens=max_tokens, **kwargs)

    def forward(self, prompt=None, messages=None, **kwargs):

        response = self.model.generate(prompt=prompt, messages=messages, **kwargs)
        return [response.content]
    
    def __call__(self, prompt=None, messages=None, **kwargs):
        return self.forward(prompt=prompt, messages=messages, **kwargs)

    def copy(self, **kwargs):

        new_config = deepcopy(self.model.config)
        new_kwargs = {}

        for key, value in kwargs.items():
            if hasattr(new_config, key):
                setattr(new_config, key, value)
            if (key in self.kwargs) or (not hasattr(self, key)):
                new_kwargs[key] = value
        
        new_model = self.model.__class__(config=new_config)
        return MiproLMWrapper(new_model, **new_kwargs)
    
    def generate(self, *args, **kwargs):
        # to be compatible with BaseLLM.generate()
        return self.model.generate(*args, **kwargs)
    
    async def async_generate(self, *args, **kwargs):
        # to be compatible with BaseLLM.async_generate()
        return await self.model.async_generate(*args, **kwargs)

class MiproEvaluator:

    def __init__(
        self, 
        benchmark: Benchmark, 
        num_threads: Optional[int] = None, 
        display_progress: Optional[bool] = None, 
        max_errors: int = 5, 
        return_all_scores: bool = False, 
        return_outputs: bool = False, 
        provide_traceback: bool = False, 
        failure_score: float = 0.0, 
        metric_name: Optional[str] = None,
        **kwargs
    ):
        self.benchmark = benchmark 
        self.num_threads = num_threads 
        self.display_progress = display_progress 
        self.max_errors = max_errors 
        self.return_all_scores = return_all_scores 
        self.return_outputs = return_outputs 
        self.provide_traceback = provide_traceback 
        self.failure_score = failure_score
        self.metric_name = metric_name
        self.kwargs = kwargs
        # Add a thread-safe counter for logging
        self._log_counter = 0
        self._log_lock = threading.Lock()

    def _extract_score_from_dict(self, score_dict: Dict[str, float]) -> float:
        """Extract a single score from a dictionary of scores.
        
        Args:
            score_dict (Dict[str, float]): Dictionary containing metric scores
            
        Returns:
            float: The extracted score based on the following rules:
                1. If dict has only one score, return that score
                2. If metric_name is specified, return that metric's score
                3. Otherwise, return average of all scores
        """
        if len(score_dict) == 1:
            return list(score_dict.values())[0]
        elif self.metric_name is not None:
            return score_dict[self.metric_name]
        else:
            avg_score = sum(score_dict.values()) / len(score_dict)
            # Use thread-safe counter to ensure message is only logged once
            with self._log_lock:
                if self._log_counter == 0:
                    logger.info(f"`{type(self.benchmark)}.evaluate` returned a dictionary of scores, but no metric name was provided. Will return the average score across all metrics.")
                    self._log_counter += 1
            return avg_score

    def metric(self, example: dspy.Example, prediction: Any, *args, **kwargs):

        if isinstance(self.benchmark.get_train_data()[0], dspy.Example):
            # the data in original benchmark is a dspy.Example
            score = self.benchmark.evaluate(
                prediction=prediction, 
                label=self.benchmark.get_label(example)
            )
        elif isinstance(self.benchmark.get_train_data()[0], dict):
            # the data in original benchmark is a dict, convert the dspy.Example to a dict
            score = self.benchmark.evaluate(
                prediction=prediction, 
                label=self.benchmark.get_label(example.toDict()) # convert the dspy.Example to a dict
            )
        else:
            raise ValueError(f"Unsupported example type in `{type(self.benchmark)}`! Expected `dspy.Example` or `dict`, got {type(self.benchmark.get_train_data()[0])}")
        
        if isinstance(score, dict):
            score = self._extract_score_from_dict(score)
        
        return score
        
    def __call__(
        self, 
        program: Callable, 
        evalset: List[Any], 
        **kwargs, 
    ) -> float:
        
        return_all_scores = kwargs.get("return_all_scores", None) or self.return_all_scores
        return_outputs = kwargs.get("return_outputs", None) or self.return_outputs
        
        tqdm.tqdm._instances.clear()
        
        # Get the current suppress_cost_logs value
        from ..core.callbacks import suppress_cost_logs
        current_suppress_cost = suppress_cost_logs.get()
        
        if self.num_threads and self.num_threads > 1: 
            executor = ParallelExecutor(
                num_threads=self.num_threads,
                disable_progress_bar=not self.display_progress,
                max_errors=self.max_errors,
                provide_traceback=self.provide_traceback,
                compare_results=True,
            )
        else:
            executor = None 

        def process_item(example):
            # Set the suppress_cost_logs context in the worker thread
            token = suppress_cost_logs.set(current_suppress_cost)
            try:
                if not isinstance(example, dspy.Example):
                    raise ValueError(f"Example from benchmark must be a dspy.Example object, got {type(example)}")
                
                try:
                    # prediction = program(**example)
                    # score = metric(example, prediction)
                    prediction = program(**example.inputs())
                    score = self.metric(example, prediction)
                    # score = self.benchmark.evaluate(prediction=prediction, label=self.benchmark.get_label(example))
                except Exception as e:
                    logger.error(f"Error evaluating example {example}: {e}")
                    return None, self.failure_score

                # Increment assert and suggest failures to program's attributes
                if hasattr(program, "_assert_failures"):
                    program._assert_failures += dspy.settings.get("assert_failures")
                if hasattr(program, "_suggest_failures"):
                    program._suggest_failures += dspy.settings.get("suggest_failures")

                return prediction, score
            finally:
                # Reset the context
                suppress_cost_logs.reset(token)

        if executor:
            results = executor.execute(process_item, evalset)
        else:
            # Use tqdm for single-threaded execution
            results = []
            pbar = tqdm.tqdm(
                total=len(evalset),
                dynamic_ncols=True,
                disable=not self.display_progress,
                desc="Processing examples"
            )
            for example in evalset:
                result = process_item(example)
                results.append(result)
                # Update progress bar with current score if available
                if result and result[1] is not None:
                    current_scores = [r[1] for r in results if r and r[1] is not None]
                    avg_score = sum(current_scores) / len(current_scores) if current_scores else 0
                    pbar.set_description(f"Average Metric: {avg_score:.2f}")
                pbar.update(1)
            pbar.close()

        assert len(evalset) == len(results)

        results = [(example, prediction, score) for example, (prediction, score) in zip(evalset, results)]
        ncorrect, ntotal = sum(score for *_, score in results), len(evalset)
        logger.info(f"Average Metric: {ncorrect} / {ntotal} ({round(100 * ncorrect / ntotal, 1)}%)")

        if return_all_scores and return_outputs:
            return round(100 * ncorrect / ntotal, 2), results, [score for *_, score in results]
        if return_all_scores:
            return round(100 * ncorrect / ntotal, 2), [score for *_, score in results]
        if return_outputs:
            return round(100 * ncorrect / ntotal, 2), results

        return round(100 * ncorrect / ntotal, 2)


class MiproOptimizer(BaseOptimizer, MIPROv2):

    def __init__(
        self,
        registry: ParamRegistry,
        program: Callable,
        optimizer_llm: BaseLLM,
        evaluator: Optional[Callable] = None,
        eval_rounds: Optional[int] = 1, 
        metric_threshold: Optional[float] = None,
        max_bootstrapped_demos: int = 4, 
        max_labeled_demos: int = 4, 
        auto: Optional[Literal["light", "medium", "heavy"]] = "medium", 
        max_steps: int = None, 
        num_candidates: Optional[int] = None, 
        num_threads: Optional[int] = None, 
        max_errors: int = 10, 
        seed: int = 9, 
        init_temperature: float = 0.5, 
        track_stats: bool = True, 
        save_path: Optional[str] = None,  
        minibatch: bool = True, 
        minibatch_size: int = 35, 
        minibatch_full_eval_steps: int = 5, 
        program_aware_proposer: bool = True,
        data_aware_proposer: bool = True,
        view_data_batch_size: int = 10,
        tip_aware_proposer: bool = True,
        fewshot_aware_proposer: bool = True,
        requires_permission_to_run: bool = False,
        provide_traceback: Optional[bool] = None,
        verbose: bool = False, 
        **kwargs
    ):
        """
        Base MiproOptimizer class that supports plug-and-play usage. 

        Args: 
            registry (ParamRegistry): a ParamRegistry object that contains the parameters to optimize. 
            program (Callable): a program to optimize. Must be a callable object with save(path) and load(path) methods.
            optimizer_llm (BaseLLM): a language model to use for optimization. 
            evaluator (Optional[Callable]): a function that evaluates the performance of the program. 
                Required to have a `__call__(program, evalset, *kwargs) -> float` method that receives a program and a list of 
                examples from a benchmark's train/dev/test set and return a float score. Must also have a `metric(example, prediction) -> float` 
                method that evaluates a single example. If not provided, will construct a default evaluator using the benchmark's evaluate method.
            eval_rounds (Optional[int]): number of rounds to evaluate the program. Defaults to 1. 
            metric_threshold (Optional[float]): threshold for the metric score. If provided, only examples with scores above this threshold will be used as demonstrations. 
                If not provided, examples with scores above 0 will be used as demonstrations. 
            max_bootstrapped_demos (int): maximum number of bootstrapped demonstrations to use. Defaults to 4.
            max_labeled_demos (int): maximum number of labeled demonstrations to use. Defaults to 4.
            auto (Optional[Literal["light", "medium", "heavy"]]): automatic configuration mode. If set, will override num_candidates and max_steps. 
                "light": n=6, val_size=100; "medium": n=12, val_size=300; "heavy": n=18, val_size=1000. Defaults to "medium".
            max_steps (int): maximum number of optimization steps. Required if auto is None.
            num_candidates (Optional[int]): number of candidates to generate for each optimization step. Required if auto is None.
            num_threads (Optional[int]): number of threads to use for parallel evaluation. If None, will use single thread. Only used if evaluator is not provided. 
            max_errors (int): maximum number of errors allowed during evaluation before stopping. Defaults to 10.
            seed (int): random seed for reproducibility. Defaults to 9.
            init_temperature (float): initial temperature for instruction generation. Defaults to 0.5.
            track_stats (bool): whether to track optimization statistics. Defaults to True.
            save_path (Optional[str]): path to save optimization results. If None, results will not be saved.
            minibatch (bool): whether to use minibatch evaluation during optimization. Defaults to True.
            minibatch_size (int): size of minibatch for evaluation. Defaults to 35.
            minibatch_full_eval_steps (int): number of minibatch steps between full evaluations. Defaults to 5.
            program_aware_proposer (bool): whether to use program-aware instruction proposer. Defaults to True.
            data_aware_proposer (bool): whether to use data-aware instruction proposer. Defaults to True.
            view_data_batch_size (int): batch size for viewing data during instruction proposal. Defaults to 10.
            tip_aware_proposer (bool): whether to use tip-aware instruction proposer. Defaults to True.
            fewshot_aware_proposer (bool): whether to use fewshot-aware instruction proposer. Defaults to True.
            requires_permission_to_run (bool): whether to require user permission before running optimization. Defaults to False.
            provide_traceback (Optional[bool]): whether to provide traceback for evaluation errors. If None, will use default setting.
            **kwargs: additional keyword arguments to pass to the evaluator.

        Raises:
            TypeError: If program is not callable or evaluator doesn't return float
            ValueError: If program doesn't have required methods (save and load) or if evaluator doesn't have required methods
        """

        # initialize base optimizer
        BaseOptimizer.__init__(self, registry=registry, program=program, evaluator=evaluator)

        # convert the registry and program to dspy-compatible module
        self._validate_program(program=program)
        self.model = self._convert_to_dspy_module(registry, program)
        self.optimizer_llm = MiproLMWrapper(optimizer_llm)
        dspy.configure(lm=self.optimizer_llm)
        self.task_model = dspy.settings.lm 
        self.prompt_model = dspy.settings.lm 
        self.metric_threshold = metric_threshold
        self.metric_name = None 
        self.teacher_settings = {"use_teacher": True} 

        # Validate 'auto' parameter
        allowed_modes = {None, "light", "medium", "heavy"}
        if auto not in allowed_modes:
            raise ValueError(f"Invalid value for auto: {auto}. Must be one of {allowed_modes}.")
        self.auto = auto
        self.num_fewshot_candidates = num_candidates
        self.num_instruct_candidates = num_candidates
        self.num_candidates = num_candidates
        self.init_temperature = init_temperature
        self.max_bootstrapped_demos = max_bootstrapped_demos
        self.max_labeled_demos = max_labeled_demos
        self.max_steps = max_steps
        self.num_threads = num_threads
        self.max_errors = max_errors
        
        self.track_stats = track_stats
        self.eval_rounds = eval_rounds 
        self.save_path = save_path
        self.prompt_model_total_calls = 0
        self.total_calls = 0
        self.seed = seed
        self.rng = None

        self.minibatch = minibatch 
        self.minibatch_size = minibatch_size 
        self.minibatch_full_eval_steps = minibatch_full_eval_steps 
        self.program_aware_proposer = program_aware_proposer 
        self.data_aware_proposer = data_aware_proposer 
        self.view_data_batch_size = view_data_batch_size 
        self.tip_aware_proposer = tip_aware_proposer 
        self.fewshot_aware_proposer = fewshot_aware_proposer 
        self.requires_permission_to_run = requires_permission_to_run 
        self.provide_traceback = provide_traceback 
        self.verbose = verbose
        self.kwargs = kwargs 

    def _validate_program(self, program: Callable):
        """
        Validate that the program meets the required interface.
        
        Args:
            program (Callable): The program to validate
            
        Raises:
            TypeError: If program is not callable
            ValueError: If program doesn't have required methods (save and load)
        """
        if not callable(program):
            raise TypeError("program must be callable")
        
        # Check if program has save method
        if not hasattr(program, 'save'):
            # raise ValueError("program must have a `save` method")
            logger.warning("program does not have a `save(path=...)` method, will use the default save method in dspy.Module")
        else:
            # Check save method signature
            save_sig = inspect.signature(program.save)
            save_params = list(save_sig.parameters.keys())
            if 'path' not in save_params:
                raise ValueError("program.save must accept a 'path' parameter")
        
        # Check if program has load method
        if not hasattr(program, 'load'):
            # raise ValueError("program must have a `load` method")
            logger.warning("program does not have a `load(path=...)` method, will use the default load method in dspy.Module")
        else:
            # Check load method signature
            load_sig = inspect.signature(program.load)
            load_params = list(load_sig.parameters.keys())
            if 'path' not in load_params:
                raise ValueError("program.load must accept a 'path' parameter")

    def _validate_evaluator(self, evaluator: Callable = None, benchmark: Benchmark = None, metric_name: Optional[str] = None) -> Callable:
        """
        Validate that the evaluator meets the required interface and wrap it with runtime checks.
        
        Args:
            evaluator (Callable): The evaluator to validate. 
                If provided, it must have a `__call__(program, evalset, *kwargs) -> float` method that receives a program and a list of examples from a benchmark's train/dev/test set and return a float score. 
                It must also have a `metric(example: dspy.Example, prediction: Any) -> float/int/bool` method that evaluates a single example. 
            benchmark (Benchmark): The benchmark to use for evaluation. Only used if evaluator is not provided. In this case, the evaluator will be constructed using the `evaluate` method (return a dictionary of scores) in the benchmark. 
            metric_name (Optional[str]): The name of the metric to use for evaluation. Only used if evaluator is not provided. It will be used to select the metric for optimization from the dictionary of scores returned by the benchmark's `evaluate` method. 
            
        Raises:
            TypeError: If evaluator is not callable or doesn't return float
            ValueError: If evaluator doesn't have required parameters
        """

        if evaluator is None:
            if not hasattr(benchmark, "evaluate"):
                raise ValueError("`evaluator` is not provided and the benchmark does not have a `evaluate` method.")
            logger.info("`evaluator` is not provided. Will construct a default evaluator using the `evaluate` method in the benchmark.")
            evaluator = MiproEvaluator(
                benchmark=benchmark, 
                num_threads=self.num_threads, 
                max_errors=self.max_errors, 
                display_progress=True, 
                provide_traceback=self.provide_traceback,
                metric_name=metric_name,
                **self.kwargs
            )

        if not callable(evaluator):
            raise TypeError("evaluator must be callable, i.e., a function or a class with interface `__call__(program, evalset, *kwargs) -> float`")
        
        # Check if evaluator has __call__ method with correct signature
        
        sig = inspect.signature(evaluator.__call__ if hasattr(evaluator, '__call__') else evaluator)
        params = list(sig.parameters.keys())
        
        if len(params) < 2:
            raise ValueError("evaluator must accept at least two parameters (program and evalset)")
        
        # Check return type annotation if available
        if sig.return_annotation != inspect.Signature.empty:
            if sig.return_annotation not in [float, int, bool]:
                raise TypeError("evaluator must return a float, int, or bool")
            
        # check if the evaluator has a `metric` method with correct signature 
        if not hasattr(evaluator, 'metric'):
            raise ValueError("evaluator must have a `metric(example: dspy.Example, prediction: Any) -> float/int/bool` method")
        
        metric_sig = inspect.signature(evaluator.metric)
        metric_params = list(metric_sig.parameters.keys())
        
        if len(metric_params) < 2:
            raise ValueError("evaluator.metric must accept at least two parameters (example and prediction)")
        
        if metric_params[0] != 'example' or metric_params[1] != 'prediction':
            raise ValueError("evaluator.metric must have parameters in order: example, prediction")
        
        # if '*args' not in str(metric_sig):
        #     raise ValueError("evaluator.metric must accept *args")
        
        # Wrap the metric method with runtime checks
        # original_metric = evaluator.metric
        
        # @wraps(original_metric)
        # def wrapped_metric(example, prediction, *args, **kwargs):
        #     result = original_metric(example, prediction, *args, **kwargs)
            
        #     # Runtime check for return value
        #     if not isinstance(result, (float, int, bool)):
        #         raise TypeError(f"evaluator.metric must return a float, int, or bool, got {type(result)}")
            
        #     return result
        
        # evaluator.metric = types.MethodType(wrapped_metric, evaluator)
        
        # Wrap the evaluator with runtime checks
        original_evaluator = evaluator.__call__ if hasattr(evaluator, '__call__') else evaluator
        
        @wraps(original_evaluator)
        def wrapped_evaluator(*args, **kwargs):
            result = original_evaluator(*args, **kwargs)
            
            # Runtime check for return value
            if not isinstance(result, (float, int, bool)):
                raise TypeError(f"evaluator must return a float, int, or bool, got {type(result)}")
            
            return result
        
        # Replace the evaluator with our wrapped version
        if hasattr(evaluator, '__call__'):
            evaluator.__call__ = wrapped_evaluator
        else:
            # If it's a function, we need to create a new callable object
            class WrappedEvaluator:
                def __init__(self, func):
                    self._func = func
                
                def __call__(self, *args, **kwargs):
                    return wrapped_evaluator(*args, **kwargs)
            
            return WrappedEvaluator(evaluator)
        
        return evaluator

    def _convert_to_dspy_module(self, registry: ParamRegistry, program: Callable):

        if isinstance(program, dspy.Module):
            return program
        
        program = PromptTuningModule.from_registry(
            program=program,
            registry=registry,
        )

        return program 
                
    def optimize(self, dataset: Benchmark, metric_name: Optional[str] = None, **kwargs):

        """
        Optimize the program using the Mipro algorithm. 

        Args:
            dataset (Benchmark): a Benchmark object that contains the training and validation data. 
            metric_name (Optional[str]): the name of the metric to use for optimization. Only used when `self.evaluator` is not provided. 
                In this case, the evaluator will be constructed using the `evaluate` method (return a dictionary of scores) in the benchmark, 
                and the metric specified by `metric_name` will be used for optimization. If not provided, the average of all scores returned by the evaluator will be used. 
                If `self.evaluator` is provided, this argument will be ignored. 
            **kwargs: additional keyword arguments to pass to the evaluator. 
        """

        zeroshot_opt = (self.max_bootstrapped_demos == 0) and (self.max_labeled_demos == 0)
        student = self.model
        num_trials = self.max_steps
        minibatch = self.minibatch
        self.metric_name = metric_name

        # If auto is None, and num_trials is not provided (but num_candidates is), raise an error that suggests a good num_trials value
        if self.auto is None and (self.num_candidates is not None and num_trials is None):
            raise ValueError(f"If auto is None, max_steps must also be provided. Given num_candidates={self.num_candidates}, we'd recommend setting max_steps to ~{self._set_num_trials_from_num_candidates(self.model, zeroshot_opt, self.num_candidates)}.")

        # If auto is None, and num_candidates or num_trials is None, raise an error
        if self.auto is None and (self.num_candidates is None or num_trials is None):
            raise ValueError("If auto is None, num_candidates must also be provided.")

        # If auto is provided, and either num_candidates or num_trials is not None, raise an error
        if self.auto is not None and (self.num_candidates is not None or num_trials is not None):
            raise ValueError("If auto is not None, num_candidates and max_steps cannot be set, since they would be overrided by the auto settings. Please either set auto to None, or do not specify num_candidates and max_steps.")

        # Set random seeds
        seed = self.seed
        self._set_random_seeds(seed)

        # Set training & validation sets
        trainset, valset = self._set_and_validate_datasets(dataset=dataset)

        # Set hyperparameters based on run mode (if set)
        num_trials, valset, minibatch = self._set_hyperparams_from_run_mode(
            student, num_trials, minibatch, zeroshot_opt, valset
        )

        if self.auto: 
            self._print_auto_run_settings(num_trials, minibatch, valset)

        if minibatch and self.minibatch_size > len(valset):
            raise ValueError(f"Minibatch size cannot exceed the size of the valset. Valset size: {len(valset)}.")
        
        # # Estimate LM calls and get user confirmation
        if self.requires_permission_to_run:
            if not self._get_user_confirmation(
                student,
                num_trials,
                minibatch,
                self.minibatch_size,
                self.minibatch_full_eval_steps,
                valset,
                self.program_aware_proposer,
            ):
                logger.info("Compilation aborted by the user.")
                return student  # Return the original student program
        
        program = student.deepcopy()

        # check the evaluator (If None, will construct a default evaluator using the `evaluate` method in the benchmark) and wrap it with runtime checks
        evaluator = self._validate_evaluator(evaluator=self.evaluator, benchmark=dataset, metric_name=metric_name)
        self.metric = evaluator.metric

        # Step 1: Bootstrap few-shot examples 
        demo_candidates = self._bootstrap_fewshot_examples(program, trainset, seed, teacher=None)

        # Step 2: Propose instruction candidates 
        with suppress_cost_logging():
            instruction_candidates = self._propose_instructions(
                program,
                trainset,
                demo_candidates,
                self.view_data_batch_size,
                self.program_aware_proposer,
                self.data_aware_proposer,
                self.tip_aware_proposer,
                self.fewshot_aware_proposer,
            )

        # Step 3: Find optimal prompt parameters 
        with suppress_cost_logging():
            best_program = self._optimize_prompt_parameters(
                program,
                instruction_candidates,
                demo_candidates,
                evaluator,
                valset,
                num_trials,
                minibatch,
                self.minibatch_size,
                self.minibatch_full_eval_steps,
                seed,
            )

        if self.save_path:
            os.makedirs(self.save_path, exist_ok=True)
            self.best_program_path = os.path.join(self.save_path, "best_program.json")
            best_program.save(self.best_program_path)
        
        # reset the self.model. After optimization, the model will be reset to the original state.
        # This is necessary to avoid the model being modified by the optimization process. 
        # Use self.restore_best_program() to restore the best program. 
        self.model.reset()

    def restore_best_program(self):
        # todo: implement this
        pass 

    def _get_input_keys(self, dataset: Benchmark) -> Optional[List[str]]:

        input_keys = None
        if hasattr(dataset, "get_input_keys"):
            candidate_input_keys = dataset.get_input_keys()
            if isinstance(candidate_input_keys, (list, tuple)) and all(isinstance(key, str) for key in candidate_input_keys):
                input_keys = candidate_input_keys
        return input_keys

    def _set_and_validate_datasets(self, dataset: Benchmark):

        trainset = dataset.get_train_data() 
        if not trainset:
            raise ValueError("No training data found in the dataset. Please set `_train_data` in the benchmark.")
        if trainset and not isinstance(trainset[0], (dict, dspy.Example)):
            raise ValueError("Training set in the benchmark must be a list of dictionaries or dspy.Example objects.")
        
        valset = dataset.get_dev_data() 
        if not valset:
            if len(trainset) < 2: 
                raise ValueError("Training set in the benchmark must have at least 2 examples if no validation set is provided.")
            valset_size = min(1000, max(1, int(len(trainset) * 0.80))) 
            cutoff = len(trainset) - valset_size
            valset = trainset[cutoff:]
            trainset = trainset[:cutoff] 
        else: 
            if len(valset) < 1: 
                raise ValueError("Validation set in the benchmark must have at least 1 example.")
        
        # convert the trainset and valset to a list of dspy Example
        input_keys = self._get_input_keys(dataset)
        if input_keys is None:
            logger.warning("`get_input_keys` is not implemented in the benchmark. Will use all keys as input keys. This may cause unexpected behavior if the program does not use all the keys.")
            input_keys = trainset[0].keys()
        
        dspy_trainset = self._convert_benchmark_data_to_dspy_examples(trainset, input_keys)
        dspy_valset = self._convert_benchmark_data_to_dspy_examples(valset, input_keys)
        
        return dspy_trainset, dspy_valset
    
    def _convert_benchmark_data_to_dspy_examples(self, data: List[dict], input_keys: List[str]) -> List[dspy.Example]:

        """
        Convert the benchmark data to a list of dspy Example. This is required since the evaluator accepts a list of dspy Example. 
        """
        dspy_examples = [
            example.with_inputs(*input_keys)
            if isinstance(example, dspy.Example) else dspy.Example(**example).with_inputs(*input_keys)
            for example in data
        ]

        return dspy_examples

    def _bootstrap_fewshot_examples(self, program: Any, trainset: List, seed: int, teacher: Any) -> Optional[List]:

        logger.info("==> STEP 1: BOOTSTRAP FEWSHOT EXAMPLES <==")
        if self.max_bootstrapped_demos > 0:
            logger.info(
                "These will be used as few-shot example candidates for our program and for creating instructions.\n"
            )
        else:
            logger.info("These will be used for informing instruction proposal.\n")

        logger.info(f"Bootstrapping N={self.num_fewshot_candidates} sets of demonstrations...")

        zeroshot = self.max_bootstrapped_demos == 0 and self.max_labeled_demos == 0

        try:
            with suppress_logger_info():
                demo_candidates = create_n_fewshot_demo_sets(
                    student=program,
                    num_candidate_sets=self.num_fewshot_candidates,
                    trainset=trainset,
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
        logger.info("==> STEP 2: PROPOSE INSTRUCTION CANDIDATES <==")
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

        logger.info(f"Proposing N={self.num_instruct_candidates} instructions...")
        instruction_candidates = proposer.propose_instructions_for_program(
            trainset=trainset,
            program=program,
            demo_candidates=demo_candidates,
            N=self.num_instruct_candidates,
            T=self.init_temperature,
            trial_logs={},
        )

        for i, pred in enumerate(program.predicts):
            logger.info(f"Proposed Instructions for Predictor {i}:\n")
            instruction_candidates[i][0] = get_signature(pred).instructions
            for j, instruction in enumerate(instruction_candidates[i]):
                logger.info(f"{j}: {instruction}\n")
            logger.info("\n")

        return instruction_candidates
    
    def _optimize_prompt_parameters(
        self,
        program: Any,
        instruction_candidates: Dict[int, List[str]],
        demo_candidates: Optional[List],
        evaluator: Callable, 
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

        # default_score = eval_candidate_program(
        #     len(valset), valset, program, evaluator, self.rng,
        # )
        default_score = self.evaluate(
            evalset=valset, 
            program=program, 
            evaluator=evaluator, 
            batch_size=len(valset)
        )
        logger.info(f"Default program score: {default_score}\n")

        trial_logs = {}
        trial_logs[1] = {}
        trial_logs[1]["full_eval_program_path"] = save_candidate_program(program, self.save_path, -1)
        trial_logs[1]["full_eval_score"] = default_score
        trial_logs[1]["total_eval_calls_so_far"] = len(valset)
        trial_logs[1]["full_eval_program"] = program.deepcopy()

        # Initialize optimization variables
        best_score = default_score
        best_program = program.deepcopy()
        total_eval_calls = len(valset)
        score_data = [{"score": best_score, "program": program.deepcopy(), "full_eval": True}]
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
            candidate_program = program.deepcopy()

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
                print_full_program(candidate_program)

            # Evaluate the candidate program (on minibatch if minibatch=True)
            batch_size = minibatch_size if minibatch else len(valset)
            # score = eval_candidate_program(batch_size, valset, candidate_program, evaluator, self.rng)
            score = self.evaluate(
                evalset=valset, 
                program=candidate_program,
                evaluator=evaluator, 
                batch_size=batch_size
            )
            total_eval_calls += batch_size

            # Update best score and program
            if not minibatch and score > best_score:
                best_score = score
                best_program = candidate_program.deepcopy()
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
                    evaluator,
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

        default_params = {f"{i}_predictor_instruction": 0 for i in range(len(program.predicts))}
        if demo_candidates:
            default_params.update({f"{i}_predictor_demos": 0 for i in range(len(program.predicts))})

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

        for i, predictor in enumerate(candidate_program.predictors()):
            # Select instruction
            instruction_idx = trial.suggest_categorical(
                f"{i}_predictor_instruction", range(len(instruction_candidates[i]))
            )
            selected_instruction = instruction_candidates[i][instruction_idx]
            # updated_signature = get_signature(predictor).with_instructions(selected_instruction)
            # set_signature(predictor, updated_signature)
            predictor.signature.instructions = selected_instruction 
            trial_logs[trial_num][f"{i}_predictor_instruction"] = instruction_idx
            chosen_params.append(f"Predictor {i}: Instruction {instruction_idx}")
            raw_chosen_params[f"{i}_predictor_instruction"] = instruction_idx
            # Select demos if available
            if demo_candidates:
                demos_idx = trial.suggest_categorical(f"{i}_predictor_demos", range(len(demo_candidates[i])))
                predictor.demos = demo_candidates[i][demos_idx]
                trial_logs[trial_num][f"{i}_predictor_demos"] = demos_idx
                chosen_params.append(f"Predictor {i}: Few-Shot Set {demos_idx}")
                raw_chosen_params[f"{i}_predictor_demos"] = instruction_idx

        return chosen_params, raw_chosen_params
    
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
        trial_logs[trial_num]["mb_program_path"] = save_candidate_program(candidate_program, self.save_path, trial_num=trial_num, note="mb")
        trial_logs[trial_num]["mb_score"] = score
        trial_logs[trial_num]["total_eval_calls_so_far"] = total_eval_calls
        trial_logs[trial_num]["mb_program"] = candidate_program.deepcopy()

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
        trial_logs[trial_num]["full_eval_program_path"] = save_candidate_program(
            candidate_program, self.save_path, trial_num
        )
        trial_logs[trial_num]["full_eval_score"] = score
        trial_logs[trial_num]["total_eval_calls_so_far"] = total_eval_calls
        trial_logs[trial_num]["full_eval_program"] = candidate_program.deepcopy()

        logger.info(f"Score: {score} with parameters {chosen_params}.")
        full_eval_scores = ", ".join([f"{s['score']}" for s in score_data if s["full_eval"]])
        logger.info(f"Scores so far: {'[' + full_eval_scores + ']'}")
        logger.info(f"Best score so far: {best_score}")
        logger.info(f"{'=' * len(f'===== Trial {trial.number + 1} / {num_trials} =====')}\n\n")

    def _perform_full_evaluation(
        self,
        trial_num: int,
        adjusted_num_trials: int,
        param_score_dict: Dict,
        fully_evaled_param_combos: Dict,
        evaluator: Callable, 
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
        # full_eval_score = eval_candidate_program(len(valset), valset, highest_mean_program, evaluator, self.rng)
        full_eval_score = self.evaluate(
            evalset=valset, 
            program=highest_mean_program, 
            evaluator=evaluator, 
            batch_size=len(valset)
        )
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
        trial_logs[trial_num + 1]["full_eval_program_path"] = save_candidate_program(
            program=highest_mean_program,
            log_dir=self.save_path,
            trial_num=trial_num + 1,
            note="full_eval",
        )
        trial_logs[trial_num + 1]["full_eval_program"] = highest_mean_program
        trial_logs[trial_num + 1]["full_eval_score"] = full_eval_score

        # Update best score and program if necessary
        if full_eval_score > best_score:
            logger.info(f"{GREEN}New best full eval score!{ENDC} Score: {full_eval_score}")
            best_score = full_eval_score
            best_program = highest_mean_program.deepcopy()
        full_eval_scores = ", ".join([f"{s['score']}" for s in score_data if s["full_eval"]])
        trajectory = "[" + full_eval_scores + "]"
        logger.info(f"Full eval scores so far: {trajectory}")
        logger.info(f"Best full score so far: {best_score}")
        logger.info(len(f"===== Full Eval {len(fully_evaled_param_combos) + 1} =====") * "=")
        logger.info("\n")

        return best_score, best_program, total_eval_calls

    def evaluate(
        self, 
        evalset: Optional[List[dspy.Example]] = None, 
        dataset: Optional[Benchmark] = None, 
        eval_mode: Optional[str] = "dev", 
        program: Optional[PromptTuningModule] = None, 
        evaluator: Optional[Callable] = None, 
        indices: Optional[List[int]] = None, 
        sample_k: Optional[int] = None, 
        batch_size: Optional[int] = None, # if provided, sample `batch_size` examples from the evalset
        **kwargs
    ):
        # if program is not provided, use the model as the program
        if program is None:
            program = self.model
        
        if evaluator is None:
            evaluator = self._validate_evaluator(evaluator=self.evaluator, benchmark=dataset, metric_name=self.metric_name)

        # if evalset is not provided, use the dataset to get the evalset
        if evalset is None:
            assert dataset is not None, "Either `evalset` or `dataset` must be provided."
            data_map = {"train": dataset.get_train_data, "dev": dataset.get_dev_data, "test": dataset.get_test_data} 
            evaldata = data_map[eval_mode](indices=indices, sample_k=sample_k)
            if not evaldata:
                logger.warning(f"No data found for {eval_mode} set. Return 0.0.")
                return 0.0 
            input_keys = self._get_input_keys(dataset=dataset)
            if not input_keys:
                input_keys = evaldata[0].keys()
            evalset = self._convert_benchmark_data_to_dspy_examples(evaldata, input_keys)
        
        batch_size = batch_size or len(evalset)

        score_list = [] 
        for _ in range(self.eval_rounds):
            score = eval_candidate_program(
                batch_size=batch_size, 
                evalset=evalset, 
                candidate_program=program, 
                evaluator=evaluator, 
                rng=self.rng
            )
            score_list.append(score)
        
        return sum(score_list) / len(score_list)


def eval_candidate_program(
    batch_size: int, 
    evalset: list, 
    candidate_program: Any, 
    evaluator: Callable, 
    rng = None,  
    return_all_scores: bool = False, 
) -> Union[float, Tuple[float, List[float]]]:

    try:
        if batch_size >= len(evalset):
            return evaluator(
                program=candidate_program, 
                evalset=evalset, 
                return_all_scores=return_all_scores, 
            ) 
        else:
            return evaluator(
                program=candidate_program, 
                evalset=create_minibatch(evalset, batch_size, rng), 
                return_all_scores=return_all_scores, 
            )
    
    except Exception as e:
        logger.error(f"An exception occurred during evaluation: {str(e)}", exc_info=True)
        if return_all_scores:
            return 0.0, [0.0] * len(evalset)
        return 0.0


class WorkFlowGraphProgram:

    def __init__(
        self, 
        graph: WorkFlowGraph, 
        agent_manager: AgentManager,
        executor_llm: BaseLLM, 
        collate_func: Optional[Callable] = None, 
        output_postprocess_func: Optional[Callable] = None, 
    ):
        self.graph = graph 
        self.agent_manager = agent_manager
        self.executor_llm = executor_llm
        self.collate_func = collate_func or (lambda x: x)
        self.output_postprocess_func = output_postprocess_func or (lambda x: x)

    def __call__(self, **input_data):

        new_config = deepcopy(self.graph.get_config())
        new_graph: WorkFlowGraph = WorkFlowGraph.from_dict(new_config)
        new_graph.reset_graph() 

        # execute the graph with WorkFlow 
        use_teacher = dspy.settings.get("use_teacher", False)
        if use_teacher:
            # use teacher model to execute the graph, used for optimization
            new_graph, new_agent_manager = self.inject_teacher_settings(new_graph, self.agent_manager)
            workflow = WorkFlow(llm=self.executor_llm, graph=new_graph, agent_manager=new_agent_manager)
        else:
            # use the original executor llm to execute the graph
            workflow = WorkFlow(llm=self.executor_llm, graph=new_graph, agent_manager=self.agent_manager)
        output: str = workflow.execute(inputs=self.collate_func(input_data))
        output = self.output_postprocess_func(output)

        # extract all the input and output data from the workflow execution
        all_execution_data = workflow.environment.execution_data
        all_input_output_keys = self._extract_input_output_keys(new_graph)
        execution_data = {k: v for k, v in all_execution_data.items() if k in all_input_output_keys}

        return output, execution_data 
    
    def inject_teacher_settings(self, graph: WorkFlowGraph, agent_manager: AgentManager):
        """
        Inject the teacher settings into the graph and agent manager.
        """
        # dspy.settings.lm is configured in MiproOptimizer, which is a MiproLMWrapper instance
        optimizer_llm_config = dspy.settings.lm.model.config.to_dict()
        for node in graph.nodes:
            for agent in node.agents:
                agent["llm_config"] = optimizer_llm_config
        
        # create a new agent manager with the teacher settings
        new_agent_manager = agent_manager.copy()
        new_agent_manager.clear_agents()
        new_agent_manager.add_agents_from_workflow(graph, llm_config=optimizer_llm_config)
        return graph, new_agent_manager
    
    def _extract_input_output_keys(self, graph: WorkFlowGraph) -> Set[str]:
        """
        Extract all the input and output keys from the graph.
        """
        all_input_output_keys = set()
        for node in graph.nodes:
            for inp in node.inputs:
                all_input_output_keys.add(inp.name)
            for out in node.outputs:
                all_input_output_keys.add(out.name)
            for agent in node.agents:
                for agent_inp in agent.get("inputs", []):
                    agent_inp_name = agent_inp.get("name", None)
                    if agent_inp_name:
                        all_input_output_keys.add(agent_inp_name)
                for agent_out in agent.get("outputs", []):
                    agent_out_name = agent_out.get("name", None)
                    if agent_out_name:
                        all_input_output_keys.add(agent_out_name)

        return all_input_output_keys

    def save(self, path: str):
        self.graph.save_module(path=path)

    def load(self, path: str):
        return WorkFlowGraph.from_file(path=path) 

class MiproEvaluatorWrapper(MiproEvaluator): 

    def __init__(
        self, 
        evaluator: Evaluator, 
        benchmark: Benchmark, 
        metric_name: str = None,
        return_all_scores: bool = False, 
        return_outputs: bool = False, 
    ):

        self.evaluator = evaluator 
        self.benchmark = benchmark 
        self.metric_name = metric_name 
        self.return_all_scores = return_all_scores
        self.return_outputs = return_outputs

    def metric(self, example: dspy.Example, prediction: Any, *args, **kwargs):
        return super().metric(example, prediction, *args, **kwargs)

    def __call__(self, program: PromptTuningModule, evalset: List[dspy.Example], **kwargs) -> float: 

        # sync the candidate prompts and instructions to the workflow graph
        program.sync_predict_inputs_to_program()

        return_all_scores = kwargs.get("return_all_scores", None) or self.return_all_scores
        return_outputs = kwargs.get("return_outputs", None) or self.return_outputs
        
        if isinstance(program, PromptTuningModule):
            graph = program.program.graph 
        elif isinstance(program, WorkFlowGraphProgram):
            graph = program.graph
        else:
            raise ValueError(f"Invalid program type: {type(program)}. Must be PromptTuningModule or WorkFlowGraphProgram.")
        
        self.evaluator._evaluation_records.clear()

        # update agents
        self.evaluator.agent_manager.update_agents_from_workflow(workflow_graph=graph, llm_config=self.evaluator.llm.config, **kwargs)
        
        if isinstance(self.benchmark.get_train_data()[0], dspy.Example):
            data = evalset
        else:
            data = [example.toDict() for example in evalset]
        
        with suppress_logger_info():
            metrics = self.evaluator._evaluate_graph(
                graph=graph, data=data, benchmark=self.benchmark, verbose=True, **kwargs
            )
        if isinstance(metrics, dict):
            score = self._extract_score_from_dict(metrics)
        else:
            score = metrics 

        # extract all outputs and predictions 
        all_scores, all_predictions = [], []
        for example in data:
            example_id = self.benchmark.get_id(example=example)
            evaluation_record = self.evaluator._evaluation_records.get(example_id, None)
            if evaluation_record is None:
                all_scores.append(0.0)
                all_predictions.append(None)
            else:
                example_metrics = evaluation_record["metrics"]
                example_score = self._extract_score_from_dict(example_metrics) if isinstance(example_metrics, dict) else example_metrics
                all_scores.append(example_score)
                all_predictions.append(evaluation_record["prediction"])

        if return_all_scores and return_outputs:
            return score, all_predictions, all_scores
        if return_all_scores:
            return score, all_scores 
        if return_outputs:
            return score, all_predictions

        return score 


class WorkFlowMiproOptimizer(MiproOptimizer):

    def __init__(
        self, 
        graph: WorkFlowGraph,
        evaluator: Evaluator, 
        optimizer_llm: Optional[BaseLLM] = None, 
        **kwargs, 
    ):
        """
        MiproOptimizer tailored for workflow graphs. 

        Args:
            graph (WorkFlowGraph): the workflow graph to optimize.
            evaluator (Evaluator): the evaluator to use for the optimization.
            optimizer_llm (BaseLLM): the LLM to use for the optimization. If None, will use the LLM model in the evaluator.
            **kwargs: additional keyword arguments to pass to the MiproOptimizer. Available options:
                - metric_threshold (Optional[int]): threshold for the metric score. If provided, only examples with scores above this threshold will be used as demonstrations.
                - max_bootstrapped_demos (int): maximum number of bootstrapped demonstrations to use. Defaults to 4.
                - max_labeled_demos (int): maximum number of labeled demonstrations to use. Defaults to 4.
                - auto (Optional[Literal["light", "medium", "heavy"]]): automatic configuration mode. If set, will override num_candidates and max_steps. 
                    "light": n=6, val_size=100; "medium": n=12, val_size=300; "heavy": n=18, val_size=1000. Defaults to "medium".
                - max_steps (int): maximum number of optimization steps. Required if auto is None.
                - num_candidates (Optional[int]): number of candidates to generate for each optimization step. Required if auto is None.
                - num_threads (Optional[int]): number of threads to use for parallel evaluation. If None, will use single thread.
                - max_errors (int): maximum number of errors allowed during evaluation before stopping. Defaults to 10.
                - seed (int): random seed for reproducibility. Defaults to 9.
                - init_temperature (float): initial temperature for instruction generation. Defaults to 0.5.
                - track_stats (bool): whether to track optimization statistics. Defaults to True.
                - save_path (Optional[str]): path to save optimization results. If None, results will not be saved.
                - minibatch (bool): whether to use minibatch evaluation during optimization. Defaults to True.
                - minibatch_size (int): size of minibatch for evaluation. Defaults to 35.
                - minibatch_full_eval_steps (int): number of minibatch steps between full evaluations. Defaults to 5.
                - program_aware_proposer (bool): whether to use program-aware instruction proposer. Defaults to True.
                - data_aware_proposer (bool): whether to use data-aware instruction proposer. Defaults to True.
                - view_data_batch_size (int): batch size for viewing data during instruction proposal. Defaults to 10.
                - tip_aware_proposer (bool): whether to use tip-aware instruction proposer. Defaults to True.
                - fewshot_aware_proposer (bool): whether to use fewshot-aware instruction proposer. Defaults to True.
                - requires_permission_to_run (bool): whether to require user permission before running optimization. Defaults to False.
                - provide_traceback (Optional[bool]): whether to provide traceback for evaluation errors. If None, will use default setting.
        """

        # check if the graph is compatible with the WorkFlowMipro optimizer.
        graph = self._validate_graph_compatibility(graph=graph)
        
        # convert the workflow graph to a callable program  
        workflow_graph_program = WorkFlowGraphProgram(
            graph=graph, 
            agent_manager=evaluator.agent_manager, 
            executor_llm=evaluator.llm, 
            collate_func=evaluator.collate_func, 
            output_postprocess_func=evaluator.output_postprocess_func, 
        )

        # register optimizable parameters 
        registry = self._register_optimizable_parameters(program=workflow_graph_program)

        super().__init__(
            registry=registry, 
            program=workflow_graph_program, 
            optimizer_llm=optimizer_llm or evaluator.llm, 
            evaluator=evaluator,
            **kwargs
        )

    def _validate_graph_compatibility(self, graph: WorkFlowGraph):
        """
        Check if the graph is compatible with the WorkFlowMipro optimizer. Also, convert the MiproPromptTemplate data to MiproPromptTemplate instances. 
        """
        for node in graph.nodes:
            if len(node.agents) > 1:
                raise ValueError("WorkFlowMiproOptimizer only supports workflows where every node only has a single agent.")
            else:
                agent = node.agents[0]
                if not isinstance(agent, dict):
                    raise ValueError(f"Unsupported agent type {type(agent)}. Expected 'dict'.")
                else:
                    if "actions" in agent:
                        # Agent has actions in its dict
                        # All agents have a `ContextExtraction` action, filter it out
                        non_ContextExtraction_actions = [
                            action for action in agent["actions"] if action["class_name"] != "ContextExtraction"
                        ]
                        if len(non_ContextExtraction_actions) > 1:
                            raise ValueError(f"WorkFlowMiproOptimizer only supports workflows where every agent only has a single action. {agent['name']} has {len(non_ContextExtraction_actions)} actions.")
                        # if "prompt_template" not in non_ContextExtraction_actions[0]:
                        if non_ContextExtraction_actions[0].get("prompt_template", None) is None:
                            # raise ValueError(f"Please provide a PromptTemplate for {agent['name']}.")
                            logger.warning(f"{agent['name']} does not have a MiproPromptTemplate, its prompt will not be optimized.")
                        else:
                            prompt_template = non_ContextExtraction_actions[0]["prompt_template"]
                            if isinstance(prompt_template, dict):
                                prompt_template = PromptTemplate.from_dict(prompt_template)
                            if isinstance(prompt_template, MiproPromptTemplate):
                                # in some cases, the raw `prompt_template` can be a dict, convert it to a MiproPromptTemplate instance
                                non_ContextExtraction_actions[0]["prompt_template"] = prompt_template 
                            else:
                                logger.warning(f"{agent['name']} has a non-MiproPromptTemplate, its prompt will not be optimized. You should use `MiproPromptTemplate` to define the optimizable prompt.")
                    else:
                        # CustomizeAgent does not have actions in its dict
                        # if "prompt_template" not in agent:
                            # raise ValueError(f"Please provide a PromptTemplate for {agent['name']}.")
                        if agent.get("prompt_template", None) is None:
                            logger.warning(f"{agent['name']} does not have a MiproPromptTemplate, its prompt will not be optimized.")
                        else:
                            prompt_template = agent["prompt_template"]
                            if isinstance(prompt_template, dict):
                                prompt_template = PromptTemplate.from_dict(prompt_template)
                            if isinstance(prompt_template, MiproPromptTemplate):
                                # in some cases, the raw `prompt_template` can be a dict, convert it to a MiproPromptTemplate instance
                                agent["prompt_template"] = prompt_template
                            else:
                                logger.warning(f"{agent['name']} has a non-MiproPromptTemplate, its prompt will not be optimized. You should use `MiproPromptTemplate` to define the optimizable prompt.")
        return graph 

    def _validate_evaluator(self, evaluator: Callable = None, benchmark: Benchmark = None, metric_name: str = None) -> Callable:
        if evaluator and isinstance(evaluator, Evaluator):
            # if evaluator is an Evaluator, convert it to a MiproEvaluatorWrapper
            evaluator = MiproEvaluatorWrapper(evaluator=evaluator, benchmark=benchmark, metric_name=metric_name)
        return super()._validate_evaluator(evaluator, benchmark, metric_name)
    
    def _register_optimizable_parameters(self, program: WorkFlowGraphProgram):

        registry = MiproRegistry()
        workflow_graph = program.graph
        for i, node in enumerate(workflow_graph.nodes):
            agent = node.agents[0] # only one agent per node is allowed
            if "actions" in agent:
                # Agent Instance 
                for j, action in enumerate(agent["actions"]):
                    # only one action is allowed per agent. Use for loop because all the agent 
                    # will have the ContextExtraction action, which will be filtered out by default 
                    # since it does not have a prompt template. 
                    action_prompt_template = action.get("prompt_template", None)
                    if action_prompt_template and isinstance(action_prompt_template, MiproPromptTemplate):
                        registry.track(
                            root_or_obj=program, 
                            path_or_attr=f"graph.nodes[{i}].agents[0]['actions'][{j}]['prompt_template']",
                            name=f"{agent['name']}_prompt_template",
                            input_names=node.get_input_names(),
                            output_names=node.get_output_names()
                        )
            else:
                # CustomizeAgent Instance
                prompt_template = agent.get("prompt_template", None)
                if prompt_template and isinstance(prompt_template, MiproPromptTemplate):
                    registry.track(
                        root_or_obj=program, 
                        path_or_attr=f"graph.nodes[{i}].agents[0]['prompt_template']",
                        name=f"{agent['name']}_prompt_template",
                        input_names=node.get_input_names(),
                        output_names=node.get_output_names()
                    )
        
        if not registry.fields:
            raise ValueError(
                "No optimizable parameters found in the workflow graph. "
                "Please check if the workflow graph is compatible with the WorkFlowMiproOptimizer. "
                "You should use `MiproPromptTemplate` to define the optimizable prompt."
            )
        
        return registry
    
        