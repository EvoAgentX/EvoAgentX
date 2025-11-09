# -----------------------------------------------------------------------------
# This file re-implements algorithms from the DSPy project:
#   Repo: https://github.com/stanfordnlp/dspy
#   Paper: "GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning"
#   Authors: Agrawal et al.
#
# Re-implementation integrated into EvoAgentX with permission from the authors.
# All mistakes or modifications are our own.
#
# Code of Conduct: This project follows the Microsoft Open Source Code of Conduct.
#   https://opensource.microsoft.com/codeofconduct/
# -----------------------------------------------------------------------------

import os
import tqdm
import inspect
import threading
from copy import deepcopy
from functools import wraps
from typing import Optional, Callable, Literal, List, Any, Dict, Union, Tuple, Set

import dspy
from dspy import GEPA
from dspy.clients import LM, Provider
from dspy.utils.callback import BaseCallback

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
from ..prompts.template import PromptTemplate, MiproPromptTemplate, GEPAPromptTemplate
from ..utils.mipro_utils.module_utils import PromptTuningModule

# Constants
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

OPTIMIZABLE_PROMPT_TYPES = (MiproPromptTemplate, GEPAPromptTemplate)
GEPA_SUPPORTED_INIT_PARAMS = set(inspect.signature(GEPA.__init__).parameters.keys())
GEPA_SUPPORTED_INIT_PARAMS.discard("self")


class GEPALMWrapper(LM):
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
        return GEPALMWrapper(new_model, **new_kwargs)

    def generate(self, *args, **kwargs):
        # to be compatible with BaseLLM.generate()
        return self.model.generate(*args, **kwargs)

    async def async_generate(self, *args, **kwargs):
        # to be compatible with BaseLLM.async_generate()
        return await self.model.async_generate(*args, **kwargs)


class GEPAEvaluator:
    """
    Base evaluator for GEPA with minimal fallback feedback.

    IMPORTANT: This base class provides only MINIMAL feedback. For best results,
    users MUST create custom evaluators with domain-specific feedback that explains:
    - What went wrong (specific errors)
    - Why it went wrong (reasoning flaws)
    - How to fix it (reference solutions, recommendations)

    Example of custom evaluator:
        >>> class MyCustomEvaluator:
        >>>     def metric(self, example, prediction, trace=None, pred_name=None, pred_trace=None):
        >>>         score = compute_score(example, prediction)
        >>>         feedback = generate_detailed_feedback(example, prediction, score)
        >>>         return dspy.Prediction(score=score, feedback=feedback)
    """

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
        generate_feedback: bool = True,
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
        self.generate_feedback = generate_feedback
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

    def _generate_feedback(
        self,
        example: dspy.Example,
        prediction: Any,
        score: float,
        error: Optional[Exception] = None
    ) -> str:
        """
        Minimal fallback feedback generator.

        WARNING: This provides only basic feedback. Users should provide custom
        evaluators with domain-specific feedback for better optimization results.
        """
        if not self.generate_feedback:
            return ""

        if error is not None:
            return f"Error during evaluation: {str(error)}"

        if score >= 0.9:
            return f"Correct (score: {score:.2f})"
        else:
            return f"Incorrect (score: {score:.2f}). Expected better performance."

    def metric(
        self,
        example: dspy.Example,
        prediction: Any,
        trace=None,
        pred_name=None,
        pred_trace=None
    ) -> Union[float, dspy.Prediction]:
        """
        GEPA-compatible metric that returns score with optional feedback.

        Args:
            example: Gold/reference example
            prediction: Model prediction
            trace: Full program execution trace (optional)
            pred_name: Name of the predictor being optimized (optional)
            pred_trace: Trace of specific predictor (optional)

        Returns:
            Union[float, dspy.Prediction]: Score or Prediction with score and feedback
        """
        error = None
        score = None

        try:
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
                    label=self.benchmark.get_label(example.toDict())  # convert the dspy.Example to a dict
                )
            else:
                raise ValueError(f"Unsupported example type in `{type(self.benchmark)}`! Expected `dspy.Example` or `dict`, got {type(self.benchmark.get_train_data()[0])}")

            if isinstance(score, dict):
                score = self._extract_score_from_dict(score)

        except Exception as e:
            logger.error(f"Error evaluating example: {e}")
            error = e
            score = self.failure_score

        # Generate feedback for GEPA
        if self.generate_feedback:
            feedback = self._generate_feedback(example, prediction, score, error)
            return dspy.Prediction(score=score, feedback=feedback)
        else:
            return score

    def __call__(
        self,
        program: Callable,
        evalset: List[Any],
        **kwargs,
    ) -> float:
        """
        Evaluate a program on a dataset.

        Args:
            program: The DSPy program to evaluate
            evalset: List of examples to evaluate on
            **kwargs: Additional arguments

        Returns:
            float: Average score across all examples
        """
        return_all_scores = kwargs.get("return_all_scores", None) or self.return_all_scores
        return_outputs = kwargs.get("return_outputs", None) or self.return_outputs

        tqdm.tqdm._instances.clear()

        # Get the current suppress_cost_logs value
        from ..core.callbacks import suppress_cost_logs
        current_suppress_cost = suppress_cost_logs.get()

        if self.num_threads and self.num_threads > 1:
            from dspy.utils.parallelizer import ParallelExecutor
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
                    prediction = program(**example.inputs())
                    score = self.metric(example, prediction)

                    # Extract score if returned as Prediction
                    if isinstance(score, dspy.Prediction):
                        score = score.score
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


class GEPAOptimizer(BaseOptimizer):
    """
    GEPA optimizer for EvoAgentX that uses reflective prompt evolution.

    Key differences from MIPRO:
    - No few-shot bootstrapping (instruction-only optimization)
    - Requires reflection_llm (strong model for analyzing failures)
    - Uses Pareto frontier for multi-objective optimization
    - Generates textual feedback for reflective learning
    - Supports variant merging
    """

    def __init__(
        self,
        registry: ParamRegistry,
        program: Callable,
        optimizer_llm: BaseLLM,
        reflection_llm: BaseLLM,
        evaluator: Optional[Callable] = None,
        eval_rounds: Optional[int] = 1,
        auto: Optional[Literal["light", "medium", "heavy"]] = "medium",
        max_full_evals: Optional[int] = None,
        max_metric_calls: Optional[int] = None,
        reflection_minibatch_size: int = 16,
        candidate_selection_strategy: Literal['pareto', 'current_best'] = 'pareto',
        component_selector: str = 'round_robin',
        use_merge: bool = True,
        max_merge_invocations: Optional[int] = 5,
        skip_perfect_score: bool = True,
        failure_score: float = 0.0,
        perfect_score: float = 1.0,
        num_threads: Optional[int] = None,
        max_errors: int = 10,
        seed: int = 42,
        track_stats: bool = True,
        save_path: Optional[str] = None,
        requires_permission_to_run: bool = False,
        provide_traceback: Optional[bool] = None,
        verbose: bool = False,
        **kwargs
    ):
        """
        Initialize GEPA optimizer for prompt evolution.

        Args:
            registry (ParamRegistry): Parameter registry containing optimizable fields
            program (Callable): Program to optimize with save/load methods
            optimizer_llm (BaseLLM): LLM for task execution (student model)
            reflection_llm (BaseLLM): Strong LLM for reflective prompt evolution
            evaluator (Optional[Callable]): Evaluator with metric(example, pred, trace, pred_name, pred_trace) method
            eval_rounds (Optional[int]): Number of evaluation rounds. Defaults to 1.
            auto (Optional[Literal["light", "medium", "heavy"]]): Auto configuration mode
            max_full_evals (Optional[int]): Maximum number of full evaluations
            max_metric_calls (Optional[int]): Maximum total metric invocations
            reflection_minibatch_size (int): Number of examples per reflection step
            candidate_selection_strategy (str): 'pareto' or 'current_best'
            component_selector (str): 'round_robin' or 'all'
            use_merge (bool): Enable variant merging
            max_merge_invocations (Optional[int]): Maximum merge operations
            skip_perfect_score (bool): Skip examples with perfect scores
            failure_score (float): Score for failed evaluations
            perfect_score (float): Score threshold for perfect performance
            num_threads (Optional[int]): Number of threads for parallel evaluation
            max_errors (int): Maximum errors before stopping
            seed (int): Random seed for reproducibility
            track_stats (bool): Track optimization statistics
            save_path (Optional[str]): Path to save results
            requires_permission_to_run (bool): Require user confirmation
            provide_traceback (Optional[bool]): Provide error tracebacks
            verbose (bool): Verbose logging
            **kwargs: Additional arguments

        Raises:
            TypeError: If program is not callable
            ValueError: If program lacks required methods or invalid parameters
        """
        # Initialize base optimizer
        BaseOptimizer.__init__(self, registry=registry, program=program, evaluator=evaluator)

        # Validate program
        self._validate_program(program=program)

        # Convert to DSPy-compatible module
        self.model = self._convert_to_dspy_module(registry, program)

        # Configure LLMs
        self.optimizer_llm = GEPALMWrapper(optimizer_llm)
        self.reflection_llm = GEPALMWrapper(reflection_llm)
        dspy.configure(lm=self.optimizer_llm)
        # Enable detailed tracebacks in dspy parallel workers to surface errors
        try:
            dspy.settings.configure(provide_traceback=True)
        except Exception:
            pass
        self.task_model = dspy.settings.lm

        # Validate budget configuration
        budget_params = [auto, max_full_evals, max_metric_calls]
        budget_count = sum(p is not None for p in budget_params)
        if budget_count == 0:
            raise ValueError("Must provide exactly one of: auto, max_full_evals, or max_metric_calls")
        if budget_count > 1:
            raise ValueError("Cannot specify multiple budget parameters. Choose one of: auto, max_full_evals, or max_metric_calls")

        # Validate 'auto' parameter
        if auto is not None:
            allowed_modes = {"light", "medium", "heavy"}
            if auto not in allowed_modes:
                raise ValueError(f"Invalid value for auto: {auto}. Must be one of {allowed_modes}.")

        self.auto = auto
        self.max_full_evals = max_full_evals
        self.max_metric_calls = max_metric_calls
        self.reflection_minibatch_size = reflection_minibatch_size
        self.candidate_selection_strategy = candidate_selection_strategy
        self.component_selector = component_selector
        self.use_merge = use_merge
        self.max_merge_invocations = max_merge_invocations
        self.skip_perfect_score = skip_perfect_score
        self.failure_score = failure_score
        self.perfect_score = perfect_score

        self.num_threads = num_threads
        self.max_errors = max_errors
        self.seed = seed
        self.track_stats = track_stats
        self.eval_rounds = eval_rounds
        self.save_path = save_path
        self.requires_permission_to_run = requires_permission_to_run
        self.provide_traceback = provide_traceback
        self.verbose = verbose
        self.kwargs = kwargs

        self.metric_name = None
        self.metric = None

    def _validate_program(self, program: Callable):
        """Validate that the program meets the required interface."""
        if not callable(program):
            raise TypeError("program must be callable")

        # Check if program has save method
        if not hasattr(program, 'save'):
            logger.warning("program does not have a `save(path=...)` method, will use the default save method in dspy.Module")
        else:
            save_sig = inspect.signature(program.save)
            save_params = list(save_sig.parameters.keys())
            if 'path' not in save_params:
                raise ValueError("program.save must accept a 'path' parameter")

        # Check if program has load method
        if not hasattr(program, 'load'):
            logger.warning("program does not have a `load(path=...)` method, will use the default load method in dspy.Module")
        else:
            load_sig = inspect.signature(program.load)
            load_params = list(load_sig.parameters.keys())
            if 'path' not in load_params:
                raise ValueError("program.load must accept a 'path' parameter")

    def _validate_evaluator(
        self,
        evaluator: Callable = None,
        benchmark: Benchmark = None,
        metric_name: Optional[str] = None
    ) -> Callable:
        """
        Validate and wrap evaluator with GEPA-compatible metric.

        GEPA metrics must accept (gold, pred, trace, pred_name, pred_trace) parameters
        and should return score with feedback for best results.
        """
        if evaluator is None:
            if not hasattr(benchmark, "evaluate"):
                raise ValueError("`evaluator` is not provided and the benchmark does not have an `evaluate` method.")
            logger.info("`evaluator` is not provided. Will construct a default evaluator using the `evaluate` method in the benchmark.")
            evaluator = GEPAEvaluator(
                benchmark=benchmark,
                num_threads=self.num_threads,
                max_errors=self.max_errors,
                display_progress=True,
                provide_traceback=self.provide_traceback,
                metric_name=metric_name,
                generate_feedback=True,  # Enable feedback for GEPA
                **self.kwargs
            )

        if not callable(evaluator):
            raise TypeError("evaluator must be callable")

        # Check if evaluator has __call__ method
        sig = inspect.signature(evaluator.__call__ if hasattr(evaluator, '__call__') else evaluator)
        params = list(sig.parameters.keys())

        if len(params) < 2:
            raise ValueError("evaluator must accept at least two parameters (program and evalset)")

        # Check if the evaluator has a `metric` method
        if not hasattr(evaluator, 'metric'):
            raise ValueError("evaluator must have a `metric(example, prediction, trace=None, pred_name=None, pred_trace=None)` method")

        metric_sig = inspect.signature(evaluator.metric)
        metric_params = list(metric_sig.parameters.keys())

        if len(metric_params) < 2:
            raise ValueError("evaluator.metric must accept at least two parameters (example and prediction)")

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
            class WrappedEvaluator:
                def __init__(self, func):
                    self._func = func

                def __call__(self, *args, **kwargs):
                    return wrapped_evaluator(*args, **kwargs)

            return WrappedEvaluator(evaluator)

        return evaluator

    def _convert_to_dspy_module(self, registry: ParamRegistry, program: Callable):
        """Convert program to DSPy module."""
        if isinstance(program, dspy.Module):
            return program

        program = PromptTuningModule.from_registry(
            program=program,
            registry=registry,
        )

        return program

    def _get_input_keys(self, dataset: Benchmark) -> Optional[List[str]]:
        """Extract input keys from dataset."""
        input_keys = None
        if hasattr(dataset, "get_input_keys"):
            candidate_input_keys = dataset.get_input_keys()
            if isinstance(candidate_input_keys, (list, tuple)) and all(isinstance(key, str) for key in candidate_input_keys):
                input_keys = candidate_input_keys
        return input_keys

    def _set_and_validate_datasets(self, dataset: Benchmark) -> Tuple[List[dspy.Example], List[dspy.Example]]:
        """Validate and convert datasets to DSPy Examples."""
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

        # Convert to DSPy Examples
        input_keys = self._get_input_keys(dataset)
        if input_keys is None:
            logger.warning("`get_input_keys` is not implemented in the benchmark. Will use all keys as input keys.")
            input_keys = list(trainset[0].keys()) if isinstance(trainset[0], dict) else list(trainset[0].inputs().keys())

        dspy_trainset = self._convert_benchmark_data_to_dspy_examples(trainset, input_keys)
        dspy_valset = self._convert_benchmark_data_to_dspy_examples(valset, input_keys)

        return dspy_trainset, dspy_valset

    def _convert_benchmark_data_to_dspy_examples(self, data: List[dict], input_keys: List[str]) -> List[dspy.Example]:
        """Convert benchmark data to DSPy Examples."""
        dspy_examples = [
            example.with_inputs(*input_keys)
            if isinstance(example, dspy.Example) else dspy.Example(**example).with_inputs(*input_keys)
            for example in data
        ]
        return dspy_examples

    def optimize(self, dataset: Benchmark, metric_name: Optional[str] = None, **kwargs):
        """
        Optimize the program using GEPA's reflective prompt evolution.

        Args:
            dataset (Benchmark): Benchmark containing training and validation data
            metric_name (Optional[str]): Metric name for optimization (if evaluator not provided)
            **kwargs: Additional arguments
        """
        self.metric_name = metric_name

        # Set training & validation sets
        trainset, valset = self._set_and_validate_datasets(dataset=dataset)

        # Apply auto settings if specified
        if self.auto:
            settings = AUTO_RUN_SETTINGS[self.auto]
            if len(valset) > settings["val_size"]:
                logger.info(f"Auto mode '{self.auto}': Limiting valset to {settings['val_size']} examples")
                valset = valset[:settings["val_size"]]

        # Check user permission if required
        if self.requires_permission_to_run:
            if self.max_full_evals:
                estimated_calls = self.max_full_evals * len(valset)
            elif self.max_metric_calls:
                estimated_calls = self.max_metric_calls
            elif self.auto:
                estimated_calls = AUTO_RUN_SETTINGS[self.auto]["n"] * len(valset)
            else:
                estimated_calls = None

            if estimated_calls is not None:
                logger.info(f"GEPA will perform approximately {estimated_calls} metric calls")
            else:
                logger.info("GEPA metric call budget could not be estimated.")
            response = input("Continue with optimization? (y/n): ")
            if response.lower() != 'y':
                logger.info("Optimization aborted by user.")
                return self.model

        program = self.model.deepcopy()

        # Validate evaluator and wrap it
        evaluator = self._validate_evaluator(evaluator=self.evaluator, benchmark=dataset, metric_name=metric_name)
        self.metric = evaluator.metric

        logger.info("==> STARTING GEPA OPTIMIZATION <==")
        logger.info(f"Training set size: {len(trainset)}")
        logger.info(f"Validation set size: {len(valset)}")
        logger.info(f"Reflection minibatch size: {self.reflection_minibatch_size}")
        logger.info(f"Candidate selection strategy: {self.candidate_selection_strategy}")
        logger.info(f"Component selector: {self.component_selector}")
        logger.info(f"Use merge: {self.use_merge}\n")

        # Initialize GEPA optimizer
        with suppress_cost_logging():
            gepa_kwargs = {
                "metric": self.metric,
                "auto": self.auto,
                "max_full_evals": self.max_full_evals,
                "max_metric_calls": self.max_metric_calls,
                "reflection_lm": self.reflection_llm,
                "reflection_minibatch_size": self.reflection_minibatch_size,
                "candidate_selection_strategy": self.candidate_selection_strategy,
                "component_selector": self.component_selector,
                "use_merge": self.use_merge,
                "max_merge_invocations": self.max_merge_invocations,
                "skip_perfect_score": self.skip_perfect_score,
                "failure_score": self.failure_score,
                "perfect_score": self.perfect_score,
                "num_threads": self.num_threads,
                "seed": self.seed,
                "track_stats": self.track_stats,
                "log_dir": self.save_path,
            }

            unsupported_kwargs = [key for key in list(gepa_kwargs.keys()) if key not in GEPA_SUPPORTED_INIT_PARAMS]
            for key in unsupported_kwargs:
                if key == "component_selector" and self.component_selector != "round_robin":
                    logger.warning("Installed dspy.GEPA does not support `component_selector`. Ignoring this setting.")
                gepa_kwargs.pop(key)

            gepa = GEPA(**gepa_kwargs)

            # Run optimization
            logger.info("Running GEPA.compile()...")
            best_program = gepa.compile(
                student=program,
                trainset=trainset,
                valset=valset,
            )

        # Save results
        if self.save_path:
            os.makedirs(self.save_path, exist_ok=True)
            self.best_program_path = os.path.join(self.save_path, "best_program.json")
            best_program.save(self.best_program_path)
            logger.info(f"Saved best program to {self.best_program_path}")

        # Log results
        if self.track_stats and hasattr(best_program, 'detailed_results'):
            results = best_program.detailed_results
            logger.info(f"\n{GREEN}==> OPTIMIZATION COMPLETE <=={ENDC}")
            logger.info(f"Best score: {results.best_candidate}")
            logger.info(f"Total metric calls: {results.total_metric_calls}")
            logger.info(f"Candidates evaluated: {len(results.candidates)}")

        # Reset model to original state
        self.model.reset()

    def restore_best_program(self):
        """Restore the best optimized program."""
        if hasattr(self, 'best_program_path') and os.path.exists(self.best_program_path):
            self.model.load(self.best_program_path)
            logger.info(f"Restored best program from {self.best_program_path}")
        else:
            logger.warning("No saved program found to restore")

    def evaluate(
        self,
        evalset: Optional[List[dspy.Example]] = None,
        dataset: Optional[Benchmark] = None,
        eval_mode: Optional[str] = "dev",
        program: Optional[PromptTuningModule] = None,
        evaluator: Optional[Callable] = None,
        indices: Optional[List[int]] = None,
        sample_k: Optional[int] = None,
        **kwargs
    ):
        """
        Evaluate the program on a dataset.

        Args:
            evalset: List of examples (if not provided, will use dataset)
            dataset: Benchmark to evaluate on
            eval_mode: 'train', 'dev', or 'test'
            program: Program to evaluate (defaults to self.model)
            evaluator: Evaluator to use (defaults to self.evaluator)
            indices: Specific indices to evaluate
            sample_k: Number of samples to evaluate
            **kwargs: Additional arguments

        Returns:
            float: Average score
        """
        if program is None:
            program = self.model

        if evaluator is None:
            evaluator = self._validate_evaluator(evaluator=self.evaluator, benchmark=dataset, metric_name=self.metric_name)

        # Get evalset from dataset if not provided
        if evalset is None:
            if dataset is None:
                raise ValueError("Either `evalset` or `dataset` must be provided.")
            data_map = {"train": dataset.get_train_data, "dev": dataset.get_dev_data, "test": dataset.get_test_data}
            evaldata = data_map[eval_mode](indices=indices, sample_k=sample_k)
            if not evaldata:
                logger.warning(f"No data found for {eval_mode} set. Return 0.0.")
                return 0.0
            input_keys = self._get_input_keys(dataset=dataset)
            if not input_keys:
                input_keys = list(evaldata[0].keys()) if isinstance(evaldata[0], dict) else list(evaldata[0].inputs().keys())
            evalset = self._convert_benchmark_data_to_dspy_examples(evaldata, input_keys)

        # Run evaluation rounds
        score_list = []
        for _ in range(self.eval_rounds):
            score = evaluator(program=program, evalset=evalset)
            score_list.append(score)

        return sum(score_list) / len(score_list)


# Reuse WorkFlowGraphProgram from mipro_optimizer
from .mipro_optimizer import WorkFlowGraphProgram, MiproEvaluatorWrapper


class GEPAEvaluatorWrapper(MiproEvaluatorWrapper):
    """
    Wrapper for Evaluator class that provides minimal GEPA-compatible feedback.

    NOTE: This wrapper provides basic fallback feedback. For best results,
    override the metric in your optimizer with a custom feedback generator.
    """

    def __init__(
        self,
        evaluator: Evaluator,
        benchmark: Benchmark,
        metric_name: str = None,
        return_all_scores: bool = False,
        return_outputs: bool = False,
        generate_feedback: bool = True,
    ):
        super().__init__(
            evaluator=evaluator,
            benchmark=benchmark,
            metric_name=metric_name,
            return_all_scores=return_all_scores,
            return_outputs=return_outputs,
        )
        self.generate_feedback = generate_feedback
        # Propagate executor settings and enable tracebacks for clearer errors
        try:
            self.num_threads = getattr(evaluator, "num_workers", None)
            self.display_progress = True
            self.max_errors = 10
            self.provide_traceback = True
        except Exception:
            pass

    def metric(self, example: dspy.Example, prediction: Any, trace=None, pred_name=None, pred_trace=None) -> Union[float, dspy.Prediction]:
        """
        GEPA-compatible metric with feedback generation.
        """
        # Normalize Prediction objects to raw values for the underlying benchmark metric
        try:
            import dspy as _dspy
            # Unwrap dspy.Prediction
            if isinstance(prediction, _dspy.Prediction):
                if hasattr(prediction, "solution"):
                    prediction = prediction.solution
                elif hasattr(prediction, "answer"):
                    prediction = prediction.answer
                else:
                    items = [(k, v) for k, v in prediction._store.items() if not str(k).startswith("dspy_")]
                    prediction = items[0][1] if items else str(prediction)

            # If a dict slips through, try common keys or first value
            if isinstance(prediction, dict):
                if "solution" in prediction:
                    prediction = prediction["solution"]
                elif "answer" in prediction:
                    prediction = prediction["answer"]
                elif len(prediction) == 1:
                    prediction = next(iter(prediction.values()))
                else:
                    prediction = str(prediction)

            # If a list/tuple, take the first element
            if isinstance(prediction, (list, tuple)):
                prediction = prediction[0] if prediction else ""

            # Ensure it's a string for downstream regex extractors
            if not isinstance(prediction, str):
                prediction = str(prediction)
        except Exception:
            # Best-effort fallback
            prediction = str(prediction)

        # Get base score from parent
        score = super().metric(example, prediction)

        if not self.generate_feedback:
            return score

        # Generate feedback
        feedback_parts = []
        if score >= 0.9:
            feedback_parts.append("Excellent! The solution is correct.")
        elif score >= 0.5:
            feedback_parts.append(f"Partial success (score: {score:.2f}). Some elements are correct.")
        else:
            feedback_parts.append(f"Needs improvement (score: {score:.2f}). The answer is incorrect.")

        # Add reference information
        if hasattr(example, 'answer'):
            feedback_parts.append(f"Expected: {example.answer}")
        if hasattr(prediction, 'answer'):
            feedback_parts.append(f"Got: {prediction.answer}")

        feedback = "\n".join(feedback_parts)
        return dspy.Prediction(score=score, feedback=feedback)


class WorkFlowGEPAOptimizer(GEPAOptimizer):
    """
    GEPA optimizer tailored for workflow graphs.
    """

    def __init__(
        self,
        graph: WorkFlowGraph,
        evaluator: Evaluator,
        optimizer_llm: Optional[BaseLLM] = None,
        reflection_llm: Optional[BaseLLM] = None,
        **kwargs,
    ):
        """
        Initialize workflow-specific GEPA optimizer.

        Args:
            graph: WorkFlowGraph to optimize
            evaluator: Evaluator for the workflow
            optimizer_llm: LLM for task execution (defaults to evaluator.llm)
            reflection_llm: Strong LLM for reflection (defaults to evaluator.llm)
            **kwargs: Additional arguments (see GEPAOptimizer)
        """
        # Validate graph compatibility
        graph = self._validate_graph_compatibility(graph=graph)

        # Convert workflow graph to callable program
        workflow_graph_program = WorkFlowGraphProgram(
            graph=graph,
            agent_manager=evaluator.agent_manager,
            executor_llm=evaluator.llm,
            collate_func=evaluator.collate_func,
            output_postprocess_func=evaluator.output_postprocess_func,
        )

        # Register optimizable parameters
        registry = self._register_optimizable_parameters(program=workflow_graph_program)

        # Initialize parent
        super().__init__(
            registry=registry,
            program=workflow_graph_program,
            optimizer_llm=optimizer_llm or evaluator.llm,
            reflection_llm=reflection_llm or evaluator.llm,
            evaluator=evaluator,
            **kwargs
        )

    def _validate_graph_compatibility(self, graph: WorkFlowGraph):
        """
        Validate graph compatibility with GEPA optimizer.
        Convert MiproPromptTemplate data to instances.
        """
        for node in graph.nodes:
            if len(node.agents) > 1:
                raise ValueError("WorkFlowGEPAOptimizer only supports workflows where every node has a single agent.")

            agent = node.agents[0]
            if not isinstance(agent, dict):
                raise ValueError(f"Unsupported agent type {type(agent)}. Expected 'dict'.")

            if "actions" in agent:
                # Agent with actions
                non_context_actions = [
                    action for action in agent["actions"]
                    if action["class_name"] != "ContextExtraction"
                ]
                if len(non_context_actions) > 1:
                    raise ValueError(f"WorkFlowGEPAOptimizer only supports workflows where every agent has a single action. {agent['name']} has {len(non_context_actions)} actions.")

                if non_context_actions and non_context_actions[0].get("prompt_template"):
                    prompt_template = non_context_actions[0]["prompt_template"]
                    if isinstance(prompt_template, dict):
                        prompt_template = PromptTemplate.from_dict(prompt_template)
                    if isinstance(prompt_template, OPTIMIZABLE_PROMPT_TYPES):
                        non_context_actions[0]["prompt_template"] = prompt_template
                    else:
                        logger.warning(
                            f"{agent['name']} has a non-optimizable prompt template. "
                            "Use MiproPromptTemplate or GEPAPromptTemplate for optimization."
                        )
            else:
                # CustomizeAgent
                if agent.get("prompt_template"):
                    prompt_template = agent["prompt_template"]
                    if isinstance(prompt_template, dict):
                        prompt_template = PromptTemplate.from_dict(prompt_template)
                    if isinstance(prompt_template, OPTIMIZABLE_PROMPT_TYPES):
                        agent["prompt_template"] = prompt_template
                    else:
                        logger.warning(
                            f"{agent['name']} has a non-optimizable prompt template. "
                            "Use MiproPromptTemplate or GEPAPromptTemplate for optimization."
                        )

        return graph

    def _validate_evaluator(self, evaluator: Callable = None, benchmark: Benchmark = None, metric_name: str = None) -> Callable:
        """Convert Evaluator to GEPA-compatible evaluator."""
        if evaluator and isinstance(evaluator, Evaluator):
            evaluator = GEPAEvaluatorWrapper(
                evaluator=evaluator,
                benchmark=benchmark,
                metric_name=metric_name,
                generate_feedback=True,
            )
        return super()._validate_evaluator(evaluator, benchmark, metric_name)

    def _register_optimizable_parameters(self, program: WorkFlowGraphProgram):
        """Register optimizable parameters from workflow graph."""
        registry = MiproRegistry()
        workflow_graph = program.graph

        for i, node in enumerate(workflow_graph.nodes):
            agent = node.agents[0]  # Only one agent per node

            if "actions" in agent:
                # Agent Instance
                for j, action in enumerate(agent["actions"]):
                    action_prompt_template = action.get("prompt_template")
                    if action_prompt_template and isinstance(action_prompt_template, OPTIMIZABLE_PROMPT_TYPES):
                        registry.track(
                            root_or_obj=program,
                            path_or_attr=f"graph.nodes[{i}].agents[0]['actions'][{j}]['prompt_template']",
                            name=f"{agent['name']}_prompt_template",
                            input_names=node.get_input_names(),
                            output_names=node.get_output_names()
                        )
            else:
                # CustomizeAgent
                prompt_template = agent.get("prompt_template")
                if prompt_template and isinstance(prompt_template, OPTIMIZABLE_PROMPT_TYPES):
                    registry.track(
                        root_or_obj=program,
                        path_or_attr=f"graph.nodes[{i}].agents[0]['prompt_template']",
                        name=f"{agent['name']}_prompt_template",
                        input_names=node.get_input_names(),
                        output_names=node.get_output_names()
                    )

        if not registry.fields:
            raise ValueError(
                "No optimizable parameters found in workflow graph. "
                "Use MiproPromptTemplate or GEPAPromptTemplate to define optimizable prompts."
            )

        return registry
