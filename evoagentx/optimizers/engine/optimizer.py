import abc
import asyncio
from concurrent.futures import ThreadPoolExecutor
from pydantic import Field
from typing import Any, Awaitable, Callable, FrozenSet, Optional, List, Dict, Literal

from ...core.module import BaseModule
from .base import OptimizationUnitType, OptimizationUnit, UnitChange
from .adapter import SnapShot, ProgramAdapter, ApplyResult


class OptimizationProposal(BaseModule):
    source_snapshot_id: str = Field(description="The snapshot_id of the base snapshot that the proposed changes will be applied on")
    changes: List[UnitChange] = Field(description="List of proposed changes to apply on the base snapshot for the next trial")


class TrialRecord(BaseModule):
    """Data model for recording the results of a single optimization trial."""
    trial_id: int = Field(description="Unique identifier for the trial")
    changes: List[UnitChange] = Field(description="List of changes applied in this trial")
    source_snapshot_id: str = Field(description="The snapshot_id of the base snapshot that the proposed changes were applied on in this trial")
    status: Literal["completed", "failed"] = Field(description="Status of the trial")

    snapshot_id: Optional[str] = Field(default=None, description="The snapshot_id of the program state after applying the changes in this trial")
    metrics: Optional[Dict[str, Any]] = Field(default=None, description="Evaluation metrics collected for this trial")
    error: Optional[str] = Field(default=None, description="Error message if the trial failed")


class Objective:
    """
    Defines the optimization objective: which metric to track and in which direction.

    The default implementation compares a single named scalar metric. Subclass and
    override `is_better` for multi-objective or custom comparison logic.
    """

    def __init__(
        self,
        metric: str = "score",
        direction: Literal["maximize", "minimize"] = "maximize",
    ) -> None:
        self.metric = metric
        self.direction = direction

    def is_better(self, metrics_a: Dict[str, Any], metrics_b: Dict[str, Any]) -> bool:
        """
        Return True if metrics_a represents a strictly better outcome than metrics_b.

        The default implementation compares `self.metric` as a scalar according to
        `self.direction`. Returns False if either dict is missing the key.

        Override for custom comparison logic (e.g. multi-objective, weighted scores).

        Args:
            metrics_a: Metrics dict from the candidate trial.
            metrics_b: Metrics dict from the reference / current-best trial.

        Returns:
            True if metrics_a is strictly better than metrics_b, False otherwise.
        """
        pass

    def select_best_snapshot_id(self, trial_records: List["TrialRecord"]) -> Optional[str]:
        """
        Return the snapshot_id of the best trial among all completed records.

        Iterates over completed (status == "completed") trial records and returns
        the snapshot_id of the one with the best metrics according to `is_better`.

        Args:
            trial_records: All trial records accumulated in the optimization run.

        Returns:
            The snapshot_id of the best completed trial, or None if none exist.
        """
        pass


class OptimizationRunState(BaseModule):
    
    snapshots: List[SnapShot] = Field(default_factory=list, description="List of snapshots for all completed trials in the optimization run")
    trial_records: List[TrialRecord] = Field(default_factory=list, description="List of records for all completed trials in the optimization run")
    best_snapshot_id: Optional[str] = Field(default=None, description="The snapshot_id of the best snapshot found so far in the optimization run, according to the defined objective and evaluation metrics")
    
    current_step: int = Field(default=0, description="Current trial step (0-indexed, not started) in the optimization run")
    save_dir: Optional[str] = Field(default="./", description="Directory to save optimization state and results")

    def get_snapshot(self, snapshot_id: str) -> Optional[SnapShot]:
        """
        Look up a snapshot by its snapshot_id.

        Args:
            snapshot_id: The unique identifier of the target snapshot.

        Returns:
            The matching SnapShot, or None if no snapshot with that ID exists.
        """
        pass

    @staticmethod
    def load_state(path: str) -> "OptimizationRunState":
        """
        Deserialize and return an OptimizationRunState from a previously saved file.

        Used by `Optimizer.optimize` when `resume_from` is provided, so that
        already-completed trials are skipped and the run continues from where it
        left off.

        Args:
            path: File path (or directory) written by a prior `save_state` call.

        Returns:
            The restored OptimizationRunState, including all past snapshots,
            trial records, best_snapshot_id, and current_step.
        """
        pass

    def save_state(self) -> str:
        """
        Serialize the current OptimizationRunState to disk under `self.save_dir`.

        Called after every completed trial so the run can be resumed if interrupted.

        Returns:
            The file path where the state was written.
        """
        pass



class TrialRuntime:

    def __init__(self, adapter: ProgramAdapter) -> None:
        self.adapter = adapter
    
    def run(
        self,
        state: OptimizationRunState,
        proposal: OptimizationProposal,
        evaluate_fn: Callable[[ProgramAdapter], Dict[str, Any]],
        objective: Objective,
        **kwargs
    ) -> OptimizationRunState:
        """
        Execute a single optimization trial and return the updated run state.

        Pipeline:
        1. Fetch `base_snapshot = state.get_snapshot(proposal.source_snapshot_id)`, then
           call `result = self.adapter.apply(base_snapshot, proposal.changes)`.
           `self.adapter` is never modified and remains the stable base for all trials.
           If `result.ok` is False, record the trial as status="failed" with `result.error`
           and return without updating best_snapshot_id.
        2. Use `result.snapshot` (produced by `merge_changes` inside `apply`) directly as
           the canonical snapshot for this trial. Do NOT call `new_adapter.take_snapshot()`
           — the snapshot from `apply` is guaranteed consistent with the adapter's state.
        3. Run evaluation: call `evaluate_fn(result.adapter)` to obtain a metrics dict.
           On exception, record the trial as status="failed" with the error message
           and append the failed record to state without updating best_snapshot_id.
        4. Build a TrialRecord (trial_id = len(state.trial_records)) with the
           snapshot_id, metrics, and status.
        5. Append `result.snapshot` and the TrialRecord to `state`.
        6. Update `state.best_snapshot_id` using `objective.select_best_snapshot_id`.

        Note: `state.current_step` is NOT incremented here; the caller (`Optimizer.optimize`)
        is responsible for that.

        Args:
            state: The current optimization run state.
            proposal: The proposed changeset and its source snapshot.
            evaluate_fn: Callable that executes the adapted program and returns metrics.
            objective: Used to update `best_snapshot_id` after the trial completes.
            **kwargs: Forwarded to `base_adapter.apply`.

        Returns:
            Updated OptimizationRunState with the new snapshot, trial record, and
            (if the trial succeeded) a potentially updated best_snapshot_id.
        """
        # when calling self.adapter.apply, pass the copied base_snapshot, since the it might be modified in-place by the apply method
        pass

    async def async_run(
        self,
        state: OptimizationRunState,
        proposal: OptimizationProposal,
        evaluate_fn: Callable[[ProgramAdapter], Awaitable[Dict[str, Any]]],
        objective: Objective,
        **kwargs
    ) -> OptimizationRunState:
        """
        Async variant of `run`. Identical pipeline but `evaluate_fn` is awaited.

        Args:
            state: The current optimization run state.
            proposal: The proposed changeset and its source snapshot.
            evaluate_fn: Async callable that executes the adapted program and returns metrics.
            objective: Used to update `best_snapshot_id` after the trial completes.
            **kwargs: Forwarded to `self.adapter.apply`.

        Returns:
            Updated OptimizationRunState (see `run` for full pipeline description).
        """
        pass

    def run_batch(
        self,
        state: OptimizationRunState,
        proposals: List[OptimizationProposal],
        evaluate_fn: Callable[[ProgramAdapter], Dict[str, Any]],
        objective: Objective,
        execution_mode: Literal["sequential", "concurrent"] = "sequential",
        **kwargs
    ) -> OptimizationRunState:
        """
        Run a batch of proposals and return the updated state.

        Sequential mode evaluates proposals one after another; each trial sees the
        state already updated by its predecessors.

        Concurrent mode submits all trials to a ThreadPoolExecutor so that
        I/O-bound evaluate calls (e.g. LLM API requests) run in parallel. All
        trials start from the same read-only snapshot of the current state;
        results are merged in submission order after all threads finish. Only use
        when `adapter.supports_concurrent_execution` is True.

        Args:
            state: Current optimization run state.
            proposals: List of proposals to evaluate.
            evaluate_fn: Sync callable returning evaluation metrics.
            objective: Used to update `best_snapshot_id` after the batch.
            execution_mode: "sequential" (default) or "concurrent".
            **kwargs: Forwarded to each `run` call.

        Returns:
            Updated OptimizationRunState after all proposals have been evaluated.
        """
        if execution_mode == "sequential":
            for proposal in proposals:
                state = self.run(state, proposal, evaluate_fn, objective, **kwargs)
            return state

        # concurrent: each thread runs one trial independently from the current state.
        # Results are merged in submission order after all threads finish.
        # TODO: implement _single_trial_outcome that returns (SnapShot, TrialRecord)
        #   without modifying state, then submit all via ThreadPoolExecutor and merge.
        pass

    async def async_run_batch(
        self,
        state: OptimizationRunState,
        proposals: List[OptimizationProposal],
        evaluate_fn: Callable[[ProgramAdapter], Awaitable[Dict[str, Any]]],
        objective: Objective,
        execution_mode: Literal["sequential", "concurrent"] = "sequential",
        **kwargs
    ) -> OptimizationRunState:
        """
        Run a batch of proposals and return the updated state.

        Sequential mode evaluates proposals one after another, so each trial sees the
        state already updated by its predecessors.

        Concurrent mode launches all trials from the *same* read-only snapshot of the
        current state via `asyncio.gather`, collects (snapshot, record) pairs without
        touching shared state, then merges all results in order. This avoids race
        conditions at the cost of trials not seeing each other's intermediate results.
        Only use when `adapter.supports_concurrent_execution` is True.

        Args:
            state: Current optimization run state.
            proposals: List of proposals to evaluate.
            evaluate_fn: Async callable returning evaluation metrics.
            objective: Used to update `best_snapshot_id` after the batch.
            execution_mode: "sequential" (default) or "concurrent".
            **kwargs: Forwarded to each `async_run` call.

        Returns:
            Updated OptimizationRunState after all proposals have been evaluated.
        """
        if execution_mode == "sequential":
            for proposal in proposals:
                state = await self.async_run(state, proposal, evaluate_fn, objective, **kwargs)
            return state

        # concurrent: each trial reads the current state independently, results merged after
        # TODO: implement _async_single_outcome helper that runs one trial without
        #   modifying state, then gather all, append snapshots/records, and update
        #   best_snapshot_id once at the end.
        pass


class Optimizer(abc.ABC):
    """
    Abstract base class for all optimizers. 
    """

    def __init__(
        self,
        adapter: ProgramAdapter,
        **kwargs         
    ) -> None:
        
        self.adapter = adapter
        if not isinstance(adapter, ProgramAdapter):
            raise TypeError(f"Expected adapter to be an instance of ProgramAdapter, got {type(adapter)}")
        
        self.runtime = TrialRuntime(adapter)
        self.kwargs = kwargs

        # TODO: check the compatibility between adapter's units and optimizer's supported unit types, warn or raise error if incompatible

    def _init_run_state(
        self,
        save_dir: Optional[str],
        resume_from: Optional[str],
    ) -> OptimizationRunState:
        """Load or create OptimizationRunState and seed the initial snapshot."""
        if resume_from:
            state = OptimizationRunState.load_state(resume_from)
        else:
            state = OptimizationRunState(save_dir=save_dir or "./")

        # Seed the initial snapshot so propose() always has a valid source_snapshot_id,
        # including the first trial of a fresh run.
        if not state.snapshots:
            initial_snapshot = self.adapter.take_snapshot()
            state.snapshots.append(initial_snapshot)
            state.best_snapshot_id = initial_snapshot.snapshot_id

        return state

    def optimize(
        self,
        evaluate_fn: Callable[[ProgramAdapter], Dict[str, Any]],
        objective: Optional[Objective] = None,
        max_trials: Optional[int] = 3,
        save_dir: Optional[str] = None,
        resume_from: Optional[str] = None,
        **kwargs
    ) -> ProgramAdapter:

        if not isinstance(max_trials, int) or max_trials <= 0:
            raise ValueError(f"`max_trials` must be a positive integer, got {max_trials}")

        if objective is None:
            objective = Objective()

        state = self._init_run_state(save_dir, resume_from)

        while state.current_step < max_trials:
            remaining = max_trials - state.current_step
            proposals = self.batch_propose(state, objective, **kwargs)[:remaining]
            state = self.runtime.run_batch(
                state=state,
                proposals=proposals,
                evaluate_fn=evaluate_fn,
                objective=objective,
                **kwargs
            )
            state.current_step += len(proposals)
            state.save_state()

        best_snapshot = state.get_snapshot(state.best_snapshot_id)
        return self.adapter.load_snapshot(best_snapshot)

    async def async_optimize(
        self,
        evaluate_fn: Callable[[ProgramAdapter], Awaitable[Dict[str, Any]]],
        objective: Optional[Objective] = None,
        max_trials: Optional[int] = 3,
        save_dir: Optional[str] = None,
        resume_from: Optional[str] = None,
        execution_mode: Optional[Literal["sequential", "concurrent"]] = None,
        **kwargs
    ) -> ProgramAdapter:
        """
        Async variant of `optimize`. Supports both sequential and concurrent trial execution.

        Args:
            evaluate_fn: Async callable `(adapter) -> Awaitable[Dict[str, Any]]` returning metrics.
            objective: Optimization objective. Defaults to maximizing "score".
            max_trials: Total number of individual trials to run.
            save_dir: Directory to write per-trial state checkpoints.
            resume_from: Path to a prior checkpoint to resume from.
            execution_mode: "sequential" or "concurrent". Defaults to "concurrent" when
                            `adapter.supports_concurrent_execution` is True, otherwise "sequential".
            **kwargs: Forwarded to `batch_propose` and `async_run_batch`.

        Returns:
            A new ProgramAdapter reflecting the best configuration found.
        """
        if not isinstance(max_trials, int) or max_trials <= 0:
            raise ValueError(f"`max_trials` must be a positive integer, got {max_trials}")

        if objective is None:
            objective = Objective()

        if execution_mode is None:
            execution_mode = "concurrent" if self.adapter.supports_concurrent_execution else "sequential"

        state = self._init_run_state(save_dir, resume_from)

        while state.current_step < max_trials:
            remaining = max_trials - state.current_step
            proposals = self.batch_propose(state, objective, **kwargs)[:remaining]
            state = await self.runtime.async_run_batch(
                state=state,
                proposals=proposals,
                evaluate_fn=evaluate_fn,
                objective=objective,
                execution_mode=execution_mode,
                **kwargs
            )
            state.current_step += len(proposals)
            state.save_state()

        best_snapshot = state.get_snapshot(state.best_snapshot_id)
        return self.adapter.load_snapshot(best_snapshot)
    
    @property
    @abc.abstractmethod
    def supported_unit_types(self) -> FrozenSet[OptimizationUnitType]:
        """
        Return a set of OptimizationUnitType that this optimizer can handle.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def propose(
        self,
        state: OptimizationRunState,
        objective: Objective,
        **kwargs
    ) -> OptimizationProposal:
        """
        Propose a single changeset for the next trial.

        Select a source snapshot from `state`, propose changes to optimization units,
        and return them as an OptimizationProposal. Called by the default `batch_propose`;
        algorithms that natively generate multiple candidates should override `batch_propose`
        directly instead.

        Args:
            state: Current optimization run state (past snapshots, trial records, best id).
            objective: The objective the proposed changes should aim to improve.
            **kwargs: Algorithm-specific arguments forwarded from `optimize`.

        Returns:
            An OptimizationProposal carrying the source snapshot id and the list of changes.
        """
        raise NotImplementedError

    def batch_propose(
        self,
        state: OptimizationRunState,
        objective: Objective,
        **kwargs
    ) -> List[OptimizationProposal]:
        """
        Propose one or more changesets for the next optimization step.

        Default implementation delegates to `propose()` and returns a single-element list,
        which gives sequential algorithms free compatibility with the batch execution path.

        Override to return multiple proposals when the algorithm naturally generates a
        population of candidates (e.g. evolutionary search, beam search, TPE with parallel
        workers). The caller (`optimize` / `async_optimize`) will trim the list to the
        remaining trial budget before passing it to the runtime.

        Args:
            state: Current optimization run state.
            objective: The objective the proposed changes should aim to improve.
            **kwargs: Forwarded to `propose` (or used directly if overriding).

        Returns:
            A non-empty list of OptimizationProposal objects.
        """
        return [self.propose(state, objective, **kwargs)]
