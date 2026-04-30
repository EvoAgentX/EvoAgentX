import abc
import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from pydantic import Field
from typing import Any, Awaitable, Callable, ClassVar, FrozenSet, Optional, List, Dict, Literal, Tuple

from ...core.module import BaseModule
from .base import OptimizationUnitType, OptimizationProposal, TrialRecord
from .adapter import SnapShot, ProgramAdapter, ApplyResult
from .objective import Objective

BASELINE_TRIAL_ID = 0


def _validate_execution_config(
    execution_mode: Any,
    max_workers: Any,
) -> Tuple[Literal["sequential", "concurrent"], Optional[int]]:
    """Validate and normalize trial execution settings."""
    if execution_mode not in ("sequential", "concurrent"):
        raise ValueError(f"execution_mode must be 'sequential' or 'concurrent', got {execution_mode!r}")

    if max_workers is not None and (
        isinstance(max_workers, bool) or not isinstance(max_workers, int) or max_workers < 1
    ):
        raise ValueError(f"max_workers must be a positive integer or None, got {max_workers!r}")

    if execution_mode == "sequential":
        if max_workers not in (None, 1):
            raise ValueError(
                "max_workers is only valid with execution_mode='concurrent'; "
                "use max_workers=None or 1 for sequential execution."
            )
        return "sequential", None

    if max_workers is None:
        raise ValueError("max_workers must be explicitly provided when execution_mode='concurrent'.")
    if max_workers <= 1:
        raise ValueError("max_workers must be greater than 1 when execution_mode='concurrent'.")

    return "concurrent", max_workers


class OptimizationRunState(BaseModule):

    snapshots: List[SnapShot] = Field(default_factory=list, description="List of snapshots for all completed trials in the optimization run")
    trial_records: List[TrialRecord] = Field(default_factory=list, description="List of records for all completed trials in the optimization run")
    best_snapshot_id: Optional[str] = Field(default=None, description="The snapshot_id of the best snapshot found so far in the optimization run, according to the defined objective and evaluation metrics")
    best_metrics: Optional[Dict[str, Any]] = Field(default=None, description="The evaluation metrics associated with best_snapshot_id; None until the baseline has been evaluated")

    current_step: int = Field(default=0, description="Current trial step (0-indexed, not started) in the optimization run")
    save_dir: Optional[str] = Field(default="./", description="Directory to save optimization state and results")

    def get_snapshot_by_id(self, snapshot_id: str) -> Optional[SnapShot]:
        """
        Look up a snapshot by its snapshot_id.

        Args:
            snapshot_id: The unique identifier of the target snapshot.

        Returns:
            The matching SnapShot, or None if no snapshot with that ID exists.
        """
        for snapshot in self.snapshots:
            if snapshot.snapshot_id == snapshot_id:
                return snapshot
        return None

    def get_trial_record_by_id(self, trial_id: int) -> Optional[TrialRecord]:
        """
        Look up a TrialRecord by its trial_id.

        Args:
            trial_id: The unique identifier of the target trial.

        Returns:
            The matching TrialRecord, or None if no record with that ID exists.
        """
        for record in self.trial_records:
            if record.trial_id == trial_id:
                return record
        return None

    def get_baseline_record(self) -> Optional[TrialRecord]:
        """
        Return the baseline TrialRecord (trial_id == BASELINE_TRIAL_ID), or None if not present.
        """
        for record in self.trial_records:
            if record.trial_id == BASELINE_TRIAL_ID:
                return record
        return None

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
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} does not exist, cannot load optimization state")
        
        if os.path.isdir(path):
            path = os.path.join(path, "optimization_state.json")
        return OptimizationRunState.from_file(path=path)

    def save_state(self) -> str:
        """
        Serialize the current OptimizationRunState to disk under `self.save_dir`.

        Called after every completed trial so the run can be resumed if interrupted.

        Returns:
            The file path where the state was written.
        """
        os.makedirs(self.save_dir, exist_ok=True)
        path = os.path.join(self.save_dir, "optimization_state.json")
        self.save_module(path=path)
        return path


def _sync_best(state: OptimizationRunState, objective: Objective) -> None:
    """Update state.best_snapshot_id and state.best_metrics from objective."""
    best_record = objective.select_best_trial_record(state.trial_records)
    if best_record is not None:
        state.best_snapshot_id = best_record.snapshot_id
        state.best_metrics = best_record.metrics


def _validate_metrics(metrics: Any, trial_id: int) -> None:
    """Raise TypeError if evaluate_fn did not return a dict."""
    if not isinstance(metrics, dict):
        raise TypeError(
            f"evaluate_fn must return a dict, got {type(metrics).__name__!r} "
            f"(trial_id={trial_id})"
        )


def _run_trial(
    adapter: ProgramAdapter,
    base_snapshot: SnapShot,
    proposal: OptimizationProposal,
    trial_id: int,
    evaluate_fn: Callable[[ProgramAdapter], Dict[str, Any]],
    **kwargs,
) -> Tuple[Optional[SnapShot], TrialRecord]:
    """
    Apply a proposal and evaluate it synchronously.

    Returns (snapshot, record): snapshot is None when the trial failed.
    Does not touch shared state — callers are responsible for merging the result.
    """
    if base_snapshot is None:
        return None, TrialRecord(
            trial_id=trial_id,
            changes=proposal.changes,
            source_snapshot_id=proposal.source_snapshot_id,
            status="failed",
            error=f"source_snapshot_id '{proposal.source_snapshot_id}' not found in run state",
        )
    result: ApplyResult = adapter.apply(base_snapshot.model_copy(deep=True), proposal.changes, **kwargs)
    if not result.ok:
        return None, TrialRecord(
            trial_id=trial_id,
            changes=proposal.changes,
            source_snapshot_id=proposal.source_snapshot_id,
            status="failed",
            error=result.error,
        )
    try:
        metrics = evaluate_fn(result.adapter)
        _validate_metrics(metrics, trial_id)
    except Exception as exc:
        return None, TrialRecord(
            trial_id=trial_id,
            changes=proposal.changes,
            source_snapshot_id=proposal.source_snapshot_id,
            status="failed",
            error=str(exc),
        )
    return result.snapshot, TrialRecord(
        trial_id=trial_id,
        changes=proposal.changes,
        source_snapshot_id=proposal.source_snapshot_id,
        status="completed",
        snapshot_id=result.snapshot.snapshot_id,
        metrics=metrics,
    )


async def _async_run_trial(
    adapter: ProgramAdapter,
    base_snapshot: SnapShot,
    proposal: OptimizationProposal,
    trial_id: int,
    evaluate_fn: Callable[[ProgramAdapter], Awaitable[Dict[str, Any]]],
    sem: Optional[asyncio.Semaphore] = None,
    **kwargs,
) -> Tuple[Optional[SnapShot], TrialRecord]:
    """
    Apply a proposal and evaluate it asynchronously.

    `sem` throttles concurrent evaluate calls when set; pass None for uncapped concurrency.
    Returns (snapshot, record): snapshot is None when the trial failed.
    Does not touch shared state — callers are responsible for merging the result.
    """
    if base_snapshot is None:
        return None, TrialRecord(
            trial_id=trial_id,
            changes=proposal.changes,
            source_snapshot_id=proposal.source_snapshot_id,
            status="failed",
            error=f"source_snapshot_id '{proposal.source_snapshot_id}' not found in run state",
        )
    result: ApplyResult = adapter.apply(base_snapshot.model_copy(deep=True), proposal.changes, **kwargs)
    if not result.ok:
        return None, TrialRecord(
            trial_id=trial_id,
            changes=proposal.changes,
            source_snapshot_id=proposal.source_snapshot_id,
            status="failed",
            error=result.error,
        )
    try:
        if sem is not None:
            async with sem:
                metrics = await evaluate_fn(result.adapter)
        else:
            metrics = await evaluate_fn(result.adapter)
        _validate_metrics(metrics, trial_id)
    except Exception as exc:
        return None, TrialRecord(
            trial_id=trial_id,
            changes=proposal.changes,
            source_snapshot_id=proposal.source_snapshot_id,
            status="failed",
            error=str(exc),
        )
    return result.snapshot, TrialRecord(
        trial_id=trial_id,
        changes=proposal.changes,
        source_snapshot_id=proposal.source_snapshot_id,
        status="completed",
        snapshot_id=result.snapshot.snapshot_id,
        metrics=metrics,
    )


def _merge_outcomes(
    state: OptimizationRunState,
    outcomes: List[Tuple[Optional[SnapShot], TrialRecord]],
    objective: Objective,
) -> OptimizationRunState:
    """Append (snapshot, record) pairs to state and sync the best pointer."""
    for snapshot, record in outcomes:
        if snapshot is not None:
            state.snapshots.append(snapshot)
        state.trial_records.append(record)
    _sync_best(state, objective)
    return state


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

        Calls `_run_trial` to apply the proposal and evaluate it, then merges the
        result into `state`. `state.current_step` is NOT incremented here; the
        caller (`Optimizer.optimize`) is responsible for that.

        Args:
            state: The current optimization run state.
            proposal: The proposed changeset and its source snapshot.
            evaluate_fn: Callable that executes the adapted program and returns metrics.
            objective: Used to update `best_snapshot_id` after the trial completes.
            **kwargs: Forwarded to `adapter.apply`.

        Returns:
            Updated OptimizationRunState with the new snapshot, trial record, and
            (if the trial succeeded) a potentially updated best_snapshot_id.
        """
        base_snapshot = state.get_snapshot_by_id(proposal.source_snapshot_id)
        trial_id = len(state.trial_records)
        outcome = _run_trial(self.adapter, base_snapshot, proposal, trial_id, evaluate_fn, **kwargs)
        return _merge_outcomes(state, [outcome], objective)

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
            **kwargs: Forwarded to `adapter.apply`.

        Returns:
            Updated OptimizationRunState (see `run` for full pipeline description).
        """
        base_snapshot = state.get_snapshot_by_id(proposal.source_snapshot_id)
        trial_id = len(state.trial_records)
        outcome = await _async_run_trial(self.adapter, base_snapshot, proposal, trial_id, evaluate_fn, **kwargs)
        return _merge_outcomes(state, [outcome], objective)

    def run_batch(
        self,
        state: OptimizationRunState,
        proposals: List[OptimizationProposal],
        evaluate_fn: Callable[[ProgramAdapter], Dict[str, Any]],
        objective: Objective,
        execution_mode: Literal["sequential", "concurrent"] = "sequential",
        max_workers: Optional[int] = None,
        **kwargs
    ) -> OptimizationRunState:
        """
        Run a batch of proposals and return the updated state.

        Sequential mode evaluates proposals one after another; each trial sees the
        state already updated by its predecessors.

        Concurrent mode submits all trials to a ThreadPoolExecutor so that
        I/O-bound evaluate calls (e.g. LLM API requests) run in parallel. All
        trials start from the same read-only snapshot of the current state;
        results are merged in submission order after all threads finish.

        Args:
            state: Current optimization run state.
            proposals: List of proposals to evaluate.
            evaluate_fn: Sync callable returning evaluation metrics.
            objective: Used to update `best_snapshot_id` after the batch.
            execution_mode: "sequential" (default) or "concurrent".
            max_workers: Maximum number of parallel threads in concurrent mode.
                         Required and must be greater than 1 in concurrent mode.
                         Use None or 1 in sequential mode.
            **kwargs: Forwarded to each trial.

        Returns:
            Updated OptimizationRunState after all proposals have been evaluated.
        """
        execution_mode, max_workers = _validate_execution_config(execution_mode, max_workers)

        if execution_mode == "sequential":
            for proposal in proposals:
                state = self.run(state, proposal, evaluate_fn, objective, **kwargs)
            return state
        if not proposals:
            return state

        # concurrent: each thread runs one trial independently from a frozen copy of
        # the current state; results are merged in submission order after all finish.
        base_trial_count = len(state.trial_records)
        effective_workers = min(max_workers, len(proposals))

        def _single_outcome(idx_proposal: Tuple[int, OptimizationProposal]) -> Tuple[Optional[SnapShot], TrialRecord]:
            idx, proposal = idx_proposal
            base_snapshot = state.get_snapshot_by_id(proposal.source_snapshot_id)
            return _run_trial(self.adapter, base_snapshot, proposal, base_trial_count + idx, evaluate_fn, **kwargs)

        with ThreadPoolExecutor(max_workers=effective_workers) as pool:
            outcomes = list(pool.map(_single_outcome, enumerate(proposals)))

        return _merge_outcomes(state, outcomes, objective)

    async def async_run_batch(
        self,
        state: OptimizationRunState,
        proposals: List[OptimizationProposal],
        evaluate_fn: Callable[[ProgramAdapter], Awaitable[Dict[str, Any]]],
        objective: Objective,
        execution_mode: Literal["sequential", "concurrent"] = "sequential",
        max_workers: Optional[int] = None,
        **kwargs
    ) -> OptimizationRunState:
        """
        Async variant of `run_batch`.

        Sequential mode evaluates proposals one after another, so each trial sees the
        state already updated by its predecessors.

        Concurrent mode launches all trials via `asyncio.gather` from the same read-only
        snapshot of the current state, then merges results in order. `max_workers` caps
        the number of coroutines that may call `evaluate_fn` simultaneously via a semaphore.

        Args:
            state: Current optimization run state.
            proposals: List of proposals to evaluate.
            evaluate_fn: Async callable returning evaluation metrics.
            objective: Used to update `best_snapshot_id` after the batch.
            execution_mode: "sequential" (default) or "concurrent".
            max_workers: Maximum number of coroutines running concurrently in concurrent mode.
                         Required and must be greater than 1 in concurrent mode.
                         Use None or 1 in sequential mode.
            **kwargs: Forwarded to each trial.

        Returns:
            Updated OptimizationRunState after all proposals have been evaluated.
        """
        execution_mode, max_workers = _validate_execution_config(execution_mode, max_workers)

        if execution_mode == "sequential":
            for proposal in proposals:
                state = await self.async_run(state, proposal, evaluate_fn, objective, **kwargs)
            return state
        if not proposals:
            return state

        # concurrent: all trials launched together; semaphore throttles evaluate calls.
        base_trial_count = len(state.trial_records)
        effective_workers = min(max_workers, len(proposals))
        sem = asyncio.Semaphore(effective_workers)

        outcomes = await asyncio.gather(*[
            _async_run_trial(
                self.adapter,
                state.get_snapshot_by_id(proposal.source_snapshot_id),
                proposal,
                base_trial_count + idx,
                evaluate_fn,
                sem,
                **kwargs,
            )
            for idx, proposal in enumerate(proposals)
        ])

        return _merge_outcomes(state, outcomes, objective)


class Optimizer(abc.ABC):
    """
    Abstract base class for all optimizers.

    Subclasses MUST define `supported_unit_types` as a class-level frozenset::

        class MyOptimizer(Optimizer):
            supported_unit_types: ClassVar[FrozenSet[OptimizationUnitType]] = frozenset({
                OptimizationUnitType.PROMPT,
            })

    Defining it as an instance property is not allowed: `Optimizer.__init__` reads
    `supported_unit_types` before the subclass constructor body has run, so any property
    that depends on instance state will observe a partially-constructed object.
    """

    supported_unit_types: ClassVar[FrozenSet[OptimizationUnitType]]

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        if "supported_unit_types" in cls.__dict__:
            val = cls.__dict__["supported_unit_types"]
            if not isinstance(val, frozenset):
                raise TypeError(
                    f"{cls.__name__}.supported_unit_types must be a frozenset, "
                    f"got {type(val).__name__!r}"
                )

    def __init__(
        self,
        adapter: ProgramAdapter,
        **kwargs
    ) -> None:

        if not isinstance(adapter, ProgramAdapter):
            raise TypeError(f"Expected adapter to be an instance of ProgramAdapter, got {type(adapter)}")

        self.adapter = adapter
        self.runtime = TrialRuntime(adapter)
        self.kwargs = kwargs

        if not isinstance(getattr(type(self), "supported_unit_types", None), frozenset):
            raise TypeError(
                f"{self.__class__.__name__}.supported_unit_types must be defined as a "
                f"frozenset[OptimizationUnitType] class attribute"
            )

        unsupported = {
            unit.unit_type
            for unit in self.adapter.units
            if unit.unit_type not in self.supported_unit_types
        }
        if unsupported:
            raise TypeError(
                f"{self.__class__.__name__} supports unit type(s): "
                f"{{{', '.join(t.value for t in sorted(self.supported_unit_types, key=lambda t: t.value))}}}; "
                f"but the adapter declares unsupported unit type(s): "
                f"{{{', '.join(t.value for t in sorted(unsupported, key=lambda t: t.value))}}}."
            )

    def _init_run_state(
        self,
        save_dir: Optional[str] = None,
        resume_from: Optional[str] = None,
    ) -> OptimizationRunState:
        """Load or create OptimizationRunState and seed the initial snapshot and baseline record."""
        if resume_from:
            state = OptimizationRunState.load_state(resume_from)
        else:
            state = OptimizationRunState(save_dir=save_dir or "./")

        # Seed the initial snapshot so propose() always has a valid source_snapshot_id,
        # including the first trial of a fresh run.
        if not state.snapshots:
            initial_snapshot = self.adapter.take_snapshot()
            state.snapshots.append(initial_snapshot)
            state.trial_records.append(TrialRecord(
                trial_id=BASELINE_TRIAL_ID,
                changes=[],
                source_snapshot_id=initial_snapshot.snapshot_id,
                status="completed",
                snapshot_id=initial_snapshot.snapshot_id,
                metrics=None,
            ))

        return state

    @staticmethod
    def _validate_optimize_args(
        objective: Any,
        max_trials: Any,
        execution_mode: Any,
        max_workers: Any,
    ) -> Tuple[Literal["sequential", "concurrent"], Optional[int]]:
        """Validate optimize / async_optimize arguments and return normalized execution settings."""
        if not isinstance(objective, Objective):
            raise TypeError(f"`objective` must be an instance of Objective, got {type(objective).__name__!r}")
        if isinstance(max_trials, bool) or not isinstance(max_trials, int) or max_trials <= 0:
            raise ValueError(f"`max_trials` must be a positive integer, got {max_trials}")
        return _validate_execution_config(execution_mode, max_workers)

    def optimize(
        self,
        evaluate_fn: Callable[[ProgramAdapter], Dict[str, Any]],
        objective: Objective,
        max_trials: Optional[int] = 3,
        save_dir: Optional[str] = None,
        resume_from: Optional[str] = None,
        execution_mode: Literal["sequential", "concurrent"] = "sequential",
        max_workers: Optional[int] = None,
        **kwargs
    ) -> ProgramAdapter:

        execution_mode, max_workers = self._validate_optimize_args(objective, max_trials, execution_mode, max_workers)

        # Initialize or load the optimization run state
        state = self._init_run_state(save_dir, resume_from)

        # Evaluate baseline once if not yet done; persists so resume skips this.
        baseline_record = state.get_baseline_record()
        if baseline_record is not None and baseline_record.metrics is None:
            metrics = evaluate_fn(self.adapter)
            _validate_metrics(metrics, BASELINE_TRIAL_ID)
            baseline_record.metrics = metrics
            state.save_state()

        # Sync best_snapshot_id / best_metrics from objective (covers both fresh and resume).
        _sync_best(state, objective)

        while state.current_step < max_trials:
            remaining = max_trials - state.current_step
            proposals = self.batch_propose(state, objective, **kwargs)[:remaining]
            if not proposals:
                break
            state = self.runtime.run_batch(
                state=state,
                proposals=proposals,
                evaluate_fn=evaluate_fn,
                objective=objective,
                execution_mode=execution_mode,
                max_workers=max_workers,
                **kwargs
            )
            state.current_step += len(proposals)
            state.save_state()

        if state.best_snapshot_id is None:
            raise RuntimeError(
                "Optimization finished but no best snapshot was recorded. "
                "This usually means all trials failed or the run state is corrupted."
            )
        best_snapshot = state.get_snapshot_by_id(state.best_snapshot_id)
        if best_snapshot is None:
            raise RuntimeError(
                f"best_snapshot_id '{state.best_snapshot_id}' not found in run state. "
                "The saved state may be incomplete or corrupted."
            )
        return self.adapter.load_snapshot(best_snapshot)

    async def async_optimize(
        self,
        evaluate_fn: Callable[[ProgramAdapter], Awaitable[Dict[str, Any]]],
        objective: Objective,
        max_trials: Optional[int] = 3,
        save_dir: Optional[str] = None,
        resume_from: Optional[str] = None,
        execution_mode: Literal["sequential", "concurrent"] = "sequential",
        max_workers: Optional[int] = None,
        **kwargs
    ) -> ProgramAdapter:
        """
        Async variant of `optimize`. Supports both sequential and concurrent trial execution.

        Args:
            evaluate_fn: Async callable `(adapter) -> Awaitable[Dict[str, Any]]` returning metrics.
            objective: Optimization objective.
            max_trials: Total number of individual trials to run.
            save_dir: Directory to write per-trial state checkpoints.
            resume_from: Path to a prior checkpoint to resume from.
            execution_mode: "sequential" (default) or "concurrent". Pass "concurrent" when the
                            underlying program is stateless or safe to evaluate in parallel
                            (e.g. pure API calls, read-only inference).
            max_workers: Maximum number of coroutines running concurrently in concurrent mode.
                         Required and must be greater than 1 in concurrent mode.
                         Use None or 1 in sequential mode.
            **kwargs: Forwarded to `batch_propose` and `async_run_batch`.

        Returns:
            A new ProgramAdapter reflecting the best configuration found.
        """
        execution_mode, max_workers = self._validate_optimize_args(objective, max_trials, execution_mode, max_workers)

        state = self._init_run_state(save_dir, resume_from)

        # Evaluate baseline once if not yet done; persists so resume skips this.
        baseline_record = state.get_baseline_record()
        if baseline_record is not None and baseline_record.metrics is None:
            metrics = await evaluate_fn(self.adapter)
            _validate_metrics(metrics, BASELINE_TRIAL_ID)
            baseline_record.metrics = metrics
            state.save_state()

        # Sync best_snapshot_id / best_metrics from objective (covers both fresh and resume).
        _sync_best(state, objective)

        while state.current_step < max_trials:
            remaining = max_trials - state.current_step
            proposals = self.batch_propose(state, objective, **kwargs)[:remaining]
            if not proposals:
                break
            state = await self.runtime.async_run_batch(
                state=state,
                proposals=proposals,
                evaluate_fn=evaluate_fn,
                objective=objective,
                execution_mode=execution_mode,
                max_workers=max_workers,
                **kwargs
            )
            state.current_step += len(proposals)
            state.save_state()

        if state.best_snapshot_id is None:
            raise RuntimeError(
                "Optimization finished but no best snapshot was recorded. "
                "This usually means all trials failed or the run state is corrupted."
            )
        best_snapshot = state.get_snapshot_by_id(state.best_snapshot_id)
        if best_snapshot is None:
            raise RuntimeError(
                f"best_snapshot_id '{state.best_snapshot_id}' not found in run state. "
                "The saved state may be incomplete or corrupted."
            )
        return self.adapter.load_snapshot(best_snapshot)

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
