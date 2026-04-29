import abc
from pydantic import Field
from typing import Any, Callable, FrozenSet, Optional, List, Dict, Literal

from ...core.module import BaseModule
from .base import OptimizationUnitType, OptimizationUnit, UnitChange
from .adapter import SnapShot, ProgramAdapter


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
           call `self.adapter.apply(base_snapshot, proposal.changes)` to produce
           `new_adapter`. `self.adapter` is never modified and remains the stable base
           for all trials in the run.
        2. Call `new_adapter.take_snapshot()` immediately to capture the clean post-apply
           configuration before any evaluation side-effects can pollute it.
        3. Run evaluation: call `evaluate_fn(new_adapter)` to obtain a metrics dict.
           On exception, record the trial as status="failed" with the error message
           and append the failed record to state without updating best_snapshot_id.
        4. Build a TrialRecord (trial_id = len(state.trial_records)) with the
           snapshot_id, metrics, and status.
        5. Append the new SnapShot and TrialRecord to `state`.
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
    
    def optimize(
        self,
        evaluate_fn: Callable[[ProgramAdapter], dict[str, Any]],
        objective: Optional[Objective] = None,
        max_trials: Optional[int] = 3,
        save_dir: Optional[str] = None,
        resume_from: Optional[str] = None,
        **kwargs
    ) -> ProgramAdapter:
        
        if not isinstance(max_trials, int) or max_trials <= 0:
            raise ValueError(f"`max_trials` must be a positive integer, got {max_trials}")
        
        # define default objective if not provided
        if objective is None:
            objective = Objective() # TODO: placehoder, change when Objective is defined

        # load or initialize optimization state
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

        for step in range(max_trials):
            if step < state.current_step:
                continue # skip already completed trials when resuming
            
            # the algorithm chooses a snapshot to start from in this trial, 
            # and propose a set of changes to apply on the snapshot
            proposal: OptimizationProposal = self.propose(state, objective, **kwargs)

            # apply the proposed changes on the base snapshot specified in the proposal,
            # generate a new snapshot and record the generated snapshot,
            # then execute, evaluate and record the trial results
            state: OptimizationRunState = self.runtime.run(
                state=state,
                proposal=proposal,
                evaluate_fn=evaluate_fn,
                objective=objective,
                **kwargs
            )

            # save the candidate snapshot, trial results
            state.current_step = step + 1
            state.save_state() # TODO: implement save_state method
        
        # reconstruct and return the adapter corresponding to the best snapshot found
        best_snapshot = state.get_snapshot(state.best_snapshot_id)
        optimized_adapter = self.adapter.load_snapshot(best_snapshot)

        return optimized_adapter
    
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
        Propose a set of changes to apply on the program for the next trial step.

        It should first select a snapshot as the base to apply changes on. 
        If there is no trial yet, use the initial snapshot from the adapter.
        Then it should propose changes to the optimization units, and return the 
        proposed changes along with the source snapshot id in an OptimizationProposal.

        Parameters:
        - state: current optimization run state, including past trial records and metadata
        - objective: the optimization objective that the proposed changes should aim to improve
        - kwargs: any additional information that might be useful for proposing changes (e.g. current parameter values, trial history, etc.)

        Returns:
        - A list of UnitChange objects representing the proposed changes for the next trial.
        """
        raise NotImplementedError
