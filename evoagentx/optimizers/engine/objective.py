import abc
from typing import Any, Dict, List, Optional, Literal

from .base import TrialRecord

class Objective(abc.ABC):
    """
    Abstract base class for optimization objectives.

    Subclass and implement `is_better` to define the comparison logic.
    For single-metric scalar optimization use the concrete `ScalarObjective`.
    """

    @abc.abstractmethod
    def is_better(self, metrics_a: Dict[str, Any], metrics_b: Dict[str, Any]) -> bool:
        """
        Return True if metrics_a represents a strictly better outcome than metrics_b.

        Args:
            metrics_a: Metrics dict from the candidate trial.
            metrics_b: Metrics dict from the reference / current-best trial.

        Returns:
            True if metrics_a is strictly better than metrics_b, False otherwise.
        """
        raise NotImplementedError

    def select_best_trial_record(self, trial_records: List["TrialRecord"]) -> Optional["TrialRecord"]:
        """
        Return the best completed TrialRecord, or None if no completed record with metrics exists.

        The returned record gives callers direct access to snapshot_id, metrics, and all
        other trial metadata without a secondary lookup.

        Iterates over completed (status == "completed") trial records that have non-None
        metrics and returns the one with the best metrics according to `is_better`.

        Args:
            trial_records: All trial records accumulated in the optimization run.

        Returns:
            The best completed TrialRecord, or None if none exist.
        """
        best: Optional[TrialRecord] = None
        for record in trial_records:
            if record.status != "completed" or record.metrics is None:
                continue
            if best is None or self.is_better(record.metrics, best.metrics):
                best = record
        return best


class ScalarObjective(Objective):
    """
    Concrete objective that compares a single named scalar metric.

    Args:
        metric: Key to look up in the metrics dict (default "score").
        direction: "maximize" (default) or "minimize".
    """

    def __init__(
        self,
        metric: str = "score",
        direction: Literal["maximize", "minimize"] = "maximize",
    ) -> None:
        self.metric = metric
        self.direction = direction

    def is_better(self, metrics_a: Dict[str, Any], metrics_b: Dict[str, Any]) -> bool:
        val_a = metrics_a.get(self.metric) if metrics_a else None
        val_b = metrics_b.get(self.metric) if metrics_b else None
        if val_a is None:
            return False
        if val_b is None:
            return True
        return val_a > val_b if self.direction == "maximize" else val_a < val_b

