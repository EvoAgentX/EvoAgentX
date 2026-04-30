import abc
from typing import Any, Dict, List, Optional, Literal, Union

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
        if not isinstance(metric, str) or not metric:
            raise ValueError(f"metric must be a non-empty string, got {metric!r}")
        if direction not in ("maximize", "minimize"):
            raise ValueError(f"direction must be 'maximize' or 'minimize', got {direction!r}")
        self.metric = metric
        self.direction = direction

    def is_better(self, metrics_a: Dict[str, Any], metrics_b: Dict[str, Any]) -> bool:
        val_a = metrics_a.get(self.metric) if metrics_a else None
        val_b = metrics_b.get(self.metric) if metrics_b else None
        if val_a is None:
            return False
        if val_b is None:
            return True
        try:
            return val_a > val_b if self.direction == "maximize" else val_a < val_b
        except TypeError:
            raise TypeError(
                f"Cannot compare values for metric '{self.metric}': "
                f"{val_a!r} (type {type(val_a).__name__}) vs {val_b!r} (type {type(val_b).__name__}). "
                "Metric values must be comparable scalars (e.g. int, float)."
            )


class ParetoObjective(Objective):
    """
    Multi-objective Pareto-dominance objective.

    metrics_a is considered "better" than metrics_b when metrics_a Pareto-dominates
    metrics_b: at least as good on every tracked metric and strictly better on at least one.
    When neither dominates the other, `is_better` returns False (no strict ordering).

    `select_best_trial_record` returns the completed record that dominates the most others;
    ties are broken by iteration order (first record encountered wins).

    Args:
        metrics: Non-empty list of metric keys to compare.
        directions: Either a single direction applied to all metrics, or a per-metric list.
                    Each direction is "maximize" (default) or "minimize".
    """

    def __init__(
        self,
        metrics: List[str],
        directions: Union[Literal["maximize", "minimize"], List[Literal["maximize", "minimize"]]] = "maximize",
    ) -> None:
        if not metrics:
            raise ValueError("metrics must be a non-empty list")
        if isinstance(directions, str):
            if directions not in ("maximize", "minimize"):
                raise ValueError(f"direction must be 'maximize' or 'minimize', got {directions!r}")
            directions = [directions] * len(metrics)
        else:
            if len(directions) != len(metrics):
                raise ValueError(
                    f"directions length ({len(directions)}) must match metrics length ({len(metrics)})"
                )
            for d in directions:
                if d not in ("maximize", "minimize"):
                    raise ValueError(f"each direction must be 'maximize' or 'minimize', got {d!r}")
        self.metrics = metrics
        self.directions: List[Literal["maximize", "minimize"]] = list(directions)

    def _dominates(self, metrics_a: Dict[str, Any], metrics_b: Dict[str, Any]) -> bool:
        """Return True if metrics_a Pareto-dominates metrics_b."""
        at_least_as_good = True
        strictly_better_on_one = False
        for key, direction in zip(self.metrics, self.directions):
            val_a = metrics_a.get(key) if metrics_a else None
            val_b = metrics_b.get(key) if metrics_b else None
            if val_a is None or val_b is None:
                return False
            try:
                if direction == "maximize":
                    if val_a < val_b:
                        at_least_as_good = False
                        break
                    if val_a > val_b:
                        strictly_better_on_one = True
                else:
                    if val_a > val_b:
                        at_least_as_good = False
                        break
                    if val_a < val_b:
                        strictly_better_on_one = True
            except TypeError:
                raise TypeError(
                    f"Cannot compare values for metric '{key}': "
                    f"{val_a!r} vs {val_b!r}. Metric values must be comparable scalars."
                )
        return at_least_as_good and strictly_better_on_one

    def is_better(self, metrics_a: Dict[str, Any], metrics_b: Dict[str, Any]) -> bool:
        """Return True if metrics_a Pareto-dominates metrics_b."""
        return self._dominates(metrics_a, metrics_b)

    def select_best_trial_record(self, trial_records: List["TrialRecord"]) -> Optional["TrialRecord"]:
        """
        Return the completed record that dominates the most others.

        Among all completed records with non-None metrics, computes each record's
        dominance count (number of other candidates it Pareto-dominates) and returns
        the one with the highest count. Ties are broken by iteration order.
        Returns None if no eligible records exist.
        """
        candidates = [
            r for r in trial_records
            if r.status == "completed" and r.metrics is not None
        ]
        if not candidates:
            return None
        best = candidates[0]
        best_count = sum(1 for other in candidates if self._dominates(best.metrics, other.metrics))
        for record in candidates[1:]:
            count = sum(1 for other in candidates if self._dominates(record.metrics, other.metrics))
            if count > best_count:
                best = record
                best_count = count
        return best

