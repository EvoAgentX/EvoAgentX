from __future__ import annotations

import sys
import textwrap
from typing import Any, Dict, Optional

from .base import TrialRecord
from .objective import Objective, ScalarObjective

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - exercised only when tqdm is absent.
    tqdm = None

GREEN = "\033[32m"
ENDC = "\033[0m"
REPORT_WIDTH = 56


class OptimizationProgress:
    """Small wrapper around tqdm with a text-only fallback."""

    def __init__(self, total: int, initial: int = 0, desc: str = "Optimization trials") -> None:
        self.total = total
        self.current = min(initial, total)
        self._bar = None
        self._closed = False
        if tqdm is not None:
            self._bar = tqdm(
                total=total,
                initial=self.current,
                desc=desc,
                unit="trial",
            )
        elif total > 0:
            self.write(f"{desc}: {self.current}/{total}")

    def update(self, n: int = 1) -> None:
        if n <= 0:
            return
        self.current = min(self.current + n, self.total)
        if self._bar is not None:
            self._bar.update(n)
        elif self.total > 0:
            self.write(f"Optimization progress: {self.current}/{self.total}")

    def write(self, message: str) -> None:
        if self._bar is not None and not self._closed:
            self._bar.write(message)
        else:
            print(message, file=sys.stderr)

    def close(self) -> None:
        if self._bar is not None and not self._closed:
            self._bar.close()
        self._closed = True

    def __enter__(self) -> "OptimizationProgress":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return repr(value)


def _format_delta(value: Any) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{value:+.6g}"
    return _format_value(value)


def _format_metrics(metrics: Optional[Dict[str, Any]]) -> str:
    if metrics is None:
        return "None"
    return "{" + ", ".join(f"{key}={_format_value(value)}" for key, value in metrics.items()) + "}"


def _scalar_score(record: TrialRecord, objective: Objective) -> Optional[Any]:
    if not isinstance(objective, ScalarObjective) or record.metrics is None:
        return None
    return record.metrics.get(objective.metric)


def _scalar_improvement(record: TrialRecord, baseline_record: Optional[TrialRecord], objective: Objective) -> Optional[Any]:
    if not isinstance(objective, ScalarObjective) or baseline_record is None:
        return None
    score = _scalar_score(record, objective)
    baseline_score = _scalar_score(baseline_record, objective)
    if score is None or baseline_score is None:
        return None
    try:
        return score - baseline_score if objective.direction == "maximize" else baseline_score - score
    except TypeError:
        return None


def format_trial_progress_message(
    record: TrialRecord,
    objective: Objective,
    baseline_record: Optional[TrialRecord],
    best_snapshot_id: Optional[str],
) -> str:
    """Format one completed trial summary for progress output."""
    prefix = "Baseline" if record.trial_id == 0 else f"Trial {record.trial_id}"
    is_best = bool(record.snapshot_id and record.snapshot_id == best_snapshot_id)
    best_text = f" {GREEN}[* new best]{ENDC}" if is_best else ""

    if record.status != "completed":
        return f"{prefix}: failed, error={record.error!r}{best_text}"

    score = _scalar_score(record, objective)
    score_text = _format_value(score) if score is not None else _format_metrics(record.metrics)
    improvement = _scalar_improvement(record, baseline_record, objective)
    delta_text = _format_delta(improvement) if improvement is not None else "n/a"
    return f"{prefix}: score={score_text}, delta_vs_baseline={delta_text}{best_text}"


def report_trial_progress(
    progress: OptimizationProgress,
    record: TrialRecord,
    objective: Objective,
    baseline_record: Optional[TrialRecord],
    best_snapshot_id: Optional[str],
) -> None:
    progress.write(format_trial_progress_message(record, objective, baseline_record, best_snapshot_id))


def _format_duration(seconds: float) -> str:
    total_seconds = int(round(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {seconds:02d}s"
    if minutes:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"


def _truncate_text(text: str, max_width: int) -> str:
    if len(text) <= max_width:
        return text
    if max_width <= 1:
        return "." * max_width
    return text[: max_width - 1] + "."


def _box_line(label: str, value: Any, width: int = REPORT_WIDTH) -> str:
    content_width = width - 2
    left = f"  {label}"
    right = str(value)
    min_dots = 2
    max_right_width = max(1, content_width - len(left) - min_dots)
    right = _truncate_text(right, max_right_width)
    dots = "." * max(min_dots, content_width - len(left) - len(right))
    content = f"{left}{dots}{right}"
    return f"│{content.ljust(content_width)}│"


def _box_wrapped_line(label: str, value: Any, width: int = REPORT_WIDTH) -> list[str]:
    content_width = width - 2
    left = f"  {label}"
    text = str(value)
    min_dots = 2
    first_width = max(1, content_width - len(left) - min_dots)
    chunks = textwrap.wrap(
        text,
        width=first_width,
        break_long_words=True,
        break_on_hyphens=False,
    ) or [""]
    first = chunks[0]
    dots = "." * max(min_dots, content_width - len(left) - len(first))
    lines = [f"│{left}{dots}{first}".ljust(width - 1) + "│"]

    continuation_indent = " " * len(left)
    continuation_width = max(1, content_width - len(continuation_indent))
    for chunk in chunks[1:]:
        for part in textwrap.wrap(
            chunk,
            width=continuation_width,
            break_long_words=True,
            break_on_hyphens=False,
        ) or [""]:
            content = f"{continuation_indent}{part}"
            lines.append(f"│{content.ljust(content_width)}│")
    return lines


def _box_title(title: str, width: int = REPORT_WIDTH) -> str:
    content_width = width - 2
    return f"║{title.center(content_width)}║"


def _box_rule(left: str, fill: str, right: str, width: int = REPORT_WIDTH) -> str:
    return f"{left}{fill * (width - 2)}{right}"


def format_optimization_report(
    state: Any,
    objective: Objective,
    baseline_record: Optional[TrialRecord],
    elapsed_seconds: float,
    start_step: int,
    max_trials: int,
    total_cost: Optional[float] = None,
) -> str:
    """Format the final optimization report."""
    trial_records = [record for record in state.trial_records if record.trial_id != 0]
    completed_trials = [record for record in trial_records if record.status == "completed"]
    failed_trials = [record for record in trial_records if record.status == "failed"]
    best_record = objective.select_best_trial_record(state.trial_records)

    trials_this_run = max(0, state.current_step - start_step)
    total_trials = state.current_step

    lines = [
        _box_rule("╔", "═", "╗"),
        _box_title("Optimization Report"),
        _box_rule("╠", "═", "╣"),
        _box_line("Max trials", max_trials),
        _box_line("Trials this run", trials_this_run),
        _box_line("Total trials in state", total_trials),
        _box_line("Snapshots retained", len(state.snapshots)),
        _box_line("Completed trials", len(completed_trials)),
        _box_line("Failed trials", len(failed_trials)),
        _box_line("Elapsed time", _format_duration(elapsed_seconds)),
    ]

    if total_cost is not None:
        lines.append(_box_line("Total cost (USD)", f"${total_cost:.4f}"))

    if baseline_record is not None and baseline_record.metrics is not None:
        baseline_score = _scalar_score(baseline_record, objective)
        baseline_text = _format_value(baseline_score) if baseline_score is not None else _format_metrics(baseline_record.metrics)
        lines.extend([
            _box_rule("╠", "═", "╣"),
            _box_line("Baseline score", baseline_text),
        ])

    lines.append(_box_rule("╠", "═", "╣"))
    if best_record is not None:
        best_label = "Baseline" if best_record.trial_id == 0 else f"Trial {best_record.trial_id}"
        best_score = _scalar_score(best_record, objective)
        best_text = _format_value(best_score) if best_score is not None else _format_metrics(best_record.metrics)
        lines.append(_box_line("Best trial", best_label))
        lines.append(_box_line("Best score", best_text))
        improvement = _scalar_improvement(best_record, baseline_record, objective)
        if improvement is not None:
            lines.append(_box_line("Delta vs baseline", _format_delta(improvement)))
        lines.extend(_box_wrapped_line("Best metrics", _format_metrics(best_record.metrics)))
    else:
        lines.append(_box_line("Best trial", "n/a"))

    if failed_trials:
        lines.extend([
            _box_rule("╠", "═", "╣"),
            _box_line("Last failure", f"Trial {failed_trials[-1].trial_id}: {failed_trials[-1].error!r}"),
        ])

    lines.append(_box_rule("╚", "═", "╝"))
    return "\n".join(lines)
