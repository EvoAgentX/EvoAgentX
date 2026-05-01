from __future__ import annotations

import os
import threading
from typing import Any, ClassVar, Dict, FrozenSet, List

import pytest

from evoagentx.optimizers.engine.adapter import ProgramAdapter, SnapShot
from evoagentx.optimizers.engine.base import (
    CodeOptimizationUnit,
    EvaluationResult,
    OptimizationProposal,
    OptimizationUnit,
    OptimizationUnitType,
    TrialRecord,
    UnitChange,
    ValidationResult,
)
from evoagentx.optimizers.engine.objective import ParetoObjective, ScalarObjective
from evoagentx.optimizers.engine.optimizer import OptimizationRunState, Optimizer


class DummyAdapter(ProgramAdapter):
    def __init__(self, prompt: str = "base", memories: List[str] | None = None):
        self.prompt = prompt
        self.memories = list(memories or [])

    def execute(self, *args, **kwargs) -> Dict[str, Any]:
        return {"prompt": self.prompt, "memories": self.memories}

    def register_units(self) -> List[OptimizationUnit]:
        return [
            OptimizationUnit(
                name="prompt",
                uid="prompt",
                unit_type=OptimizationUnitType.PROMPT,
                json_schema={"type": "string"},
                metadata={"role": "system"},
            ),
            OptimizationUnit(
                name="memories",
                uid="memories",
                unit_type=OptimizationUnitType.MEMORY,
                json_schema={"type": "array", "items": {"type": "string"}},
                allowed_operations=["replace", "append"],
                operation_schemas={"append": {"type": "string"}},
                metadata={"store": "experience"},
            ),
        ]

    def take_snapshot(self) -> SnapShot:
        return SnapShot(unit_values={"prompt": self.prompt, "memories": list(self.memories)})

    def merge_changes(self, snapshot: SnapShot, changes: List[UnitChange], **kwargs) -> SnapShot:
        values = dict(snapshot.unit_values)
        values["memories"] = list(values.get("memories", []))
        for change in changes:
            if change.operation == "replace":
                values[change.uid] = change.value
            elif change.operation == "append":
                values[change.uid].append(change.value)
            else:
                raise ValueError(f"unsupported operation {change.operation}")
        return SnapShot(unit_values=values, program_config=snapshot.program_config)

    def from_snapshot(self, snapshot: SnapShot, **kwargs) -> ProgramAdapter:
        return DummyAdapter(
            prompt=snapshot.unit_values["prompt"],
            memories=list(snapshot.unit_values.get("memories", [])),
        )


class PromptOnlyOptimizer(Optimizer):
    
    supported_unit_types: ClassVar[FrozenSet[OptimizationUnitType]] = frozenset({OptimizationUnitType.PROMPT})

    def propose(self, state: OptimizationRunState, objective: ScalarObjective, **kwargs) -> OptimizationProposal:
        unit = self.target_units_by_uid["prompt"]
        source_snapshot_id = state.best_snapshot_id or state.snapshots[-1].snapshot_id
        return OptimizationProposal(
            source_snapshot_id=source_snapshot_id,
            changes=[UnitChange.create(unit, "improved prompt", metadata={"reason": "test"})],
        )


class MemoryAppendOptimizer(Optimizer):
    supported_unit_types: ClassVar[FrozenSet[OptimizationUnitType]] = frozenset({OptimizationUnitType.MEMORY})

    def propose(self, state: OptimizationRunState, objective: ScalarObjective, **kwargs) -> OptimizationProposal:
        unit = self.target_units_by_uid["memories"]
        source_snapshot_id = state.best_snapshot_id or state.snapshots[-1].snapshot_id
        return OptimizationProposal(
            source_snapshot_id=source_snapshot_id,
            changes=[UnitChange.create(unit, "learned experience", operation="append")],
        )

    def observe(
        self,
        state: OptimizationRunState,
        trial_records: List[Any],
        objective: ScalarObjective,
        **kwargs,
    ) -> None:
        state.optimizer_state["observed_trials"] = state.optimizer_state.get("observed_trials", 0) + len(trial_records)


class SkillFileAdapter(ProgramAdapter):
    def __init__(self, content: str = "def base_skill():\n    return 'base'\n"):
        self.content = content

    def execute(self, *args, **kwargs) -> Dict[str, Any]:
        workspace_file = self.workspace.path("skills.py") if self.workspace is not None else None
        return {"content": self.content, "workspace_file": workspace_file}

    def register_units(self) -> List[OptimizationUnit]:
        return [
            CodeOptimizationUnit(
                uid="skills.py",
                name="skills.py",
                language="python",
                entrypoint="base_skill",
                metadata={"path": "skills.py"},
            )
        ]

    def take_snapshot(self) -> SnapShot:
        return SnapShot(unit_values={"skills.py": self.content})

    def merge_changes(self, snapshot: SnapShot, changes: List[UnitChange], **kwargs) -> SnapShot:
        return SnapShot(
            unit_values=self.merge_unit_values(snapshot, changes),
            program_config=snapshot.program_config,
        )

    def from_snapshot(self, snapshot: SnapShot, **kwargs) -> ProgramAdapter:
        return SkillFileAdapter(content=snapshot.unit_values["skills.py"])

    def prepare_workspace(self, workspace, snapshot: SnapShot, **kwargs) -> None:
        path = workspace.path("skills.py")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(snapshot.unit_values["skills.py"])

    def validate_trial(self, snapshot: SnapShot, changes: List[UnitChange], workspace=None, **kwargs) -> List[ValidationResult]:
        content = snapshot.unit_values["skills.py"]
        if "def " not in content:
            return [ValidationResult(validator="python-static", status="failed", message="no function definition")]
        return [ValidationResult(validator="python-static", status="passed", message="contains function definition")]


class SkillCodeOptimizer(Optimizer):
    supported_unit_types: ClassVar[FrozenSet[OptimizationUnitType]] = frozenset({OptimizationUnitType.CODE})

    def propose(self, state: OptimizationRunState, objective: ScalarObjective, **kwargs) -> OptimizationProposal:
        unit = self.target_units_by_uid["skills.py"]
        source_snapshot_id = state.best_snapshot_id or state.snapshots[-1].snapshot_id
        return OptimizationProposal(
            source_snapshot_id=source_snapshot_id,
            changes=[
                UnitChange.create(
                    unit,
                    "def new_skill():\n    return 'new'\n",
                    operation="append",
                )
            ],
        )


class BrokenSkillCodeOptimizer(SkillCodeOptimizer):
    def propose(self, state: OptimizationRunState, objective: ScalarObjective, **kwargs) -> OptimizationProposal:
        unit = self.target_units_by_uid["skills.py"]
        source_snapshot_id = state.best_snapshot_id or state.snapshots[-1].snapshot_id
        return OptimizationProposal(
            source_snapshot_id=source_snapshot_id,
            changes=[UnitChange.create(unit, "print('not a skill')", operation="replace")],
        )


class EpisodeAdapter(DummyAdapter):
    def __init__(self, prompt: str = "base", memories: List[str] | None = None):
        super().__init__(prompt=prompt, memories=memories)
        self.events: List[Dict[str, Any]] = []

    def on_task_begin(self, task: Any = None, **kwargs) -> Any:
        self.events.append({"event": "task_begin", "task": task})
        return self.events[-1]

    def on_step(self, step: Any = None, **kwargs) -> Any:
        self.events.append({"event": "step", "step": step})
        return self.events[-1]

    def on_task_end(self, trajectory: Any = None, result: Any = None, **kwargs) -> Any:
        self.events.append({"event": "task_end", "result": result})
        return self.events[-1]

    def from_snapshot(self, snapshot: SnapShot, **kwargs) -> ProgramAdapter:
        return EpisodeAdapter(
            prompt=snapshot.unit_values["prompt"],
            memories=list(snapshot.unit_values.get("memories", [])),
        )


def test_optimizer_targets_supported_units_by_default_and_persists_evaluation_artifacts(tmp_path):
    optimizer = PromptOnlyOptimizer(DummyAdapter())

    assert optimizer.target_unit_uids == ["prompt"]

    def evaluate(adapter: ProgramAdapter) -> EvaluationResult:
        output = adapter.execute()
        return EvaluationResult(
            metrics={"score": len(output["prompt"])},
            traces=[{"prompt": output["prompt"]}],
            artifacts={"length": len(output["prompt"])},
            metadata={"split": "dev"},
        )

    best = optimizer.optimize(
        evaluate_fn=evaluate,
        objective=ScalarObjective(metric="score", direction="maximize"),
        max_trials=1,
        save_dir=str(tmp_path),
    )

    assert best.execute()["prompt"] == "improved prompt"

    state = OptimizationRunState.load_state(str(tmp_path))
    trial = state.trial_records[-1]
    assert trial.metrics == {"score": len("improved prompt")}
    assert trial.traces == [{"prompt": "improved prompt"}]
    assert trial.artifacts == {"length": len("improved prompt")}
    assert trial.metadata == {"split": "dev"}
    assert state.adapter_fingerprint is not None


def test_optimizer_can_target_memory_unit_and_use_append_changes(tmp_path):
    optimizer = MemoryAppendOptimizer(DummyAdapter(), target_unit_types=OptimizationUnitType.MEMORY)

    assert optimizer.target_unit_uids == ["memories"]

    def evaluate(adapter: ProgramAdapter) -> Dict[str, Any]:
        return {"score": len(adapter.execute()["memories"])}

    best = optimizer.optimize(
        evaluate_fn=evaluate,
        objective=ScalarObjective(metric="score", direction="maximize"),
        max_trials=1,
        save_dir=str(tmp_path),
    )

    assert best.execute()["memories"] == ["learned experience"]

    state = OptimizationRunState.load_state(str(tmp_path))
    assert state.optimizer_state["observed_trials"] == 1


def test_explicit_unsupported_target_unit_is_rejected():
    try:
        PromptOnlyOptimizer(DummyAdapter(), target_unit_uids=["memories"])
    except TypeError as exc:
        assert "unsupported type" in str(exc)
    else:
        raise AssertionError("expected unsupported target unit to be rejected")


def test_code_unit_uses_trial_workspace_and_validation_pipeline(tmp_path):
    optimizer = SkillCodeOptimizer(SkillFileAdapter())

    def evaluate(adapter: ProgramAdapter) -> Dict[str, Any]:
        output = adapter.execute()
        assert output["workspace_file"] is not None
        assert os.path.exists(output["workspace_file"])
        return {"score": int("new_skill" in output["content"])}

    best = optimizer.optimize(
        evaluate_fn=evaluate,
        objective=ScalarObjective(metric="score", direction="maximize"),
        max_trials=1,
        save_dir=str(tmp_path),
    )

    assert "new_skill" in best.execute()["content"]

    state = OptimizationRunState.load_state(str(tmp_path))
    trial = state.trial_records[-1]
    assert trial.status == "completed"
    assert trial.validation_results[0].status == "passed"
    assert trial.workspace_dir is not None
    workspace_file = os.path.join(trial.workspace_dir, "skills.py")
    assert os.path.exists(workspace_file)
    with open(workspace_file, encoding="utf-8") as handle:
        assert "new_skill" in handle.read()


def test_validation_failure_skips_trial_evaluation(tmp_path):
    optimizer = BrokenSkillCodeOptimizer(SkillFileAdapter())
    calls = {"count": 0}

    def evaluate(adapter: ProgramAdapter) -> Dict[str, Any]:
        calls["count"] += 1
        return {"score": int("def " in adapter.execute()["content"])}

    best = optimizer.optimize(
        evaluate_fn=evaluate,
        objective=ScalarObjective(metric="score", direction="maximize"),
        max_trials=1,
        save_dir=str(tmp_path),
    )

    assert calls["count"] == 1
    assert best.execute()["content"] == "def base_skill():\n    return 'base'\n"

    state = OptimizationRunState.load_state(str(tmp_path))
    trial = state.trial_records[-1]
    assert trial.status == "failed"
    assert trial.validation_results[0].status == "failed"
    assert "python-static" in trial.error


def test_episode_lifecycle_hooks_can_feed_evaluation_traces(tmp_path):
    optimizer = PromptOnlyOptimizer(EpisodeAdapter())

    def evaluate(adapter: ProgramAdapter) -> EvaluationResult:
        adapter.on_task_begin(task={"id": "task-1"})
        adapter.on_step(step={"action": "search"})
        adapter.on_task_end(trajectory=list(adapter.events), result={"ok": True})
        output = adapter.execute()
        return EvaluationResult(
            metrics={"score": len(output["prompt"]) + len(adapter.events)},
            traces=[{"events": list(adapter.events)}],
        )

    best = optimizer.optimize(
        evaluate_fn=evaluate,
        objective=ScalarObjective(metric="score", direction="maximize"),
        max_trials=1,
        save_dir=str(tmp_path),
    )

    assert best.execute()["prompt"] == "improved prompt"

    state = OptimizationRunState.load_state(str(tmp_path))
    trial = state.trial_records[-1]
    assert [event["event"] for event in trial.traces[0]["events"]] == ["task_begin", "step", "task_end"]


# ---------------------------------------------------------------------------
# Multi-trial: minimize direction + current_step tracking
# ---------------------------------------------------------------------------

def test_multi_trial_minimize_objective(tmp_path):
    """Three trials; the one with the lowest score wins under minimize."""
    score_map = {"base": 100, "prompt-0": 50, "prompt-1": 75, "prompt-2": 60}
    call_n = {"n": 0}

    class SeqOptimizer(Optimizer):
        supported_unit_types: ClassVar[FrozenSet[OptimizationUnitType]] = frozenset({OptimizationUnitType.PROMPT})

        def propose(self, state, objective, **kwargs):
            unit = self.target_units_by_uid["prompt"]
            source = state.best_snapshot_id or state.snapshots[-1].snapshot_id
            name = f"prompt-{call_n['n']}"
            call_n["n"] += 1
            return OptimizationProposal(source_snapshot_id=source, changes=[UnitChange.create(unit, name)])

    optimizer = SeqOptimizer(DummyAdapter())
    best = optimizer.optimize(
        evaluate_fn=lambda a: {"score": score_map[a.execute()["prompt"]]},
        objective=ScalarObjective(metric="score", direction="minimize"),
        max_trials=3,
        save_dir=str(tmp_path),
    )

    assert best.execute()["prompt"] == "prompt-0"
    state = OptimizationRunState.load_state(str(tmp_path))
    assert state.best_metrics["score"] == 50
    assert state.current_step == 3


# ---------------------------------------------------------------------------
# should_stop hook
# ---------------------------------------------------------------------------

def test_should_stop_halts_before_max_trials(tmp_path):
    class EarlyStopOptimizer(PromptOnlyOptimizer):
        def should_stop(self, state, objective, **kwargs):
            return state.current_step >= 1

    optimizer = EarlyStopOptimizer(DummyAdapter())
    eval_count = {"n": 0}

    def evaluate(adapter):
        eval_count["n"] += 1
        return {"score": eval_count["n"]}

    optimizer.optimize(
        evaluate_fn=evaluate,
        objective=ScalarObjective(metric="score"),
        max_trials=5,
        save_dir=str(tmp_path),
    )

    # baseline + exactly 1 trial before should_stop fires
    assert eval_count["n"] == 2
    state = OptimizationRunState.load_state(str(tmp_path))
    assert state.current_step == 1


# ---------------------------------------------------------------------------
# on_run_start hook
# ---------------------------------------------------------------------------

def test_on_run_start_receives_state_with_evaluated_baseline(tmp_path):
    started = {}

    class StartTrackingOptimizer(PromptOnlyOptimizer):
        def on_run_start(self, state, objective, **kwargs):
            started["baseline_metrics"] = state.best_metrics
            started["step"] = state.current_step

    optimizer = StartTrackingOptimizer(DummyAdapter())
    optimizer.optimize(
        evaluate_fn=lambda a: {"score": 1},
        objective=ScalarObjective(metric="score"),
        max_trials=1,
        save_dir=str(tmp_path),
    )

    assert started["baseline_metrics"] is not None
    assert started["step"] == 0


# ---------------------------------------------------------------------------
# Population-based batch_propose (multiple proposals per step)
# ---------------------------------------------------------------------------

def test_batch_propose_returns_multiple_proposals(tmp_path):
    """batch_propose returning N proposals evaluates all N in one step."""
    score_map = {"candidate-0": 10, "candidate-1": 30, "candidate-2": 20}

    class PopOptimizer(Optimizer):
        supported_unit_types: ClassVar[FrozenSet[OptimizationUnitType]] = frozenset({OptimizationUnitType.PROMPT})

        def propose(self, state, objective, **kwargs):
            raise NotImplementedError

        def batch_propose(self, state, objective, max_proposals=None, **kwargs):
            unit = self.target_units_by_uid["prompt"]
            source = state.best_snapshot_id or state.snapshots[-1].snapshot_id
            names = ["candidate-0", "candidate-1", "candidate-2"]
            if max_proposals is not None:
                names = names[:max_proposals]
            return [
                OptimizationProposal(source_snapshot_id=source, changes=[UnitChange.create(unit, n)])
                for n in names
            ]

    optimizer = PopOptimizer(DummyAdapter())
    best = optimizer.optimize(
        evaluate_fn=lambda a: {"score": score_map.get(a.execute()["prompt"], 0)},
        objective=ScalarObjective(metric="score", direction="maximize"),
        max_trials=3,
        save_dir=str(tmp_path),
    )

    assert best.execute()["prompt"] == "candidate-1"
    state = OptimizationRunState.load_state(str(tmp_path))
    assert state.current_step == 3


# ---------------------------------------------------------------------------
# Concurrent (sync) execution + unique monotonic trial IDs
# ---------------------------------------------------------------------------

def test_concurrent_sync_execution_evaluates_all_proposals_with_unique_ids(tmp_path):
    evaluated = []
    lock = threading.Lock()

    class ConcurrentPopOptimizer(Optimizer):
        supported_unit_types: ClassVar[FrozenSet[OptimizationUnitType]] = frozenset({OptimizationUnitType.PROMPT})

        def propose(self, state, objective, **kwargs):
            raise NotImplementedError

        def batch_propose(self, state, objective, max_proposals=None, **kwargs):
            unit = self.target_units_by_uid["prompt"]
            source = state.snapshots[-1].snapshot_id
            names = ["cp-0", "cp-1", "cp-2"]
            if max_proposals is not None:
                names = names[:max_proposals]
            return [
                OptimizationProposal(source_snapshot_id=source, changes=[UnitChange.create(unit, n)])
                for n in names
            ]

    optimizer = ConcurrentPopOptimizer(DummyAdapter())

    def evaluate(adapter):
        prompt = adapter.execute()["prompt"]
        with lock:
            evaluated.append(prompt)
        return {"score": 1}

    optimizer.optimize(
        evaluate_fn=evaluate,
        objective=ScalarObjective(metric="score"),
        max_trials=3,
        execution_mode="concurrent",
        max_workers=3,
        save_dir=str(tmp_path),
    )

    assert set(evaluated) >= {"cp-0", "cp-1", "cp-2"}

    state = OptimizationRunState.load_state(str(tmp_path))
    trial_ids = [r.trial_id for r in state.trial_records]
    assert len(set(trial_ids)) == len(trial_ids), "all trial IDs must be unique"
    assert sorted(trial_ids) == list(range(len(trial_ids))), "trial IDs must be a contiguous range starting at 0"


# ---------------------------------------------------------------------------
# evaluate_fn exception → failed trial, baseline remains best
# ---------------------------------------------------------------------------

def test_evaluate_fn_exception_records_failed_trial_and_keeps_baseline(tmp_path):
    call_n = {"n": 0}

    def evaluate(adapter):
        call_n["n"] += 1
        if call_n["n"] == 2:
            raise RuntimeError("eval crashed")
        return {"score": 1}

    optimizer = PromptOnlyOptimizer(DummyAdapter())
    best = optimizer.optimize(
        evaluate_fn=evaluate,
        objective=ScalarObjective(metric="score"),
        max_trials=1,
        save_dir=str(tmp_path),
    )

    assert best.execute()["prompt"] == "base"
    state = OptimizationRunState.load_state(str(tmp_path))
    failed = [r for r in state.trial_records if r.status == "failed"]
    assert len(failed) == 1
    assert "eval crashed" in failed[0].error


# ---------------------------------------------------------------------------
# target_unit_uids selects specific unit
# ---------------------------------------------------------------------------

def test_target_unit_uids_selects_specific_unit():
    class MultiTypeOptimizer(Optimizer):
        supported_unit_types: ClassVar[FrozenSet[OptimizationUnitType]] = frozenset({
            OptimizationUnitType.PROMPT, OptimizationUnitType.MEMORY,
        })

        def propose(self, state, objective, **kwargs):
            raise NotImplementedError

    optimizer = MultiTypeOptimizer(DummyAdapter(), target_unit_uids=["prompt"])
    assert optimizer.target_unit_uids == ["prompt"]
    assert len(optimizer.target_units) == 1
    assert optimizer.target_units[0].unit_type == OptimizationUnitType.PROMPT


# ---------------------------------------------------------------------------
# Multiple changes in a single proposal
# ---------------------------------------------------------------------------

def test_proposal_with_multiple_changes_updates_all_units(tmp_path):
    class MultiChangeOptimizer(Optimizer):
        supported_unit_types: ClassVar[FrozenSet[OptimizationUnitType]] = frozenset({
            OptimizationUnitType.PROMPT, OptimizationUnitType.MEMORY,
        })

        def propose(self, state, objective, **kwargs):
            prompt_unit = self.target_units_by_uid["prompt"]
            memory_unit = self.target_units_by_uid["memories"]
            source = state.best_snapshot_id or state.snapshots[-1].snapshot_id
            return OptimizationProposal(
                source_snapshot_id=source,
                changes=[
                    UnitChange.create(prompt_unit, "updated prompt"),
                    UnitChange.create(memory_unit, "new memory", operation="append"),
                ],
            )

    optimizer = MultiChangeOptimizer(DummyAdapter())

    def evaluate(adapter):
        out = adapter.execute()
        # trial gets score 2 (has memory); baseline gets score 1 (empty memories)
        return {"score": 1 + len(out["memories"])}

    best = optimizer.optimize(
        evaluate_fn=evaluate,
        objective=ScalarObjective(metric="score"),
        max_trials=1,
        save_dir=str(tmp_path),
    )

    output = best.execute()
    assert output["prompt"] == "updated prompt"
    assert "new memory" in output["memories"]


# ---------------------------------------------------------------------------
# max_proposals limits how many proposals batch_propose should generate
# ---------------------------------------------------------------------------

def test_max_proposals_is_passed_to_batch_propose(tmp_path):
    received_max = []

    class TrackingPopOptimizer(Optimizer):
        supported_unit_types: ClassVar[FrozenSet[OptimizationUnitType]] = frozenset({OptimizationUnitType.PROMPT})

        def propose(self, state, objective, **kwargs):
            raise NotImplementedError

        def batch_propose(self, state, objective, max_proposals=None, **kwargs):
            received_max.append(max_proposals)
            unit = self.target_units_by_uid["prompt"]
            source = state.snapshots[-1].snapshot_id
            n = min(5, max_proposals) if max_proposals is not None else 5
            return [
                OptimizationProposal(source_snapshot_id=source, changes=[UnitChange.create(unit, f"p-{i}")])
                for i in range(n)
            ]

    optimizer = TrackingPopOptimizer(DummyAdapter())
    optimizer.optimize(
        evaluate_fn=lambda a: {"score": 1},
        objective=ScalarObjective(metric="score"),
        max_trials=3,
        save_dir=str(tmp_path),
    )

    assert received_max[0] == 3
    state = OptimizationRunState.load_state(str(tmp_path))
    assert state.current_step == 3


# ---------------------------------------------------------------------------
# keep_trial_workspaces=False deletes workspace after evaluation
# ---------------------------------------------------------------------------

def test_keep_trial_workspaces_false_deletes_workspace(tmp_path):
    optimizer = SkillCodeOptimizer(SkillFileAdapter())

    def evaluate(adapter):
        return {"score": int("new_skill" in adapter.execute()["content"])}

    optimizer.optimize(
        evaluate_fn=evaluate,
        objective=ScalarObjective(metric="score"),
        max_trials=1,
        save_dir=str(tmp_path),
        keep_trial_workspaces=False,
    )

    state = OptimizationRunState.load_state(str(tmp_path))
    non_baseline = [r for r in state.trial_records if r.trial_id != 0]
    for record in non_baseline:
        assert record.workspace_dir is not None
        assert not os.path.exists(record.workspace_dir), "workspace should be deleted when keep=False"


# ---------------------------------------------------------------------------
# Resume from checkpoint skips already-evaluated trials
# ---------------------------------------------------------------------------

def test_resume_from_checkpoint_skips_evaluated_trials(tmp_path):
    eval_count = {"n": 0}

    def evaluate(adapter):
        eval_count["n"] += 1
        return {"score": eval_count["n"]}

    optimizer1 = PromptOnlyOptimizer(DummyAdapter())
    optimizer1.optimize(
        evaluate_fn=evaluate,
        objective=ScalarObjective(metric="score"),
        max_trials=2,
        save_dir=str(tmp_path),
    )
    count_after_first_run = eval_count["n"]  # baseline + 2 trials = 3

    optimizer2 = PromptOnlyOptimizer(DummyAdapter())
    optimizer2.optimize(
        evaluate_fn=evaluate,
        objective=ScalarObjective(metric="score"),
        max_trials=3,
        resume_from=str(tmp_path),
        save_dir=str(tmp_path),
    )

    assert eval_count["n"] == count_after_first_run + 1, "resume should only evaluate the remaining 1 trial"
    state = OptimizationRunState.load_state(str(tmp_path))
    assert state.current_step == 3


# ---------------------------------------------------------------------------
# ParetoObjective: dominance logic and best-record selection
# ---------------------------------------------------------------------------

def test_pareto_objective_dominance():
    obj = ParetoObjective(metrics=["accuracy", "speed"], directions=["maximize", "maximize"])

    # a dominates b: better on accuracy, equal on speed
    assert obj.is_better({"accuracy": 0.9, "speed": 100}, {"accuracy": 0.8, "speed": 100})
    # a does not dominate b: worse on speed even though better on accuracy
    assert not obj.is_better({"accuracy": 0.9, "speed": 80}, {"accuracy": 0.8, "speed": 100})
    # a clearly dominated by b on both axes
    assert not obj.is_better({"accuracy": 0.7, "speed": 90}, {"accuracy": 0.9, "speed": 100})
    # equal on all: not strictly better
    assert not obj.is_better({"accuracy": 0.9, "speed": 100}, {"accuracy": 0.9, "speed": 100})


def test_pareto_objective_select_best_by_dominance_count():
    obj = ParetoObjective(metrics=["f1", "latency"], directions=["maximize", "minimize"])

    records = [
        TrialRecord(trial_id=1, changes=[], source_snapshot_id="s0", status="completed",
                    snapshot_id="s1", metrics={"f1": 0.9, "latency": 50}),
        TrialRecord(trial_id=2, changes=[], source_snapshot_id="s0", status="completed",
                    snapshot_id="s2", metrics={"f1": 0.7, "latency": 100}),
        TrialRecord(trial_id=3, changes=[], source_snapshot_id="s0", status="completed",
                    snapshot_id="s3", metrics={"f1": 0.8, "latency": 60}),
        TrialRecord(trial_id=4, changes=[], source_snapshot_id="s0", status="failed",
                    snapshot_id=None, metrics=None),
    ]

    best = obj.select_best_trial_record(records)
    # s1: dominates s2 (0.9>0.7, 50<100) and s3 (0.9>0.8, 50<60) → 2 dominances
    # s3: dominates s2 (0.8>0.7, 60<100) → 1 dominance
    # s2: dominates nobody → 0 dominances
    assert best.snapshot_id == "s1"


def test_pareto_objective_per_metric_directions():
    obj = ParetoObjective(metrics=["loss", "size"], directions=["minimize", "minimize"])
    assert obj.is_better({"loss": 0.1, "size": 10}, {"loss": 0.2, "size": 20})
    assert not obj.is_better({"loss": 0.1, "size": 25}, {"loss": 0.2, "size": 20})


def test_pareto_objective_missing_metric_returns_false():
    obj = ParetoObjective(metrics=["a", "b"], directions="maximize")
    assert not obj.is_better({"a": 1}, {"a": 0, "b": 0})  # b missing in a → not dominated


def test_pareto_objective_validation_errors():
    with pytest.raises(ValueError, match="non-empty"):
        ParetoObjective(metrics=[])
    with pytest.raises(ValueError, match="length"):
        ParetoObjective(metrics=["a", "b"], directions=["maximize"])


# ---------------------------------------------------------------------------
# Async optimize: sequential and concurrent paths
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_async_optimize_sequential(tmp_path):
    optimizer = PromptOnlyOptimizer(DummyAdapter())

    async def evaluate(adapter):
        return {"score": len(adapter.execute()["prompt"])}

    best = await optimizer.async_optimize(
        evaluate_fn=evaluate,
        objective=ScalarObjective(metric="score"),
        max_trials=1,
        save_dir=str(tmp_path),
    )

    assert best.execute()["prompt"] == "improved prompt"
    state = OptimizationRunState.load_state(str(tmp_path))
    assert state.current_step == 1


@pytest.mark.asyncio
async def test_async_optimize_concurrent(tmp_path):
    class AsyncPopOptimizer(Optimizer):
        supported_unit_types: ClassVar[FrozenSet[OptimizationUnitType]] = frozenset({OptimizationUnitType.PROMPT})

        def propose(self, state, objective, **kwargs):
            raise NotImplementedError

        def batch_propose(self, state, objective, max_proposals=None, **kwargs):
            unit = self.target_units_by_uid["prompt"]
            source = state.snapshots[-1].snapshot_id
            names = ["ap-0", "ap-1", "ap-2"]
            if max_proposals is not None:
                names = names[:max_proposals]
            return [
                OptimizationProposal(source_snapshot_id=source, changes=[UnitChange.create(unit, n)])
                for n in names
            ]

    optimizer = AsyncPopOptimizer(DummyAdapter())

    async def evaluate(adapter):
        return {"score": 1}

    await optimizer.async_optimize(
        evaluate_fn=evaluate,
        objective=ScalarObjective(metric="score"),
        max_trials=3,
        execution_mode="concurrent",
        max_workers=3,
        save_dir=str(tmp_path),
    )

    state = OptimizationRunState.load_state(str(tmp_path))
    assert state.current_step == 3
    trial_ids = [r.trial_id for r in state.trial_records]
    assert len(set(trial_ids)) == len(trial_ids), "concurrent async trials must have unique IDs"


# ---------------------------------------------------------------------------
# async_batch_propose override is used in async_optimize (not sync batch_propose)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_async_batch_propose_override_is_used_in_async_optimize(tmp_path):
    sync_calls = {"n": 0}

    class AsyncProposalOptimizer(Optimizer):
        supported_unit_types: ClassVar[FrozenSet[OptimizationUnitType]] = frozenset({OptimizationUnitType.PROMPT})

        def propose(self, state, objective, **kwargs):
            raise NotImplementedError

        def batch_propose(self, state, objective, max_proposals=None, **kwargs):
            sync_calls["n"] += 1
            raise AssertionError("sync batch_propose must not be called from async_optimize")

        async def async_batch_propose(self, state, objective, max_proposals=None, **kwargs):
            unit = self.target_units_by_uid["prompt"]
            source = state.snapshots[-1].snapshot_id
            return [OptimizationProposal(source_snapshot_id=source, changes=[UnitChange.create(unit, "async-p")])]

    optimizer = AsyncProposalOptimizer(DummyAdapter())

    async def evaluate(adapter):
        # baseline prompt="base" (len 4), trial prompt="async-p" (len 7) → trial wins
        return {"score": len(adapter.execute()["prompt"])}

    best = await optimizer.async_optimize(
        evaluate_fn=evaluate,
        objective=ScalarObjective(metric="score"),
        max_trials=1,
        save_dir=str(tmp_path),
    )

    assert sync_calls["n"] == 0
    assert best.execute()["prompt"] == "async-p"
