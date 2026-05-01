import abc
import json
import os
import shutil
from dataclasses import dataclass, field
from uuid import uuid4
from pydantic import Field, field_validator
from typing import final, List, Any, Optional, Dict, Literal, Iterable

from ...core.module import BaseModule
from .base import ChangeOperation, OptimizationUnit, OptimizationUnitType, UnitChange, ValidationResult


class SnapShot(BaseModule):
    snapshot_id: str = Field(default_factory=lambda: uuid4().hex[:8], description="Unique identifier for the snapshot")
    unit_values: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Mapping from unit uid to its current value; covers only the optimizable units declared by the adapter. All values must be JSON-serializable (str, int, float, bool, None, list, dict) so snapshots can be persisted and resumed without data loss.")
    program_config: Optional[Dict[str, Any]] = Field(default=None, description="Complete adapter-defined configuration required to fully reconstruct the underlying program (e.g. model settings, pipeline paths, non-optimizable params). Opaque to the framework; populated and consumed exclusively by the adapter's take_snapshot / from_snapshot. Set to None if unit_values alone is sufficient for reconstruction. Must be JSON-serializable (str, int, float, bool, None, list, dict) — non-serializable objects will be rejected at construction time.")

    @field_validator("unit_values", mode="before")
    @classmethod
    def _require_unit_values_json_safe(cls, v: Any) -> Any:
        if not isinstance(v, dict):
            return v  # let pydantic's type check handle non-dict
        for uid, value in v.items():
            try:
                json.dumps(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"unit_values[{uid!r}] must be JSON-serializable; serialization failed: {exc}"
                ) from exc
        return v

    @field_validator("program_config", mode="before")
    @classmethod
    def _require_program_config_json_safe(cls, v: Any) -> Any:
        if v is None:
            return v
        try:
            json.dumps(v)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"program_config must be JSON-serializable; serialization failed: {exc}"
            ) from exc
        return v


@dataclass
class ApplyResult:
    """Return value of ProgramAdapter.apply, carrying either a successful (adapter, snapshot) pair or a failure description."""
    status: Literal["success", "failed"]
    adapter: Optional["ProgramAdapter"] = None
    snapshot: Optional[SnapShot] = None
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.status == "success"


@dataclass
class TrialWorkspace:
    """Filesystem sandbox for one optimization trial."""
    root_dir: str
    trial_id: int
    source_snapshot_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)
    keep: Optional[bool] = True

    @classmethod
    def create(
        cls,
        root_dir: str,
        trial_id: int,
        source_snapshot_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        keep: Optional[bool] = True,
    ) -> "TrialWorkspace":
        workspace = cls(
            root_dir=os.path.abspath(root_dir),
            trial_id=trial_id,
            source_snapshot_id=source_snapshot_id,
            metadata=metadata or {},
            keep=keep,
        )
        os.makedirs(workspace.root_dir, exist_ok=True)
        return workspace

    def path(self, *parts: str) -> str:
        """Return an absolute path inside this workspace."""
        path = os.path.abspath(os.path.join(self.root_dir, *parts))
        if os.path.commonpath([self.root_dir, path]) != self.root_dir:
            raise ValueError(f"Workspace path escapes root_dir: {path}")
        return path

    def ensure_dir(self, *parts: str) -> str:
        """Create and return a directory inside the workspace."""
        path = self.path(*parts)
        os.makedirs(path, exist_ok=True)
        return path

    def cleanup(self) -> None:
        """Remove the workspace when `keep` is False."""
        if not self.keep and os.path.exists(self.root_dir):
            shutil.rmtree(self.root_dir)

    def as_artifact(self) -> Dict[str, Any]:
        return {
            "root_dir": self.root_dir,
            "trial_id": self.trial_id,
            "source_snapshot_id": self.source_snapshot_id,
            "metadata": self.metadata,
            "kept": self.keep,
        }


class ProgramAdapter(abc.ABC):

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)

        # Ensure that subclasses implement either execute() or async_execute()
        has_execute = cls.execute is not ProgramAdapter.execute
        has_async_execute = cls.async_execute is not ProgramAdapter.async_execute
        if not has_execute and not has_async_execute:
            raise TypeError(
                f"{cls.__name__} must implement execute() or async_execute()."
            )

    def _validate_units(self, units: List[OptimizationUnit]) -> List[OptimizationUnit]:
        if not units:
            raise ValueError(f"{self.__class__.__name__} requires at least one optimization unit.")
        if not all(isinstance(unit, OptimizationUnit) for unit in units):
            raise TypeError("register_units() must return a list of OptimizationUnit objects.")

        unit_ids = [unit.uid for unit in units]
        if len(unit_ids) != len(set(unit_ids)):
            raise ValueError("OptimizationUnit uid values must be unique.")

        unit_names = [unit.name for unit in units]
        if len(unit_names) != len(set(unit_names)):
            raise ValueError("OptimizationUnit name values must be unique.")

        return units

    @property
    def units(self) -> List[OptimizationUnit]:
        if not hasattr(self, "_units"):
            self._units = self._validate_units(self.register_units())
        return self._units

    def get_unit(self, uid: str) -> Optional[OptimizationUnit]:
        """Return a registered optimization unit by uid, or None if it is unknown."""
        for unit in self.units:
            if unit.uid == uid:
                return unit
        return None

    def select_units(
        self,
        *,
        unit_types: Optional[Iterable[OptimizationUnitType]] = None,
        uids: Optional[Iterable[str]] = None,
    ) -> List[OptimizationUnit]:
        """
        Return registered units filtered by type and/or uid.

        Args:
            unit_types: Optional set of unit types to include.
            uids: Optional set of unit uids to include.

        Returns:
            Matching units in adapter registration order.
        """
        unit_type_set = set(unit_types) if unit_types is not None else None
        uid_set = set(uids) if uids is not None else None
        return [
            unit for unit in self.units
            if (unit_type_set is None or unit.unit_type in unit_type_set)
            and (uid_set is None or unit.uid in uid_set)
        ]

    def fingerprint(self) -> Dict[str, Any]:
        """
        Return a stable adapter compatibility fingerprint for persisted optimization runs.

        Subclasses may override when reconstruction depends on additional adapter-level
        contracts. The default intentionally avoids live object identity and includes only
        declared optimization-unit shape.
        """
        return {
            "adapter_class": f"{self.__class__.__module__}.{self.__class__.__name__}",
            "units": [
                {
                    "uid": unit.uid,
                    "name": unit.name,
                    "unit_type": unit.unit_type.value,
                    "allowed_operations": list(unit.allowed_operations),
                }
                for unit in self.units
            ],
        }

    def _validate_changes(self, changes: List[UnitChange]) -> None:
        if not isinstance(changes, list):
            raise TypeError("changes must be a list of UnitChange objects.")
        if not all(isinstance(change, UnitChange) for change in changes):
            raise TypeError("changes must be a list of UnitChange objects.")

        units_by_uid = {unit.uid: unit for unit in self.units}
        unknown_uids = [change.uid for change in changes if change.uid not in units_by_uid]
        if unknown_uids:
            raise ValueError(
                "Changes reference unknown OptimizationUnit uid values: "
                f"{', '.join(unknown_uids)}."
            )

        for change in changes:
            unit = units_by_uid[change.uid]
            UnitChange.validate_value(change.value, unit, operation=change.operation)

    @property
    def workspace(self) -> Optional[TrialWorkspace]:
        """Workspace bound to this adapter for the current trial, if any."""
        return getattr(self, "_workspace", None)

    def bind_workspace(self, workspace: Optional[TrialWorkspace]) -> None:
        """Attach a trial workspace to this adapter before validation/evaluation."""
        self._workspace = workspace

    def prepare_workspace(
        self,
        workspace: TrialWorkspace,
        snapshot: SnapShot,
        **kwargs,
    ) -> None:
        """
        Optional hook to materialize file-backed state into a trial workspace.

        File/code/skills adapters can override this to write snapshot contents into
        `workspace` and then execute against those isolated files.
        """
        pass

    def validate_trial(
        self,
        snapshot: SnapShot,
        changes: List[UnitChange],
        workspace: Optional[TrialWorkspace] = None,
        **kwargs,
    ) -> List[ValidationResult]:
        """
        Optional validation pipeline run after apply/workspace preparation and before evaluation.

        Override this for static checks, import checks, smoke tests, schema checks, or
        adapter-specific consistency validation. The engine records all returned results
        and skips evaluation when any result has status="failed".
        """
        return []

    async def async_validate_trial(
        self,
        snapshot: SnapShot,
        changes: List[UnitChange],
        workspace: Optional[TrialWorkspace] = None,
        **kwargs,
    ) -> List[ValidationResult]:
        """Async variant of `validate_trial`."""
        return self.validate_trial(snapshot, changes, workspace=workspace, **kwargs)

    def on_task_begin(self, task: Any = None, **kwargs) -> Any:
        """Optional per-task lifecycle hook used by evaluators/agents."""
        return None

    def on_step(self, step: Any = None, **kwargs) -> Any:
        """Optional per-step lifecycle hook used by evaluators/agents."""
        return None

    def on_task_end(self, trajectory: Any = None, result: Any = None, **kwargs) -> Any:
        """Optional per-task completion hook used by evaluators/agents."""
        return None

    async def async_on_task_begin(self, task: Any = None, **kwargs) -> Any:
        return self.on_task_begin(task=task, **kwargs)

    async def async_on_step(self, step: Any = None, **kwargs) -> Any:
        return self.on_step(step=step, **kwargs)

    async def async_on_task_end(self, trajectory: Any = None, result: Any = None, **kwargs) -> Any:
        return self.on_task_end(trajectory=trajectory, result=result, **kwargs)

    @staticmethod
    def _recursive_merge(base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
        merged = dict(base)
        for key, value in patch.items():
            if isinstance(merged.get(key), dict) and isinstance(value, dict):
                merged[key] = ProgramAdapter._recursive_merge(merged[key], value)
            else:
                merged[key] = value
        return merged

    @staticmethod
    def _append_text_block(current: str, value: str) -> str:
        if not current:
            return value
        if current.endswith("\n") or value.startswith("\n"):
            return f"{current}{value}"
        return f"{current}\n{value}"

    @staticmethod
    def apply_change_to_value(current_value: Any, change: UnitChange) -> Any:
        """
        Apply one standard UnitChange operation to a single unit value.

        Domain-specific operations should be handled by subclass `merge_changes`.
        This helper intentionally covers only generic value/file/code semantics.
        """
        operation = change.operation
        value = change.value

        if operation == ChangeOperation.REPLACE.value:
            return value
        if operation == ChangeOperation.NOOP.value:
            return current_value
        if operation == ChangeOperation.PATCH.value:
            if not isinstance(current_value, dict) or not isinstance(value, dict):
                raise ValueError(f"Operation '{operation}' requires dict current value and dict payload.")
            return ProgramAdapter._recursive_merge(current_value, value)
        if operation == ChangeOperation.APPEND.value:
            if isinstance(current_value, list):
                return [*current_value, value]
            if isinstance(current_value, str) and isinstance(value, str):
                return ProgramAdapter._append_text_block(current_value, value)
            raise ValueError("Operation 'append' requires a list current value or string current value with string payload.")
        if operation == ChangeOperation.EXTEND.value:
            if isinstance(current_value, list) and isinstance(value, list):
                return [*current_value, *value]
            if isinstance(current_value, dict) and isinstance(value, dict):
                return {**current_value, **value}
            if isinstance(current_value, str) and isinstance(value, str):
                return ProgramAdapter._append_text_block(current_value, value)
            raise ValueError("Operation 'extend' requires matching list, dict, or string values.")
        if operation == ChangeOperation.DELETE.value:
            if isinstance(current_value, dict):
                keys = value
                if isinstance(value, dict):
                    keys = value.get("keys", value.get("key"))
                if not isinstance(keys, list):
                    keys = [keys]
                return {key: item for key, item in current_value.items() if key not in keys}
            if isinstance(current_value, list):
                if isinstance(value, int):
                    return [item for idx, item in enumerate(current_value) if idx != value]
                if isinstance(value, list) and all(isinstance(idx, int) for idx in value):
                    indices = set(value)
                    return [item for idx, item in enumerate(current_value) if idx not in indices]
                return [item for item in current_value if item != value]
            if isinstance(current_value, str) and isinstance(value, str):
                return current_value.replace(value, "")
            raise ValueError(f"Operation '{operation}' is not supported for {type(current_value).__name__} values.")

        raise ValueError(
            f"Operation '{operation}' has no default merge semantics. "
            "Handle it in the adapter's merge_changes method."
        )

    def merge_unit_values(self, snapshot: SnapShot, changes: List[UnitChange]) -> Dict[str, Any]:
        """
        Apply standard change operations to snapshot.unit_values and return a new dict.

        Subclasses can call this from `merge_changes` when their units use standard
        value/file/code operations.
        """
        values = dict(snapshot.unit_values)
        for change in changes:
            values[change.uid] = self.apply_change_to_value(values.get(change.uid), change)
        return values

    @final
    def apply(self, snapshot: SnapShot, changes: List[UnitChange], **kwargs) -> ApplyResult:
        """
        Apply a changeset on top of a snapshot and return an ApplyResult.

        Subclasses should NOT override this method. Use `pre_apply_hook` / `post_apply_hook`
        for lifecycle customization, and `from_snapshot` / `merge_changes` for the core logic.

        The original adapter and the input snapshot are never modified. On success the returned
        ApplyResult carries both the new adapter and the SnapShot produced by `merge_changes`,
        so the caller never needs to call `take_snapshot()` again to obtain a consistent snapshot.

        Args:
            snapshot: The baseline snapshot to apply changes on top of.
            changes:  List of UnitChange objects specifying which units to update.

        Returns:
            ApplyResult with status="success" and (adapter, snapshot) on success, or
            status="failed" and an error message string on any exception.
        """
        try:
            processed_changes = self.pre_apply_hook(snapshot, changes, **kwargs)
            self._validate_changes(processed_changes)
            new_snapshot = self.merge_changes(snapshot, processed_changes, **kwargs)
            if not isinstance(new_snapshot, SnapShot):
                raise TypeError(
                    f"merge_changes() must return a SnapShot instance, "
                    f"got {type(new_snapshot).__name__}."
                )
            new_adapter = self.from_snapshot(new_snapshot, **kwargs)
            if not isinstance(new_adapter, ProgramAdapter):
                raise TypeError(
                    f"from_snapshot() must return a ProgramAdapter instance, "
                    f"got {type(new_adapter).__name__}."
                )
            self.post_apply_hook(new_adapter, processed_changes, **kwargs)
            return ApplyResult(status="success", adapter=new_adapter, snapshot=new_snapshot)
        except Exception as e:
            return ApplyResult(status="failed", error=str(e))

    @final
    def load_snapshot(self, snapshot: SnapShot, **kwargs) -> "ProgramAdapter":
        """
        Reconstruct a new adapter from a snapshot, delegating to `from_snapshot`.

        Called once at the end of `Optimizer.optimize()` to restore the best-found adapter.
        The current adapter is not modified.

        Args:
            snapshot: A SnapShot previously produced by `take_snapshot`.
            **kwargs: Forwarded to `from_snapshot`.

        Returns:
            A new ProgramAdapter instance whose state matches the given snapshot.
        """
        return self.from_snapshot(snapshot, **kwargs)

    def execute(self, *args, **kwargs) -> Any:
        raise NotImplementedError(f"{self.__class__.__name__} does not implement execute().")

    async def async_execute(self, *args, **kwargs) -> Any:
        return self.execute(*args, **kwargs)

    @abc.abstractmethod
    def register_units(self) -> List[OptimizationUnit]:
        """
        Declare the optimization units exposed by this adapter.

        Called once during initialization; the result is cached in `self._units`.
        Each unit describes one optimizable component (e.g. a prompt string, model name,
        temperature value).

        Subclasses must ensure:
            - At least one unit is returned.
            - All unit `uid` and `name` values are unique.
            - If a unit has a `json_schema`, it reflects the expected type/structure
              of the value in a corresponding `UnitChange`.

        Returns:
            A list of OptimizationUnit objects.

        Example:
            def register_units(self) -> List[OptimizationUnit]:
                return [
                    OptimizationUnit(name="system_prompt", unit_type=OptimizationUnitType.PROMPT),
                    OptimizationUnit(name="model_name", unit_type=OptimizationUnitType.MODEL),
                ]
        """
        raise NotImplementedError

    @abc.abstractmethod
    def take_snapshot(self) -> SnapShot:
        """
        Capture the current state of the program as a SnapShot.

        Must populate `unit_values` as a dict mapping each registered unit's `uid`
        to its current value. If `from_snapshot` needs additional context beyond
        `unit_values` to reconstruct the adapter (e.g. non-optimizable model settings,
        pipeline paths), also populate `program_config`. All values in `program_config`
        must be JSON-serializable (str, int, float, bool, None, list, dict) so the
        snapshot can be persisted and restored without data loss.

        Returns:
            A SnapShot capturing the current value of every registered optimization unit.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def merge_changes(self, snapshot: SnapShot, changes: List[UnitChange], **kwargs) -> SnapShot:
        """
        Produce a new SnapShot reflecting the proposed changes applied on top of the given snapshot.

        The input snapshot must remain unmodified. Return a fresh SnapShot whose
        `unit_values` (and optionally `program_config`) reflect the result of applying
        each change. The simplest implementation is a plain dict update, but you may
        apply custom merging semantics (e.g. additive updates, cross-unit dependencies,
        config recomputation).

        Called by `apply` after validation and before `from_snapshot`.

        Args:
            snapshot: The baseline snapshot to merge changes into.
            changes:  Validated list of UnitChange objects (all uids are guaranteed registered).
            **kwargs: Optional subclass-specific arguments forwarded from `apply`.

        Returns:
            A new SnapShot whose state reflects the applied changes.

        Example:
            def merge_changes(self, snapshot: SnapShot, changes: List[UnitChange], **kwargs) -> SnapShot:
                new_values = dict(snapshot.unit_values)
                for change in changes:
                    new_values[change.uid] = change.value
                return SnapShot(unit_values=new_values, program_config=snapshot.program_config)
        """
        raise NotImplementedError

    @abc.abstractmethod
    def from_snapshot(self, snapshot: SnapShot, **kwargs) -> "ProgramAdapter":
        """
        Build a new ProgramAdapter whose state matches the given SnapShot.

        This is the single construction point shared by `apply` (produces a trial adapter)
        and `load_snapshot` (restores the best adapter after optimization). Keeping all
        reconstruction logic here avoids duplication and makes the contract clear:
        given a snapshot, produce a fully initialized adapter.

        The current adapter must remain unmodified. Create and return a fresh instance
        initialized from `snapshot.unit_values` (and `snapshot.program_config` if needed).

        Args:
            snapshot: A SnapShot previously produced by `take_snapshot`.
            **kwargs: Optional subclass-specific arguments.

        Returns:
            A new ProgramAdapter instance whose state matches the given snapshot.
        """
        raise NotImplementedError

    def pre_apply_hook(self, snapshot: SnapShot, changes: List[UnitChange], **kwargs) -> List[UnitChange]:
        """
        Optional hook called before changes are validated and merged.
        Override to preprocess or transform the changeset.
        Returns the changeset unchanged by default.

        Args:
            snapshot: The baseline snapshot to merge changes into.
            changes: The original list of UnitChange objects.
            **kwargs: Optional subclass-specific arguments forwarded from `apply`.

        Returns:
            A (possibly modified) list of UnitChange objects.
        """
        return changes
    
    def post_apply_hook(self, new_adapter: "ProgramAdapter", changes: List[UnitChange], **kwargs) -> None:  # noqa: ARG002
        """
        Optional hook called after `from_snapshot`.
        Override to perform post-processing on the newly created adapter `new_adapter`.

        Args:
            new_adapter: The ProgramAdapter instance returned by `from_snapshot`.
            changes:     The list of UnitChange objects that were applied (after preprocessing using `pre_apply_hook`).
            **kwargs:    Optional subclass-specific arguments forwarded from `apply`.
        """
        pass
