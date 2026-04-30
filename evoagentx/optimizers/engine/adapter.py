import abc
from dataclasses import dataclass
from uuid import uuid4
from pydantic import Field
from typing import final, List, Any, Optional, Dict, Literal

from ...core.module import BaseModule
from .base import OptimizationUnit, UnitChange


class SnapShot(BaseModule):
    snapshot_id: str = Field(default_factory=lambda: uuid4().hex[:8], description="Unique identifier for the snapshot")
    unit_values: Dict[str, Any] = Field(default_factory=dict, description="Mapping from unit uid to its current value; covers only the optimizable units declared by the adapter")
    program_config: Optional[Dict[str, Any]] = Field(default=None, description="Complete adapter-defined configuration required to fully reconstruct the underlying program (e.g. model settings, pipeline paths, non-optimizable params). Opaque to the framework; populated and consumed exclusively by the adapter's take_snapshot / from_snapshot. Set to None if unit_values alone is sufficient for reconstruction.")


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
            UnitChange.validate_value(change.value, unit)

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
            new_adapter = self.from_snapshot(new_snapshot, **kwargs)
            if not isinstance(new_adapter, ProgramAdapter):
                raise TypeError(
                    f"from_snapshot() must return a ProgramAdapter instance, "
                    f"got {type(new_adapter).__name__}."
                )
            self.post_apply_hook(new_adapter, processed_changes)
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
                    OptimizationUnit(name="system_prompt", unitType=OptimizationUnitType.PROMPT),
                    OptimizationUnit(name="model_name", unitType=OptimizationUnitType.MODEL),
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
        pipeline paths), also populate `program_config`.

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
