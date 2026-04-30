import abc
from enum import Enum
from pydantic import Field, model_validator
from jsonschema import validate, ValidationError
from typing import Any, Callable, Dict, List, Optional, Literal

from ...core.module import BaseModule
from .decorators import EntryPoint
from .registry import ParamRegistry

class OptimizationUnitType(str, Enum):
    # Generic structured field or scalar parameter that has no more specific type.
    FIELD = "field"

    # Prompt text or prompt template content used by an agent or workflow.
    PROMPT = "prompt"

    # Model selection or model configuration exposed as an optimizable unit.
    MODEL = "model"

    # Memory contents, retrieval policy, or memory-store configuration.
    MEMORY = "memory"

    # File-backed artifact whose semantics are adapter-defined.
    FILE = "file"

    # Source-code artifact; use metadata to mark domain roles such as skills/tools.
    CODE = "code"


UnitChangeOperation = str


class ChangeOperation(str, Enum):
    """Minimal standard value operations understood by optimizer engine helpers."""

    # Replace the current unit value with the provided payload.
    REPLACE = "replace"

    # Recursively update a dict/object-like value with fields from the payload.
    PATCH = "patch"

    # Append one payload item to a list, or append a text block to a string.
    APPEND = "append"

    # Extend a list/string with multiple payload values, or shallow-extend a dict.
    EXTEND = "extend"

    # Delete an item/key/substring identified by the payload.
    DELETE = "delete"

    # Leave the current value unchanged while still recording a proposed operation.
    NOOP = "noop"


STANDARD_CHANGE_OPERATIONS = frozenset(operation.value for operation in ChangeOperation)


class OptimizationUnit(BaseModule):
    name: str = Field(description="Name of the optimization unit; also serves as the default uid")
    unit_type: OptimizationUnitType = Field(description="Type of the optimization unit")
    uid: str = Field(default="", description="Stable unique ID for this unit across adapter reconstructions; defaults to name if not set explicitly")
    json_schema: Optional[dict] = Field(default=None, description="Optional schema for the optimization unit, used to validate the parameters associated with this unit")
    allowed_operations: List[UnitChangeOperation] = Field(default_factory=lambda: ["replace"], description="Operations this unit accepts in UnitChange. Defaults to replacement updates for backward compatibility.")
    operation_schemas: Dict[str, dict] = Field(default_factory=dict, description="Optional per-operation JSON schemas for UnitChange.value payloads. For replace operations, json_schema is used when no operation-specific schema is provided.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Adapter-defined unit metadata used by optimizers to understand semantics such as file path, prompt role, retrieval policy, constraints, or dependencies.")

    @model_validator(mode="before")
    @classmethod
    def _default_uid_to_name(cls, data: Any) -> Any:
        if isinstance(data, dict) and not data.get("uid"):
            data = {**data, "uid": data.get("name", "")}
        return data


class FileOptimizationUnit(OptimizationUnit):
    """Optimization unit for file-backed artifacts such as prompts, skills, or configs."""
    unit_type: OptimizationUnitType = Field(default=OptimizationUnitType.FILE, description="File-backed optimization unit type")
    path: str = Field(description="Adapter-relative or workspace-relative file path for this unit")
    encoding: str = Field(default="utf-8", description="Text encoding used when materializing this file")
    content_type: str = Field(default="text", description="Logical content type, e.g. text, json, yaml, python")
    json_schema: Optional[dict] = Field(default_factory=lambda: {"type": "string"})
    allowed_operations: List[UnitChangeOperation] = Field(
        default_factory=lambda: [
            ChangeOperation.REPLACE.value,
            ChangeOperation.PATCH.value,
            ChangeOperation.APPEND.value,
            ChangeOperation.EXTEND.value,
            ChangeOperation.DELETE.value,
            ChangeOperation.NOOP.value,
        ],
        description="File units accept standard value replacement, patch, append, extend, delete, and no-op updates by default.",
    )

    @model_validator(mode="before")
    @classmethod
    def _default_file_identity(cls, data: Any) -> Any:
        if isinstance(data, dict):
            path = data.get("path", "")
            if path:
                data = {
                    **data,
                    "name": data.get("name") or path,
                    "uid": data.get("uid") or path,
                }
        return data


class CodeOptimizationUnit(FileOptimizationUnit):
    """Optimization unit for source-code artifacts such as generated skills or providers."""
    unit_type: OptimizationUnitType = Field(default=OptimizationUnitType.CODE, description="Code-backed optimization unit type")
    language: str = Field(default="python", description="Programming language of the code artifact")
    entrypoint: Optional[str] = Field(default=None, description="Optional callable/module entrypoint exposed by this code unit")
    validation_commands: List[str] = Field(
        default_factory=list,
        description="Optional adapter-owned validation commands or labels. The engine records this metadata but does not execute commands automatically.",
    )
    allowed_operations: List[UnitChangeOperation] = Field(
        default_factory=lambda: [
            ChangeOperation.REPLACE.value,
            ChangeOperation.PATCH.value,
            ChangeOperation.APPEND.value,
            ChangeOperation.EXTEND.value,
            ChangeOperation.DELETE.value,
            ChangeOperation.NOOP.value,
        ],
        description="Code units accept standard value replacement, patch, append, extend, delete, and no-op updates by default.",
    )


class UnitChange(BaseModule):
    uid: str = Field(description="Unique ID of the optimization unit to change")
    value: Any = Field(description="Payload for this change. For operation='replace', this is the new unit value. Other operations may interpret it as a patch, appended item, deletion selector, etc.")
    old_value: Optional[Any] = Field(default=None, description="Previous value of the optimization unit before change (optional)")
    operation: UnitChangeOperation = Field(default="replace", description="Operation to apply to the target unit. Adapters define the concrete merge semantics for non-replace operations.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Optimizer-defined metadata for provenance, rationale, confidence, or algorithm-specific bookkeeping.")

    @staticmethod
    def validate_value(value: Any, unit: OptimizationUnit, operation: UnitChangeOperation = "replace") -> None:
        """
        Validate a value against the json_schema of the given OptimizationUnit.

        Args:
            value: The candidate value to validate.
            unit: The OptimizationUnit whose json_schema defines the validation rules.
            operation: The change operation whose payload is being validated.

        Raises:
            ValueError: If the value does not conform to the unit's json_schema.
        """
        if operation not in unit.allowed_operations:
            raise ValueError(
                f"Operation '{operation}' is not allowed for unit '{unit.name}' "
                f"(uid={unit.uid}). Allowed operations: {unit.allowed_operations}"
            )

        schema = unit.operation_schemas.get(operation)
        if schema is None and operation == "replace":
            schema = unit.json_schema

        if schema is None:
            return # No schema means no validation needed
        
        try:
            validate(instance=value, schema=schema)
        except ValidationError as e:
            raise ValueError(
                f"Value for unit '{unit.name} (uid={unit.uid}) "
                f"does not conform to the schema for operation '{operation}': {e.message}"
            )
            
    @classmethod
    def create(
        cls,
        unit: OptimizationUnit,
        new_value: Any = None,
        old_value: Optional[Any] = None,
        operation: UnitChangeOperation = "replace",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "UnitChange":
        """
        Preferred constructor for optimizer-side change construction.
        Validates the value against the unit's json_schema before constructing the instance, ensuring changes applied to a unit are always valid. 

        Args:
            unit: The OptimizationUnit being changed.
            new_value: The new value to apply to the unit.
            old_value: The previous value of the unit before change (optional).
            operation: The operation to apply to the unit.
            metadata: Optional optimizer-defined metadata for this change.
        
        Returns:
            A UnitChange instance with the provided values if validation passes.
        """
        cls.validate_value(new_value, unit, operation=operation)
        return cls(uid=unit.uid, value=new_value, old_value=old_value, operation=operation, metadata=metadata or {})


class OptimizationProposal(BaseModule):
    source_snapshot_id: str = Field(description="The snapshot_id of the base snapshot that the proposed changes will be applied on")
    changes: List[UnitChange] = Field(description="List of proposed changes to apply on the base snapshot for the next trial")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Optimizer-defined metadata for the proposal, such as parent IDs, generation number, sampler params, or rationale.")


class EvaluationResult(BaseModule):
    """Structured result returned by evaluation functions."""
    metrics: Dict[str, Any] = Field(description="Objective-facing metrics for this trial.")
    traces: List[Any] = Field(default_factory=list, description="Optional execution traces or trajectory data collected during evaluation.")
    artifacts: Dict[str, Any] = Field(default_factory=dict, description="Optional produced artifacts such as summaries, retrieved memories, file paths, generated skills, or debug payloads.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Optional evaluation metadata such as split name, cost, latency, seeds, or evaluator configuration.")


ValidationStatus = Literal["passed", "failed", "skipped"]


class ValidationResult(BaseModule):
    """Result from one adapter-defined validation step before a trial is evaluated."""
    validator: str = Field(description="Name of the validation step")
    status: ValidationStatus = Field(description="Validation status")
    message: Optional[str] = Field(default=None, description="Human-readable validation message")
    details: Dict[str, Any] = Field(default_factory=dict, description="Structured validation details")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Validator-owned metadata")

    @property
    def ok(self) -> bool:
        return self.status in ("passed", "skipped")


class TrialRecord(BaseModule):
    """Data model for recording the results of a single optimization trial."""
    trial_id: int = Field(description="Unique identifier for the trial")
    changes: List[UnitChange] = Field(description="List of changes applied in this trial")
    source_snapshot_id: str = Field(description="The snapshot_id of the base snapshot that the proposed changes were applied on in this trial")
    status: Literal["completed", "failed"] = Field(description="Status of the trial")

    snapshot_id: Optional[str] = Field(default=None, description="The snapshot_id of the program state after applying the changes in this trial")
    metrics: Optional[Dict[str, Any]] = Field(default=None, description="Evaluation metrics collected for this trial")
    traces: List[Any] = Field(default_factory=list, description="Execution traces or trajectory data collected during evaluation.")
    artifacts: Dict[str, Any] = Field(default_factory=dict, description="Artifacts produced or consumed during evaluation.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Trial-level metadata, including structured evaluator metadata or optimizer bookkeeping.")
    validation_results: List[ValidationResult] = Field(default_factory=list, description="Validation results collected before evaluation.")
    workspace_dir: Optional[str] = Field(default=None, description="Trial workspace directory used to isolate file-backed artifacts, if any.")
    error: Optional[str] = Field(default=None, description="Error message if the trial failed")


class BaseOptimizer(abc.ABC):
    # def __init__(
    #     self,
    #     registry: ParamRegistry,
    #     program: Callable, 
    #     evaluator: Callable[[Dict[str, Any]], float],
    #     **kwargs

    def __init__(
        self,
        registry: ParamRegistry,
        program: Callable[..., Dict[str, Any]] = None,
        evaluator: Optional[Callable[..., Any]] = None,
    ):
        """
        Abstract base class for optimization routines.

        Parameters:
        - registry (ParamRegistry): parameter access layer
        - evaluator (Callable): function that evaluates the result dict and returns a float
        """
        self.program = program
        self.registry = registry
        self.program = program
        self.evaluator = evaluator
        
    def get_param(self, name: str) -> Any:
        """Retrieve the current value of a parameter by name."""
        return self.registry.get(name)

    def set_param(self, name: str, value: Any):
        """Set the value of a parameter by name."""
        self.registry.set(name, value)

    def param_names(self) -> List[str]:
        """Return the list of all registered parameter names."""
        return self.registry.names()
    
    def get_current_cfg(self) -> Dict[str, Any]:
        """Return current config as a dictionary."""
        return {name: self.get_param(name) for name in self.param_names()}

    def apply_cfg(self, cfg: Dict[str, Any]):
        """Apply a configuration dictionary to the registered parameters."""
        for k, v in cfg.items():
            if k in self.registry.fields:
                self.registry.set(k, v)

    @abc.abstractmethod
    def optimize(self):
        """
        Abstract optimization loop. Should be implemented by subclasses.

        Parameters:
        - program_entry: callable that runs the program and returns output dict

        Returns:
        - (best_cfg, history): best config found and full search history
        """
        if self.program is None:
            self.program = EntryPoint.get_entry()
        if self.program is None:
            raise RuntimeError("No entry function provided or registered.")
        print(f"Starting optimization from entry: {self.program.__name__}")
        raise NotImplementedError
