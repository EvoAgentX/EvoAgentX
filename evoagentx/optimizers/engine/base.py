import abc
from enum import Enum
from pydantic import Field, model_validator
from jsonschema import validate, ValidationError
from typing import Any, Callable, Dict, List, Optional

from ...core.module import BaseModule
from .decorators import EntryPoint
from .registry import ParamRegistry

class OptimizationUnitType(str, Enum):
    FIELD = "field"
    PROMPT = "prompt"
    MODEL = "model"
    MEMORY = "memory"
    SKILLS = "skills"


class OptimizationUnit(BaseModule):
    name: str = Field(description="Name of the optimization unit; also serves as the default uid")
    unitType: OptimizationUnitType = Field(description="Type of the optimization unit")
    uid: str = Field(default="", description="Stable unique ID for this unit across adapter reconstructions; defaults to name if not set explicitly")
    json_schema: Optional[dict] = Field(default=None, description="Optional schema for the optimization unit, used to validate the parameters associated with this unit")

    @model_validator(mode="before")
    @classmethod
    def _default_uid_to_name(cls, data: Any) -> Any:
        if isinstance(data, dict) and not data.get("uid"):
            data = {**data, "uid": data.get("name", "")}
        return data


class UnitChange(BaseModule):
    uid: str = Field(description="Unique ID of the optimization unit to change")
    value: Any = Field(description="New value to apply to the optimization unit")
    old_value: Optional[Any] = Field(default=None, description="Previous value of the optimization unit before change (optional)")

    @staticmethod
    def validate_value(value: Any, unit: OptimizationUnit) -> None:
        """
        Validate a value against the json_schema of the given OptimizationUnit.

        Args:
            value: The candidate value to validate.
            unit: The OptimizationUnit whose json_schema defines the validation rules.

        Raises:
            ValueError: If the value does not conform to the unit's json_schema.
        """
        if unit.json_schema is None:
            return # No schema means no validation needed
        
        try:
            validate(instance=value, schema=unit.json_schema)
        except ValidationError as e:
            raise ValueError(
                f"Value for unit '{unit.name} (uid={unit.uid}) "
                f"does not conform to its json_schema: {e.message}"
            )
            
    @classmethod
    def create(cls, unit: OptimizationUnit, new_value: Any, old_value: Optional[Any] = None) -> "UnitChange":
        """
        Preferred constructor for optimizer-side change construction.
        Validates the value against the unit's json_schema before constructing the instance, ensuring changes applied to a unit are always valid. 

        Args:
            unit: The OptimizationUnit being changed.
            new_value: The new value to apply to the unit.
            old_value: The previous value of the unit before change (optional).
        
        Returns:
            A UnitChange instance with the provided values if validation passes.
        """
        cls.validate_value(new_value, unit)
        return cls(uid=unit.uid, value=new_value, old_value=old_value)


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