import asyncio
import functools
import inspect
import threading
import time
from abc import ABC
from collections import defaultdict
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Callable, DefaultDict, Dict, List, Optional, Type, TypeVar, overload

from jsonschema import Draft202012Validator
from pydantic import Field

from ..core.decorators import atomic_method
from ..core.logging import logger
from ..core.metadata import Metadata
from ..core.module import BaseModule, MetaModule
from ..utils.utils import (
    add_dict,
    compose_decorators,
    get_cost_per_tool,
    get_provider_tool_cost,
    get_total_tool_cost,
    json_to_python_type,
)
from .tool_utils import (
    ALLOWED_TYPES,
    get_schema,
    is_valid_description,
    normalize_component_name,
)

_ToolT = TypeVar("_ToolT", bound="Tool")


class ToolMetadata(Metadata):
    tool_name: str = ""
    args: Dict[str, Any] = Field(default_factory=dict, description="The arguments passed to the tool")
    cost_breakdown: DefaultDict[str, float] = Field(default_factory=lambda: defaultdict(float), description="The cost of the tool call")
    execution_time: float = Field(default=0., description="The execution time of the tool call in seconds")
 
    def init_module(self):
        self._lock = threading.Lock()

    @property
    def cost(self) -> float:
        return sum(self.cost_breakdown.values())

    @atomic_method
    def add_cost_breakdown(self, cost_breakdown: Dict[str, float]):
        self.cost_breakdown = add_dict(self.cost_breakdown, cost_breakdown)


class ToolResult(BaseModule):    
    result: Any
    metadata: ToolMetadata

    def __await__(self):
        async def _return_self():
            return self
        return _return_self().__await__()


def _get_tool_name(func, args) -> str:
    if args:
        tool_name = getattr(args[0], "name", None)
        if tool_name:
            return tool_name
    return getattr(func, "__name__", "tool")


def ensure_tool_result(value: Any, tool_name: str, args: Dict[str, Any]) -> ToolResult:
    if isinstance(value, ToolResult):
        return value
    return ToolResult(
        result=value,
        metadata=ToolMetadata(tool_name=tool_name, args=args)
    )


class ToolUsageMetadata(Metadata):
    tool_calls: DefaultDict[str, int] = Field(default_factory=lambda: defaultdict(int), description="The number of times each tool was called")
    cost_breakdown: Dict[str, Dict[str, float]] = Field(default_factory=lambda: defaultdict(lambda: defaultdict(float)), description="Cost incurred by each tool")
    execution_time: DefaultDict[str, float] = Field(default_factory=lambda: defaultdict(float), description="The total execution time of each tool in seconds")
    history: List[ToolResult] = Field(default_factory=list, description="History of tool calls")

    def init_module(self):
        self._lock = threading.Lock()

    @property
    def total_tool_calls(self) -> int:
        return sum(self.tool_calls.values())

    @property
    def cost_per_tool(self) -> Dict[str, float]:
        return get_cost_per_tool(self.cost_breakdown)

    @property
    def total_tool_cost(self) -> float:
        return get_total_tool_cost(self.cost_breakdown)

    @property
    def openrouter_cost(self) -> float:
        return get_provider_tool_cost(self.cost_breakdown, "openrouter")

    @property
    def total_execution_time(self) -> float:
        return sum(self.execution_time.values())

    @atomic_method
    def update(self, tool_result: ToolResult):
        self._update_metadata(tool_result)
        tracker = tool_usage_tracker.get()
        if tracker is not None and tracker is not self:
            tracker._update_metadata(tool_result)

    def _update_metadata(self, tool_result: ToolResult):
        tool_name = tool_result.metadata.tool_name
        self.tool_calls[tool_name] += 1
        self.cost_breakdown[tool_name] = add_dict(
            self.cost_breakdown[tool_name],
            tool_result.metadata.cost_breakdown
        )
        self.execution_time[tool_name] += tool_result.metadata.execution_time
        self.history.append(tool_result)


tool_usage_tracker = ContextVar("tool_usage_tracker", default=None)

@contextmanager
def track_tool_usage():
    tracker = ToolUsageMetadata()
    token = tool_usage_tracker.set(tracker)
    try:
        yield tracker
    finally:
        tool_usage_tracker.reset(token)


def track_tool_metadata(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        tool_name = _get_tool_name(func, args)
        with track_tool_usage():
            result = func(*args, **kwargs)
        result = ensure_tool_result(result, tool_name, kwargs)

        tracker = tool_usage_tracker.get()
        if tracker is not None:
            tracker._update_metadata(result)
        return result
    
    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        tool_name = _get_tool_name(func, args)
        with track_tool_usage():
            result = await func(*args, **kwargs)
        result = ensure_tool_result(result, tool_name, kwargs)

        tracker = tool_usage_tracker.get()
        if tracker is not None:
            tracker._update_metadata(result)
        return result

    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    return wrapper


def track_tool_cost(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        tool_name = _get_tool_name(func, args)
        from ..models.model_utils import cost_manager, cost_tracker, track_cost
        with track_cost() as tracker:
            result = func(*args, **kwargs)
            result = ensure_tool_result(result, tool_name, kwargs)

            for cost_breakdown in tracker.tool_cost_breakdown.values():
                result.metadata.add_cost_breakdown(cost_breakdown)

            llm_cost_breakdown = {"openrouter:" + k: v for k, v in tracker.cost_per_model.items()}
            result.metadata.add_cost_breakdown(llm_cost_breakdown)
        
        cost_breakdown = result.metadata.cost_breakdown
        tracker = cost_tracker.get()

        if tracker is not None:
            tracker.update_tool_cost(tool_name, cost_breakdown)
        else:
            cost_manager.update_tool_cost(tool_name, cost_breakdown)
        return result

    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        tool_name = _get_tool_name(func, args)
        from ..models.model_utils import cost_manager, cost_tracker, track_cost
        with track_cost() as tracker:
            result = await func(*args, **kwargs)
            result = ensure_tool_result(result, tool_name, kwargs)

            for cost_breakdown in tracker.tool_cost_breakdown.values():
                result.metadata.add_cost_breakdown(cost_breakdown)

            llm_cost_breakdown = {"openrouter:" + k: v for k, v in tracker.cost_per_model.items()}
            result.metadata.add_cost_breakdown(llm_cost_breakdown)
        
        cost_breakdown = result.metadata.cost_breakdown
        tracker = cost_tracker.get()

        if tracker is not None:
            tracker.update_tool_cost(tool_name, cost_breakdown)
        else:
            cost_manager.update_tool_cost(tool_name, cost_breakdown)
        return result

    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    return wrapper


def track_tool_execution_time(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        tool_name = _get_tool_name(func, args)
        start_time = time.time()
        result = func(*args, **kwargs)
        result = ensure_tool_result(result, tool_name, kwargs)
        execution_time = time.time() - start_time
        result.metadata.execution_time = execution_time
        return result

    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        tool_name = _get_tool_name(func, args)
        start_time = time.time()
        result = await func(*args, **kwargs)
        result = ensure_tool_result(result, tool_name, kwargs)
        execution_time = time.time() - start_time
        result.metadata.execution_time = execution_time
        return result
    
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    return wrapper


def log_tool_error(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        tool_name = _get_tool_name(func, args)
        result = func(*args, **kwargs)
        result = ensure_tool_result(result, tool_name, kwargs)
        log_error(result)
        return result
    
    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        tool_name = _get_tool_name(func, args)
        result = await func(*args, **kwargs)
        result = ensure_tool_result(result, tool_name, kwargs)
        log_error(result)
        return result

    def log_error(result: ToolResult):
        tool_result = result.result
        
        if not isinstance(tool_result, dict):
            return
        
        error = tool_result.get("error")
        if error is not None:
            tool_name = result.metadata.tool_name
            logger.error(f"Tool `{tool_name}` error: {error}")
    
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    return wrapper


tool_wrapper = compose_decorators(
    track_tool_execution_time,
    log_tool_error,
    track_tool_cost,
    track_tool_metadata,
)


class ToolMetadataMeta(type):
    def __new__(mcs, name, bases, namespace, **kwargs):
        exclude = ["ToolCollection"]
        
        if "__call__" in namespace:
            if bases and bases[0].__name__ not in exclude:
                namespace["__call__"] = tool_wrapper(namespace["__call__"])
        else:
            for base in bases:
                if callable(base.__call__) and base.__name__ not in exclude:
                    namespace["__call__"] = tool_wrapper(base.__call__)
                    break

        cls = super().__new__(mcs, name, bases, namespace, **kwargs)
        return cls


class ToolMeta(ToolMetadataMeta, MetaModule):
    pass


class Tool(BaseModule, ABC, metaclass=ToolMeta):
    name: str
    description: str
    extra_description: Optional[str] = None
    inputs: Dict[str, Dict[str, Any]]
    required: Optional[List[str]] = None

    """
    inputs: {"input_name": {"type": "string", "description": "input description"}, ...}
    """

    def __init_subclass__(cls):
        super().__init_subclass__()
        cls.validate_attributes()

    def get_tool_schema(self, extra_description: bool = True) -> Dict:
        description = self.description

        if extra_description and self.extra_description is not None:
            description += "\n\n" + self.extra_description
        
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": self.inputs,
                    "required": self.required
                }
            }
        }

    @classmethod
    def validate_attributes(cls):
        required_attributes = {
            "name": str,
            "description": str,
            "inputs": dict
        }
        
        for attr, attr_type in required_attributes.items():
            if not hasattr(cls, attr):
                raise ValueError(f"Attribute '{attr}' is required")
            if not isinstance(getattr(cls, attr), attr_type):
                raise ValueError(f"Attribute '{attr}' must be of type `{attr_type.__name__}`")

        for input_name, input_content in cls.inputs.items():
            if not isinstance(input_content, dict):
                raise ValueError(f"Input '{input_name}' must be a dictionary")
            if "type" not in input_content or "description" not in input_content:
                raise ValueError(f"Input '{input_name}' must have 'type' and 'description'")
            if input_content["type"] not in ALLOWED_TYPES:
                raise ValueError(f"Input '{input_name}' must have a valid type, should be one of {ALLOWED_TYPES}")
            
            call_signature = inspect.signature(cls.__call__)
            if input_name not in call_signature.parameters:
                raise ValueError(f"Input '{input_name}' is not found in __call__")
            if call_signature.parameters[input_name].annotation != json_to_python_type[input_content["type"]]:
                raise ValueError(f"Input '{input_name}' has a type mismatch in __call__")

            try:
                Draft202012Validator.check_schema(input_content)
            except Exception as e:
                raise ValueError(f"Invalid JSON schema for input '{input_name}'") from e

        required_inputs = getattr(cls, "required", None)
        if required_inputs is not None:
            for required_input in required_inputs:
                if required_input not in cls.inputs:
                    raise ValueError(f"Required input '{required_input}' is not found in inputs")

    def init_module(self):
        desc = getattr(self, "description", None)
        if not is_valid_description(desc):
            logger.warning(
                f"Tool '{self.name}' has invalid description (missing or < 10 chars)."
            )

    def __call__(self, **kwargs) -> ToolResult:
        raise NotImplementedError("All tools must implement __call__")

    @classmethod
    def build(cls, *args, **kwargs) -> Optional["Tool"]:
        try:
            return cls(*args, **kwargs)
        except Exception as e:
            logger.warning(f"Failed to initialize {cls.__name__}: {e}")
            return None


class Toolkit(BaseModule):
    name: str
    tools: List[Tool]

    def get_tool_names(self) -> List[str]:
        return [tool.name for tool in self.tools]

    def get_tool_descriptions(self) -> List[str]:
        return [tool.description for tool in self.tools]

    def get_tool_inputs(self) -> List[Dict]:
        return [tool.inputs for tool in self.tools]

    def add_tool(self, tool: Tool):
        self.tools.append(tool)

    def remove_tool(self, tool_name: str):
        self.tools = [tool for tool in self.tools if tool.name != tool_name]

    @overload
    def get_tool(self, tool_name: str) -> "Tool": ...
    @overload
    def get_tool(self, tool_name: str, tool_cls: Type[_ToolT]) -> _ToolT: ...
    def get_tool(self, tool_name: str, tool_cls: Optional[Type["Tool"]] = None) -> "Tool":
        for tool in self.tools:
            if tool.name == tool_name:
                return tool
        raise ValueError(f"Tool '{tool_name}' not found")
    
    def init_module(self):
        raw_name = getattr(self, "name", None) or self.__class__.__name__
        self.name = normalize_component_name(raw_name, "Toolkit")
        self.tools = [t for t in (self.tools or []) if t is not None]
    
    def get_tools(self) -> List[Tool]:
        return self.tools
    
    def get_tool_schemas(self, extra_description: bool = True) -> List[Dict]:
        return [tool.get_tool_schema(extra_description=extra_description) for tool in self.tools]

    @classmethod
    def build(cls, *args, **kwargs) -> Optional["Toolkit"]:
        try:
            return cls(*args, **kwargs)
        except Exception as e:
            logger.warning(f"Failed to initialize {cls.__name__}: {e}")
            return None


class CustoimzeFunctionTool(Tool):
    name: str = "CustoimzeFunctionTool"
    description: str = "CustoimzeFunctionTool"
    inputs: Dict[str, Dict[str, Any]] = {}
    required: Optional[List[str]] = None
    function: Callable = None
    
    def __init__(self, name: str, description: str, inputs: Dict[str, Dict[str, str]], required: Optional[List[str]] = None, function: Callable = None):
        super().__init__(name=name, description=description, inputs=inputs, required=required)
        self.function = function
    
    @property
    def __name__(self):
        return self.name
    
    async def __call__(self, **kwargs):
        if not self.function:
            raise ValueError("Function not set for MCPTool")
        if asyncio.iscoroutinefunction(self.function):
            result = await self.function(**kwargs)
        else:
            result = self.function(**kwargs)
            if inspect.isawaitable(result):
                result = await result
        return result


def tool(func) -> Optional[Tool]:
    def wrapper_func(*args, **kwargs):
        return func(*args, **kwargs)
    
    tool_structure = get_schema(func)
    name = tool_structure.get("name", "custoimzed_tool")
    description = tool_structure.get("description", tool_structure.get("name", "custoimzed_tool"))
    inputs = tool_structure.get("parameters", {})
    required = tool_structure.get("required", {})

    if not is_valid_description(description):
        logger.warning(
            f"Function-derived Tool '{name}' has invalid description (missing or < 10 chars). Returning None."
        )
        return None
    norm_name = normalize_component_name(name, "Tool")
    
    new_tool = CustoimzeFunctionTool.build(name=norm_name, description=description, inputs=inputs, required=required, function=wrapper_func)
    return new_tool
