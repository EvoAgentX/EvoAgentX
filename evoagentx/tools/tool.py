      
import inspect
from typing import Dict, List, Optional, Any

from ..core.module import BaseModule

ALLOWED_TYPES = ["string", "number", "integer", "boolean", "object", "array"]


class Tool(BaseModule):
    name: str
    description: str
    inputs: Dict[str, Dict[str, Any]]
    required: Optional[List[str]] = None

    """
    inputs: {"input_name": {"type": "string", "description": "input description"}, ...}
    """

    def __init_subclass__(cls):
        super().__init_subclass__()
        cls.validate_attributes()

    def get_tool_schema(self) -> Dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
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

        json_to_python = {
            "string": str,
            "integer": int,
            "number": float,
            "boolean": bool,
            "object": dict,
            "array": list,
        }
        
        for attr, attr_type in required_attributes.items():
            if not hasattr(cls, attr):
                raise ValueError(f"Attribute {attr} is required")
            if not isinstance(getattr(cls, attr), attr_type):
                raise ValueError(f"Attribute {attr} must be of type {attr_type}")

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
            if call_signature.parameters[input_name].annotation != json_to_python[input_content["type"]]:
                raise ValueError(f"Input '{input_name}' has a type mismatch in __call__")

        if cls.required:
            for required_input in cls.required:
                if required_input not in cls.inputs:
                    raise ValueError(f"Required input '{required_input}' is not found in inputs")
    
    def __call__(self, **kwargs):
        raise NotImplementedError("All tools must implement __call__")

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

    def get_tool(self, tool_name: str) -> Tool:
        for tool in self.tools:
            if tool.name == tool_name:
                return tool
        raise ValueError(f"Tool '{tool_name}' not found")
    
    def get_tools(self) -> List[Tool]:
        return self.tools
    
    def get_tool_schemas(self) -> List[Dict]:
        return [tool.get_tool_schema() for tool in self.tools]
    