      
import inspect
import re
from typing import Dict, List, Optional, Any, Callable, get_origin

from ..core.module import BaseModule


json_to_python_type = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "object": dict,
    "array": list,
}

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
            if call_signature.parameters[input_name].annotation != json_to_python_type[input_content["type"]]:
                raise ValueError(f"Input '{input_name}' has a type mismatch in __call__")

        if cls.required:
            for required_input in cls.required:
                if required_input not in cls.inputs:
                    raise ValueError(f"Required input '{required_input}' is not found in inputs")
    
    def __call__(self, **kwargs):
        raise NotImplementedError("All tools must implement __call__")

    def to_dict(self, exclude_none: bool = True, ignore: List[str] = [], **kwargs) -> dict:
        """
        Convert the Tool to a dictionary with shallow unpacking (only one layer).
        This prevents extremely long output when there are errors by not recursively 
        unpacking nested BaseModule objects.
        
        Args:
            exclude_none: Whether to exclude fields with None values
            ignore: List of field names to ignore
            **kwargs: Additional keyword arguments
        
        Returns:
            dict: Dictionary containing the object data with shallow unpacking
        """
        data = {}
        for field_name, _ in type(self).model_fields.items():
            if field_name in ignore:
                continue
            field_value = getattr(self, field_name, None)
            if exclude_none and field_value is None:
                continue
            
            # Only unpack one layer - don't recursively unpack nested BaseModule objects
            if hasattr(field_value, 'to_dict') and hasattr(field_value, '__class__'):
                # For BaseModule objects, just include their class name and basic info
                data[field_name] = {
                    'class_name': field_value.__class__.__name__,
                    'type': str(type(field_value))
                }
            elif isinstance(field_value, list):
                # For lists, handle each item but don't go deeper
                list_data = []
                for item in field_value:
                    if hasattr(item, 'to_dict') and hasattr(item, '__class__'):
                        list_data.append({
                            'class_name': item.__class__.__name__,
                            'type': str(type(item))
                        })
                    else:
                        list_data.append(item)
                data[field_name] = list_data
            elif isinstance(field_value, dict):
                # For dicts, handle values but don't go deeper
                dict_data = {}
                for key, value in field_value.items():
                    if hasattr(value, 'to_dict') and hasattr(value, '__class__'):
                        dict_data[key] = {
                            'class_name': value.__class__.__name__,
                            'type': str(type(value))
                        }
                    elif callable(value):
                        # Handle functions/methods with full name representation
                        dict_data[key] = f"<function: {getattr(value, '__name__', str(value))}>"
                    else:
                        dict_data[key] = value
                data[field_name] = dict_data
            elif callable(field_value):
                # Handle functions/methods with full name
                data[field_name] = f"<function: {getattr(field_value, '__name__', str(field_value))}>"
            else:
                # For primitive types, include as-is
                data[field_name] = field_value
        
        return data

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
    
    def to_dict(self, exclude_none: bool = True, ignore: List[str] = [], **kwargs) -> dict:
        """
        Convert the Toolkit to a dictionary with shallow unpacking (only one layer).
        This prevents extremely long output when there are errors by not recursively 
        unpacking nested BaseModule objects.
        
        Args:
            exclude_none: Whether to exclude fields with None values
            ignore: List of field names to ignore
            **kwargs: Additional keyword arguments
        
        Returns:
            dict: Dictionary containing the object data with shallow unpacking
        """
        data = {}
        for field_name, _ in type(self).model_fields.items():
            if field_name in ignore:
                continue
            field_value = getattr(self, field_name, None)
            if exclude_none and field_value is None:
                continue
            
            # Only unpack one layer - don't recursively unpack nested BaseModule objects
            if hasattr(field_value, 'to_dict') and hasattr(field_value, '__class__'):
                # For BaseModule objects, just include their class name and basic info
                data[field_name] = {
                    'class_name': field_value.__class__.__name__,
                    'type': str(type(field_value))
                }
            elif isinstance(field_value, list):
                # For lists, handle each item but don't go deeper
                list_data = []
                for item in field_value:
                    if hasattr(item, 'to_dict') and hasattr(item, '__class__'):
                        list_data.append({
                            'class_name': item.__class__.__name__,
                            'type': str(type(item))
                        })
                    else:
                        list_data.append(item)
                data[field_name] = list_data
            elif isinstance(field_value, dict):
                # For dicts, handle values but don't go deeper
                dict_data = {}
                for key, value in field_value.items():
                    if hasattr(value, 'to_dict') and hasattr(value, '__class__'):
                        dict_data[key] = {
                            'class_name': value.__class__.__name__,
                            'type': str(type(value))
                        }
                    elif callable(value):
                        # Handle functions/methods with full name representation
                        dict_data[key] = f"<function: {getattr(value, '__name__', str(value))}>"
                    else:
                        dict_data[key] = value
                data[field_name] = dict_data
            elif callable(field_value):
                # Handle functions/methods with full name
                data[field_name] = f"<function: {getattr(field_value, '__name__', str(field_value))}>"
            else:
                # For primitive types, include as-is
                data[field_name] = field_value
        
        return data
    
    def get_tools(self) -> List[Tool]:
        return self.tools
    
    def get_tool_schemas(self) -> List[Dict]:
        return [tool.get_tool_schema() for tool in self.tools]




TYPE_MAP = {
    int: "integer",
    float: "number",
    str: "string",
    bool: "boolean",
}
ORIGIN_MAP = {
    list: "array",
    tuple: "array",
    set: "array",
    dict: "object",
}
DOC_TYPE_MAP = {
    "int": "integer",
    "integer": "integer",
    "float": "number",
    "double": "number",
    "number": "number",
    "str": "string",
    "string": "string",
    "bool": "boolean",
    "boolean": "boolean",
    "list": "array",
    "tuple": "array",
    "set": "array",
    "array": "array",
    "dict": "object",
    "mapping": "object",
    "object": "object",
}

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
    
    def __call__(self, **kwargs):
        if not self.function:
            raise ValueError("Function not set for MCPTool")
        result = self.function(**kwargs)
        return result

def map_doc_type(label: str) -> str:
    base_match = re.match(r"([A-Za-z_]+)", label.strip())
    base = base_match.group(1).lower() if base_match else label.strip().lower()
    return DOC_TYPE_MAP.get(base, "string")

# Map Python type annotations to schema types (reused by parameters and returns fallback)
def _map_type_to_schema(py_type: Any) -> str:
    if py_type is None or py_type is inspect._empty:
        return "string"
    origin = get_origin(py_type)
    if origin in ORIGIN_MAP:
        return ORIGIN_MAP[origin]
    if isinstance(py_type, type) and py_type in TYPE_MAP:
        return TYPE_MAP[py_type]
    return "string"

def extract_descriptions(func):
    """
    Extract the function-level description, parameter descriptions, return entries, and required params from the docstring.
    Returns a tuple: (
        function_description,
        {param_name: description},
        [{"type": t, "description": d}, ...],
        [required_param_names]
    )
    """
    doc = inspect.getdoc(func) or ""
    if not doc.strip():
        return "", {}, [], []

    # Helper to normalize whitespace
    def collapse(parts_or_text):
        if isinstance(parts_or_text, str):
            parts = parts_or_text.splitlines()
        else:
            parts = parts_or_text
        return re.sub(r"\s+", " ", " ".join(p.strip() for p in parts if p is not None and p.strip())).strip()

    # Split docstring into sections purely via regex (no manual indices)
    # tokens: [preface, section_name_1, section_content_1, section_name_2, section_content_2, ...]
    tokens = re.split(r"^\s*(Args|Arguments|Parameters|Returns|Required|Raises|Examples|Yields):\s*$", doc, flags=re.MULTILINE)
    preface = tokens[0] if tokens else doc
    func_desc = collapse(preface)

    sections = {}
    for i in range(1, len(tokens), 2):
        label = tokens[i].strip().lower()
        content = tokens[i + 1]
        sections[label] = content

    # Choose parameter section content
    param_content = None
    for key in ("args", "arguments", "parameters"):
        if key in sections:
            param_content = sections[key]
            break

    returns_content = sections.get("returns")
    required_content = sections.get("required")

    # Entry patterns
    param_re = re.compile(r"^\s*([A-Za-z_]\w*)\s*(?:\([^)]+\))?\s*:\s*(.*)$")
    ret_re = re.compile(r"^\s*([A-Za-z_][\w\[\], ]*)\s*:\s*(.*)$")

    # Generic parser over a section's content (no indices)
    def parse_entries_from(content: str, entry_re: re.Pattern, postprocess_first=None):
        if not content:
            return []
        entries = []
        current_first = None
        current_desc_parts = []
        for raw in content.splitlines():
            if not raw.strip():
                # allow blank lines inside sections without flushing
                continue
            m = entry_re.match(raw)
            if m:
                # flush previous
                if current_first is not None:
                    first = postprocess_first(current_first) if postprocess_first else current_first
                    entries.append((first, collapse(current_desc_parts)))
                current_first = m.group(1).strip()
                first_desc = (m.group(2) or "").strip()
                current_desc_parts = [first_desc] if first_desc else []
            else:
                # continuation of description
                if current_first is not None:
                    current_desc_parts.append(raw.strip())
        if current_first is not None:
            first = postprocess_first(current_first) if postprocess_first else current_first
            entries.append((first, collapse(current_desc_parts)))
        return entries

    # Parse parameters and returns using the shared parser
    param_entries = parse_entries_from(param_content, param_re)
    returns_entries_raw = parse_entries_from(returns_content, ret_re, postprocess_first=map_doc_type)

    param_descs = {name: desc for name, desc in param_entries}
    returns_entries = [{"type": typ, "description": desc} for typ, desc in returns_entries_raw]

    # Parse required names: default handled by caller; here we only extract names present in params
    required_names = []
    if required_content:
        seen = set()
        param_names_order = [name for name, _ in param_entries]
        param_name_set = set(param_names_order)
        for raw in required_content.splitlines():
            # support comma-separated or bullet lines
            # extract any identifier tokens and keep ones matching parameter names
            for tok in re.findall(r"[A-Za-z_]\w*", raw):
                if tok in param_name_set and tok not in seen:
                    seen.add(tok)
                    required_names.append(tok)

    return func_desc, param_descs, returns_entries, required_names


def extract_name_and_types(func):
    """
    Extract the function name and a mapping of parameter names to their JSON Schema types.
    Returns a tuple: (function_name, {param_name: type_str})
    """
    sig = inspect.signature(func)
    annotations = getattr(func, "__annotations__", {}) or {}
    types = {}
    for name, param in sig.parameters.items():
        anno = annotations.get(name, param.annotation)
        types[name] = _map_type_to_schema(anno)
    return func.__name__, types


def extract_return_info(func):
    """
    Extract return type from annotations and return description from the docstring.
    Returns a dict: {"type": <json type>, "description": <text>}
    """
    # Type from annotation
    ret_anno = getattr(func, "__annotations__", {}).get("return", inspect._empty)
    ret_type = _map_type_to_schema(ret_anno)

    # Description from docstring
    doc = inspect.getdoc(func) or ""
    desc = ""
    if doc:
        lines = doc.splitlines()
        returns_header_re = re.compile(r"^\s*Returns:\s*$", re.IGNORECASE)
        stop_re = re.compile(r"^\s*(Args|Arguments|Parameters|Raises|Examples|Yields):\s*$", re.IGNORECASE)

        ret_idx = None
        for i, line in enumerate(lines):
            if returns_header_re.match(line):
                ret_idx = i
                break
        if ret_idx is not None:
            ret_lines = []
            for raw in lines[ret_idx + 1:]:
                if stop_re.match(raw):
                    break
                if raw.strip():
                    ret_lines.append(raw.strip())
            if ret_lines:
                # If first line is like "int: description", drop the leading type part
                m = re.match(r"^\s*[^:]+:\s*(.*)$", ret_lines[0])
                if m:
                    first_desc = m.group(1).strip()
                    rest = " ".join(line.strip() for line in ret_lines[1:])
                    desc = (first_desc + (" " + rest if rest else "")).strip()
                else:
                    desc = " ".join(ret_lines).strip()

    return {"type": ret_type, "description": desc}


def get_schema(func):
    """
    Assemble the function metadata into the final dict schema.
    {
        "name": <function name>,
        "description": <function description>,
        "parameters": {
            <param>: {"type": <json type>, "description": <doc description>}
        },
        "returns": [{"type": <json type>, "description": <doc description>}, ...],
        "required": [<param_name>, ...]
    }
    """
    func_name, type_map = extract_name_and_types(func)
    func_desc, param_descs, returns_entries, required_names = extract_descriptions(func)

    # Fallback to annotation if no returns found in docstring
    if not returns_entries:
        ret_anno = getattr(func, "__annotations__", {}).get("return", inspect._empty)
        returns_entries = [{"type": _map_type_to_schema(ret_anno), "description": ""}]

    # Compute required: default to all parameters if docstring didn't specify
    param_order = list(type_map.keys())
    if not required_names:
        required_names = param_order
    else:
        # keep only valid param names and preserve the doc order
        required_names = [n for n in required_names if n in type_map]

    schema = {
        "name": func_name,
        "description": func_desc,
        "parameters": {},
        "returns": returns_entries,
        "required": required_names,
    }
    for p, t in type_map.items():
        schema["parameters"][p] = {
            "type": t,
            "description": param_descs.get(p, "")
        }
    return schema

def tool(func) -> Tool:
    def wrapper_func(*args, **kwargs):
        return func(*args, **kwargs)
    
    tool_structure = get_schema(func)
    name = tool_structure.get("name", "custoimzed_tool")
    description = tool_structure.get("description", tool_structure.get("name", "custoimzed_tool"))
    inputs = tool_structure.get("parameters", {})
    required = tool_structure.get("required", {})
    
    new_tool = CustoimzeFunctionTool(name=name, description=description, inputs=inputs, required=required, function=wrapper_func)
    return new_tool
