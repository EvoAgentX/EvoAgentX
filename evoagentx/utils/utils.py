import contextvars
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from math import tanh
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Set,
    Type,
    Union,
    get_args,
    get_origin,
)

import regex
import requests
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
)
from pydantic import BaseModel, ValidationError
from pydantic_core import PydanticUndefined
from tqdm import tqdm

from ..core.base_config import Parameter
from ..core.logging import logger
from ..core.registry import MODULE_REGISTRY

# Import for type hints (avoiding circular imports with TYPE_CHECKING)
if TYPE_CHECKING:
    from ..agents import Agent
    from ..models import LLMConfig
    from ..tools.tool import Tool, Toolkit


def make_parent_folder(path: str):
    """Checks if the parent folder of a given path exists, and creates it if not.

    Args:
        path (str): The file path for which to create the parent folder.
    """
    dir_folder = os.path.dirname(path)
    if not os.path.exists(dir_folder):
        logger.info(f"creating folder {dir_folder} ...")
        os.makedirs(dir_folder, exist_ok=True)

def safe_remove(data: Union[List[Any], Set[Any]], remove_value: Any):
    try:
        data.remove(remove_value)
    except ValueError:
        pass

def generate_dynamic_class_name(base_name: str) -> str:

    base_name = base_name.strip()

    cleaned_name = re.sub(r'[^a-zA-Z0-9\s]', ' ', base_name)
    components = cleaned_name.split()
    class_name = ''.join(x.capitalize() for x in components)

    return class_name if class_name else 'DefaultClassName'


def get_unique_class_name(candidate_name: str) -> str:
    """
    Get a unique class name by checking if it already exists in the registry.
    If it does, append "Vx" to make it unique.
    """
    if not MODULE_REGISTRY.has_module(candidate_name):
        return candidate_name

    i = 1
    while True:
        unique_name = f"{candidate_name}V{i}"
        if not MODULE_REGISTRY.has_module(unique_name):
            break
        i += 1
    return unique_name


def normalize_text(s: str) -> str:

    def remove_articles(text):
        return regex.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        return text.replace("_", " ")
        # exclude = set(string.punctuation)
        # return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def download_file(url: str, save_file: str, max_retries=3, timeout=10):

    make_parent_folder(save_file)
    for attempt in range(max_retries):
        try:
            resume_byte_pos = 0
            if os.path.exists(save_file):
                resume_byte_pos = os.path.getsize(save_file)

            response_head = requests.head(url=url)
            total_size = int(response_head.headers.get("content-length", 0))

            if resume_byte_pos >= total_size:
                logger.info("File already downloaded completely.")
                return

            headers = {'Range': f'bytes={resume_byte_pos}-'} if resume_byte_pos else {}
            response = requests.get(url=url, stream=True, headers=headers, timeout=timeout)
            response.raise_for_status()
            # total_size = int(response.headers.get("content-length", 0))
            mode = 'ab' if resume_byte_pos else 'wb'
            progress_bar = tqdm(total=total_size, unit="iB", unit_scale=True, initial=resume_byte_pos)

            with open(save_file, mode) as file:
                for chunk_data in response.iter_content(chunk_size=1024):
                    if chunk_data:
                        size = file.write(chunk_data)
                        progress_bar.update(size)

            progress_bar.close()

            if os.path.getsize(save_file) >= (total_size + resume_byte_pos):
                logger.info("Download completed successfully.")
                break
            else:
                logger.warning("File size mismatch, retrying...")
                time.sleep(5)
        except (requests.ConnectionError, requests.Timeout) as e:
            logger.warning(f"Download error: {e}. Retrying ({attempt+1}/{max_retries})...")
            time.sleep(5)
        except Exception as e:
            error_message = f"Unexpected error: {e}"
            logger.error(error_message)
            raise ValueError(error_message)
    else:
        error_message = "Exceeded maximum retries. Download failed."
        logger.error(error_message)
        raise RuntimeError(error_message)


def recursive_remove(data: Any, keys: List[str]) -> Any:
    """
    Recursively removes specified keys from dictionaries and their nested structures within a
    dictionary or list, if an object is not a list or dictionary return as is.

    Args:
        data (Any): Specified keys will be removed from `data` if it is a dictionary or a list containing dictionaries.
        keys (List[str]): A list of string keys to be removed.
    """
    if isinstance(data, dict):
        new_dict = {}
        for k, v in data.items():
            if k not in keys:
                new_dict[k] = recursive_remove(v, keys)
        return new_dict
    elif isinstance(data, list):
        new_list = [recursive_remove(item, keys) for item in data]
        return new_list
    else:
        return data


def tool_names_to_tools(
    tool_names: Optional[List[str]] = None,
    tools: Optional[List] = None,
) -> Optional[List]:

    if not tool_names:
        return None

    if not tools:
        raise ValueError(f"Must provide the following tools: {tool_names}")

    tool_map = {tool.name: tool for tool in tools}

    tool_list = []
    for tool_name in tool_names:
        if tool_name not in tool_map:
            raise ValueError(f"'{tool_name}' not found in provided tools")
        tool_list.append(tool_map[tool_name])
    return tool_list


def add_llm_config_to_agent_dict(agent_dict: Dict, llm_config: Optional['LLMConfig'] = None) -> Dict:
    """Assign the llm_config to agent_dict, overwriting any existing value.
    If `llm` exists, it will be overwritten by `llm_config` to prevent conflicts.
    If `is_human` is True, llm_config will not be added.
    """

    agent_dict_copy = agent_dict.copy()

    if agent_dict_copy.get("is_human", False):
        return agent_dict_copy

    agent_llm_config = agent_dict_copy.get("llm_config", None)
    agent_llm = agent_dict_copy.get("llm", None)

    if llm_config is None and agent_llm_config is None and agent_llm is None:
        raise ValueError("Must provide `llm_config` or `llm` for agent")

    if llm_config is not None:
        agent_dict_copy.pop("llm", None)
        agent_dict_copy["llm_config"] = llm_config
    return agent_dict_copy


def create_agent_from_dict(
    agent_dict: Dict,
    llm_config: Optional['LLMConfig'] = None,
    tools: Optional[List] = None,
    agents: Optional[List] = None,
    **kwargs
) -> 'Agent':

    agent_class_name = agent_dict.pop("class_name", None)

    if agent_class_name is None:
        agent_class_name = "CustomizeAgent"

    cls = MODULE_REGISTRY.get_module(agent_class_name)
    agent = cls.from_dict(data=agent_dict, llm_config=llm_config, tools=tools, agents=agents, **kwargs)
    return agent


def pydantic_to_parameters(base_model: Type[BaseModel], ignore: List[str] = []) -> List[Parameter]:
    """
    Converts a Pydantic BaseModel class into a list of Parameter instances.

    Args:
        model: A Pydantic BaseModel class.

    Returns:
        A list of Parameter objects, where each object corresponds to a field
        in the input BaseModel.
    """
    parameters = []
    for field_name, field_info in base_model.model_fields.items():
        if field_name in ignore:
            continue

        # Determine the description
        description = field_info.description if field_info.description else field_name

        # Determine if the field is required
        # A field is considered required if it doesn't have a default value
        # and isn't Optional.
        required = field_info.is_required()

        field_type = python_to_json_type[extract_type(field_info.annotation)]

        # Create the Parameter instance
        param = Parameter(
            name=field_name,
            type=field_type,
            description=description,
            required=required,
            json_schema=field_info.json_schema_extra,
        )
        parameters.append(param)
    return parameters


def validate_param(
    required_param: Parameter,
    actual_param: Parameter,
    required_params_name: str,
    actual_params_name: str,
):
    """
    Checks if `actual_param` is compatible with `required_param`.

    Only the attributes that affect runtime behavior are strictly enforced: `type` and
    `required`. `description` is free-form text and is not compared. If the required
    parameter provides a `json_schema`, the actual parameter must provide the same
    schema so structured contracts cannot be dropped downstream.
    """

    def format_error_msg(
        attr_name: str,
        required_value: Any,
        actual_value: Any,
    ) -> str:
        return f"Mismatch for '{required_param.name}': {required_params_name} ({attr_name}={required_value}) vs. {actual_params_name} ({attr_name}={actual_value})"

    try:
        actual_type = string_to_python_type[actual_param.type]
        required_type = string_to_python_type[required_param.type]
    except KeyError as e:
        logger.warning(f"Unsupported type in {actual_param.name}: {e}")
        actual_type = actual_param.type
        required_type = required_param.type

    if required_type != actual_type:
        raise ValueError(format_error_msg("type", required_param.type, actual_param.type))

    if required_param.required != actual_param.required:
        raise ValueError(format_error_msg("required", required_param.required, actual_param.required))

    # `json_schema` is optional for parameters in general, but once an upstream
    # contract provides one, downstream params must preserve it.
    if required_param.json_schema is not None:
        if required_param.json_schema != actual_param.json_schema:
            raise ValueError(format_error_msg("json_schema", required_param.json_schema, actual_param.json_schema))


def format_validation_error(error: ValidationError) -> str:
    """
    Formats a Pydantic ValidationError into a nicely formatted string.

    Args:
        error: The Pydantic ValidationError object.

    Returns:
        A formatted string containing all error details.
    """
    formatted_messages: List[str] = []

    for e in error.errors():
        path_parts = []
        for item in e['loc']:
            if isinstance(item, int):
                path_parts[-1] += f"[{item}]"
            else:
                path_parts.append(str(item))

        error_location_str = ".".join(path_parts)

        if error_location_str:
            formatted_message = (
                f"Location: {error_location_str}\n"
                f"{e['msg']}\n"
            )
        else:
            formatted_message = f"{e['msg']}\n"

        formatted_messages.append(formatted_message)

    return "\n".join(formatted_messages)


def params_to_json(params: List[Parameter], ignore: List[str] = []) -> str:
    params_dict = [param.to_dict(ignore=ignore) for param in params]
    params_json = json.dumps(params_dict, indent=4, ensure_ascii=False)
    return params_json


def resolve_json_schema_ref(json_schema: Any, root_schema: Optional[Dict] = None) -> Any:
    """
    Recursively resolve all $ref in a JSON schema.

    Parameters:
        json_schema (Any): The current schema to resolve.
        root_schema (Optional[Dict]): The root schema used to resolve references. If not provided, it will be set to `json_schema`.

    Returns:
        Any: A new schema with all $ref replaced by their actual definitions.
    """
    if root_schema is None:
        if not isinstance(json_schema, dict):
            raise ValueError("`root_schema` must be a dictionary")
        root_schema = json_schema

    if isinstance(json_schema, dict):
        if "$ref" in json_schema:
            ref_path = json_schema["$ref"]

            # Only support internal references (starting with "#/")
            if not ref_path.startswith("#/"):
                raise ValueError(f"External references not supported: {ref_path}")

            # Navigate the path
            parts = ref_path.lstrip("#/").split("/")
            target = root_schema

            for part in parts:
                target = target[part]

            resolved = resolve_json_schema_ref(deepcopy(target), root_schema)
            return resolved

        # Recurse into dict values
        return {k: resolve_json_schema_ref(v, root_schema) for k, v in json_schema.items()}

    elif isinstance(json_schema, list):
        return [resolve_json_schema_ref(item, root_schema) for item in json_schema]

    else:
        return json_schema


def remove_none(obj):
    """
    Recursively removes all keys where the value is None.
    """
    if isinstance(obj, dict):
        obj_without_none = dict()

        for k, v in obj.items():
            if v is not None:
                if isinstance(v, dict):
                    obj_without_none[k] = remove_none(v)
                elif isinstance(v, list):
                    obj_without_none[k] = [remove_none(item) for item in v]
                else:
                    obj_without_none[k] = v
        return obj_without_none

    elif isinstance(obj, list):
        return [remove_none(item) for item in obj]

    else:
        return obj


def extract_type(annotation: Type) -> Type:
    """
    If `annotation` is Optional/Union, return the first type in the union.
    """
    if get_origin(annotation) is Union:
        return get_args(annotation)[0]
    return annotation


def get_name_to_value_map(names: List[str], lookup_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Takes a list of names and a dictionary, and returns a dictionary mapping each name to its corresponding value in the lookup dictionary.

    Args:
        names (List[str]): The list of names to look up.
        lookup_dict (Dict[str, Any]): The dictionary to look up the names in.

    Returns:
        Dict[str, Any]: A dictionary mapping each name to its corresponding value in the lookup dictionary.
    """
    return {name: lookup_dict.get(name, None) for name in names}


def compute_score(
    score: float,
    improvement_score: float,
    min_score: float = 1.,
    max_score: float = 10.,
    decimal_places: int = 2
) -> float:
    new_score = score + (max_score - score) * tanh(improvement_score)
    return round(min(max_score, max(min_score, new_score)), decimal_places)


def transform_score(old_score: float, old_min: float, old_max: float, new_min: float, new_max: float) -> float:
    """Transforms a score from one scale to another using linear scaling."""
    assert old_min < old_max, "`old_min` must be less than `old_max`"
    assert new_min < new_max, "`new_min` must be less than `new_max`"

    transformed_score = ((old_score - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min
    return transformed_score


def format_execution_data(input_data: List[Dict], output_data: List[Dict]) -> List[Dict]:
    execution_data = [
        {
            "execution_input": input,
            "execution_output": output
        }
        for input, output in zip(input_data, output_data, strict=True)
    ]

    return execution_data


def compose_decorators(*decorators):
    def combined(func):
        wrapped = func
        for decorator in decorators:
            wrapped = decorator(wrapped)
        return wrapped
    return combined


def add_dict(a: Dict[str, Union[float, int]], b: Dict[str, Union[float, int]]) -> Dict[str, Union[float, int]]:
    """
    Adds the values from two dict together if they share the same key.
    Also keeps the values that don't share keys in the final output.
    """
    if not a:
        return b

    if not b:
        return a

    dict_sum = a.copy()

    for key, value in b.items():
        dict_sum[key] = dict_sum.get(key, 0) + value

    return dict_sum



def compile_tool_schemas(tools: List[Union['Tool', 'Toolkit']]) -> List[dict]:
    """
    Compiles the schemas of a list of tools or toolkits.

    Args:
        tools: A list of tools or toolkits

    Returns:
        A list of dictionaries containing the schemas of the tools or toolkits
    """
    from ..tools.tool import Tool, Toolkit

    if len(tools) == 0:
        return []

    schemas = []
    for tool in tools:
        if isinstance(tool, Tool):
            schemas.append(tool.get_tool_schema())
        elif isinstance(tool, Toolkit):
            schemas.extend(tool.get_tool_schemas())
        else:
            raise ValueError(f"Unknown tool type: {type(tool)}")
    return schemas


def format_tool_calls(tool_calls: List[ChatCompletionMessageToolCall]) -> List[Dict]:
    """
    Formats a list of tool calls into a EAX format.

    Args:
        tool_calls: A list of tool calls

    Returns:
        A string containing the formatted tool calls
    """
    formatted_tool_calls = []

    for tool_call in tool_calls:
        tool_name = tool_call.function.name
        try:
            tool_args = json.loads(tool_call.function.arguments)
        except Exception:
            logger.error(f"Failed to parse tool call arguments for `{tool_name}`:\n{tool_call.function.arguments}")
            continue

        formatted_tool_calls.append(
            {
                "id": tool_call.id,
                "function_name": tool_name,
                "function_args": tool_args,
            }
        )

    return formatted_tool_calls


def get_field_default(model: Type[BaseModel], field_name: str) -> Optional[Any]:
    """
    Retrieves the default value for a specified field in a Pydantic model.

    If the field has a default value, returns it. If the field does not have a default value but
    has a default factory, calls the factory to generate and return the default value.
    Returns None if no default or factory is defined.

    Args:
        model: The Pydantic model class.
        field_name: The name of the field to check.

    Returns:
        The default value of the field or None if no default is defined.
    """

    field = model.model_fields.get(field_name)

    if field is not None:
        if field.default is not PydanticUndefined:
            return field.default

        if field.default_factory is not None:
            return field.default_factory()

    return None


def to_params(items: List[Union[Parameter, dict]]) -> List[Parameter]:
    """
    Convert a list of dictionaries or Parameter objects into a list of Parameter objects.
    """
    params: List[Parameter] = []
    for item in items:
        if isinstance(item, dict):
            params.append(Parameter(**item))
        elif isinstance(item, Parameter):
            params.append(item)
        else:
            raise TypeError(f"Expects dict or Parameter, but got {type(item).__name__}")
    return params



class ContextualThreadPoolExecutor(ThreadPoolExecutor):
    """ThreadPoolExecutor that preserves context variables"""

    def __init__(self, max_workers: Optional[int] = None, **kwargs):
        super().__init__(max_workers=max_workers, **kwargs)

    def submit(self, fn, *args, **kwargs):
        current_context = contextvars.copy_context()

        def wrapped_fn(*args, **kwargs):
            return current_context.run(fn, *args, **kwargs)

        return super().submit(wrapped_fn, *args, **kwargs)


string_to_python_type = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "object": dict,
    "array": list,

    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
    "dict": dict,
    "list": list,
}

# Common aliases LLMs emit for parameter types, mapped to canonical names.
_param_type_aliases = {
    "text": "string",
    "char": "string",
    "double": "number",
    "long": "integer",
    "json": "object",
    "dictionary": "object",
    "map": "object",
    "tuple": "array",
    "set": "array",
}


def normalize_param_type(type_str: str) -> Optional[str]:
    """Best-effort normalization of a parameter `type` string into a canonical type.

    Handles casing/whitespace, common aliases, and parametrized forms such as
    ``List[str]`` or ``Dict[str, int]``. Returns ``None`` when the type cannot be
    recognized, so the caller can decide how to handle it (e.g. raise an error).
    """
    if not isinstance(type_str, str):
        return None
    normalized = type_str.strip().lower()
    # Strip parametrization, e.g. "list[str]" -> "list", "dict[str, int]" -> "dict".
    base = normalized.split("[", 1)[0].strip()
    if base in string_to_python_type:
        return base
    if base in _param_type_aliases:
        return _param_type_aliases[base]
    return None


json_to_python_type = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "object": dict,
    "array": list,
}

python_to_json_type = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    dict: "object",
    list: "array",
}

string_to_json_schema_type = {
    "string": "string",
    "integer": "integer",
    "number": "number",
    "boolean": "boolean",
    "object": "object",
    "array": "array",
    "str": "string",
    "int": "integer",
    "float": "number",
    "bool": "boolean",
    "dict": "object",
    "list": "array",
}
