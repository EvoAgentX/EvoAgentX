import os
import re
import time
from typing import Any, Dict, List, Optional, Set, Union, get_args, get_origin, Type

import regex
import requests
from pydantic import BaseModel
from tqdm import tqdm

from ..core.base_config import Parameter
from ..core.logging import logger
from ..core.registry import MODULE_REGISTRY


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

    if tool_names is None:
        return None

    if len(tool_names) == 0:
        return None

    if tools is None:
        raise ValueError(f"Must provide the following tools: {tool_names}")

    tool_map = {tool.name: tool for tool in tools}
    
    tool_list = []
    for tool_name in tool_names:
        if tool_name not in tool_map:
            raise ValueError(f"'{tool_name}' not found in provided tools")
        tool_list.append(tool_map[tool_name])
    return tool_list


def add_llm_config_to_agent_dict(agent_dict: Dict, llm_config: Optional['LLMConfig'] = None) -> Dict:
    """Add llm_config to agent_dict if it is not present and converts llm_config dict to LLMConfig
    If `is_human` is True, llm_config will not be added.
    """
    from ..models import LLMConfig

    if agent_dict.get("is_human", False):
        return agent_dict

    data_llm_config = agent_dict.get("llm_config", None)

    if data_llm_config is None:
        if llm_config is None:
            raise ValueError("Must provide `llm_config` for agent")
        agent_dict["llm_config"] = llm_config
    else:
        if isinstance(data_llm_config, dict):
            agent_dict["llm_config"] = LLMConfig.from_dict(data_llm_config)
    
    return agent_dict


def create_agent_from_dict(
    agent_dict: Dict, 
    llm_config: Optional['LLMConfig'] = None,
    tools: Optional[List] = None,
    agents: Optional[List] = None,
) -> 'Agent':

    agent_class_name = agent_dict.get("class_name", None)

    if agent_class_name is None:
        agent_class_name = "CustomizeAgent"
    
    cls = MODULE_REGISTRY.get_module(agent_class_name)
    agent = cls.from_dict(data=agent_dict, llm_config=llm_config, tools=tools, agents=agents)
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

        if required:
            field_type = python_to_json_type[field_info.annotation]
        else:
            if get_origin(field_info.annotation) is Union:
                # field is not required because it can be None
                field_type = python_to_json_type[get_args(field_info.annotation)[0]]
            else:
                # field is not required because it has a default value
                field_type = python_to_json_type[field_info.annotation]

        # Create the Parameter instance
        param = Parameter(
            name=field_name,
            type=field_type,
            description=description,
            required=required
        )
        parameters.append(param)
    return parameters


def validate_params(
    required_params: List[Parameter], 
    actual_params: List[Parameter], 
    required_params_name: str, 
    actual_params_name: str,
    auto_fix: bool = False
)-> Dict[str, Parameter]:
    """
    Checks if `actual_params` have `required_params` and if the `required_params` in `actual_params` have the same type and required value.
    If `actual_params` doesn't have `required_params`, raises ValueError.
    If `actual_params` has `required_params` but with different `type` or `required` value, raises ValueError. Fix the mismatch if `auto_fix` is True.
    If `actual_params` has `required_params` but with different `description`, it will only warn. Fix the mismatch if `auto_fix` is True.

    Args:
        required_params: A list of parameters that are required. Raises error if `actual_params` doesn't contain these parameters.
        actual_params: A list of parameters to check if `required_params` exist.
        required_params_name: A name for `required params` to be shown in error messages.
        actual_params_name: A name for `actual_params` to be shown in error messages.

    Returns:
        Dict[str, Parameter]: A dictionary of actual parameters with their names as keys.
            If `auto_fix` is True, it will fix the mismatched `type`, `required` and `description` fields.
    """
    actual_params_dict = {param.name: param for param in actual_params}

    def check_attr(
        pass_condition: bool, 
        param_name: str, 
        attr_name: str, 
        required_value: Any, 
        actual_value: Any, 
        auto_fix: bool
    ):
        if not pass_condition:
            if auto_fix:
                logger.warning(
                    f"Mismatch for '{param_name}': {required_params_name} ({attr_name}={required_value}) vs. {actual_params_name} ({attr_name}={actual_value})"
                )
                logger.info(f"Fixing '{param_name}' `{attr_name}` from '{actual_value}' to '{required_value}'")
                setattr(actual_params_dict[param_name], attr_name, required_value)
            else:
                raise ValueError(
                    f"Mismatch for '{param_name}': {required_params_name} ({attr_name}={required_value}) vs. {actual_params_name} ({attr_name}={actual_value})"
                )


    for param in required_params:
        if param.name not in actual_params_dict:
            raise ValueError(f"{required_params_name} '{param.name}' is not found in {actual_params_name}: {list(actual_params_dict.keys())}")

        actual_type = actual_params_dict[param.name].type

        try:
            # see if they map to the same python type
            actual_python_type = string_to_python_type[actual_type]
            required_param_python_type = string_to_python_type[param.type]

            check_attr(
                pass_condition=required_param_python_type is actual_python_type,
                param_name=param.name,
                attr_name="type",
                required_value=required_param_python_type,
                actual_value=actual_python_type,
                auto_fix=auto_fix
            )

        except KeyError:
            # if not supported type, issue an warning and check if they are the same
            logger.warning(
                f"Unsupported type '{actual_type}' for parameter '{param.name}' in {actual_params_name}"
            )

            check_attr(
                pass_condition=actual_type == param.type,
                param_name=param.name,
                attr_name="type",
                required_value=param.type,
                actual_value=actual_type,
                auto_fix=auto_fix
            )

        actual_required = actual_params_dict[param.name].required
        actual_description = actual_params_dict[param.name].description

        check_attr(
            pass_condition=actual_required == param.required,
            param_name=param.name,
            attr_name="required",
            required_value=param.required,
            actual_value=actual_required,
            auto_fix=auto_fix
        )

        if actual_description != param.description:
            # only warn if description is different
            logger.warning(
                f"Mismatch for '{param.name}': {required_params_name} (description={param.description}) vs. {actual_params_name} (description={actual_description})"
            )

            if auto_fix:
                logger.info(f"Fixing '{param.name}' `description` from '{actual_description}' to '{param.description}'")
                actual_params_dict[param.name].description = param.description

    return actual_params_dict


def update_params(old_params: List[Parameter], new_params: Dict[str, Parameter]) -> List[Parameter]:
    """
    Updates a list of parameters with new parameters.

    Args:
        old_params: The list of parameters to update.
        new_params: A dictionary of parameters to update with, where the key is the parameter name.

    Returns:
        List[Parameter]: The updated list of parameters.
    """
    for param_idx, param in enumerate(old_params):
        if param.name in new_params:
            old_params[param_idx] = new_params[param.name]
    return old_params


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