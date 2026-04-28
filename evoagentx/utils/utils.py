import os 
import re 
import time
import regex
import requests
from tqdm import tqdm
from typing import Union, Any, List, Set, Optional, Dict

from ..core.logging import logger
from ..core.registry import MODULE_REGISTRY

def make_parent_folder(path: str):
    """Checks if the parent folder of a given path exists, and creates it if not.

    Args:
        path (str): The file path for which to create the parent folder.
    """
    dir_folder = os.path.dirname(path)
    if dir_folder and not os.path.exists(dir_folder):
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
    if not MODULE_REGISTRY.has_module(candidate_name):
        return candidate_name 
    
    i = 1 
    while True:
        unique_name = f"{candidate_name}V{i}"
        if not MODULE_REGISTRY.has_module(unique_name):
            break
        i += 1 
    return unique_name 

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


def fix_property_name(object: Any, json_schema: Dict) -> Any:
    """
    Recursively fixes the property names of `object` to match the provided JSON schema.
    """
    if object is None:
        return object

    if json_schema["type"] == "array" and json_schema["items"]["type"] == "object":
        return [fix_property_name(item, json_schema["items"]) for item in object]

    elif json_schema["type"] == "object":
        fixed_object = dict()
        properties = json_schema.get("properties")

        if properties is None:
            return object
        
        for property_name, property_schema in properties.items():

            if property_schema["type"] == "array":
                property = object.get(property_name, None)
                if property is not None:
                    fixed_object[property_name] = [fix_property_name(item, property_schema["items"]) for item in property]
            
            elif property_schema["type"] == "object":
                property = object.get(property_name, None)
                if property is not None:
                    fixed_object[property_name] = fix_property_name(property, property_schema)
            
            else:
                object_properties_lower = {name.lower(): name for name in object}
                schema_properties_lower = {name.lower(): name for name in properties}
                
                for name in object_properties_lower:
                    if name in schema_properties_lower:
                        fixed_object[schema_properties_lower[name]] = object[object_properties_lower[name]]
                    else:
                        fixed_object[object_properties_lower[name]] = object[object_properties_lower[name]]
        
        return fixed_object

    else:
        return object
