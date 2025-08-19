import os
import re
import time
from typing import Any, Dict, List, Optional, Set, Union

import regex
import requests
from tqdm import tqdm

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

def fix_json_booleans(json_string: str) -> str:
    """
    Finds and replaces isolated "True" and "False" with "true" and "false".

    The '\b' in the regex stands for a "word boundary", which ensures that
    we only match the full words and not substrings like "True" in "IsTrue".

    Args:
        json_string (str): The input JSON string.

    Returns:
        str: The modified JSON string with booleans in lowercase.
    """
    # Use re.sub() with a word boundary (\b) to ensure we only match
    # the isolated words 'True' and 'False' and not substrings like "True" in "IsTrue"
    modified_string = re.sub(r'\bTrue\b', 'true', json_string)
    modified_string = re.sub(r'\bFalse\b', 'false', modified_string)
    return modified_string


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