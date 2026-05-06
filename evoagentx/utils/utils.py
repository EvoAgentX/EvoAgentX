import os 
import re 
import time
import contextvars
import regex
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from typing import Union, Any, List, Set, Dict, Optional, Type
from pydantic import BaseModel
from pydantic_core import PydanticUndefined

from ..core.logging import logger

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

def get_cost_per_tool(cost_breakdown: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """
    Returns the total cost for each tool.
    """
    cost_per_tool = dict()
    
    for tool_name, costs in cost_breakdown.items():
        cost_per_tool[tool_name] = sum(costs.values())

    return cost_per_tool

def get_total_tool_cost(cost_breakdown: Dict[str, Dict[str, float]]) -> float:
    return sum(get_cost_per_tool(cost_breakdown).values())

def get_provider_tool_cost(cost_breakdown: Dict[str, Dict[str, float]], provider: str) -> float:

    provider_cost = 0.

    for costs in cost_breakdown.values():
        for key, value in costs.items():
            if key.startswith(provider):
                provider_cost += value

    return provider_cost

class ContextualThreadPoolExecutor(ThreadPoolExecutor):
    """ThreadPoolExecutor that preserves context variables"""
    
    def submit(self, fn, *args, **kwargs):
        current_context = contextvars.copy_context()
        
        def wrapped_fn(*args, **kwargs):
            return current_context.run(fn, *args, **kwargs)
            
        return super().submit(wrapped_fn, *args, **kwargs)

json_to_python_type = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "object": dict,
    "array": list,
}

def get_field_default(model: Type[BaseModel], field_name: str) -> Optional[Any]:
    field = model.model_fields.get(field_name)

    if field is not None:
        if field.default is not PydanticUndefined:
            return field.default

        if field.default_factory is not None:
            return field.default_factory()
    
    return None

def generate_dynamic_class_name(base_name: str) -> str:

    base_name = base_name.strip()
    
    cleaned_name = re.sub(r'[^a-zA-Z0-9\s]', ' ', base_name)
    components = cleaned_name.split()
    class_name = ''.join(x.capitalize() for x in components)

    return class_name if class_name else 'DefaultClassName'

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
