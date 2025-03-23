import os 
import re 
import time
import regex
import requests
from tqdm import tqdm
from typing import Union, Any, List, Set, Tuple
from enum import Enum
import json


from ..core.logging import logger

def make_parent_folder(path: str):

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


def download_file(url: str, save_file: str) -> None:
    """Download a file from the given URL and show progress."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024
    progress_bar = tqdm(total=total_size, unit="iB", unit_scale=True)

    with open(save_file, "wb") as file:
        for data in response.iter_content(block_size):
            size = file.write(data)
            progress_bar.update(size)
    progress_bar.close()



class CodeDataset(Enum):
    HUMAN_EVAL = "HumanEval"
    MBPP = "MBPP"


def extract_test_cases_from_jsonl(entry_point: str, dataset: CodeDataset = CodeDataset.HUMAN_EVAL):
    # 尝试多个可能的路径位置
    possible_paths = [
        # 相对于项目根目录的路径
        os.path.join("evoagentx", "ext", "aflow", "data", "humaneval_public_test.jsonl"),
        # 相对于当前文件的路径
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                    "ext", "aflow", "data", "humaneval_test.jsonl"),
        # 用户主目录下的路径
        os.path.expanduser("~/.evoagentx/data/HumanEval/humaneval_public_test.jsonl"),
        # 原始路径
        "data/aflow_benchmark_data/HumanEval/humaneval_public_test.jsonl"
    ]
    
    if dataset == CodeDataset.HUMAN_EVAL.value:
        # Retain the original hardcoded test cases
        hardcoded_cases = {
            "find_zero": "",
            "decode_cyclic": "",
            "decode_shift": "",
            "by_length": "",
            "add": "",
            "triangle_area": "",
            "correct_bracketing": "",
            "solve": "",
            "sum_squares": "",
            "starts_one_ends": "",
        }
        
        # 如果有硬编码的测试用例，直接返回
        if entry_point in hardcoded_cases:
            return hardcoded_cases[entry_point]
        
        # 尝试所有可能的路径
        file_path = None
        for path in possible_paths:
            if os.path.exists(path):
                file_path = path
                break
        
        # 如果找不到文件，尝试下载
        if file_path is None:
            logger.warning(f"Could not find HumanEval test cases file. Tried paths: {possible_paths}")
            return None
            
    elif dataset == CodeDataset.MBPP.value:
        file_path = "data/aflow_benchmark_data/mbpp_public_test.jsonl"
        hardcoded_cases = {
            "remove_odd": "",
            "replace_spaces": "",
            "snake_to_camel": "",
            "Split": "",
            "swap_List": "",
            "square_Sum": "",
            "sort_sublists": "",
            "unique_sublists": "",
        }
        
        # 如果有硬编码的测试用例，直接返回
        if entry_point in hardcoded_cases:
            return hardcoded_cases[entry_point]
    
    # 尝试读取文件
    # breakpoint()
    try:
        with open(file_path, "r") as file:
            for line in file:
                data = json.loads(line)
                if data.get("entry_point") == entry_point:
                    # breakpoint()
                    return data.get("test")
    except FileNotFoundError:
        logger.error(f"Test cases file not found: {file_path}")
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON in test cases file: {file_path}")
    except Exception as e:
        logger.error(f"Error reading test cases file: {str(e)}")

    return None


def extract_test_cases(docstring: str) -> List[Tuple[str, List[Any], Any]]:
    # Use regular expressions to match test cases, now capturing function names and any output
    pattern = r">>> (\w+)\((.*?)\)\n\s*(.*?)(?=\n|$)"
    matches = re.findall(pattern, docstring, re.DOTALL)

    test_cases = []
    for match in matches:
        func_name, input_str, expected_output = match

        # Process input
        input_list = []
        for item in input_str.split(","):
            item = item.strip()
            try:
                # Try to convert input to numeric type
                if "." in item:
                    input_list.append(float(item))
                else:
                    input_list.append(int(item))
            except ValueError:
                # If unable to convert to numeric, keep as string
                input_list.append(item.strip("'\""))

        # Process output
        try:
            # Try to convert output to numeric or boolean value
            if expected_output.lower() == "true":
                expected_output = True
            elif expected_output.lower() == "false":
                expected_output = False
            elif "." in expected_output:
                expected_output = float(expected_output)
            else:
                expected_output = int(expected_output)
        except ValueError:
            # If unable to convert, keep as string
            expected_output = expected_output.strip("'\"")

        test_cases.append([func_name, input_list, expected_output])

    return test_cases


def test_cases_2_test_functions(solution: str, test_cases: str):
    tester_function = f"""
{solution}

{test_cases}
"""
    return tester_function


def test_case_2_test_function(solution: str, test_case: str, entry_point: str):
    tester_function = f"""
{solution}


def check(candidate):
    {test_case}

def test_check():
    check({entry_point})

test_check()
"""
    return tester_function


# def download_file(url: str, save_file: str, max_retries=3, timeout=10):

#     make_parent_folder(save_file)
#     for attempt in range(max_retries):
#         try:
#             resume_byte_pos = 0
#             if os.path.exists(save_file):
#                 resume_byte_pos = os.path.getsize(save_file)
            
#             response_head = requests.head(url=url)
#             total_size = int(response_head.headers.get("content-length", 0))

#             # Only return early if file exists AND has correct size
#             if os.path.exists(save_file) and resume_byte_pos >= total_size and total_size > 0:
#                 logger.info("File already downloaded completely.")
#                 return

#             headers = {'Range': f'bytes={resume_byte_pos}-'} if resume_byte_pos else {}
#             response = requests.get(url=url, stream=True, headers=headers, timeout=timeout)
#             response.raise_for_status()
#             mode = 'ab' if resume_byte_pos else 'wb'
#             progress_bar = tqdm(total=total_size, unit="iB", unit_scale=True, initial=resume_byte_pos)
            
#             with open(save_file, mode) as file:
#                 for chunk_data in response.iter_content(chunk_size=1024):
#                     if chunk_data:
#                         size = file.write(chunk_data)
#                         progress_bar.update(size)
            
#             progress_bar.close()

#             # Verify file was downloaded correctly
#             if os.path.exists(save_file) and os.path.getsize(save_file) >= total_size and total_size > 0:
#                 logger.info("Download completed successfully.")
#                 break
#             else:
#                 logger.warning("File size mismatch, retrying...")
#                 if os.path.exists(save_file):
#                     os.remove(save_file)
#                 time.sleep(5)
#         except (requests.ConnectionError, requests.Timeout) as e:
#             logger.warning(f"Download error: {e}. Retrying ({attempt+1}/{max_retries})...")
#             time.sleep(5)
#         except Exception as e:
#             error_message = f"Unexpected error: {e}"
#             logger.error(error_message)
#             raise ValueError(error_message)
#     else:
#         error_message = "Exceeded maximum retries. Download failed."
#         logger.error(error_message)
#         raise RuntimeError(error_message)
