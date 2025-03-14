import os 
import regex
from typing import Union, Dict, Any, List
from ..core.logging import logger
from .benchmark import CodingBenchmark 
from ..utils.utils import download_file
from ..core.module_utils import load_json


def download_raw_mbpp_data(name: str, save_folder: str):
    url = "https://raw.githubusercontent.com/google-research/google-research/master/mbpp/sanitized-mbpp.json"
    logger.info(f"Downloading MBPP data from: {url}")
    download_file(url=url, save_file=os.path.join(save_folder, name))


def load_mbpp_data(data_path: str):

    """
    load MBPP data from the given path and convert to HumanEval format
    """

    def extract_func_name(func_header: str) -> str:
        func_name_pattern = r"def\s+([a-zA-Z_]\w*)\s*\("
        match = regex.search(func_name_pattern, func_header)
        if match:
            return match.group(1)
        else:
            return None

    def extract_func_header(code: str, test_list: List[str]) -> str:
        lines = code.split("\n")
        imports, defs = [], []
        for line in lines:
            if line.startswith("def "):
                break
            imports.append(line)
        for line in lines:
            if line.startswith("def "):
                defs.append(line)
        func_head = None
        for header in defs:
            func_name = extract_func_name(header)
            if func_name is None:
                continue 
            if all(func_name in test for test in test_list):
                func_head = header 
                break 
        if func_head is None:
            logger.warning(f"No function header found for {code}")
        return ("\n".join(imports) + "\n\n" + func_head).strip() 

    data = load_json(data_path, type="json")

    for example in data:
        original_prompt = example["prompt"] 
        code = example["code"]
        test_list = [assert_str.strip() for assert_str in example["test_list"]]
        func_header = extract_func_header(code, test_list)

        if example["task_id"] == 56:
            # change the `check` function to `check_answer`
            func_header = func_header.replace("check", "check_answer")
            code = code.replace("check", "check_answer")
            test_list = [test.replace("check", "check_answer") for test in test_list]

        prompt = example["prompt"] + "\n\n" + func_header + "\n"
        canonical_solution = code 
        test = "def check(candidate):\n    " + "\n    ".join(test_list) + "\n" 
        entry_point = extract_func_name(func_header)

        example["prompt"] = prompt 
        example["entry_point"] = entry_point 
        example["canonical_solution"] = canonical_solution 
        example["test"] = test 
        example["original_prompt"] = original_prompt 
    
    return data


class MBPP(CodingBenchmark):

    """
    {
        "task_id" (int): 2, 
        "prompt" (str): "Write a function to find the shared elements from the given two lists.",
        "code" (str): "def similar_elements(test_tup1, test_tup2):\n  res = tuple(set(test_tup1) & set(test_tup2))\n  return (res) ", 
        "test_imports": [] 
        "test_list" (List[str]): ['assert set(similar_elements((3, 4, 5, 6),(5, 7, 4, 10))) == set((4, 5))', 'assert set(similar_elements((1, 2, 3, 4),(5, 4, 3, 7))) == set((3, 4))', 'assert set(similar_elements((11, 12, 14, 13),(17, 15, 14, 13))) == set((13, 14))']
    }
    """

    def __init__(self, path: str = None, mode: str = "all", timeout: int = 60, k: Union[int, list] = 1,**kwargs):
        path = os.path.expanduser(path or "~/.evoagentx/data/mbpp")
        self.k = k 
        super().__init__(name=type(self).__name__, path=path, mode=mode, timeout=timeout, **kwargs)
    
    def _load_data(self):

        data_path = os.path.join(self.path, "sanitized-mbpp.json")
        if not os.path.exists(data_path):
            download_raw_mbpp_data(name="sanitized-mbpp.json", save_folder=self.path)
        
        # load data 
        if self.mode == "train" or self.mode == "all":
            self._train_data = None 
        if self.mode == "dev" or self.mode == "all":
            self._dev_data = None 
        if self.mode == "test" or self.mode == "all":
            self._test_data = load_mbpp_data(data_path)
    
    def _get_id(self, example: Any) -> Any:
        return example["task_id"]

    def _get_label(self, example: Any) -> Any:
        # return the unit test code
        return {
            "task_id": example["task_id"],
            "canonical_solution": example["canonical_solution"],
            "test": example["test"],
            "entry_point": example["entry_point"]
        }
    
    def evaluate(self, prediction: Any, label: Any) -> dict:
        """
        Evaluate the solution code.

        Args:
            prediction (str | List[str]): The solution code(s).
            label (dict | List[dict]): The unit test code(s).

        Returns:
            dict: The evaluation metrics (pass@k).
        """
        assert isinstance(prediction, str) or isinstance(prediction, list), "prediction must be a string or a list of strings, but got {}".format(type(prediction))
        assert isinstance(label, dict) or isinstance(label, list), "label must be a string or a list of strings, but got {}".format(type(label))
        prediction = [prediction] if isinstance(prediction, str) else prediction
        label = [label] if isinstance(label, dict) else label

        results = []
        for solution in prediction:
            solution_states = []
            for label_data in label:
                task_id = label_data["task_id"]
                prompt = self.get_example_by_id(task_id, "test")["prompt"]
                unit_test = label_data["test"]
                entry_point = label_data["entry_point"]
                state, message = self.check_solution(
                    task_id=task_id, 
                    solution=prompt + solution,
                    test=unit_test, 
                    entry_point=entry_point
                )
                if state != self.SUCCESS:
                    break 
                solution_states.append(state)
            results.append(len(solution_states)==len(label) and all(state==self.SUCCESS for state in solution_states))
        
        k_list = [self.k] if isinstance(self.k, int) else self.k
        pass_at_k = {}
        for k in k_list:
            pass_at_k[f"pass@{k}"] = sum(results[:min(k, len(results))]) / min(k, len(results))
        
        return pass_at_k
    
    