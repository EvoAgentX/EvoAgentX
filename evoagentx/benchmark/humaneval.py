import os 
import gzip 
import shutil
from typing import Union, Any, List, Callable, Tuple
from .benchmark import CodingBenchmark
from ..core.logging import logger 
from ..utils.utils import download_file 
from ..core.module_utils import load_json
import pandas as pd
from datetime import datetime
from ..utils.aflow.aflow_utils import AFLOW_DATASET_FILES_MAP, download_aflow_benchmark_data


def download_raw_humaneval_data(save_folder: str): 
    url = "https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz"
    logger.info(f"Downloading HumanEval data from {url} ...")
    save_file_path = os.path.join(save_folder, "HumanEval.jsonl.gz")
    download_file(url=url, save_file=save_file_path)
    with gzip.open(save_file_path, "rb") as f_in, open(os.path.join(save_folder, "HumanEval.jsonl"), "wb") as f_out: 
        shutil.copyfileobj(f_in, f_out) 
    if os.path.exists(save_file_path):
        os.remove(save_file_path)


def load_humaneval_data(data_path: str):
    data = load_json(data_path, type="jsonl") 
    
    logger.info(f"Loaded {len(data)} examples")
    # breakpoint()
    # Handle 115 prompt to make its docstring well-formed
    for example in data:
        if example["task_id"] == "HumanEval/115":
            example["prompt"] = "import math\n" + example["prompt"].replace("import math", "")
    return data 


class HumanEval(CodingBenchmark):

    """
    {
        "task_id": "HumanEval/0", 
        "prompt": "from typing import List\n\ndef func_name(*args, **kwargs) -> return_type\n    "function description"\n\n", 
        "entry_point": "func_name",
        "canonical_solution": "canonical solution (code)",
        "test": "METADATA = {xxx}\n\n\ndef check(candidate):\n assert candidate(inputs) == output\n"
    }
    """

    def __init__(self, path: str = None, mode: str = "all", timeout: int = 60, k: Union[int, list] = 1, **kwargs):
        path = os.path.expanduser(path or "~/.evoagentx/data/humaneval")
        self.k = k 
        super().__init__(name=type(self).__name__, path=path, mode=mode, timeout=timeout, **kwargs)

    def _load_data(self):

        data_path = os.path.join(self.path, "HumanEval.jsonl")
        
        logger.info(f"Loading HumanEval data from {data_path} ...")
        
        if not os.path.exists(data_path):
            download_raw_humaneval_data(self.path)
        
        # load data  
        if self.mode == "train" or self.mode == "all":
            self._train_data = None 
        if self.mode == "dev" or self.mode == "all":
            self._dev_data = None 
        if self.mode == "test" or self.mode == "all":
            self._test_data = load_humaneval_data(data_path)
            
        logger.info(f"Loaded {len(self._test_data)} test examples")

    def _get_label(self, example: Any):
        # return the unit test code
        return {
            "task_id": example["task_id"],
            "canonical_solution": example["canonical_solution"],
            "test": example["test"],
            "entry_point": example["entry_point"]
        }
    
    def _get_id(self, example: Any):
        return example["task_id"]
    
    def handle_special_cases(self, task_id: str, solution: str, test: str) -> bool:
        """
        Handle special cases for HumanEval.
        """
        if task_id == "HumanEval/50":
            solution = (
                '\n\ndef encode_shift(s: str):\n    """\n    returns encoded string by shifting every character by 5 in the alphabet.\n    """\n    return "".join([chr(((ord(ch) + 5 - ord("a")) % 26) + ord("a")) for ch in s])\n\n\n'
                + solution
            )
            return solution, test 
        
        return super().handle_special_cases(task_id=task_id, solution=solution, test=test)

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
    
    
class HumanEvaluPlus(HumanEval):

    """
    {
        "task_id": "HumanEvalPlus/0",
        "prompt": "function signature with docstring such as: from typing import List\n\ndef func_name(*args, **kwargs) -> return_type\n    "function description"\n\n", 
        "entry_point": "func_name",
        "canonical_solution": "canonical solution (code)",
        "test": "METADATA = {xxx}\n\n\ndef check(candidate):\n assert candidate(inputs) == output\n", 
        "contract": "string", # the assertions for the function's input (validity)
        "base_input": list, # the test inputs from original HumanEval
        "plus_input": list, # the test inputs brought by EvalPlus
        "atol": int, # the absolute tolerance for diff-testing
    }
    """
    pass 


class AFlowHumanEval(HumanEval):
    def __init__(self, path: str = None, log_path: str = None, mode: str = "all", timeout: int = 60, k: Union[int, list] = 1, **kwargs):
        path = os.path.expanduser(path or "~/.evoagentx/data/humaneval")
        self.k = k
        self.log_path = log_path if log_path else os.path.join(path, "logs")
        os.makedirs(self.log_path, exist_ok=True)
        super().__init__(path=path, mode=mode, timeout=timeout, k=k, **kwargs)


    def _load_data_from_file(self, file_name: str):
        if file_name is None:
            return None
        file_path = os.path.join(self.path, file_name)
        # logger.info(f"file_path is {file_path}")
        if not os.path.exists(file_path):
            download_aflow_benchmark_data(dataset="humaneval", save_folder=self.path)
        
        return load_json(path=file_path, type="jsonl")
        
        
    def _load_data(self):
        
        if self.mode == "train" or self.mode == "all":
            self._train_data = self._load_data_from_file(file_name=AFLOW_DATASET_FILES_MAP["humaneval"]["train"])
            self.data = self._train_data
        if self.mode == "dev" or self.mode == "all":
            self._dev_data = self._load_data_from_file(file_name=AFLOW_DATASET_FILES_MAP["humaneval"]["dev"])
            self.data = self._dev_data
            # logger.info(f"self.data is {self.data}")
        if self.mode == "test" or self.mode == "all":
            self._test_data = self._load_data_from_file(file_name=AFLOW_DATASET_FILES_MAP["humaneval"]["test"])
            self.data = self._test_data
            
        return self.data


    def run_evaluation(self, graph: Callable, va_list: List[int]):
        # The _load_data method in HumanEval doesn't return data, it populates self._test_data
        # So we need to get the data from self._test_data instead
        # if self._test_data is None:
        #     self._load_data()  # Make sure data is loaded
        
        
        # logger.info(f"self.mode is {self.mode}")
        # data = self._test_data
        data = self._load_data()
        # data = data[:1]
        # logger.info(f"data is {data}")
        
        if data is None:
            logger.error("No test data available. Make sure the data is properly loaded.")
            return 0.0, 0.0, 0.0
            
        results = self.evaluate_all_problems(data, graph)
        columns = self.get_result_columns()
        average_score, average_cost, total_cost = self.save_results_to_csv(results, columns)
        logger.info(f"Average score on {self.name} dataset: {average_score:.5f}")
        logger.info(f"Total Cost: {total_cost:.5f}")
        return average_score, average_cost, total_cost

    def evaluate_all_problems(self, data: List[Any], graph: Callable, max_concurrent_tasks: int = 50):
        results = []
        for problem in data:
            results.append(self.evaluate_problem(problem, graph))
        return results

    def evaluate_problem(self, data: dict, graph: Callable):
        input_text = data["prompt"]
        expected_output = (
            "\nCorrect Solution:\ndef "
            + data["entry_point"]
            + "(params you should put here):"
            + "\n\n"
            + data["canonical_solution"]
        )

        try:
            prediction, cost = self._generate_output(graph, input_text, data["entry_point"])
            ret = self.check_solution(data["task_id"], prediction, data["test"], data["entry_point"])
            test_case_details = ret[1]
            expected_output = test_case_details + expected_output
            score = 1.0 if ret[0] == self.SUCCESS else 0.0
            return input_text, prediction, expected_output, score, cost

        except Exception as e:
            logger.info(f"Maximum retries reached. Skipping this sample. Error: {e}")
            return input_text, str(e), expected_output, 0.0, 0.0

    def _generate_output(self, graph: Callable, prompt: str, entry_point: str):
        return graph(prompt, entry_point)

    def get_result_columns(self) -> List[str]:
        return ["inputs", "prediction", "expected_output", "score", "cost"]

    def save_results_to_csv(self, results: List[Tuple[Any, ...]], columns: List[str]):
        df = pd.DataFrame(results, columns=columns)
        avg_score = df["score"].mean()
        t_cost = df["cost"].max()
        a_cost = t_cost / len(df) if len(df) > 0 else 0
        current_time = datetime.now().strftime("%Y%m%d_%H-%M-%S")
        filename = f"{avg_score:.5f}_{current_time}.csv"
        output_file = os.path.join(self.log_path, filename)
        df.to_csv(output_file, index=False)
        logger.info(f"Results saved to {output_file}")
        return avg_score, a_cost, t_cost
