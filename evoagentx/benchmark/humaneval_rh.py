import os
from ..core.logging import logger
from ..core.module_utils import load_json
from .benchmark import Benchmark
from ..utils.aflow.aflow_utils import AFLOW_DATASET_FILES_MAP, download_aflow_benchmark_data
from typing import Any, Callable, List, Dict, Tuple, Optional
from ..utils.sanitize import sanitize
import threading
import time
import pandas as pd
from datetime import datetime


class HumanEval(Benchmark):

    def __init__(self, path: str, mode: str = "all", **kwargs):
        super().__init__(name=type(self).__name__, path=path, mode=mode, **kwargs)

    def _load_data(self):
        pass
    
    def _get_label(self, example: Any):
        pass

    def _get_id(self, example: Any):
        pass
    
    def evaluate(self, prediction: Any, label: Any):
        pass
    

class AFlowHumanEval(HumanEval):
    
    def __init__(self, path: str, log_path: str = None, mode: str = "all", **kwargs):
        if 'name' in kwargs:
            del kwargs['name']  # 删除 kwargs 中的 name 参数，避免重复传递
        super().__init__(path=path, mode=mode, **kwargs)
        self.log_path = log_path if log_path else os.path.join(self.path, "logs")
        os.makedirs(self.log_path, exist_ok=True)

    PASS = "PASS"
    FAIL = "FAIL"
    
    # logger.info(f"self mode is {self.mode}")

    def _load_data_from_file(self, file_name: str):
        if file_name is None:
            return None
        file_path = os.path.join(self.path,file_name)
        if not os.path.exists(file_path):
            download_aflow_benchmark_data(dataset="humaneval", save_folder=self.path)
        # logger.info(f"loading data from {file_path} ...")
        return load_json(path=file_path, type="jsonl")
    
    
    def _load_data(self):
        
        if self.mode == "train" or self.mode == "all":
            self._train_data = self._load_data_from_file(file_name=AFLOW_DATASET_FILES_MAP["humaneval"]["train"])
            self.data = self._train_data
        if self.mode == "dev" or self.mode == "all":
            self._dev_data = self._load_data_from_file(file_name=AFLOW_DATASET_FILES_MAP["humaneval"]["dev"])
            self.data = self._dev_data
        if self.mode == "test" or self.mode == "all":
            self._test_data = self._load_data_from_file(file_name=AFLOW_DATASET_FILES_MAP["humaneval"]["test"])
            self.data = self._test_data
            
        return self.data
        
    
    def run_evaluation(self, graph: Callable, va_list: List[int]):
        data = self._load_data()
        # data = data[:1]
        # logger.info(f"data is {data}")
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
    
    def _generate_output(self, graph: Callable, prompt: str, entry_point: str):
        # logger.info(f"graph is {graph}")
        # logger.info(f"prompt is {prompt}")
        # logger.info(f"entry_point is {entry_point}")
        # logger.info(f"prompt is {prompt}")
        # logger.info(f"entry_point is {entry_point}")
        """
        example prompt:
        def can_arrange(arr):
            Create a function which returns the largest index of an element which
            is not greater than or equal to the element immediately preceding it. If
            no such element exists then return -1. The given array will not contain
            duplicate values.

            Examples:
            can_arrange([1,2,4,3,5]) = 3
            can_arrange([1,2,3]) = -1
            
        example entry_point:
            can_arrange
        """
        breakpoint()
        
        return graph(prompt, entry_point)
    
    
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
            # Generate prediction using the graph function
            prediction, cost = self._generate_output(graph, input_text, data["entry_point"])
            # logger.info(f"prediction is {prediction}")
            
            # Check the solution
            ret = self.check_solution(prediction, data["test"], data["entry_point"])
            test_case_details = ret[1]
            expected_output = test_case_details + expected_output

            # Calculate score based on the check result
            score = 1.0 if ret[0] == self.PASS else 0.0

            # Log mismatch if the score is 0
            # if score == 0:
            #     self.log_mismatch(input_text, expected_output, prediction, score)

            return input_text, prediction, expected_output, score, cost

        except Exception as e:
            logger.info(f"Maximum retries reached. Skipping this sample. Error: {e}")
            return input_text, str(e), expected_output, 0.0, 0.0
        
    class TimeoutError(Exception):
        pass
        
        
    def run_with_timeout(self, func, args, timeout):
        result = []
        stop_event = threading.Event()

        def target():
            try:
                result.append(func(*args))
            except Exception as e:
                result.append(e)
            finally:
                stop_event.set()

        thread = threading.Thread(target=target)
        thread.start()
        is_timeout = not stop_event.wait(timeout)
        
        # logger.info(f"result is {result}")

        if is_timeout:
            raise self.TimeoutError("Function execution timed out")

        if not result:
            return None
        if isinstance(result[0], Exception):
            raise result[0]
        return result[0]
        
        
    def check_solution(self, solution, test, entry_point):
        """
        First, sanitize the solution
        """
        solution = sanitize(code=solution, entrypoint=entry_point)
        try:
            global_dict = {
                "math": __import__("math"),
                "hashlib": __import__("hashlib"),
                "re": __import__("re"),
                "List": List,
                "Dict": Dict,
                "Tuple": Tuple,
                "Optional": Optional,
                "Any": Any,
            }

            # Add handling for special cases
            if entry_point == "decode_cyclic":
                solution = (
                    '\n\ndef encode_cyclic(s: str):\n    """\n    returns encoded string by cycling groups of three characters.\n    """\n    # split string to groups. Each of length 3.\n    groups = [s[(3 * i):min((3 * i + 3), len(s))] for i in range((len(s) + 2) // 3)]\n    # cycle elements in each group. Unless group has fewer elements than 3.\n    groups = [(group[1:] + group[0]) if len(group) == 3 else group for group in groups]\n    return "".join(groups)'
                    + "\n\n"
                    + solution
                )
            elif entry_point == "decode_shift":
                solution = (
                    '\n\ndef encode_shift(s: str):\n    """\n    returns encoded string by shifting every character by 5 in the alphabet.\n    """\n    return "".join([chr(((ord(ch) + 5 - ord("a")) % 26) + ord("a")) for ch in s])\n\n\n'
                    + solution
                )
            elif entry_point == "find_zero":
                solution = (
                    "\n\ndef poly(xs: list, x: float):\n    return sum(coeff * (x ** i) for i, coeff in enumerate(xs))\n\n"
                    + solution
                )

            exec(solution, global_dict)

            if entry_point not in global_dict:
                raise ValueError(f"Function {entry_point} is not defined in the solution.")

            exec(test, global_dict)

            check = global_dict["check"]
            
            # logger.info(f"check is {check}")

            result = self.run_with_timeout(check, (global_dict[entry_point],), 15)
            
            # logger.info(f"result is {result}")

            if result is None:
                result = (self.PASS, "The solution passed all test cases.")

        except self.TimeoutError:
            result = (
                self.FAIL,
                "Execution timed out. Please check if your solution contains infinite loops or overly time-consuming operations.",
            )
        except Exception as e:
            error_message = f"Error: {str(e)}.\n Solution: {solution}.\n Test: {test}"
            result = (self.FAIL, error_message)

            with open("error.log", "a", encoding="utf-8") as log_file:
                log_file.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {error_message}\n")

        return result
    
    
    def get_result_columns(self) -> List[str]:
        return ["inputs", "prediction", "expected_output", "score", "cost"]
        
        
    def save_results_to_csv(self, results: List[Tuple[Any, ...]], columns: List[str]):
        df = pd.DataFrame(results, columns=columns)
        avg_score = df["score"].mean()
        t_cost = df["cost"].max()
        a_cost = t_cost / len(df) if len(df) > 0 else 0
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{avg_score:.5f}_{current_time}.csv"
        output_file = os.path.join(self.log_path, filename)
        df.to_csv(output_file, index=False)
        logger.info(f"Results saved to {output_file}")
        return avg_score, a_cost, t_cost
    
    
    