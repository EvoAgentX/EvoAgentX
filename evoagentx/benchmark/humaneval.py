import os 
import gzip 
import shutil
from typing import Union, Any, Callable
from .benchmark import CodingBenchmark
from ..core.logging import logger 
from ..utils.utils import download_file 
from ..core.module_utils import load_json
from ..utils.aflow_utils.data_utils import AFLOW_DATASET_FILES_MAP, download_aflow_benchmark_data


def download_raw_humaneval_data(save_folder: str): 
    """Download raw HumanEval dataset from the official GitHub repository.
    
    Fetches the compressed HumanEval benchmark data, decompresses it,
    and saves it to the specified folder. The original gzip file is removed
    after extraction.
    
    Args:
        save_folder: Directory path where the data should be saved
    """
    url = "https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz"
    logger.info(f"Downloading HumanEval data from {url} ...")
    save_file_path = os.path.join(save_folder, "HumanEval.jsonl.gz")
    download_file(url=url, save_file=save_file_path)
    with gzip.open(save_file_path, "rb") as f_in, open(os.path.join(save_folder, "HumanEval.jsonl"), "wb") as f_out: 
        shutil.copyfileobj(f_in, f_out) 
    if os.path.exists(save_file_path):
        os.remove(save_file_path)


def load_humaneval_data(data_path: str):
    """Load and preprocess HumanEval data from a JSONL file.
    
    Loads the dataset from the specified path and applies special fixes
    for known issues in specific problems (e.g., fixing import statements
    in problem 115).
    
    Args:
        data_path: Path to the HumanEval JSONL file
        
    Returns:
        List of HumanEval problem objects
    """
    data = load_json(data_path, type="jsonl") 
    # Handle 115 prompt to make its docstring well-formed
    for example in data:
        if example["task_id"] == "HumanEval/115":
            example["prompt"] = "import math\n" + example["prompt"].replace("import math", "")
    return data 


class HumanEval(CodingBenchmark):
    """Benchmark class for evaluating code generation on HumanEval.
    
    HumanEval is a collection of Python programming problems designed to test
    a model's ability to generate functionally correct code from natural language
    descriptions. This class handles loading the dataset, evaluating solutions,
    and computing metrics such as pass@k.
    
    Each HumanEval example has the following structure:
    {
        "task_id": "HumanEval/0", 
        "prompt": "from typing import List\n\ndef func_name(*args, **kwargs) -> return_type\n    "function description"\n\n", 
        "entry_point": "func_name",
        "canonical_solution": "canonical solution (code)",
        "test": "METADATA = {xxx}\n\n\ndef check(candidate):\n assert candidate(inputs) == output\n"
    }
    
    Attributes:
        k: An integer or list of integers specifying which pass@k metrics to compute
    """

    def __init__(self, path: str = None, mode: str = "all", timeout: int = 60, k: Union[int, list] = 1, **kwargs):
        """Initialize the HumanEval benchmark.
        
        Args:
            path: Directory path to store/load HumanEval data. Defaults to "~/.evoagentx/data/humaneval"
            mode: Dataset mode to load ("train", "dev", "test", or "all"). Defaults to "all"
            timeout: Execution timeout in seconds for code evaluation. Defaults to 60
            k: Integer or list of integers specifying which pass@k metrics to compute. Defaults to 1
            **kwargs: Additional arguments passed to the parent class
        """
        path = os.path.expanduser(path or "~/.evoagentx/data/humaneval")
        self.k = k 
        super().__init__(name=type(self).__name__, path=path, mode=mode, timeout=timeout, **kwargs)

    def _load_data(self):
        """Load HumanEval dataset based on the specified mode.
        
        Downloads the data if not already present at the specified path.
        Sets the appropriate data attributes (_train_data, _dev_data, _test_data)
        based on the specified mode.
        """
        data_path = os.path.join(self.path, "HumanEval.jsonl")
        if not os.path.exists(data_path):
            download_raw_humaneval_data(self.path)
        
        # load data  
        if self.mode == "train" or self.mode == "all":
            self._train_data = None 
        if self.mode == "dev" or self.mode == "all":
            self._dev_data = None 
        if self.mode == "test" or self.mode == "all":
            self._test_data = load_humaneval_data(data_path)

    def _get_label(self, example: Any):
        """Extract label information from a HumanEval example.
        
        Args:
            example: A HumanEval problem object
            
        Returns:
            A dictionary containing task ID, canonical solution, 
            test code, and entry point function name
        """
        # return the unit test code
        return {
            "task_id": example["task_id"],
            "canonical_solution": example["canonical_solution"],
            "test": example["test"],
            "entry_point": example["entry_point"]
        }
    
    def _get_id(self, example: Any):
        """Extract the unique identifier from a HumanEval example.
        
        Args:
            example: A HumanEval problem object
            
        Returns:
            The task ID string (e.g., "HumanEval/42")
        """
        return example["task_id"]
    
    def handle_special_cases(self, task_id: str, solution: str, test: str) -> bool:
        """Handle special cases requiring modifications to solutions or tests.
        
        Some HumanEval problems require special handling due to quirks in the
        problem statement or test cases. This method applies those fixes.
        
        Args:
            task_id: The task identifier
            solution: The solution code
            test: The test code
            
        Returns:
            Modified solution and test code as a tuple
        """
        if task_id == "HumanEval/50":
            solution = (
                '\n\ndef encode_shift(s: str):\n    """\n    returns encoded string by shifting every character by 5 in the alphabet.\n    """\n    return "".join([chr(((ord(ch) + 5 - ord("a")) % 26) + ord("a")) for ch in s])\n\n\n'
                + solution
            )
            return solution, test 
        
        return super().handle_special_cases(task_id=task_id, solution=solution, test=test)

    def evaluate(self, prediction: Any, label: Any) -> dict:
        """Evaluate solution code against HumanEval test cases.
        
        Executes the solution code against test cases and computes the pass@k
        metric, which measures the probability that at least one correct solution
        appears in k randomly sampled solutions.
        
        Args:
            prediction: Solution code as a string or list of strings
            label: Test code and metadata as a dictionary or list of dictionaries
            
        Returns:
            Dictionary with pass@k metrics
        """
        prediction, label = self._check_evaluation_inputs(prediction, label)

        results = []
        for solution in prediction:
            solution_states = []
            for label_data in label:
                task_id = label_data["task_id"]
                prompt = self.get_example_by_id(task_id)["prompt"]
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
        pass_at_k = self.compute_pass_at_k(results, k_list)
        
        return pass_at_k
    

class HumanEvaluPlus(HumanEval):
    """Extended version of HumanEval with additional test cases and inputs.
    
    HumanEvalPlus extends the original HumanEval benchmark with additional
    test cases, input validation contracts, and more rigorous testing.
    
    Each HumanEvalPlus example has the following structure:
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
    """AFlow-specific implementation of HumanEval benchmark.
    
    This class extends the HumanEval benchmark with features specific to the
    AFlow framework, including loading from AFlow-formatted data files,
    supporting asynchronous evaluation for workflows, and handling AFlow-specific
    test case formats.
    
    Attributes:
        Same as HumanEval, with additional support for AFlow structures
    """

    def __init__(self, path: str = None, mode: str = "all", timeout: int = 60, k: Union[int, list] = 1, **kwargs):
        """Initialize the AFlow-specific HumanEval benchmark.
        
        Args:
            path: Directory path to store/load data. Defaults to "~/.evoagentx/data/aflow/humaneval"
            mode: Dataset mode to load ("train", "dev", "test", or "all"). Defaults to "all"
            timeout: Execution timeout in seconds for code evaluation. Defaults to 60
            k: Integer or list of integers specifying which pass@k metrics to compute. Defaults to 1
            **kwargs: Additional arguments passed to the parent class
        """
        path = os.path.expanduser(path or "~/.evoagentx/data/aflow/humaneval")
        super().__init__(path=path, mode=mode, timeout=timeout, k=k, **kwargs)

    def _load_data_from_file(self, file_name: str):
        """Load data from a specific AFlow benchmark file.
        
        Downloads the file if not already present in the specified path.
        
        Args:
            file_name: Name of the file to load
            
        Returns:
            Loaded data as a list of objects, or None if file_name is None
        """
        if file_name is None:
            return None
        file_path = os.path.join(self.path, file_name)
        if not os.path.exists(file_path):
            download_aflow_benchmark_data(dataset="humaneval", save_folder=self.path)
        
        return load_json(path=file_path, type="jsonl")

    def _load_data(self):
        """Load AFlow-formatted HumanEval dataset based on the specified mode.
        
        Downloads the data if not already present, and loads train/dev/test
        data from AFlow-specific files based on the specified mode.
        Additionally loads test cases for evaluation.
        """
        if self.mode == "train" or self.mode == "all":
            logger.info(f"Loading train data from {AFLOW_DATASET_FILES_MAP['humaneval']['train']}")
            self._train_data = self._load_data_from_file(file_name=AFLOW_DATASET_FILES_MAP["humaneval"]["train"])
        if self.mode == "dev" or self.mode == "all":
            logger.info(f"Loading dev data from {AFLOW_DATASET_FILES_MAP['humaneval']['dev']}")
            self._dev_data = self._load_data_from_file(file_name=AFLOW_DATASET_FILES_MAP["humaneval"]["dev"])
        if self.mode == "test" or self.mode == "all":
            logger.info(f"Loading test data from {AFLOW_DATASET_FILES_MAP['humaneval']['test']}")
            self._test_data = self._load_data_from_file(file_name=AFLOW_DATASET_FILES_MAP["humaneval"]["test"])
        
        # load test cases 
        self._test_cases = self._load_data_from_file(file_name=AFLOW_DATASET_FILES_MAP["humaneval"]["test_cases"])
    
    def extract_test_cases_with_entry_point(self, entry_point: str):
        """Extract test cases for a specific function entry point.
        
        Some entry points have hardcoded test cases, while others are
        looked up in the loaded test cases.
        
        Args:
            entry_point: The function name to find test cases for
            
        Returns:
            Test code as a string, or None if no test cases are found
        """
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
        if entry_point in hardcoded_cases:
            return hardcoded_cases[entry_point]
        
        for case in self._test_cases:
            if case["entry_point"] == entry_point:
                return case["test"]
        
        return None
    
    async def evaluate_async(self, graph: Callable, example: Any) -> float:
        """Asynchronously evaluate a workflow graph on a HumanEval example.
        
        This method is specifically designed for AFlow workflows, allowing
        asynchronous evaluation of solutions generated by workflow graphs.
        
        Args:
            graph: A callable workflow graph that generates solutions
            example: A HumanEval problem object
            
        Returns:
            pass@1 score (0.0 or 1.0) indicating whether the solution passed
        """
        # generate solution 
        prompt, entry_point = example["prompt"], example["entry_point"]
        solution = await graph(prompt, entry_point)
        label = self._get_label(example)
        metrics = await super().evaluate_async(prediction=solution, label=label)
        return metrics["pass@1"]
    
