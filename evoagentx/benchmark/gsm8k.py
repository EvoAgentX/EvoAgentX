import os
import regex 
from typing import Any, List, Callable
from ..core.logging import logger
from .benchmark import Benchmark
from ..utils.utils import download_file
from ..core.module_utils import load_json
from ..utils.aflow_utils.data_utils import AFLOW_DATASET_FILES_MAP, download_aflow_benchmark_data


GSM8K_FILES_MAP = {"train": "train.jsonl", "dev": None, "test": "test.jsonl"}
VALID_RAW_GSM8K_FILES = [file for file in list(GSM8K_FILES_MAP.values()) if file is not None]

def download_raw_gsm8k_data(name: str, save_folder: str):
    """Download raw GSM8K dataset files from the official GitHub repository.
    
    Fetches the specified GSM8K data file from the OpenAI grade-school-math
    repository and saves it to the specified folder.
    
    Args:
        name: Name of the file to download (must be in VALID_RAW_GSM8K_FILES)
        save_folder: Directory path where the data should be saved
        
    Raises:
        AssertionError: If the specified file name is not valid
    """
    assert name in VALID_RAW_GSM8K_FILES, f"'{name}' is an invalid GSM8K file name. Available file names: {VALID_RAW_GSM8K_FILES}"
    url = f"https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/{name}"
    typ = "train" if "train" in name else "test"
    logger.info(f"Downloading GSM8K {typ} data from: {url}")
    download_file(url=url, save_file=os.path.join(save_folder, name))


def load_gsm8k_data(file_path: str) -> List[dict]:
    """Load and preprocess GSM8K data from a JSONL file.
    
    Loads the dataset from the specified path and adds unique identifiers
    to each example based on the file type (train or test).
    
    Args:
        file_path: Path to the GSM8K JSONL file
        
    Returns:
        List of preprocessed GSM8K problem objects with added IDs
        
    Raises:
        AssertionError: If the specified file name is not valid
    """
    base_name = os.path.basename(file_path)
    file_type_map = {file_name: typ for typ, file_name in GSM8K_FILES_MAP.items()}
    assert base_name in file_type_map, f"'{base_name}' is an invalid gsm8k file name. Available file names: {VALID_RAW_GSM8K_FILES}"

    typ = file_type_map[base_name]
    data = load_json(path=file_path, type="jsonl")
    new_data = [] 
    for i, example in enumerate(data):
        item = {"id": f"{typ}-{i+1}"}
        item.update(example)
        new_data.append(item)
    return new_data
    

class GSM8K(Benchmark):
    """Benchmark class for evaluating math reasoning on GSM8K dataset.
    
    GSM8K (Grade School Math 8K) is a dataset of math word problems that
    test a model's ability to solve grade school level math problems requiring
    multi-step reasoning. This class handles loading the dataset, evaluating
    solutions, and computing metrics based on answer accuracy.
    
    Each GSM8K example has the following structure:
    {
        "id": "test-1", 
        "question": "the question", 
        "answer": "the answer"
    }
    
    The benchmark evaluates answers by extracting the final numerical value
    and comparing it to the ground truth answer.
    """
    
    def __init__(self, path: str = None, mode: str = "all", **kwargs):
        """Initialize the GSM8K benchmark.
        
        Args:
            path: Directory path to store/load GSM8K data. Defaults to "~/.evoagentx/data/gsm8k"
            mode: Dataset mode to load ("train", "dev", "test", or "all"). Defaults to "all"
            **kwargs: Additional arguments passed to the parent class
        """
        path = os.path.expanduser(path or "~/.evoagentx/data/gsm8k")
        super().__init__(name=type(self).__name__, path=path, mode=mode, **kwargs)

    def _load_data_from_file(self, file_name: str):
        """Load GSM8K data from a specific file.
        
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
            download_raw_gsm8k_data(name=file_name, save_folder=self.path)
        # loading data from file 
        logger.info(f"loading GSM8K data from {file_path} ...")
        return load_gsm8k_data(file_path=file_path)
    
    def _load_data(self):
        """Load GSM8K dataset based on the specified mode.
        
        Downloads data files if not already present and loads train/test
        data based on the specified mode.
        """
        if self.mode == "train" or self.mode == "all":
            self._train_data = self._load_data_from_file(file_name=GSM8K_FILES_MAP["train"])
        if self.mode == "dev" or self.mode == "all":
            self._dev_data = self._load_data_from_file(file_name=GSM8K_FILES_MAP["dev"])
        if self.mode == "test" or self.mode == "all":
            self._test_data = self._load_data_from_file(file_name=GSM8K_FILES_MAP["test"])
            
    def _get_label(self, example: Any) -> Any:
        """Extract the answer (label) from a GSM8K example.
        
        Args:
            example: A GSM8K problem object
            
        Returns:
            The answer string from the example
        """
        return example["answer"]
    
    def _get_id(self, example: Any) -> Any:
        """Extract the unique identifier from a GSM8K example.
        
        Args:
            example: A GSM8K problem object
            
        Returns:
            The example ID string (e.g., "train-42")
        """
        return example["id"]
    
    def extract_last_number(self, text: str) -> float:
        """Extract the last numerical value from a text.
        
        Uses regex to find all numbers in the text (including those with
        decimal points and commas) and returns the last one found.
        
        Args:
            text: Text to extract the number from
            
        Returns:
            The last number found as a float, or None if no numbers are found
        """
        matches = regex.findall(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?|\d+\.\d+", str(text))
        if matches:
            last_number = matches[-1].replace(",", "").strip()
            try:
                last_number = float(last_number)
                return last_number
            except ValueError:
                return None
        return None
    
    def evaluate(self, prediction: Any, label: Any) -> dict:
        """Evaluate a predicted answer against the ground truth.
        
        Extracts the final numerical values from both the prediction and ground truth,
        then checks if they match within a small tolerance.
        
        Args:
            prediction: The model's predicted answer
            label: The ground truth answer
            
        Returns:
            Dictionary with solve_rate (1.0 for correct, 0.0 for incorrect)
        """
        ground_truth_answer = self.extract_last_number(label)
        predicted_answer = self.extract_last_number(prediction)
        if predicted_answer is None:
            return {"solve_rate": 0.0}
        solve_rate = 1.0 if abs(predicted_answer - ground_truth_answer) < 1e-6 else 0.0
        return {"solve_rate": solve_rate}


class AFlowGSM8K(GSM8K): 
    """AFlow-specific implementation of GSM8K benchmark.
    
    This class extends the GSM8K benchmark with features specific to the
    AFlow framework, including loading from AFlow-formatted data files and
    supporting asynchronous evaluation for workflows.
    
    Attributes:
        Same as GSM8K, with additional support for AFlow structures
    """

    def __init__(self, path: str = None, mode: str = "all", **kwargs):
        """Initialize the AFlow-specific GSM8K benchmark.
        
        Args:
            path: Directory path to store/load data. Defaults to "~/.evoagentx/data/aflow/gsm8k"
            mode: Dataset mode to load ("train", "dev", "test", or "all"). Defaults to "all"
            **kwargs: Additional arguments passed to the parent class
        """
        path = os.path.expanduser(path or "~/.evoagentx/data/aflow/gsm8k")
        super().__init__(path=path, mode=mode, **kwargs)

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
            download_aflow_benchmark_data(dataset="gsm8k", save_folder=self.path)
        return load_json(path=file_path, type="jsonl")
        
    def _load_data(self):
        """Load AFlow-formatted GSM8K dataset based on the specified mode.
        
        Downloads data if not already present, and loads train/dev/test
        data from AFlow-specific files based on the specified mode.
        """
        if self.mode == "train" or self.mode == "all":
            logger.info(f"Loading train data from {AFLOW_DATASET_FILES_MAP['gsm8k']['train']}")
            self._train_data = self._load_data_from_file(file_name=AFLOW_DATASET_FILES_MAP["gsm8k"]["train"])
        if self.mode == "dev" or self.mode == "all":
            logger.info(f"Loading dev data from {AFLOW_DATASET_FILES_MAP['gsm8k']['dev']}")
            self._dev_data = self._load_data_from_file(file_name=AFLOW_DATASET_FILES_MAP["gsm8k"]["dev"])
        if self.mode == "test" or self.mode == "all":
            logger.info(f"Loading test data from {AFLOW_DATASET_FILES_MAP['gsm8k']['test']}")
            self._test_data = self._load_data_from_file(file_name=AFLOW_DATASET_FILES_MAP["gsm8k"]["test"])       
    
    async def evaluate_async(self, graph: Callable, example: Any) -> float:
        """Asynchronously evaluate a workflow graph on a GSM8K example.
        
        This method is specifically designed for AFlow workflows, allowing
        asynchronous evaluation of solutions generated by workflow graphs.
        
        Args:
            graph: A callable workflow graph that generates solutions
            example: A GSM8K problem object
            
        Returns:
            solve_rate score (0.0 or 1.0) indicating whether the solution is correct
        """
        input_text = example["question"] 
        label = self._get_label(example) 
        output = await graph(input_text)
        metrics = await super().evaluate_async(prediction=output, label=label)
        return metrics["solve_rate"]