import os
import regex 
import zipfile
import requests
from math import isclose
from typing import Any, List
from sympy import N, simplify
from sympy.parsing.latex import parse_latex
from sympy.parsing.sympy_parser import parse_expr

from ..core.logging import logger
from .benchmark import Benchmark
from ..utils.utils import make_parent_folder
from ..core.module_utils import load_json


def download_raw_math_data(save_folder: str):
    """Download the MATH dataset from the ModelScope website.
    
    Downloads the MATH benchmark dataset as a zip file from ModelScope,
    extracts its contents to the specified folder, and removes the zip file
    after extraction is complete.
    
    Args:
        save_folder: Directory path where the MATH dataset should be saved
    
    Raises:
        HTTPError: If the download request fails
    """
    url = "https://www.modelscope.cn/datasets/opencompass/competition_math/resolve/master/data/MATH.zip"
    logger.info(f"Downloading MATH data from {url} ...")
    save_file_path = os.path.join(save_folder, "MATH.zip")

    make_parent_folder(save_file_path)
    if not os.path.exists(save_file_path):
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(save_file_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    file.write(chunk)
    
    with zipfile.ZipFile(save_file_path, "r") as zip_ref:
        zip_ref.extractall(save_folder)
    if os.path.exists(save_file_path):
        os.remove(save_file_path)


class MATH(Benchmark):
    """Benchmark class for evaluating mathematical reasoning on the MATH dataset.
    
    MATH is a dataset of challenging competition mathematics problems,
    spanning various difficulty levels and subject areas. This class handles
    loading the dataset, extracting answers, evaluating solutions through
    symbolic and numerical comparisons, and computing accuracy metrics.
    
    The dataset includes problems across 7 subject areas (Algebra, Geometry, etc.)
    and 5 difficulty levels. Each problem contains LaTeX-formatted
    questions and solutions.
    
    Each MATH example has the following structure:
    {
        "id": "test-1", 
        "problem": "the problem", 
        "solution": "the solution",
        "level": "Level 1", # "Level 1", "Level 2", "Level 3", "Level 4", "Level 5", "Level ?"
        "type": "Algebra", # 'Geometry', 'Algebra', 'Intermediate Algebra', 'Counting & Probability', 'Precalculus', 'Number Theory', 'Prealgebra'
    }
    
    The benchmark evaluates answers using symbolic math equality checking
    and numerical approximation to handle equivalent mathematical expressions.
    """

    def __init__(self, path: str = None, mode: str = "all", **kwargs):
        """Initialize the MATH benchmark.
        
        Args:
            path: Directory path to store/load MATH data. Defaults to "~/.evoagentx/data/math"
            mode: Dataset mode to load ("train", "dev", "test", or "all"). Defaults to "all"
            **kwargs: Additional arguments passed to the parent class
        """
        path = os.path.expanduser(path or "~/.evoagentx/data/math")
        super().__init__(name=type(self).__name__, path=path, mode=mode, **kwargs)
    
    def _load_data_from_folders(self, data_folder: str) -> List[dict]:
        """Load MATH data from nested folder structure.
        
        Traverses the MATH dataset folder structure, which organizes problems
        by subject areas, and loads all problem JSON files into a list of examples.
        
        Args:
            data_folder: Path to the data folder (train or test)
            
        Returns:
            List of loaded problem examples with added IDs, or None if data_folder is None
        """
        if data_folder is None:
            return None
        data = []
        typ = "train" if "train" in data_folder else "test"
        sub_data_folders = os.listdir(data_folder)
        i = 0
        logger.info(f"loading MATH data from {data_folder} ...")
        for sub_data_folder in sub_data_folders:
            if os.path.isdir(os.path.join(data_folder, sub_data_folder)):
                files = os.listdir(os.path.join(data_folder, sub_data_folder))
                for file in files:
                    if file.endswith(".json"):
                        example = {"id": f"{typ}-{i+1}"}
                        example.update(load_json(os.path.join(data_folder, sub_data_folder, file), type="json"))
                        data.append(example)
                        i += 1
        return data
                
    def _load_data(self):
        """Load MATH dataset based on the specified mode.
        
        Downloads the MATH dataset if not already present and loads train/test
        data based on the specified mode. Currently, dev data is not available
        in the MATH dataset.
        """
        if not os.path.exists(os.path.join(self.path, "MATH")):
            download_raw_math_data(save_folder=self.path)
        data_folder = os.path.join(self.path, "MATH")

        # load data 
        if self.mode == "train" or self.mode == "all":
            self._train_data = self._load_data_from_folders(data_folder=os.path.join(data_folder, "train"))
        if self.mode == "dev" or self.mode == "all":
            self._dev_data = None 
        if self.mode == "test" or self.mode == "all":
            self._test_data = self._load_data_from_folders(data_folder=os.path.join(data_folder, "test"))
    
    def _get_label(self, example: Any) -> Any:
        """Extract the solution (label) from a MATH example.
        
        Args:
            example: A MATH problem object
            
        Returns:
            The solution string from the example
        """
        return example["solution"]
    
    def _get_id(self, example: Any) -> Any:
        """Extract the unique identifier from a MATH example.
        
        Args:
            example: A MATH problem object
            
        Returns:
            The example ID string
        """
        return example["id"] 
    
    def extract_answer(self, text: str) -> str: 
        """Extract the final answer from a solution text.
        
        Looks for boxed answers (common in LaTeX math solutions) using regex,
        and falls back to extracting the last sentence if no boxed answer is found.
        
        Args:
            text: Solution text to extract the answer from
            
        Returns:
            The extracted answer as a string
        """
        pattern = r"\\boxed{((?:[^{}]|{[^{}]*})*)}"
        boxed_matches = regex.findall(pattern, text, regex.DOTALL)
        if boxed_matches:
            return boxed_matches[-1].strip()
        
        sentence_end_pattern = r"(?<!\d)[.!?]\s+"
        sentences = regex.split(sentence_end_pattern, text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences[-1] if sentences else ""
    
    # Acknowledgement: https://github.com/geekan/MetaGPT/blob/main/metagpt/ext/aflow/benchmark/math.py#L40 
    def math_equal(self, prediction: Any, reference: Any) -> bool:
        """Check if two mathematical expressions are equivalent.
        
        Uses multiple strategies to determine equality:
        1. Direct string comparison
        2. Numerical comparison for digits
        3. Symbolic comparison for mathematical expressions
        
        Args:
            prediction: Predicted mathematical expression
            reference: Reference mathematical expression
            
        Returns:
            Boolean indicating whether the expressions are equivalent
        """
        if str(prediction) == str(reference):
            return True
        
        try:
            if self.is_digit(prediction) and self.is_digit(reference):
                prediction = self.parse_digits(prediction)
                reference = self.parse_digits(reference)
                return isclose(prediction, reference, abs_tol=1e-3)
        except Exception:
            pass

        try:
            return self.symbolic_equal(prediction, reference)
        except Exception:
            pass

        return False
    
    def is_digit(self, num: Any) -> bool:
        """Check if a string represents a numerical value.
        
        Args:
            num: Value to check
            
        Returns:
            Boolean indicating whether the value can be parsed as a number
        """
        return self.parse_digits(num) is not None
    
    def parse_digits(self, num: Any) -> float:
        """Parse a string into a numerical value.
        
        Handles various number formats including:
        - Regular numbers
        - Numbers with commas (e.g., "1,000")
        - Percentages (e.g., "50%")
        
        Args:
            num: String representing a number
            
        Returns:
            Parsed float value, or None if parsing fails
        """
        num = regex.sub(",", "", str(num))
        try:
            return float(num)
        except Exception:
            if num.endswith("%"):
                num = num[:-1]
                if num.endswith("\\"):
                    num = num[:-1]
                try:
                    return float(num) / 100
                except Exception:
                    pass
        return None

    def symbolic_equal(self, a: Any, b: Any) -> bool:
        """Check if two mathematical expressions are symbolically equivalent.
        
        Attempts to parse expressions using LaTeX and symbolic parsing,
        then compares them using symbolic simplification and numerical evaluation.
        
        Args:
            a: First mathematical expression
            b: Second mathematical expression
            
        Returns:
            Boolean indicating whether the expressions are symbolically equivalent
        """
        def _parse(s: Any) -> Any:
            for f in [parse_latex, parse_expr]:
                try:
                    return f(s)
                except Exception:
                    pass
            return s

        a = _parse(a)
        b = _parse(b)

        try:
            if simplify(a - b) == 0:
                return True
        except Exception:
            pass

        try:
            if isclose(N(a), N(b), abs_tol=1e-3):
                return True
        except Exception:
            pass
        return False

    def evaluate(self, prediction: Any, label: Any) -> dict:
        """Evaluate a predicted answer against the ground truth.
        
        Extracts the answers from both the prediction and ground truth solution,
        then checks if they are mathematically equivalent.
        
        Args:
            prediction: The model's predicted solution
            label: The ground truth solution
            
        Returns:
            Dictionary with solve_rate (1.0 for correct, 0.0 for incorrect)
        """
        ground_truth_answer = self.extract_answer(label)
        predicted_answer = self.extract_answer(prediction)
        solve_rate = 1.0 if self.math_equal(predicted_answer, ground_truth_answer) else 0.0
        return {"solve_rate": solve_rate}