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
    """
    Download the MATH data from the modelscope website.
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
        path = os.path.expanduser(path or "~/.evoagentx/data/math")
        super().__init__(name=type(self).__name__, path=path, mode=mode, **kwargs)
    
    def _load_data_from_folders(self, data_folder: str) -> List[dict]:
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
        return example["solution"]
    
    def _get_id(self, example: Any) -> Any:
        return example["id"] 
    
    def extract_answer(self, text: str) -> str: 

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
        return self.parse_digits(num) is not None
    
    def parse_digits(self, num: Any) -> float:
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
        ground_truth_answer = self.extract_answer(label)
        predicted_answer = self.extract_answer(prediction)
        solve_rate = 1.0 if self.math_equal(predicted_answer, ground_truth_answer) else 0.0
        return {"solve_rate": solve_rate}