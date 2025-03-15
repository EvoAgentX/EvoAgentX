import os
import regex 
from typing import Any, List
from ..core.logging import logger
from .benchmark import Benchmark
from ..utils.utils import download_file
from ..core.module_utils import load_json


GSM8K_FILES_MAP = {"train": "train.jsonl", "dev": None, "test": "test.jsonl"}
VALID_RAW_GSM8K_FILES = [file for file in list(GSM8K_FILES_MAP.values()) if file is not None]

def download_raw_gsm8k_data(name: str, save_folder: str):

    assert name in VALID_RAW_GSM8K_FILES, f"'{name}' is an invalid GSM8K file name. Available file names: {VALID_RAW_GSM8K_FILES}"
    url = f"https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/{name}"
    typ = "train" if "train" in name else "test"
    logger.info(f"Downloading GSM8K {typ} data from: {url}")
    download_file(url=url, save_file=os.path.join(save_folder, name))


def load_gsm8k_data(file_path: str) -> List[dict]:

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

    """
    {
        "id": "test-1", 
        "question": "the question", 
        "answer": "the answer"
    }
    """
    
    def __init__(self, path: str = None, mode: str = "all", **kwargs):
        path = os.path.expanduser(path or "~/.evoagentx/data/gsm8k")
        super().__init__(name=type(self).__name__, path=path, mode=mode, **kwargs)

    def _load_data_from_file(self, file_name: str):
        if file_name is None:
            return None
        file_path = os.path.join(self.path, file_name)
        if not os.path.exists(file_path):
            download_raw_gsm8k_data(name=file_name, save_folder=self.path)
        # loading data from file 
        logger.info(f"loading GSM8K data from {file_path} ...")
        return load_gsm8k_data(file_path=file_path)
    
    def _load_data(self):
        if self.mode == "train" or self.mode == "all":
            self._train_data = self._load_data_from_file(file_name=GSM8K_FILES_MAP["train"])
        if self.mode == "dev" or self.mode == "all":
            self._dev_data = self._load_data_from_file(file_name=GSM8K_FILES_MAP["dev"])
        if self.mode == "test" or self.mode == "all":
            self._test_data = self._load_data_from_file(file_name=GSM8K_FILES_MAP["test"])
            
    def _get_label(self, example: Any) -> Any:
        return example["answer"]
    
    def _get_id(self, example: Any) -> Any:
        return example["id"]
    
    def extract_last_number(self, text: str) -> float:
        """
        Extract the last number from a text.
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
        ground_truth_answer = self.extract_last_number(label)
        predicted_answer = self.extract_last_number(prediction)
        if predicted_answer is None:
            return {"solve_rate": 0.0}
        solve_rate = 1.0 if abs(predicted_answer - ground_truth_answer) < 1e-6 else 0.0
        return {"solve_rate": solve_rate}
         
