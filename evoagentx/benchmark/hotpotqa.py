import os 
from typing import Any
from .benchmark import Benchmark
from .measures import exact_match_score, f1_score
from ..core.logging import logger
from ..core.module_utils import load_json
from ..utils.utils import download_file
from ..utils.aflow_utils import AFLOW_DATASET_FILES_MAP, download_aflow_benchmark_data


HOTPOTQA_FILES_MAP = {"train": "hotpot_train_v1.1.json", "dev": "hotpot_dev_distractor_v1.json", "test": None}
VALIDE_RAW_HOTPOTQA_FILES = [file for file in list(HOTPOTQA_FILES_MAP.values()) if file is not None]

def download_raw_hotpotqa_data(name: str, save_folder: str):

    assert name in VALIDE_RAW_HOTPOTQA_FILES, f"'{name}' is an invalid hotpotqa file name. Available file names: {VALIDE_RAW_HOTPOTQA_FILES}"
    url = f"http://curtis.ml.cmu.edu/datasets/hotpot/{name}"
    typ = "train" if "train" in name else "dev"
    logger.info(f"Downloading HotPotQA {typ} data from: {url}")
    download_file(url=url, save_file=os.path.join(save_folder, name))


class HotPotQA(Benchmark):

    """
    {
        "_id": str, 
        "question": str, 
        "answer": str, 
        "context": [["context_title", ["context_sentence", "another_sentence"]]],
        "supporting_facts": [["supporting_title", supporting_sentence_index]],
        "type": str,
        "level": str
    }
    """

    def __init__(self, path: str, mode: str = "all", **kwargs):
        super().__init__(name=type(self).__name__, path=path, mode=mode, **kwargs)

    def _load_data_from_file(self, file_name: str):
        if file_name is None:
            return None
        file_path = os.path.join(self.path, file_name)
        if not os.path.exists(file_path):
            download_raw_hotpotqa_data(name=file_name, save_folder=self.path)
        logger.info(f"loading data from {file_path} ...")
        return load_json(path=file_path, type="json")

    def _load_data(self):
        if self.mode == "train" or self.mode == "all":
            self._train_data = self._load_data_from_file(file_name=HOTPOTQA_FILES_MAP["train"])
        if self.mode == "dev" or self.mode == "all":
            self._dev_data = self._load_data_from_file(file_name=HOTPOTQA_FILES_MAP["dev"])
        if self.mode == "test" or self.mode == "all":
            self._test_data = self._load_data_from_file(file_name=HOTPOTQA_FILES_MAP["test"])
    
    def _get_label(self, example: Any) -> Any:
        return example["answer"]
    
    def evaluate(prediction: Any, label: Any) -> dict:
        em = exact_match_score(prediction=prediction, ground_truth=label)
        f1 = f1_score(prediction=prediction, ground_truth=label)
        return {"f1": f1, "em": em}
    

class AFlowHotPotQA(HotPotQA):

    def _load_data_from_file(self, file_name: str):
        if file_name is None:
            return None
        file_path = os.path.join(self.path, file_name)
        if not os.path.exists(file_path):
            download_aflow_benchmark_data(dataset="hotpotqa", save_folder=self.path)
        logger.info(f"loading data from {file_path} ...")
        return load_json(path=file_path, type="jsonl")

    def _load_data(self):
        if self.mode == "train" or self.mode == "all":
            self._train_data = self._load_data_from_file(file_name=AFLOW_DATASET_FILES_MAP["hotpotqa"]["train"])
        if self.mode == "dev" or self.mode == "all":
            self._dev_data = self._load_data_from_file(file_name=AFLOW_DATASET_FILES_MAP["hotpotqa"]["dev"])
        if self.mode == "test" or self.mode == "all":
            self._test_data = self._load_data_from_file(file_name=AFLOW_DATASET_FILES_MAP["hotpotqa"]["test"])
    