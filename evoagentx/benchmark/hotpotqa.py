import os 
from typing import Any, Callable
from .benchmark import Benchmark
from .measures import exact_match_score, f1_score, acc_score
from ..core.logging import logger
from ..core.module_utils import load_json
from ..utils.utils import download_file
from ..utils.aflow_utils.data_utils import AFLOW_DATASET_FILES_MAP, download_aflow_benchmark_data


HOTPOTQA_FILES_MAP = {"train": "hotpot_train_v1.1.json", "dev": "hotpot_dev_distractor_v1.json", "test": None}
VALIDE_RAW_HOTPOTQA_FILES = [file for file in list(HOTPOTQA_FILES_MAP.values()) if file is not None]

def download_raw_hotpotqa_data(name: str, save_folder: str):
    """Download raw HotPotQA dataset files from the official source.
    
    Fetches the specified HotPotQA data file from the CMU dataset server
    and saves it to the specified folder.
    
    Args:
        name: Name of the file to download (must be in VALIDE_RAW_HOTPOTQA_FILES)
        save_folder: Directory path where the data should be saved
        
    Raises:
        AssertionError: If the specified file name is not valid
    """
    assert name in VALIDE_RAW_HOTPOTQA_FILES, f"'{name}' is an invalid hotpotqa file name. Available file names: {VALIDE_RAW_HOTPOTQA_FILES}"
    url = f"http://curtis.ml.cmu.edu/datasets/hotpot/{name}"
    typ = "train" if "train" in name else "dev"
    logger.info(f"Downloading HotPotQA {typ} data from: {url}")
    download_file(url=url, save_file=os.path.join(save_folder, name))


class HotPotQA(Benchmark):
    """Benchmark class for evaluating multi-hop question answering on HotPotQA dataset.
    
    HotPotQA is a question answering dataset featuring natural, multi-hop questions,
    where the model needs to use information from multiple paragraphs to answer
    a question correctly. This class handles loading the dataset, evaluating answers,
    and computing metrics like F1 score and exact match.
    
    Each HotPotQA example has the following structure:
    {
        "_id": str, 
        "question": str, 
        "answer": str, 
        "context": [["context_title", ["context_sentence", "another_sentence"]]],
        "supporting_facts": [["supporting_title", supporting_sentence_index]],
        "type": str,
        "level": str
    }
    
    The benchmark evaluates answers using exact match, F1 score, and accuracy metrics.
    """

    def __init__(self, path: str = None, mode: str = "all", **kwargs):
        """Initialize the HotPotQA benchmark.
        
        Args:
            path: Directory path to store/load HotPotQA data. Defaults to "~/.evoagentx/data/hotpotqa"
            mode: Dataset mode to load ("train", "dev", "test", or "all"). Defaults to "all"
            **kwargs: Additional arguments passed to the parent class
        """
        path = os.path.expanduser(path or "~/.evoagentx/data/hotpotqa")
        super().__init__(name=type(self).__name__, path=path, mode=mode, **kwargs)

    def _load_data_from_file(self, file_name: str):
        """Load HotPotQA data from a specific file.
        
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
            download_raw_hotpotqa_data(name=file_name, save_folder=self.path)
        logger.info(f"loading HotPotQA data from {file_path} ...")
        return load_json(path=file_path, type="json")

    def _load_data(self):
        """Load HotPotQA dataset based on the specified mode.
        
        Downloads data files if not already present and loads train/dev/test
        data based on the specified mode.
        """
        if self.mode == "train" or self.mode == "all":
            self._train_data = self._load_data_from_file(file_name=HOTPOTQA_FILES_MAP["train"])
        if self.mode == "dev" or self.mode == "all":
            self._dev_data = self._load_data_from_file(file_name=HOTPOTQA_FILES_MAP["dev"])
        if self.mode == "test" or self.mode == "all":
            self._test_data = self._load_data_from_file(file_name=HOTPOTQA_FILES_MAP["test"])
    
    def _get_label(self, example: Any) -> Any:
        """Extract the answer (label) from a HotPotQA example.
        
        Args:
            example: A HotPotQA problem object
            
        Returns:
            The answer string from the example
        """
        return example["answer"]
    
    def _get_id(self, example: Any) -> Any:
        """Extract the unique identifier from a HotPotQA example.
        
        Args:
            example: A HotPotQA problem object
            
        Returns:
            The example ID string
        """
        return example["_id"]
    
    def evaluate(self, prediction: Any, label: Any) -> dict:
        """Evaluate a predicted answer against the ground truth.
        
        Computes multiple metrics to assess the quality of the prediction:
        - Exact match: Binary score (1.0/0.0) for exact string match
        - F1 score: Harmonic mean of precision and recall based on word overlap
        - Accuracy: Binary score based on simplified string comparison
        
        Args:
            prediction: The model's predicted answer
            label: The ground truth answer
            
        Returns:
            Dictionary with f1, em (exact match), and acc (accuracy) scores
        """
        em = exact_match_score(prediction=prediction, ground_truth=label)
        f1 = f1_score(prediction=prediction, ground_truth=label)
        acc = acc_score(prediction=prediction, ground_truths=[label])
        return {"f1": f1, "em": em, "acc": acc}
    

class AFlowHotPotQA(HotPotQA):
    """AFlow-specific implementation of HotPotQA benchmark.
    
    This class extends the HotPotQA benchmark with features specific to the
    AFlow framework, including loading from AFlow-formatted data files and
    supporting asynchronous evaluation for workflows.
    
    The class handles formatting context and questions for AFlow workflows
    and evaluates generated answers using the same metrics as the base class.
    """

    def _load_data_from_file(self, file_name: str):
        """Load data from a specific AFlow benchmark file.
        
        Downloads the file if not already present in the specified path.
        AFlow data is stored in JSONL format rather than the original JSON.
        
        Args:
            file_name: Name of the file to load
            
        Returns:
            Loaded data as a list of objects, or None if file_name is None
        """
        if file_name is None:
            return None
        file_path = os.path.join(self.path, file_name)
        if not os.path.exists(file_path):
            download_aflow_benchmark_data(dataset="hotpotqa", save_folder=self.path)
        logger.info(f"loading data from {file_path} ...")
        return load_json(path=file_path, type="jsonl")

    def _load_data(self):
        """Load AFlow-formatted HotPotQA dataset based on the specified mode.
        
        Downloads data if not already present, and loads train/dev/test
        data from AFlow-specific files based on the specified mode.
        """
        if self.mode == "train" or self.mode == "all":
            self._train_data = self._load_data_from_file(file_name=AFLOW_DATASET_FILES_MAP["hotpotqa"]["train"])
        if self.mode == "dev" or self.mode == "all":
            self._dev_data = self._load_data_from_file(file_name=AFLOW_DATASET_FILES_MAP["hotpotqa"]["dev"])
        if self.mode == "test" or self.mode == "all":
            self._test_data = self._load_data_from_file(file_name=AFLOW_DATASET_FILES_MAP["hotpotqa"]["test"])
    
    async def evaluate_async(self, graph: Callable, example: Any) -> float:
        """Asynchronously evaluate a workflow graph on a HotPotQA example.
        
        This method is specifically designed for AFlow workflows, allowing
        asynchronous evaluation of answers generated by workflow graphs.
        It formats the input by combining the question with relevant context
        paragraphs and evaluates the generated answer using F1 score.
        
        Args:
            graph: A callable workflow graph that generates answers
            example: A HotPotQA problem object
            
        Returns:
            F1 score between the generated answer and ground truth
        """
        # generate solution 
        prompt = example["question"]
        paragraphs = [item[1] for item in example["context"] if isinstance(item[1], list)]
        context_str = "\n".join(" ".join(paragraph) for paragraph in paragraphs)
        inputs = f"Context: {context_str}\n\nQuestion: {prompt}\n\nAnswer:"
        solution = await graph(inputs)
        label = self._get_label(example)
        metrics = await super().evaluate_async(prediction=solution, label=label)
        return metrics["f1"]
    