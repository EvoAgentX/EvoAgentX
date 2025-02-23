import random
from abc import ABC, abstractmethod
from typing import Optional, List, Any

from ..core.logging import logger


class Benchmark(ABC):

    """
    Abstract base class for defining benchmarks. This class provides methods to load,
    retrieve, and evaluate benchmark data, with train, dev, and test splits.
    """

    def __init__(self, name: str, path: str, mode: str = "all", **kwargs):
        """
        Initializes the benchmark with a name and data path.
        
        Args:
            name (str): The name of the benchmark.
            path (str): The path to the dataset.
            mode (str): which type of data to load, choices: ["all", "train", "dev", "test"]
            **kwargs: Additional parameters for customization.
        """
        valid_mode = ["all", "train", "dev", "test"]
        assert mode in valid_mode, f"Invalid value for model: {mode}. Available choices: {valid_mode}"

        self.name = name
        self.path = path
        self.mode = mode
        self.kwargs = kwargs

        # 用于存储不同数据集的内部变量
        self._train_data: Optional[List[dict]] = None
        self._dev_data: Optional[List[dict]] = None
        self._test_data: Optional[List[dict]] = None

        # load data from `self.path`
        self._load_data()
    
    @abstractmethod
    def _load_data(self):
        """
        Abstract method to load data from `self.path` and assign it to `_train_data`, `_dev_data`, and `_test_data` if applicable.
        """
        pass

    @abstractmethod
    def _get_label(self, example: Any) -> Any:
        """
        Abstract method to return the ground-truth label for a given example.
        
        Args:
            example (Any): The input example for which the label is needed.
        
        Returns:
            Any: The ground-truth label associated with the example.
        """
        pass

    @abstractmethod
    def evaluate(prediction: Any, label: Any) -> dict:
        """
        Abstract method to evaluate a single prediction against the ground-truth label.
        
        Args:
            prediction (Any): The predicted output.
            label (Any): The actual ground-truth label.
        
        Returns:
            dict: A dictionary containing evaluation metrics.
        """
        pass 

    def get_labels(self, examples: List[Any]) -> List[Any]:
        return [self._get_label(example=example) for example in examples]
    
    def _get_data(self, data: List[dict], indices: List[int]=None, sample_k: int=None) -> List[dict]:
        """
        Retrieves a subset of data based on provided indices or a random sample.
        
        Args:
            data (List[dict]): The list of data examples.
            indices (List[int], optional): Specific indices of data to retrieve. Defaults to None.
            sample_k (int, optional): The number of random samples to retrieve. Defaults to None.
        
        Returns:
            List[dict]: The selected subset of data. If both `indices` and `sample_k` are None, it will return the original `data`.
        """
        if indices is None:
            indices = list(range(len(data)))
        if sample_k is not None:
            indices = random.sample(indices, k=min(sample_k, len(indices)))
        return_data = [data[idx] for idx in indices]
        return return_data

    def get_train_data(self, indices: List[int] = None, sample_k: int = None) -> List[dict]:
        # Retrieves training data based on specified indices or random sampling.
        if self._train_data is None:
            logger.warning(f"Train data for benchmark {type(self).__name__} is not loaded or None. Return an empty list.")
            return [] 
        
        train_data = self._get_data(self._train_data, indices=indices, sample_k=sample_k)
        return train_data 
    
    def get_dev_data(self, indices: List[int] = None, sample_k: int = None) -> List[dict]:
        # Retrieves development data based on specified indices or random sampling.
        if self._dev_data is None:
            logger.warning(f"Dev data for benchmark {type(self).__name__} is not loaded or None. Return an empty list.")
            return [] 
        
        dev_data = self._get_data(self._dev_data, indices=indices, sample_k=sample_k)
        return dev_data  

    def get_test_data(self, indices: List[int] = None, sample_k: int = None) -> List[dict]:
        # Retrieves test data based on specified indices or random sampling.
        if self._test_data is None:
            logger.warning(f"Test data for benchmark {type(self).__name__} is not loaded or None. Return an empty list.")
            return [] 
        
        test_data = self._get_data(self._test_data, indices=indices, sample_k=sample_k)
        return test_data 
