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
    def _get_id(self, example: Any) -> Any:
        """
        Abstract method to return the id for a given example.
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
    def evaluate(self, prediction: Any, label: Any) -> dict:
        """
        Abstract method to evaluate a single prediction against the ground-truth label.
        
        Args:
            prediction (Any): The predicted output.
            label (Any): The actual ground-truth label.
        
        Returns:
            dict: A dictionary containing evaluation metrics.
        """
        pass 
    
    def get_label(self, example: List[Any]) -> Any:
        return self._get_label(example=example)
    
    def get_labels(self, examples: List[Any]) -> List[Any]:
        return [self._get_label(example=example) for example in examples]
    
    def get_id(self, example: List[Any]) -> Any:
        return self._get_id(example=example)
    
    def get_ids(self, examples: List[Any]) -> List[Any]:
        return [self._get_id(example=example) for example in examples]
    
    def get_data_by_mode(self, mode: str = "test") -> List[Any]:
        """
        Get the data from the benchmark by mode.
        """
        assert mode in ["train", "dev", "test"], f"Invalid value for mode: {mode}. Available choices: ['train', 'dev', 'test']"
        if mode == "train":
            if self._train_data is None:
                logger.warning(f"Train data for benchmark {type(self).__name__} is not loaded or None. Return an empty list.")
                return []
            data = self._train_data
        elif mode == "dev":
            if self._dev_data is None:
                logger.warning(f"Dev data for benchmark {type(self).__name__} is not loaded or None. Return an empty list.")
                return []
            data = self._dev_data 
        else:
            if self._test_data is None:
                logger.warning(f"Test data for benchmark {type(self).__name__} is not loaded or None. Return an empty list.")
                return []
            data = self._test_data
        return data
    
    def get_example_by_id(self, example_id: Any, mode: str = "test") -> Optional[Any]:
        """
        Get an example from the benchmark by its id.

        Args:
            example_id (Any): The id of the example to retrieve.
            mode (str): The mode to retrieve the example from, choices: ["train", "dev", "test"]
        
        Returns:
            Optional[Any]: The example if found, otherwise None.
        """
        data = self.get_data_by_mode(mode=mode)
        for example in data:
            if self._get_id(example=example) == example_id:
                return example
        return None
    
    def get_example_by_index(self, index: int, mode: str = "test") -> Optional[Any]:
        """
        Get an example from the benchmark by its index.

        Args:
            index (int): The index of the example to retrieve.
            mode (str): The mode to retrieve the example from, choices: ["train", "dev", "test"]
        
        Returns:
            Optional[Any]: The example if found, otherwise None.
        """
        data = self.get_data_by_mode(mode=mode)
        return data[index] if index < len(data) else None
        
    def _get_data(self, data: List[dict], indices: Optional[List[int]]=None, sample_k: Optional[int]=None) -> List[dict]:
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

    def get_train_data(self, indices: Optional[List[int]] = None, sample_k: Optional[int] = None) -> List[dict]:
        # Retrieves training data based on specified indices or random sampling.
        if self._train_data is None:
            logger.warning(f"Train data for benchmark {type(self).__name__} is not loaded or None. Return an empty list.")
            return [] 
        
        train_data = self._get_data(self._train_data, indices=indices, sample_k=sample_k)
        return train_data 
    
    def get_dev_data(self, indices: Optional[List[int]] = None, sample_k: Optional[int] = None) -> List[dict]:
        # Retrieves development data based on specified indices or random sampling.
        if self._dev_data is None:
            logger.warning(f"Dev data for benchmark {type(self).__name__} is not loaded or None. Return an empty list.")
            return [] 
        
        dev_data = self._get_data(self._dev_data, indices=indices, sample_k=sample_k)
        return dev_data  

    def get_test_data(self, indices: Optional[List[int]] = None, sample_k: Optional[int] = None) -> List[dict]:
        # Retrieves test data based on specified indices or random sampling.
        if self._test_data is None:
            logger.warning(f"Test data for benchmark {type(self).__name__} is not loaded or None. Return an empty list.")
            return [] 
        
        test_data = self._get_data(self._test_data, indices=indices, sample_k=sample_k)
        return test_data 
