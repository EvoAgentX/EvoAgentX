import asyncio
import random
import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Optional, List, Dict, Any

from ..core.logging import logger
from ..core.callbacks import timeout, TimeoutException
from ..utils.sanitize import sanitize
from .lcb_utils.evaluation import estimate_pass_at_k 


class BaseBenchmark(ABC):

    """
    A ``BaseBenchmark`` is responsible only for loading / storing / providing data
    across train/dev/test splits.

    Subclasses must implement:
        - ``_load_data``: populate ``_train_data`` / ``_dev_data`` / ``_test_data``
        - ``_get_id``: return a stable unique id for an example
        - ``_get_label``: return the ground-truth label for an example

    Note:
        This coexists with the legacy ``Benchmark`` below during migration. Once all
        datasets and call sites move over, ``Benchmark`` will be removed and this can be
        renamed back to ``Benchmark``.
    """

    def __init__(self, name: str, path: str, mode: str = "all", **kwargs):
        """Initialize the benchmark.

        Args:
            name (str): The name of the benchmark.
            path (str): The path to the dataset.
            mode (str): which split(s) to load, one of ["all", "train", "dev", "test"].
            **kwargs: Additional parameters for customization.
        """
        valid_mode = ["all", "train", "dev", "test"]
        assert mode in valid_mode, f"Invalid value for mode: {mode}. Available choices: {valid_mode}"

        self.name = name
        self.path = path
        self.mode = mode
        self.kwargs = kwargs

        self._train_data: Optional[List[Any]] = None
        self._dev_data: Optional[List[Any]] = None
        self._test_data: Optional[List[Any]] = None

        self._load_data()
        self._validate_unique_ids()

    # ------------------------------------------------------------------ #
    # Abstract hooks
    # ------------------------------------------------------------------ #
    @abstractmethod
    def _load_data(self):
        """Load data from ``self.path`` into ``_train_data`` / ``_dev_data`` / ``_test_data``."""
        pass

    @abstractmethod
    def _get_id(self, example: Any) -> Any:
        """Return the stable unique id for a given example."""
        pass

    @abstractmethod
    def _get_label(self, example: Any) -> Any:
        """Return the ground-truth label for a given example."""
        pass

    def _validate_unique_ids(self) -> None:
        """Warn when loaded examples have duplicate ids within the same split."""
        for split_name, data in (
            ("train", self._train_data),
            ("dev", self._dev_data),
            ("test", self._test_data),
        ):
            if data is None:
                continue

            seen_ids = []
            duplicate_ids = []
            for example in data:
                example_id = self._get_id(example=example)
                if any(example_id == seen_id for seen_id in seen_ids):
                    if not any(example_id == duplicate_id for duplicate_id in duplicate_ids):
                        duplicate_ids.append(example_id)
                else:
                    seen_ids.append(example_id)

            if duplicate_ids:
                logger.warning(
                    f"{split_name} data for benchmark {type(self).__name__} contains non-unique example ids: "
                    f"{sorted(duplicate_ids, key=str)}. Methods such as get_example_by_id may return only the "
                    "first matching example, making id-based lookup ambiguous."
                )

    # ------------------------------------------------------------------ #
    # Id / label accessors
    # ------------------------------------------------------------------ #
    def get_id(self, example: Any) -> Any:
        return self._get_id(example=example)

    def get_ids(self, examples: List[Any]) -> List[Any]:
        return [self._get_id(example=example) for example in examples]

    def get_label(self, example: Any) -> Any:
        return self._get_label(example=example)

    def get_labels(self, examples: List[Any]) -> List[Any]:
        return [self._get_label(example=example) for example in examples]

    # ------------------------------------------------------------------ #
    # Data accessors
    # ------------------------------------------------------------------ #
    def get_data_by_mode(self, mode: str = "test") -> List[Any]:
        assert mode in ["train", "dev", "test"], \
            f"Invalid value for mode: {mode}. Available choices: ['train', 'dev', 'test']"
        data = {"train": self._train_data, "dev": self._dev_data, "test": self._test_data}[mode]
        if data is None:
            logger.warning(f"{mode} data for benchmark {type(self).__name__} is not loaded or None. Return an empty list.")
            return []
        return data

    def get_example_by_id(self, example_id: Any, mode: str = None) -> Optional[Any]:
        if mode is not None and mode not in ["train", "dev", "test", "all"]:
            raise ValueError(f"Invalid value for mode: {mode}. Available choices: ['train', 'dev', 'test', 'all']")
        if mode is None or mode == "all":
            data = []
            for split in (self._train_data, self._dev_data, self._test_data):
                if split is not None:
                    data.extend(split)
        else:
            data = self.get_data_by_mode(mode=mode)
        for example in data:
            if self._get_id(example=example) == example_id:
                return example
        return None

    def get_example_by_index(self, index: int, mode: str = "test") -> Optional[Any]:
        data = self.get_data_by_mode(mode=mode)
        return data[index] if index < len(data) else None

    def _get_data(
        self,
        data: List[Any],
        indices: Optional[List[int]] = None,
        sample_k: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> List[Any]:
        """Select a subset of ``data`` by explicit indices and/or random sampling."""
        if data is None:
            return []
        if indices is None:
            indices = list(range(len(data)))
        if sample_k is not None:
            if seed is not None:
                random.seed(seed)
            indices = random.sample(indices, k=min(sample_k, len(indices)))
        return [data[idx] for idx in indices]

    def get_train_data(self, indices: Optional[List[int]] = None, sample_k: Optional[int] = None, seed: Optional[int] = None) -> List[Any]:
        return self._get_data(self.get_data_by_mode("train"), indices=indices, sample_k=sample_k, seed=seed)

    def get_dev_data(self, indices: Optional[List[int]] = None, sample_k: Optional[int] = None, seed: Optional[int] = None) -> List[Any]:
        return self._get_data(self.get_data_by_mode("dev"), indices=indices, sample_k=sample_k, seed=seed)

    def get_test_data(self, indices: Optional[List[int]] = None, sample_k: Optional[int] = None, seed: Optional[int] = None) -> List[Any]:
        return self._get_data(self.get_data_by_mode("test"), indices=indices, sample_k=sample_k, seed=seed)


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

    async def async_evaluate(self, prediction: Any, label: Any) -> dict:
        """
        Asynchronous version of evaluate method that internally calls the synchronous evaluate.
        
        Args:
            prediction (Any): The predicted output.
            label (Any): The actual ground-truth label.
        
        Returns:
            dict: A dictionary containing evaluation metrics.
        """
        return await asyncio.to_thread(self.evaluate, prediction, label)
    
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
    
    def get_example_by_id(self, example_id: Any, mode: str = None) -> Optional[Any]:
        """
        Get an example from the benchmark by its id.

        Args:
            example_id (Any): The id of the example to retrieve.
            mode (str): The mode to retrieve the example from, choices: ["train", "dev", "test", "all"]
        
        Returns:
            Optional[Any]: The example if found, otherwise None.
        """
        # data = self.get_data_by_mode(mode=mode)
        if mode is not None and mode not in ["train", "dev", "test", "all"]:
            raise ValueError(f"Invalid value for mode: {mode}. Available choices: ['train', 'dev', 'test', 'all']")
        
        if mode is None or mode == "all":
            data = []
            if self._train_data is not None:
                data.extend(self._train_data)
            if self._dev_data is not None:
                data.extend(self._dev_data)
            if self._test_data is not None:
                data.extend(self._test_data)
        else:
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
        
    def _get_data(self, data: List[dict], indices: Optional[List[int]]=None, sample_k: Optional[int]=None, seed: Optional[int]=None) -> List[dict]:
        """
        Retrieves a subset of data based on provided indices or a random sample.
        
        Args:
            data (List[dict]): The list of data examples.
            indices (List[int], optional): Specific indices of data to retrieve. Defaults to None.
            sample_k (int, optional): The number of random samples to retrieve. Defaults to None.
            seed (int, optional): The seed for random sampling. Defaults to None. If provided, the random sampling will be deterministic.
        Returns:
            List[dict]: The selected subset of data. If both `indices` and `sample_k` are None, it will return the original `data`.
        """
        if indices is None:
            indices = list(range(len(data)))
        if sample_k is not None:
            if seed is not None:
                random.seed(seed)
            indices = random.sample(indices, k=min(sample_k, len(indices)))
        return_data = [data[idx] for idx in indices]
        return return_data

    def get_train_data(self, indices: Optional[List[int]] = None, sample_k: Optional[int] = None, seed: Optional[int] = None) -> List[dict]:
        # Retrieves training data based on specified indices or random sampling.
        if self._train_data is None:
            logger.warning(f"Train data for benchmark {type(self).__name__} is not loaded or None. Return an empty list.")
            return [] 
        
        train_data = self._get_data(self._train_data, indices=indices, sample_k=sample_k, seed=seed)
        return train_data 
    
    def get_dev_data(self, indices: Optional[List[int]] = None, sample_k: Optional[int] = None, seed: Optional[int] = None) -> List[dict]:
        # Retrieves development data based on specified indices or random sampling.
        if self._dev_data is None:
            logger.warning(f"Dev data for benchmark {type(self).__name__} is not loaded or None. Return an empty list.")
            return [] 
        
        dev_data = self._get_data(self._dev_data, indices=indices, sample_k=sample_k, seed=seed)
        return dev_data  

    def get_test_data(self, indices: Optional[List[int]] = None, sample_k: Optional[int] = None, seed: Optional[int] = None) -> List[dict]:
        # Retrieves test data based on specified indices or random sampling.
        if self._test_data is None:
            logger.warning(f"Test data for benchmark {type(self).__name__} is not loaded or None. Return an empty list.")
            return [] 
        
        test_data = self._get_data(self._test_data, indices=indices, sample_k=sample_k, seed=seed)
        return test_data 


class CodingBenchmark(Benchmark):

    """
    Abstract base class for defining coding benchmarks. This class provides methods to check the solution code.
    """
    
    def __init__(self, name: str, path: str, mode: str = "all", timeout: int = 60, **kwargs):
        
        self.SUCCESS = 0
        self.FAILED = 1 
        self.TIMEOUT = 2 
        self.timeout = timeout
        super().__init__(name=name, path=path, mode=mode, **kwargs)

    def handle_special_cases(self, task_id: str, solution: str, test: str) -> bool:
        return solution, test 
    
    def _check_evaluation_inputs(self, prediction: Any, label: Any) -> bool:
        """
        Check if the inputs are valid for evaluation.
        """
        assert isinstance(prediction, str) or isinstance(prediction, list), "prediction must be a string or a list of strings, but got {}".format(type(prediction))
        assert isinstance(label, dict) or isinstance(label, list), "label must be a string or a list of strings, but got {}".format(type(label))
        prediction = [prediction] if isinstance(prediction, str) else prediction
        label = [label] if isinstance(label, dict) else label
        return prediction, label

    def check_solution(self, task_id: str, solution: str, test: str, entry_point: Optional[str] = None, use_entrypoint_as_input: bool = True) -> Tuple[int, str]:
        """
        Execute the solution code and check if it passes the unit test.

        Args:
            task_id (str): The task id.
            solution (str): The solution code.
            test (str): The unit test code in HumanEval format. 
            entry_point (str): The entry point of the solution code.
        Returns:
            Tuple[int, str]: A tuple containing an integer indicating whether the solution passes the unit test (0: success, 1: failed, 2: timeout) and a string containing the success/error message.
        """
        solution = sanitize(solution, entrypoint=entry_point)

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
            solution, test = self.handle_special_cases(task_id=task_id, solution=solution, test=test)
            exec(solution, global_dict)
            if entry_point not in global_dict:
                raise ValueError(f"Function {entry_point} not found in the solution code.")
            exec(test, global_dict)
            unit_test_func = global_dict["check"] # check is the function name in the test code
            # run the unit test within the timeout
            with timeout(seconds=self.timeout):
                if use_entrypoint_as_input:
                    unit_test_func(global_dict[entry_point])
                else:
                    unit_test_func()
            result = (self.SUCCESS, "The solution passed the unit test.")
        
        except TimeoutException:
            result = (self.TIMEOUT, "Execution timed out.")
        
        except Exception as e:
            error_msg = f"An error occurred: {e}\nSolution:\n{solution}\nTest:\n{test}"
            result = (self.FAILED, error_msg)
        
        return result
    
    def compute_pass_at_k(self, results: List[bool], k_list: List[int]) -> Dict[str, float]:
        """
        Compute the pass@k for the given results.
        """
        pass_at_k = {}
        n = len(results)
        c = sum(results)
        for k in k_list:
            if n >= k:
                pass_at_k[f"pass@{k}"] = float(estimate_pass_at_k(np.array([n]), np.array([c]), k)[0])
        
        return pass_at_k
