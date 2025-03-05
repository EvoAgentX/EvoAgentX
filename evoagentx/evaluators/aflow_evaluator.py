# -*- coding: utf-8 -*-
# @Date    : 8/23/2024 10:00 AM
# @Author  : all
# @Desc    : Evaluation for different datasets

from typing import Dict, Literal, Tuple

from ..benchmark.humaneval import AFlowHumanEval
from ..benchmark.benchmark import Benchmark
from ..core.logging import logger

# If you want to customize tasks, add task types here and provide evaluation functions, just like the ones given above
DatasetType = Literal["HumanEval", "MBPP", "GSM8K", "MATH", "HotpotQA", "DROP"]


class Evaluator:
    """
    Complete the evaluation for different datasets here
    """

    def __init__(self, eval_path: str):
        self.eval_path = eval_path
        self.dataset_configs: Dict[DatasetType, Benchmark] = {
            # "GSM8K": GSM8KBenchmark,
            # "MATH": MATHBenchmark,
            "HumanEval": AFlowHumanEval,
            # "HotpotQA": HotpotQABenchmark,
            # "MBPP": MBPPBenchmark,
            # "DROP": DROPBenchmark,
        }

    def graph_evaluate(
        self, dataset: DatasetType, 
        graph, 
        params: dict, 
        path: str, 
        is_test: bool = False
    ) -> Tuple[float, float, float]:
        if dataset not in self.dataset_configs:
            raise ValueError(f"Unsupported dataset: {dataset}")

        data_path = self._get_data_path(dataset, is_test)
        # logger.info(f"data_path is {data_path}")
        benchmark_class = self.dataset_configs[dataset]
        benchmark = benchmark_class(file_path=data_path, 
                                    log_path=path,
                                    path=params.get("path", "evoagentx/ext/aflow/data"),
                                    mode=params.get("mode", "dev"))
        # logger.info(f"benchmark self mode is {benchmark.mode}")
        # Use params to configure the graph and benchmark
        configured_graph = self._configure_graph(dataset, 
                                                 graph, 
                                                 params)
        if is_test:
            va_list = None  # For test data, generally use None to test all
        else:
            va_list = None  # Use None to test all Validation data, or set va_list (e.g., [1, 2, 3]) to use partial data
        
        # logger.info(f"configured_graph is {configured_graph}")
        
        return benchmark.run_evaluation(configured_graph, va_list)

    def _configure_graph(self, dataset, graph, params: dict):
        # Here you can configure the graph based on params
        # For example: set LLM configuration, dataset configuration, etc.
        dataset_config = params.get("dataset", {})
        llm_config = params.get("llm", {})
        
        logger.info(f"dataset_config is {dataset_config}")
        logger.info(f"llm_config is {llm_config}")
        logger.info(f"graph is {graph}")
        return graph(name=dataset, 
                     llm=llm_config, 
                     dataset=dataset_config)

    def _get_data_path(self, dataset: DatasetType, test: bool) -> str:
        base_path = f"{dataset.lower()}"
        return f"{base_path}_test.jsonl" if test else f"{base_path}_validate.jsonl"
