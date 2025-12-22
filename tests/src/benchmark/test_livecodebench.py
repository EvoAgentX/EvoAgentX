import unittest
from unittest.mock import patch
import os
import shutil
import tempfile
from datasets import load_from_disk 
from evoagentx.benchmark.livecodebench import LiveCodeBench
from evoagentx.benchmark.lcb_utils.code_generation import CodeGenerationProblem
from evoagentx.benchmark.lcb_utils.test_output_prediction import TestOutputPredictionProblem
from evoagentx.benchmark.lcb_utils.code_execution import CodeExecutionProblem
from tests.src.benchmark.lcb_solutions import (
    codegen_solution, codegen_solution2, codegen_solution3,
    test_output_prediction_solution1, test_output_prediction_solution2, test_output_prediction_solution3,
    code_execution_solution1, code_execution_solution2, code_execution_solution3
)

class TestLiveCodeBench(unittest.TestCase):

    def setUp(self):
        self.codegen_samples = load_from_disk("tests/data/benchmark/lcb_codegen_samples")
        self.codegen_solutions = [codegen_solution, codegen_solution2, codegen_solution3]
        self.test_output_prediction_samples = load_from_disk("tests/data/benchmark/lcb_outputprediction_samples")
        self.test_output_prediction_solutions = [
            test_output_prediction_solution1, 
            test_output_prediction_solution2, 
            test_output_prediction_solution3
        ]
        self.code_execution_samples = load_from_disk("tests/data/benchmark/lcb_codeexecution_samples")
        self.code_execution_solutions = [
            code_execution_solution1, 
            code_execution_solution2, 
            code_execution_solution3
        ]
    
    @patch("evoagentx.benchmark.livecodebench.load_code_generation_dataset")
    def test_code_generation(self, mock_load_dataset):
        mock_load_dataset.return_value = [CodeGenerationProblem(**p) for p in self.codegen_samples]

        benchmark = LiveCodeBench(scenario="code_generation", version="release_v1")
        test_data = benchmark.get_test_data()

        self.assertEqual(len(test_data), len(self.codegen_samples))
        self.assertEqual(mock_load_dataset.call_count, 1)

        for example, solution in zip(test_data, self.codegen_solutions):
            label = benchmark.get_label(example)
            metrics = benchmark.evaluate(solution, label) 
            self.assertEqual(metrics, {"pass@1": 1.0})

    @patch("evoagentx.benchmark.livecodebench.load_test_prediction_dataset")
    def test_test_output_prediction(self, mock_load_dataset):
        mock_load_dataset.return_value = [TestOutputPredictionProblem(**p) for p in self.test_output_prediction_samples]

        benchmark = LiveCodeBench(scenario="test_output_prediction")
        test_data = benchmark.get_test_data()
        self.assertEqual(len(test_data), len(self.test_output_prediction_samples))
        self.assertEqual(mock_load_dataset.call_count, 1)

        for example, solution in zip(test_data, self.test_output_prediction_solutions):
            label = benchmark.get_label(example)
            metrics = benchmark.evaluate(solution, label)
            self.assertEqual(metrics, {"pass@1": 1.0}) 

    @patch("evoagentx.benchmark.livecodebench.load_code_execution_dataset")
    def test_code_execution(self, mock_load_dataset):
        mock_load_dataset.return_value = [CodeExecutionProblem(**p) for p in self.code_execution_samples]

        benchmark = LiveCodeBench(scenario="code_execution")
        test_data = benchmark.get_test_data()
        self.assertEqual(len(test_data), len(self.code_execution_samples))
        self.assertEqual(mock_load_dataset.call_count, 1)

        for example, solution in zip(test_data, self.code_execution_solutions):
            label = benchmark.get_label(example)
            metrics = benchmark.evaluate(solution, label)
            self.assertEqual(metrics, {"pass@1": 1.0})
