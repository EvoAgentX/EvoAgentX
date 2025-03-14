import unittest
from unittest.mock import patch
import os
import shutil
import tempfile
from evoagentx.core.module_utils import load_json, save_json
from evoagentx.benchmark.humaneval import HumanEval

class TestHumanEval(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp() 
        self.sample_data = load_json(path="tests/data/benchmark/humaneval_samples.jsonl", type="jsonl")
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def create_test_files(self):
        test_file = os.path.join(self.temp_dir, "HumanEval.jsonl")
        os.makedirs(os.path.dirname(test_file), exist_ok=True)
        save_json(self.sample_data, test_file, type="jsonl")

    @patch("evoagentx.benchmark.humaneval.download_raw_humaneval_data")
    def test_load_data(self, mock_download):
        self.create_test_files()
        benchmark = HumanEval(path=self.temp_dir)

        self.assertEqual(len(benchmark.get_train_data()), 0)
        self.assertEqual(len(benchmark.get_dev_data()), 0)
        self.assertEqual(len(benchmark.get_test_data()), 10)
        self.assertEqual(mock_download.call_count, 0)

    def test_get_label(self):
        self.create_test_files()
        benchmark = HumanEval(path=self.temp_dir, mode="test")
        example = benchmark.get_test_data()[0]

        label = benchmark.get_label(example)
        self.assertTrue(isinstance(label, dict))
        self.assertEqual(label["task_id"], self.sample_data[0]["task_id"])
        self.assertEqual(label["canonical_solution"], self.sample_data[0]["canonical_solution"])
        self.assertEqual(label["test"], self.sample_data[0]["test"])
        self.assertEqual(label["entry_point"], self.sample_data[0]["entry_point"])
    
    def test_evaluate(self):
        self.create_test_files()
        benchmark = HumanEval(path=self.temp_dir, mode="test")
        test_data = benchmark.get_test_data() 

        for example in test_data:
            prediction = example["prompt"] + example["canonical_solution"]
            label = benchmark.get_label(example)
            metrics = benchmark.evaluate(prediction, label)
            self.assertEqual(len(metrics), 1)
            self.assertTrue("pass@1" in metrics) 
            self.assertEqual(metrics["pass@1"], 1.0)