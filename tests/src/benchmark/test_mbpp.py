import unittest
from unittest.mock import patch
import os
import shutil
import tempfile
from evoagentx.core.module_utils import load_json, save_json
from evoagentx.benchmark.mbpp import MBPP

class TestMBPP(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.sample_data = load_json(path="tests/data/benchmark/mbpp_samples.json", type="json")
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def create_test_files(self):
        test_file = os.path.join(self.temp_dir, "sanitized-mbpp.json")
        os.makedirs(os.path.dirname(test_file), exist_ok=True)
        save_json(self.sample_data, test_file, type="json")

    @patch("evoagentx.benchmark.mbpp.download_raw_mbpp_data")
    def test_load_data(self, mock_download):
        self.create_test_files()
        benchmark = MBPP(path=self.temp_dir)

        self.assertEqual(len(benchmark.get_train_data()), 0)
        self.assertEqual(len(benchmark.get_dev_data()), 0)
        self.assertEqual(len(benchmark.get_test_data()), 10)
        self.assertEqual(mock_download.call_count, 0)

    def test_get_label(self):

        self.create_test_files()
        benchmark = MBPP(path=self.temp_dir, mode="test")
        example = benchmark.get_test_data()[0]

        label = benchmark.get_label(example)
        self.assertTrue(isinstance(label, dict))
        self.assertEqual(label["task_id"], self.sample_data[0]["task_id"])
        self.assertEqual(label["canonical_solution"], self.sample_data[0]["code"])

        # check entry point and test list 
        for i, example in enumerate(benchmark.get_test_data()): 
            label = benchmark.get_label(example)
            self.assertTrue(isinstance(label, dict))
            self.assertEqual(label["task_id"], self.sample_data[i]["task_id"])
            self.assertEqual(label["canonical_solution"], self.sample_data[i]["code"]) # except for task 56, where I change the `check` function to `check_answer`
            entry_point = label["entry_point"]
            test = label["test"]
            self.assertTrue(all(entry_point in assert_str for assert_str in self.sample_data[i]["test_list"]))
            self.assertTrue(all(assert_str in test for assert_str in self.sample_data[i]["test_list"]))
    
    def test_evaluate(self):
        self.create_test_files()
        benchmark = MBPP(path=self.temp_dir, mode="test")
        test_data = benchmark.get_test_data()

        for example in test_data:
            prediction = example["canonical_solution"]
            label = benchmark.get_label(example)
            metrics = benchmark.evaluate(prediction, label)
            self.assertEqual(len(metrics), 1)
            self.assertTrue("pass@1" in metrics)
            self.assertTrue(metrics["pass@1"] == 1.0)