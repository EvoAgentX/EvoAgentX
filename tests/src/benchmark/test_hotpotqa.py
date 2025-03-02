import unittest
from unittest.mock import patch
import os
import tempfile
import json
from evoagentx.benchmark.hotpotqa import HotPotQA, AFlowHotPotQA

class TestHotPotQA(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for test data
        self.temp_dir = tempfile.mkdtemp()
        
        # Sample test data
        self.sample_data = [
            {
                "_id": "test_id_1",
                "question": "What is the capital of France?",
                "answer": "Paris",
                "context": [["France", ["Paris is the capital of France.", "It is a beautiful city."]]],
                "supporting_facts": [["France", 0]],
                "type": "comparison",
                "level": "medium"
            },
            {
                "_id": "test_id_2",
                "question": "Who wrote Romeo and Juliet?",
                "answer": "William Shakespeare",
                "context": [["Shakespeare", ["William Shakespeare wrote many plays.", "Romeo and Juliet is one of them."]]],
                "supporting_facts": [["Shakespeare", 0]],
                "type": "bridge",
                "level": "easy"
            }
        ]

    def tearDown(self):
        # Clean up temporary files
        for filename in os.listdir(self.temp_dir):
            os.remove(os.path.join(self.temp_dir, filename))
        os.rmdir(self.temp_dir)

    def create_test_file(self, filename, data):
        filepath = os.path.join(self.temp_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(data, f)
        return filepath

    @patch('evoagentx.benchmark.hotpotqa.download_raw_hotpotqa_data')
    def test_load_data(self, mock_download):
        # Create test file
        self.create_test_file("hotpot_dev_distractor_v1.json", self.sample_data)
        
        # Initialize benchmark
        benchmark = HotPotQA(path=self.temp_dir, mode="dev")
        
        # Test data loading
        self.assertEqual(len(benchmark.get_dev_data()), 2)
        self.assertEqual(mock_download.call_count, 0)  # Should not download as file exists

    def test_get_label(self):
        # Create test file
        self.create_test_file("hotpot_dev_distractor_v1.json", self.sample_data)

        benchmark = HotPotQA(path=self.temp_dir, mode="dev")
        example = self.sample_data[0]
        
        # Test label extraction
        self.assertEqual(benchmark._get_label(example), "Paris")

    def test_get_id(self):
        # Create test file
        self.create_test_file("hotpot_dev_distractor_v1.json", self.sample_data)

        benchmark = HotPotQA(path=self.temp_dir, mode="dev")
        example = self.sample_data[0]
        
        # Test ID extraction
        self.assertEqual(benchmark._get_id(example), "test_id_1")

    def test_evaluate(self):
        # Test exact match case
        # Create test file
        self.create_test_file("hotpot_dev_distractor_v1.json", self.sample_data)

        benchmark = HotPotQA(path=self.temp_dir, mode="dev")

        result = benchmark.evaluate(prediction="Paris", label="Paris")
        self.assertEqual(result["em"], 1)
        self.assertEqual(result["f1"], 1)

        # Test partial match case
        result = benchmark.evaluate(prediction="Paris, France", label="Paris")
        self.assertEqual(result["em"], 0)
        self.assertTrue(result["f1"] > 0)

        # Test no match case
        result = benchmark.evaluate(prediction="London", label="Paris")
        self.assertEqual(result["em"], 0)
        self.assertEqual(result["f1"], 0)

    def test_data_sampling(self):
        # Create test file
        self.create_test_file("hotpot_dev_distractor_v1.json", self.sample_data)
        
        benchmark = HotPotQA(path=self.temp_dir, mode="dev")
        
        # Test sample_k
        sampled_data = benchmark.get_dev_data(sample_k=1)
        self.assertEqual(len(sampled_data), 1)

        # Test indices
        specific_data = benchmark.get_dev_data(indices=[0])
        self.assertEqual(len(specific_data), 1)
        self.assertEqual(specific_data[0]["_id"], self.sample_data[0]["_id"])


class TestAFlowHotPotQA(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.sample_data = [
            {
                "_id": "test_id_1",
                "question": "What is the capital of France?",
                "answer": "Paris",
                "context": [["France", ["Paris is the capital of France.", "It is a beautiful city."]]],
                "supporting_facts": [["France", 0]],
                "type": "comparison",
                "level": "medium"
            }
        ]

    def tearDown(self):
        for filename in os.listdir(self.temp_dir):
            os.remove(os.path.join(self.temp_dir, filename))
        os.rmdir(self.temp_dir)

    @patch('evoagentx.benchmark.hotpotqa.download_aflow_benchmark_data')
    def test_aflow_load_data(self, mock_download):
        # Create test file in jsonl format
        filepath = os.path.join(self.temp_dir, "hotpotqa_test.jsonl")
        with open(filepath, 'w') as f:
            for item in self.sample_data:
                f.write(json.dumps(item) + '\n')

        benchmark = AFlowHotPotQA(path=self.temp_dir, mode="test")
        
        # Test data loading
        self.assertEqual(mock_download.call_count, 0)

if __name__ == '__main__':
    unittest.main()