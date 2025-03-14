import unittest
from unittest.mock import patch
import os
import tempfile
from evoagentx.core.module_utils import save_json
from evoagentx.benchmark.gsm8k import GSM8K

class TestGSM8K(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

        # sample data 
        self.sample_data = [
            {
                "question": "Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?", 
                "answer": "Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.\nShe makes 9 * 2 = $<<9*2=18>>18 every day at the farmer’s market.\n#### 18"
            },
            {
                "question": "A robe takes 2 bolts of blue fiber and half that much white fiber.  How many bolts in total does it take?", 
                "answer": "It takes 2/2=<<2/2=1>>1 bolt of white fiber\nSo the total amount of fabric is 2+1=<<2+1=3>>3 bolts of fabric\n#### 3"
            }
        ]

    def tearDown(self):
        for filename in os.listdir(self.temp_dir):
            os.remove(os.path.join(self.temp_dir, filename))
        os.rmdir(self.temp_dir)

    def create_test_file(self, filename, data):
        filepath = os.path.join(self.temp_dir, filename)
        save_json(data=data, path=filepath, type="jsonl")
        return filepath
    
    @patch("evoagentx.benchmark.gsm8k.download_raw_gsm8k_data")
    def test_load_data(self, mock_download):
        # create test file 
        self.create_test_file("train.jsonl", self.sample_data)
        self.create_test_file("test.jsonl", self.sample_data)

        # initialize benchmark 
        benchmark = GSM8K(path=self.temp_dir)

        # Test data loading 
        self.assertEqual(len(benchmark.get_train_data()), 2)
        self.assertEqual(len(benchmark.get_dev_data()), 0)
        self.assertEqual(len(benchmark.get_test_data()), 2)
        self.assertEqual(mock_download.call_count, 0)

    def test_get_label(self):
        # create test file 
        self.create_test_file("train.jsonl", self.sample_data)

        benchmark = GSM8K(path=self.temp_dir, mode="train")
        example = benchmark.get_train_data()[0]

        # Test label extraction
        self.assertEqual(benchmark.get_label(example), self.sample_data[0]["answer"])
        self.assertEqual(benchmark.get_id(example), "train-1")
        
    def test_extract_last_number(self):
        # create test file 
        self.create_test_file("train.jsonl", self.sample_data)

        benchmark = GSM8K(path=self.temp_dir)

        # Test extracting last number from a string 
        self.assertEqual(benchmark.extract_last_number(benchmark.get_train_data()[0]["answer"]), 18)
        self.assertEqual(benchmark.extract_last_number(benchmark.get_train_data()[1]["answer"]), 3)
        self.assertEqual(benchmark.extract_last_number("The answer is123.45"), 123.45)
        self.assertEqual(benchmark.extract_last_number("The answer is: xxx123.45"), 123.45)
        self.assertEqual(benchmark.extract_last_number("The answer is:\n123.45"), 123.45)
        self.assertEqual(benchmark.extract_last_number("The answer is:\n #### 123.45"), 123.45)
        
    def test_evaluate(self):
        # create test file 
        self.create_test_file("train.jsonl", self.sample_data)

        benchmark = GSM8K(path=self.temp_dir, mode="train")

        # Test exact match case 
        result = benchmark.evaluate(prediction="18", label=self.sample_data[0]["answer"])
        self.assertEqual(result["solve_rate"], 1.0)

        # Test partial match case 
        result = benchmark.evaluate(prediction="reasoning process, ####18", label=self.sample_data[0]["answer"])
        self.assertEqual(result["solve_rate"], 1.0)

        # Test no match case 
        result = benchmark.evaluate(prediction="wrong answer 111", label=self.sample_data[0]["answer"])
        self.assertEqual(result["solve_rate"], 0.0)
        
        