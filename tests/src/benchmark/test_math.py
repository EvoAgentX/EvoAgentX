import unittest
from unittest.mock import patch
import os
import shutil
import tempfile
from evoagentx.core.module_utils import save_json
from evoagentx.benchmark.math_benchmark import MATH

class TestMath(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.sample_data1 = {
            'problem': 'We roll a fair 6-sided die 5 times.  What is the probability that we get a 6 in at most 2 of the rolls?',
            'level': 'Level 5', 
            'type': 'Counting & Probability',
            'solution': "The number of ways to roll exactly 2 6's is $\\binom{5}{2}5^3$, since there are $\\binom{5}{2}$ choices for which of the two dice are 6, and there are 5 choices for each of the other 3 dice. Similarly, the number of ways to roll exactly 1 6 is $\\binom{5}{1}5^4$, and the number of ways to roll no 6's is $\\binom{5}{0}5^5$. So the probability is \\[\\frac{\\binom{5}{2}5^3+\\binom{5}{1}5^4+\\binom{5}{0}5^5}{6^5}=\\boxed{\\frac{625}{648}}.\\]"
        }

        self.sample_data2 = {
            'problem': 'When counting from $3$ to $201$, $53$ is the $51^\\mathrm{st}$ number counted. When counting backwards from $201$ to $3$, $53$ is the $n^\\mathrm{th}$ number counted. What is $n$?',
            'level': 'Level 2',
            'type': 'Counting & Probability',
            'solution': 'Note that $n$ is equal to the number of integers between $53$ and $201$, inclusive. Thus, $n=201-53+1=\\boxed{149}$.'
        }

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def create_test_files(self):

        train_file = os.path.join(self.temp_dir, "MATH", "train", "Counting & Probability", "sample1.json")
        os.makedirs(os.path.dirname(train_file), exist_ok=True)
        save_json(self.sample_data1, train_file, type="json")

        test_file = os.path.join(self.temp_dir, "MATH", "test", "Counting & Probability", "sample1.json")
        os.makedirs(os.path.dirname(test_file), exist_ok=True)
        save_json(self.sample_data2, test_file, type="json")
    
    @patch("evoagentx.benchmark.math_benchmark.download_raw_math_data")
    def test_load_data(self, mock_download): 
        # create test files 
        self.create_test_files()
        benchmark = MATH(path=self.temp_dir)

        # Test data loading 
        self.assertEqual(len(benchmark.get_train_data()), 1)
        self.assertEqual(len(benchmark.get_dev_data()), 0)
        self.assertEqual(len(benchmark.get_test_data()), 1)
        self.assertEqual(mock_download.call_count, 0)
    
    def test_get_label(self):
        self.create_test_files()
        benchmark = MATH(path=self.temp_dir, mode="train") 
        example = benchmark.get_train_data()[0]

        self.assertEqual(benchmark.get_label(example), self.sample_data1["solution"])
        self.assertEqual(benchmark.get_id(example), "train-1")
        
    def test_extract_answer(self):
        self.create_test_files()
        benchmark = MATH(path=self.temp_dir, mode="train")
        example = benchmark.get_train_data()[0]

        self.assertEqual(benchmark.extract_answer(example["solution"]), "\\frac{625}{648}")
        
    def test_evaluate(self):

        self.create_test_files()
        benchmark = MATH(path=self.temp_dir, mode="train")
        example = benchmark.get_train_data()[0]

        # test math equal 
        prediction = benchmark.extract_answer(example["solution"])
        self.assertEqual(str(prediction), str("\\frac{625}{648}"))
        self.assertTrue(benchmark.math_equal(prediction, "\\frac{625}{648}"))
        self.assertFalse(benchmark.math_equal(prediction, "\\frac{625}{649}"))
        self.assertFalse(benchmark.is_digit(prediction))
        self.assertFalse(benchmark.is_digit("\\frac{625}{648}"))
        self.assertTrue(benchmark.symbolic_equal(prediction, "\\frac{625}{648}"))
        self.assertFalse(benchmark.symbolic_equal(prediction, "\\frac{625}{649}"))

        # test evaluate 
        self.assertEqual(benchmark.evaluate(example["solution"], "\\frac{625}{648}"), {"solve_rate": 1.0})
        self.assertEqual(benchmark.evaluate(example["solution"], "\\frac{625}{649}"), {"solve_rate": 0.0})

