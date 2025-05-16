import unittest
from evoagentx.utils.utils import append_inputs_to_prompt

class TestUtils(unittest.TestCase):
    def test_append_inputs_to_prompt_basic(self):
        """测试基本的输入添加功能"""
        prompt = "这是一个测试"
        inputs = ["name", "age"]
        result = append_inputs_to_prompt(prompt, inputs)
        expected = "这是一个测试\nname : {name}\nage : {age}"
        self.assertEqual(result, expected)

    def test_append_inputs_to_prompt_with_existing_placeholder(self):
        """测试当输入变量已存在时的情况"""
        prompt = "这是一个{name}测试"
        inputs = ["name", "age"]
        result = append_inputs_to_prompt(prompt, inputs)
        expected = "这是一个{name}测试\nage : {age}"
        self.assertEqual(result, expected)

    def test_append_inputs_to_prompt_single_braces_conversion(self):
        """测试单个大括号转换为双大括号的功能"""
        prompt = "这是一个{变量}测试，包含{另一个变量}"
        inputs = ["name"]
        result = append_inputs_to_prompt(prompt, inputs)
        expected = "这是一个{{变量}}测试，包含{{另一个变量}}\nname : {name}"
        self.assertEqual(result, expected)

    def test_append_inputs_to_prompt_double_braces_preservation(self):
        """测试双大括号保持不变的功能"""
        prompt = "这是一个{{变量}}测试，包含{单个变量}"
        inputs = ["name"]
        result = append_inputs_to_prompt(prompt, inputs)
        expected = "这是一个{{变量}}测试，包含{{单个变量}}\nname : {name}"
        self.assertEqual(result, expected)

    def test_append_inputs_to_prompt_empty_inputs(self):
        """测试空输入列表的情况"""
        prompt = "这是一个{变量}测试"
        inputs = []
        result = append_inputs_to_prompt(prompt, inputs)
        expected = "这是一个{{变量}}测试"
        self.assertEqual(result, expected)

    def test_append_inputs_to_prompt_complex_case(self):
        """测试复杂情况：混合单个和双大括号，以及重复变量"""
        prompt = "这是一个{变量}测试，包含{{双括号}}和{另一个变量}"
        inputs = ["变量", "新变量"]
        result = append_inputs_to_prompt(prompt, inputs)
        expected = "这是一个{变量}测试，包含{{双括号}}和{{另一个变量}}\n新变量 : {新变量}"
        self.assertEqual(result, expected)

if __name__ == '__main__':
    unittest.main() 