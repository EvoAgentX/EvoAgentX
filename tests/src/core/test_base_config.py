import unittest
from typing import List
from evoagentx.core.base_config import BaseConfig


class ToyConfig(BaseConfig):
    var1: str 
    var2: List[str]
    var3: int = 111


class TestModule(unittest.TestCase):

    def test_base_config(self):

        config = ToyConfig(var1="test", var2=["test2", "test3"])
        config_params = config.get_config_params()
        self.assertEqual(len(config_params), 3)
        self.assertTrue("var1" in config_params)
        self.assertTrue("var2" in config_params)
        self.assertTrue("var3" in config_params)

        set_params = config.get_set_params(ignore=["var2"])
        self.assertEqual(len(set_params), 1)
        self.assertEqual(set_params["var1"], "test")


if __name__ == "__main__":
    unittest.main()
