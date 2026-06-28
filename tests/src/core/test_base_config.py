import unittest
from typing import List

import pytest

from evoagentx.core.base_config import BaseConfig, Parameter


class ToyConfig(BaseConfig):
    var1: str
    var2: List[str]
    var3: int = 111


class TestBaseConfig(unittest.TestCase):

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


class TestParameter(unittest.TestCase):

    def test_basic_parameter(self):
        param = Parameter(name="x", type="string", description="a string param")
        self.assertEqual(param.name, "x")
        self.assertEqual(param.type, "string")
        self.assertTrue(param.required)
        self.assertIsNone(param.json_schema)

    def test_python_type_aliases(self):
        for type_str in ("str", "int", "float", "bool", "dict", "list"):
            param = Parameter(name="p", type=type_str, description="test")
            self.assertEqual(param.type, type_str)

    def test_invalid_type_raises(self):
        with self.assertRaises(Exception):
            Parameter(name="p", type="invalid_type", description="bad type")

    def test_object_type_allows_missing_json_schema(self):
        param = Parameter(name="p", type="object", description="missing schema")
        self.assertIsNone(param.json_schema)

    def test_array_type_allows_missing_json_schema(self):
        param = Parameter(name="p", type="array", description="missing schema")
        self.assertIsNone(param.json_schema)

    def test_object_with_valid_json_schema(self):
        schema = {"type": "object", "properties": {"key": {"type": "string"}}}
        param = Parameter(name="p", type="object", description="an object", json_schema=schema)
        self.assertEqual(param.json_schema, schema)

    def test_array_with_valid_json_schema(self):
        schema = {"type": "array", "items": {"type": "string"}}
        param = Parameter(name="p", type="array", description="an array", json_schema=schema)
        self.assertEqual(param.json_schema, schema)

    def test_python_alias_with_valid_json_schema(self):
        schema = {"type": "array", "items": {"type": "string"}}
        param = Parameter(name="p", type="list", description="an array", json_schema=schema)
        self.assertEqual(param.json_schema, schema)

    def test_json_schema_type_mismatch_raises(self):
        schema = {"type": "object", "properties": {}}
        with self.assertRaises(Exception):
            Parameter(name="p", type="string", description="mismatch", json_schema=schema)

    def test_invalid_json_schema_raises(self):
        with self.assertRaises(Exception):
            Parameter(name="p", type="object", description="bad schema", json_schema={"type": "object", "properties": "not_a_dict"})


if __name__ == "__main__":
    unittest.main()
