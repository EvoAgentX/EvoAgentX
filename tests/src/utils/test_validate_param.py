import unittest

from evoagentx.core.base_config import Parameter
from evoagentx.utils.utils import validate_param


class TestValidateParam(unittest.TestCase):

    def _validate(self, required: Parameter, actual: Parameter):
        validate_param(required, actual, "node", "agent")

    def test_differing_description_is_allowed(self):
        """Description is free-form text and must not cause a validation failure."""
        required = Parameter(name="p", type="string", description="node-side wording")
        actual = Parameter(name="p", type="string", description="agent-side wording")
        # Should not raise.
        self._validate(required, actual)

    def test_type_mismatch_raises(self):
        required = Parameter(name="p", type="string", description="d")
        actual = Parameter(name="p", type="integer", description="d")
        with self.assertRaises(ValueError):
            self._validate(required, actual)

    def test_required_mismatch_raises(self):
        required = Parameter(name="p", type="string", description="d", required=True)
        actual = Parameter(name="p", type="string", description="d", required=False)
        with self.assertRaises(ValueError):
            self._validate(required, actual)

    def test_json_schema_skipped_when_one_side_missing(self):
        """json_schema is soft: if either side omits it, it is not compared."""
        schema = {"type": "object", "properties": {"a": {"type": "string"}}}
        required = Parameter(name="p", type="object", description="d", json_schema=schema)
        actual = Parameter(name="p", type="object", description="d")  # no json_schema
        # Should not raise.
        self._validate(required, actual)
        # Other direction too.
        self._validate(actual, required)

    def test_json_schema_enforced_when_both_provided(self):
        required = Parameter(
            name="p", type="object", description="d",
            json_schema={"type": "object", "properties": {"a": {"type": "string"}}},
        )
        actual = Parameter(
            name="p", type="object", description="d",
            json_schema={"type": "object", "properties": {"a": {"type": "integer"}}},
        )
        with self.assertRaises(ValueError):
            self._validate(required, actual)

    def test_json_schema_match_passes(self):
        schema = {"type": "object", "properties": {"a": {"type": "string"}}}
        required = Parameter(name="p", type="object", description="d", json_schema=dict(schema))
        actual = Parameter(name="p", type="object", description="d", json_schema=dict(schema))
        # Should not raise.
        self._validate(required, actual)


if __name__ == "__main__":
    unittest.main()
