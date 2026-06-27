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

    def test_json_schema_required_when_required_param_has_schema(self):
        """A downstream param cannot drop a schema declared by the required param."""
        schema = {"type": "object", "properties": {"a": {"type": "string"}}}
        required = Parameter(name="p", type="object", description="d", json_schema=schema)
        actual = Parameter(name="p", type="object", description="d")  # no json_schema
        with self.assertRaises(ValueError):
            self._validate(required, actual)

    def test_extra_actual_json_schema_allowed_when_required_param_has_none(self):
        """A more specific actual param is allowed when the required param has no schema."""
        schema = {"type": "object", "properties": {"a": {"type": "string"}}}
        required = Parameter(name="p", type="object", description="d")  # no json_schema
        actual = Parameter(name="p", type="object", description="d", json_schema=schema)
        # Should not raise.
        self._validate(required, actual)

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
