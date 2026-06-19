import json
import random
from math import inf
from typing import Any, Dict, List
from jsonschema import Draft7Validator


DEFAULT_SYSTEM_PROMPT = "You are a helpful and highly intelligent assistant."


class JSONSchemaExampleGenerator:
    """
    A class to generate examples that conform to JSON schemas.
    
    This class handles recursive generation of examples for complex nested
    JSON schemas including objects, arrays, and various constraints.
    """
    
    def __init__(
        self,
        max_array_length: int = 1,
        max_object_depth: int = 4,
        max_optional_properties: int = 5
    ):
        """
        Initialize the generator with configuration options.
        
        Args:
            max_array_length: Maximum number of items to generate for arrays
            max_object_depth: Maximum nesting depth for objects to prevent infinite recursion
            max_optional_properties: Maximum number of optional properties to generate for objects
        """
        self.max_array_length = max_array_length
        self.max_object_depth = max_object_depth
        self.max_optional_properties = max_optional_properties
        self._current_depth = 0

        self.type_handlers = {
            "string": self._generate_string,
            "integer": self._generate_integer,
            "number": self._generate_number,
            "boolean": self._generate_boolean,
            "array": self._generate_array,
            "object": self._generate_object,
        }


    def generate(self, json_schema: Dict[str, Any], to_str: bool = False) -> Any:
        """
        Generate an example that matches the JSON schema.
        
        Args:
            json_schema: A JSON schema dictionary
            
        Returns:
            An example value that conforms to the schema
        """
        # Check if the schema is valid
        Draft7Validator.check_schema(json_schema)
        example = self._generate(json_schema)
        if to_str:
            example_str = json.dumps(example, indent=2, ensure_ascii=False)
            example_str = example_str.replace("\"...\"", "...").replace("...: ...", "...")
            return example_str
        return example


    def _generate(self, json_schema: Dict[str, Any]) -> Any:        
        schema_type = json_schema.get("type", "string")
        
        # Handle enum first as it takes precedence
        if "enum" in json_schema:
            return random.choice(json_schema["enum"])
        
        # Handle composition keywords
        if "allOf" in json_schema:
            # Merge all schemas and generate
            merged_schema = {"type": "string"}  # Default fallback
            for subschema in json_schema["allOf"]:
                if "type" in subschema:
                    merged_schema["type"] = subschema["type"]
                merged_schema.update(subschema)
            return self._generate(merged_schema)
        
        if "anyOf" in json_schema:
            return self._generate(random.choice(json_schema["anyOf"]))
        
        if "oneOf" in json_schema:
            return self._generate(random.choice(json_schema["oneOf"]))
        
        handler = self.type_handlers.get(schema_type, lambda _: "example")
        return handler(json_schema)


    def _generate_string(self, schema: Dict[str, Any]) -> str:
        """Generate example string based on schema constraints."""
        if "format" in schema:
            format_examples = {
                "email": "user@example.com",
                "date": "2024-01-15",
                "date-time": "2024-01-15T10:30:00Z",
                "uri": "https://example.com",
                "uuid": "550e8400-e29b-41d4-a716-446655440000"
            }
            return format_examples.get(schema["format"], "example")
        
        min_length = schema.get("minLength", 1)
        max_length = schema.get("maxLength", inf)
        
        example_str = schema.get("description") or "example"
        str_length = len(example_str)

        if min_length > str_length:
            example_str += "..."
        elif max_length < str_length:
            example_str = example_str[:max_length]
        
        return example_str
    

    def _generate_integer(self, schema: Dict[str, Any]) -> int:
        """Generate example integer based on schema constraints."""
        minimum = schema.get("minimum", 0)
        maximum = schema.get("maximum", 100)
        
        if "exclusiveMinimum" in schema:
            minimum = max(minimum, schema["exclusiveMinimum"] + 1)
        if "exclusiveMaximum" in schema:
            maximum = min(maximum, schema["exclusiveMaximum"] - 1)
        
        if "multipleOf" in schema:
            multiple = schema["multipleOf"]
            if multiple == 0:
                return minimum  # Fallback for invalid multipleOf
            base = random.randint(minimum // multiple, maximum // multiple)
            return base * multiple
        
        return random.randint(minimum, maximum)
    

    def _generate_number(self, schema: Dict[str, Any]) -> float:
        """Generate example number based on schema constraints."""
        minimum = float(schema.get("minimum", 0))
        maximum = float(schema.get("maximum", 100))
        
        if "exclusiveMinimum" in schema:
            minimum = max(minimum, float(schema["exclusiveMinimum"]) + 0.1)
        if "exclusiveMaximum" in schema:
            maximum = min(maximum, float(schema["exclusiveMaximum"]) - 0.1)
        
        return round(random.uniform(minimum, maximum), 2)
    

    def _generate_boolean(self, schema: Dict[str, Any]) -> bool:
        """Generate example boolean."""
        return random.choice([True, False])
    

    def _generate_array(self, schema: Dict[str, Any]) -> List[Any]:
        """Generate example array based on schema constraints."""
        items_schema = schema.get("items", {"type": "string"})
        min_items = max(schema.get("minItems", 0), 1)  # Generate at least 1 array item
        max_items = schema.get("maxItems", inf)
        
        self._current_depth += 1

        if self.max_object_depth is not None and self._current_depth > self.max_object_depth:
            self._current_depth -= 1
            return ["..."]

        result = []

        for i in range(min_items):
            if self.max_array_length is not None and i >= self.max_array_length:
                break
            result.append(self._generate(items_schema))
        
        self._current_depth -= 1

        if len(result) < max_items and result[-1] != "...":
            result.append("...")
        
        return result


    def _generate_object(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Generate example object based on schema constraints."""
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        
        example = {}
        
        self._current_depth += 1
        if self.max_object_depth is not None and self._current_depth > self.max_object_depth:
            self._current_depth -= 1
            return {"...": "..."}
        
        # Add required properties
        for prop_name in required:
            if prop_name in properties:
                example[prop_name] = self._generate(properties[prop_name])
        
        # Add optional properties
        optional_props = [p for p in properties if p not in required]
        num_optional = len(optional_props)

        if self.max_optional_properties is not None:
            num_optional = min(num_optional, self.max_optional_properties)
        
        if optional_props and num_optional > 0:
            for prop_name in random.sample(optional_props, num_optional):
                example[prop_name] = self._generate(properties[prop_name])
        
        self._current_depth -= 1
        return example

