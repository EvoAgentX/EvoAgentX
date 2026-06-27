from typing import List, Optional

from jsonschema import Draft7Validator
from pydantic import model_validator

from .module import BaseModule


class BaseConfig(BaseModule):

    """
    Base configuration class that serves as parent for all configuration classes.
    
    A config should inherit BaseConfig and specify the attributes and their types. 
    Otherwise this will be an empty config.
    """
    def save(self, path: str, **kwargs)-> str:

        """Save configuration to the specified path.
        
        Args:
            path: The file path to save the configuration
            **kwargs (Any): Additional keyword arguments passed to save_module method
        
        Returns:
            str: The path where the file was saved
        """
        return super().save_module(path, **kwargs)

    def get_config_params(self) -> List[str]:
        """Get a list of configuration parameters.
        
        Returns:
            List[str]: List of configuration parameter names, excluding 'class_name'
        """
        config_params = list(type(self).model_fields.keys())
        config_params.remove("class_name")
        return config_params

    def get_set_params(self, ignore: List[str] = []) -> dict:
        """Get a dictionary of explicitly set parameters.
        
        Args:
            ignore: List of parameter names to ignore
        
        Returns:
            dict: Dictionary of explicitly set parameters, excluding 'class_name' and ignored parameters
        """
        explicitly_set_fields = {field: getattr(self, field) for field in self.model_fields_set}
        if self.kwargs:
            explicitly_set_fields.update(self.kwargs)
        for field in ignore:
            explicitly_set_fields.pop(field, None)
        explicitly_set_fields.pop("class_name", None)
        return explicitly_set_fields


class Parameter(BaseModule):
    """Parameter class used to define configuration parameters.

    Attributes:
        name: Parameter name
        type: Parameter type, support json & python type.
        description: Parameter description
        required: Whether the parameter is required, defaults to True
        json_schema: the optional json schema of the parameter. Recommended when type is `object` or `array`.
    """
    name: str
    type: str
    description: str
    required: Optional[bool] = True
    json_schema: Optional[dict] = None

    @model_validator(mode="after")
    def _validate_type_and_schema(self):
        from ..utils.utils import string_to_json_schema_type, string_to_python_type
        if self.type not in string_to_python_type:
            raise ValueError(f"Invalid `type`: {self.type}. Allowed: {list(string_to_python_type.keys())}")
        if self.json_schema is not None:
            try:
                Draft7Validator.check_schema(self.json_schema)
            except Exception as e:
                raise ValueError(f"Invalid `json_schema` for '{self.name}': {self.json_schema}.") from e
            expected_schema_type = string_to_json_schema_type[self.type]
            actual_schema_type = self.json_schema.get("type")
            if expected_schema_type != actual_schema_type:
                raise ValueError(
                    "`type` and `json_schema.type` must be the same if `json_schema` is provided. "
                    f"But got `type`: {self.type}, `json_schema.type`: {actual_schema_type}"
                )
        return self
