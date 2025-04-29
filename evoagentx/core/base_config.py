# from pydantic import BaseModel
from typing import Optional, List
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
            **kwargs: Additional keyword arguments passed to save_module method
        
        Returns:
            str: The path where the file was saved
        """
        return super().save_module(path, **kwargs)

    def get_config_params(self):
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
        type: Parameter type
        description: Parameter description
        required: Whether the parameter is required, defaults to True
    """
    name: str
    type: str 
    description: str 
    required: Optional[bool] = True 

