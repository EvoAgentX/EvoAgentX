from typing import List
from functools import wraps
class ModuleRegistry:

    def __init__(self):
        self.module_dict = {}
    
    def register_module(self, cls_name: str, cls):
        if cls_name in self.module_dict:
            raise ValueError(f"Found duplicate module: `{cls_name}`!")
        self.module_dict[cls_name] = cls 
    
    def get_module(self, cls_name: str):
        if cls_name not in self.module_dict:
            raise ValueError(f"module `{cls_name}` not Found!")
        return self.module_dict[cls_name]
    
    def has_module(self, cls_name: str) -> bool:
        return cls_name in self.module_dict

MODULE_REGISTRY = ModuleRegistry()

def register_module(cls_name, cls):
    MODULE_REGISTRY.register_module(cls_name=cls_name, cls=cls)


class ModelRegistry:

    def __init__(self):
        
        self.models = {}
        self.model_configs = {}
    
    def register(self, key: str, model_cls, config_cls):
        if key in self.models:
            raise ValueError(f"model name '{key}' is already registered!")
        self.models[key] = model_cls
        self.model_configs[key] = config_cls
    
    def key_error_message(self, key: str):
        error_message = f"""`{key}` is not a registered model name. Currently availabel model names: {self.get_model_names()}. If `{key}` is a customized model, you should use @register_model({key}) to register the model."""
        return error_message
    
    def get_model(self, key: str):
        model = self.models.get(key, None)
        if model is None:
            raise KeyError(self.key_error_message(key))
        return model
    
    def get_model_config(self, key: str):
        config = self.model_configs.get(key, None)
        if config is None:
            raise KeyError(self.key_error_message(key))
        return config 

    def get_model_names(self):
        return list(self.models.keys())


MODEL_REGISTRY = ModelRegistry()

def register_model(config_cls, alias: List[str]=None):

    def decorator(cls):
        class_name = cls.__name__
        MODEL_REGISTRY.register(class_name, cls, config_cls)
        if alias is not None:
            for alia in alias:
                MODEL_REGISTRY.register(alia, cls, config_cls)
        return cls
    
    return decorator

class ParseFunctionRegistry:
    
    def __init__(self):
        self.functions = {}
    
    def register(self, func_name: str, func):
        """Register a function with a given name.
        
        Args:
            func_name: The name to register the function under
            func (Callable): The function to register
            
        Raises:
            ValueError: If a function with the same name is already registered
        """
        if func_name in self.functions:
            raise ValueError(f"Function name '{func_name}' is already registered!")
        self.functions[func_name] = func
    
    def get_function(self, func_name: str) -> callable:
        """Get a registered function by name.
        
        Args:
            func_name: The name of the function to retrieve
            
        Returns:
            Callable: The registered function
            
        Raises:
            KeyError: If no function with the given name is registered
        """
        if func_name not in self.functions:
            available_funcs = list(self.functions.keys())
            raise KeyError(f"Function '{func_name}' not found! Available functions: {available_funcs}")
        return self.functions[func_name]
    
    def has_function(self, func_name: str) -> bool:
        """Check if a function name is registered.
        
        Args:
            func_name: The name to check
            
        Returns:
            True if the function name is registered, False otherwise
        """
        return func_name in self.functions


PARSE_FUNCTION_REGISTRY = ParseFunctionRegistry()


def register_parse_function(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    PARSE_FUNCTION_REGISTRY.register(func.__name__, wrapper)
    return wrapper
    