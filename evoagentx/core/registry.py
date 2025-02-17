from typing import List

class ModuleRegistry:

    def __init__(self):
        self.module_dict = {}
    
    def register_module(self, cls_name: str, cls, force: bool=False):
        if cls_name in self.module_dict and not force:
            return self.module_dict[cls_name]
        self.module_dict[cls_name] = cls 
        return cls 
    def get_module(self, cls_name: str):
        if cls_name not in self.module_dict:
            raise ValueError(f"module `{cls_name}` not Found!")
        return self.module_dict[cls_name]

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
