# from pydantic import BaseModel
from typing import List
from .module import BaseModule

class BaseConfig(BaseModule):

    """
    a base config

    A config should inherent BaseConfig and specify the attributes and their types. 
    Otherwise this will be an empty config.
    """
    def save(self, path: str, **kwargs)-> str:
        super().save_module(path, **kwargs)

    def get_config_params(self):
        config_params = list(type(self).model_fields.keys())
        config_params.remove("class_name")
        return config_params

    def get_set_params(self, ignore: List[str] = []) -> dict:
        explicitly_set_fields = {field: getattr(self, field) for field in self.model_fields_set}
        if self.kwargs:
            explicitly_set_fields.update(self.kwargs)
        for field in ignore:
            explicitly_set_fields.pop(field, None)
        explicitly_set_fields.pop("class_name", None)
        return explicitly_set_fields


class Parameter(BaseModule):
    name: str
    type: str 
    description: str 
