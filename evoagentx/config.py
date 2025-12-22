import yaml
from pydantic import BaseModel, model_validator
from typing import Optional, Type, Union, List, Any
# from .core.base_config import BaseConfig
from .models.model_configs import LLMConfig
from .core.registry import MODEL_REGISTRY


class Config(BaseModel):

    llm_config: dict
    agents: Optional[Union[str, List[dict]]] = []
    model_config = {"arbitrary_types_allowed": True, "extra": "allow", "protected_namespaces": ()}

    @classmethod
    def from_file(cls, path: str):
        with open(path, mode="r", encoding="utf-8") as file:
            data = yaml.safe_load(file.read())
        config = cls.model_validate(data)
        return config
    
    @property
    def kwargs(self):
        return self.model_extra

    @model_validator(mode="before")
    @classmethod
    def validate_config_data(cls, data: Any) -> Any:

        # process llm config
        llm_config_data = data.get("llm_config", None)
        if not llm_config_data:
            raise ValueError("config file must contain 'llm_config'")
        data["llm_config"] = cls.process_llm_config(data=data["llm_config"])
        
        # process agent data
        agents_data = data.get("agents", None)
        if agents_data:
            data["agents"] = cls.process_agents_data(agents=agents_data, llm_config=data["llm_config"])

        return data

    @classmethod
    def process_llm_config(cls, data: dict) -> dict:

        llm_type = data.get("llm_type", None)
        if not llm_type:
            raise ValueError("must specify `llm_type` in in `llm_config`!")
        llm_config_cls: Type[LLMConfig] = MODEL_REGISTRY.get_model_config(llm_type)
        if "class_name" in data:
            assert data["class_name"] == llm_config_cls.__name__, \
                "the 'class_name' specified in 'llm_config' ({}) doesn't match the LLMConfig class ({}) registered for {} model. You should either remove 'class_name' or set it to {}.".format(
                    data["class_name"], llm_config_cls.__name__, llm_type, llm_config_cls.__name__
                )
        else:
            data["class_name"] = llm_config_cls.__name__
        
        return data
    
    @classmethod
    def process_agents_data(cls, agents: List[dict], llm_config=dict) -> List[dict]:

        for agent in agents:
            if "llm_config" not in agent:
                agent["llm_config"] = llm_config
        return agents

