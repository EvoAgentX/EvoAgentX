from .agent import Agent
from ..actions.agent_generation import AgentGeneration 
from ..prompts.agent_generator import AGENT_GENERATOR


class AgentGenerator(Agent):

    """
    An agent responsible for generating agents for a task. 
    """
    
    def __init__(self, **kwargs):

        name = kwargs.pop("name") if "name" in kwargs else AGENT_GENERATOR["name"]
        description = kwargs.pop("description") if "description" in kwargs else AGENT_GENERATOR["description"]
        system_prompt = kwargs.pop("system_prompt") if "system_prompt" in kwargs else AGENT_GENERATOR["system_prompt"]
        actions = kwargs.pop("actions") if "actions" in kwargs else [AgentGeneration(tools=kwargs.pop("tools", []))]
        super().__init__(name=name, description=description, system_prompt=system_prompt, actions=actions, **kwargs)
    
    @property
    def agent_generation_action_name(self):
        return self.get_action_name(action_cls=AgentGeneration)
    
