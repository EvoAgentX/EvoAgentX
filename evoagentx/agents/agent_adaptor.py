from typing import Dict, List, Optional

from ..actions.action import Action
from ..actions.agent_adaptor import AgentAdaptorAction
from ..agents import Agent, CustomizeAgent
from ..core.base_config import Parameter
from ..core.registry import MODEL_REGISTRY
from ..models import BaseLLM, LLMConfig, OpenAILLMConfig
from ..utils.utils import add_llm_config_to_agent_dict


class AgentAdaptor(Agent):
    """
    AgentAdaptor is a wrapper for an agent, which can transform the inputs and outputs of the agent to the specified formats.

    Args:
        name (str): Unique identifier for the agent adaptor.
        agent (Agent): The agent to be wrapped.
        inputs (List[Parameter]): List of input specifications, must contain:
            - name (str): Name of the input parameter
            - type (str): Type of the input
            - description (str): Description of what the input represents
            - required (bool, optional): Whether this input is required (default: True)
        outputs (List[Parameter]): Same format as `inputs`.
        llm_config (Optional[LLMConfig]): The LLMConfig to be used for the agent adaptor. Default: OpenAILLMConfig(model="gpt-4o-mini").
    """
    def __init__(
        self, 
        name: str,
        agent: Agent, 
        inputs: List[Parameter], 
        outputs: List[Parameter],
        llm_config: Optional[LLMConfig] = None,
        **kwargs
    ):

        if llm_config is None:
            llm_config = OpenAILLMConfig(model="gpt-4o-mini")
        
        llm_cls = MODEL_REGISTRY.get_model(llm_config.llm_type)
        llm = llm_cls(config=llm_config)

        # All agents have a `ContextExtraction` action, filter it out
        non_ContextExtraction_actions = [
            action for action in agent.actions if action.class_name != "ContextExtraction"
        ]

        if len(non_ContextExtraction_actions) > 1:
            raise ValueError(f"AgentAdaptor currently only supports agents with a single action. {agent.name} has {len(non_ContextExtraction_actions)} actions.")

        agent_adaptor_action = self.create_agent_adaptor_action(agent, inputs, outputs, llm)

        super().__init__(
            name=name, 
            description=agent.description, 
            actions=[agent_adaptor_action],
            llm=llm
        )
        self.agent = agent
        self.inputs = inputs
        self.outputs = outputs
        self.llm_config = llm_config


    @staticmethod
    def create_agent_adaptor_action(
        agent: Agent, 
        inputs: List[Parameter],
        outputs: List[Parameter],
        llm: BaseLLM
    ) -> Action:

        inputs_dict = [input.to_dict(ignore=["class_name"]) for input in inputs]
        outputs_dict = [output.to_dict(ignore=["class_name"]) for output in outputs]

        action_input_type = CustomizeAgent.create_action_input(inputs_dict, "input transformation")
        action_output_type = CustomizeAgent.create_action_output(outputs_dict, "output transformation")
        
        agent_adaptor_action = AgentAdaptorAction(
            agent=agent, 
            inputs_format=action_input_type, 
            outputs_format=action_output_type,
            llm=llm
        )

        return agent_adaptor_action


    def get_config(self) -> Dict:
        """
        Get the dictionary containing all necessary configuration to recreate this agent.
        
        Returns:
            A configuration dictionary that can be used to initialize a new Agent instance
            with the same properties as this one.
        """

        config = {
            "class_name": "AgentAdaptor",
            "name": self.name,
            "description": self.description,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "agent": self.agent.name,
            "llm_config": self.llm_config
        }

        return config


    def to_dict(self, exclude_none: bool = True, ignore: List[str] = [], **kwargs) -> Dict:
        config = self.get_config()

        for ignore_key in ignore:
            config.pop(ignore_key, None)

        return config


    @classmethod
    def from_dict(
        cls, 
        data: Dict, 
        agents: List[Agent],
        llm_config: Optional[LLMConfig] = None,
        **kwargs
    ) -> "AgentAdaptor":
        """
        Instantiate AgentAdaptor from a dictionary.
        
        Args:
            data (Dict): Dictionary containing agent adaptor configurations.
            agents (List[Agent]): The list of agents containing the agent that the agent adaptor is based on.
            llm_config (Optional[LLMConfig]): The LLMConfig to be used for the agent that the agent adaptor is based on.
            
        Returns:
            AgentAdaptor: The created agent adaptor instance
        """
        if llm_config is None:
            llm_config = OpenAILLMConfig(model="gpt-4o-mini")

        data = add_llm_config_to_agent_dict(data, llm_config)
        agent_map = {agent.name: agent for agent in agents}
        agent = agent_map.get(data["agent"], None)

        if agent is None:
            raise ValueError(f"'{data['agent']}' not found in the provided list of agents.")    

        data["agent"] = agent

        return cls(**data)

