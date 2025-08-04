import json
from typing import Dict, List, Optional, Union

from ..actions.action import Action
from ..actions.agent_adaptor import AgentAdaptorAction
from ..agents import Agent, CustomizeAgent
from ..core.registry import MODEL_REGISTRY
from ..models import BaseLLM, LLMConfig, OpenAILLMConfig
from ..tools import Tool, Toolkit
from ..utils.utils import get_unique_class_name


class AgentAdaptor(Agent):
    """
    AgentAdaptor is a wrapper for an agent, which can transform the inputs and outputs of the agent to the specified formats.

    Args:
        agent (Agent): The agent to be wrapped.
        inputs (List[Dict]): List of input specifications, where each dict 
            (e.g., `{"name": str, "type": str, "description": str, ["required": bool]}`) contains:
            - name (str): Name of the input parameter
            - type (str): Type of the input
            - description (str): Description of what the input represents
            - required (bool, optional): Whether this input is required (default: True)
        outputs (List[Dict]): Same format as `inputs`.
    """
    def __init__(
        self, 
        agent: Agent, 
        inputs: List[Dict], 
        outputs: List[Dict],
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
        
        name = get_unique_class_name(agent.name)
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
        inputs: List[Dict],
        outputs: List[Dict],
        llm: BaseLLM
    ) -> Action:

        action_input_type = CustomizeAgent.create_action_input(inputs, "input transformation")
        action_output_type = CustomizeAgent.create_action_output(outputs, "output transformation")
        
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
            "agent": self.agent.get_config(),
            "llm_config": self.llm_config
        }

        return config


    def to_dict(self, exclude_none: bool = True, ignore: List[str] = [], **kwargs) -> dict:
        config = self.get_config()

        for ignore_key in ignore:
            config.pop(ignore_key, None)
            config["agent"].pop(ignore_key, None)

        tools = config["agent"].pop("tools", None)
        if tools is not None and len(tools) > 0:
            config["agent"]["tool_names"] = [tool.name for tool in tools]

        return config


    @classmethod
    def load_module(
        cls, 
        path: str, 
        llm_config: Optional[LLMConfig] = None, 
        agent_adaptor_llm_config: Optional[LLMConfig] = None,
        tools: Optional[List[Union[Toolkit, Tool]]] = None,
        **kwargs
    ) -> Dict:
        """
        Load the agent adaptor from a JSON file.

        Args:
            path (str): The path of the JSON file.
            llm_config (Optional[LLMConfig]): The LLMConfig to be used for the agent that the agent adaptor is based on.
            agent_adaptor_llm_config (Optional[LLMConfig]): The LLMConfig to be used for the agent adaptor. Default: OpenAILLMConfig(model="gpt-4o-mini")
            tools (Optional[List[Union[Toolkit, Tool]]]): The tools to be used by the agent.

        Returns:
            A dictionary containing all necessary configuration to recreate this agent adaptor.
        """
        data = json.load(open(path, "r"))
        agent_adaptor_dict = cls._process_dict(data, llm_config, agent_adaptor_llm_config, tools)
        return agent_adaptor_dict


    @classmethod
    def from_dict(
        cls, 
        data: Dict, 
        llm_config: Optional[LLMConfig] = None, 
        agent_adaptor_llm_config: Optional[LLMConfig] = None,
        tools: Optional[List[Union[Toolkit, Tool]]] = None,
        **kwargs
    ) -> "AgentAdaptor":
        """
        Instantiate AgentAdaptor from a dictionary.
        
        Args:
            data (Dict): Dictionary containing agent adaptor configurations.
            llm_config (Optional[LLMConfig]): The LLMConfig to be used for the agent that the agent adaptor is based on.
            agent_adaptor_llm_config (Optional[LLMConfig]): The LLMConfig to be used for the agent adaptor. Default: OpenAILLMConfig(model="gpt-4o-mini")
            tools (Optional[List[Union[Toolkit, Tool]]]): The tools to be used by the agent.
            
        Returns:
            AgentAdaptor: The created agent adaptor instance
        """
        data = cls._process_dict(data, llm_config, agent_adaptor_llm_config, tools)
        agent = CustomizeAgent.from_dict(data["agent"])
        data["agent"] = agent
        data["llm_config"] = LLMConfig.from_dict(data["llm_config"])
        return cls(**data)

    
    @staticmethod
    def _process_dict( 
        data: Dict, 
        llm_config: Optional[LLMConfig] = None, 
        agent_adaptor_llm_config: Optional[LLMConfig] = None,
        tools: Optional[List[Union[Toolkit, Tool]]] = None,
    ) -> Dict:
        """
        Checks if agent adaptor and the agent it's based on have `llm_config` in the dictionary.
        Converts `tool_names` to tools.

        Args:
            data (Dict): The dictionary containing the agent adaptor's configuration.
            llm_config (Optional[LLMConfig]): The LLMConfig to be used for the agent adaptor.
            agent_adaptor_llm_config (Optional[LLMConfig]): The LLMConfig to be used for the agent adaptor. Default: OpenAILLMConfig(model="gpt-4o-mini")
            tools (Optional[List[Union[Toolkit, Tool]]]): The tools to be used by the agent.

        Returns:
            Dict: The processed dictionary containing the agent adaptor's configuration
        """

        adaptor_llm_config = data.get("llm_config", None)
        agent_llm_config = data["agent"].get("llm_config", None)

        if llm_config is not None:
            data["agent"]["llm_config"] = llm_config
        else:
            if agent_llm_config is None:
                raise ValueError("Must provide `llm_config` for agent")
        
        if agent_adaptor_llm_config is None and adaptor_llm_config is None:
            data["llm_config"] = OpenAILLMConfig(model="gpt-4o-mini")

        agent_tool_names = data["agent"].pop("tool_names", None)

        if agent_tool_names is not None:
            if len(agent_tool_names) == 0:
                return data

            if tools is None:
                raise ValueError(f"Must provide the following `tools` for agent: {agent_tool_names}")

            tool_map = {tool.name: tool for tool in tools}
            agent_tools = []

            for tool_name in agent_tool_names:
                if tool_name not in tool_map:
                    raise ValueError(f"'{tool_name}' not found in provided tools")
                agent_tools.append(tool_map[tool_name])

            data["agent"]["tools"] = agent_tools

        return data


