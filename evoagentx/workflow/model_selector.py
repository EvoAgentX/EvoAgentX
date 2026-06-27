from abc import ABC, abstractmethod
from typing import Dict, Optional, Union

from ..agents import Agent
from ..models import LLMConfig


class ModelSelector(ABC):

    def validate_agent(self, agent: Union[Agent, Dict]) -> None:
        if not isinstance(agent, Agent) and not isinstance(agent, dict):
            raise TypeError(f"Unsupported agent type: {type(agent)}")

    @abstractmethod
    def get_model(self, agent: Union[Agent, Dict]) -> LLMConfig:
        pass


class DefaultModelSelector(ModelSelector):
    """
    The default model selector, which will return the same LLMConfig for all agents.
    """

    def __init__(self, llm_config: LLMConfig, override: bool = True):
        """
        Args:
            llm_config: The LLMConfig to use for all agents.
            override: Whether to override the existingLLMConfig of agents.
        """
        self.llm_config = llm_config
        self.override = override

    def get_model(self, agent: Union[Agent, Dict]) -> LLMConfig:
        self.validate_agent(agent)

        if self.override:
            return self.llm_config
        else:
            if isinstance(agent, Agent):
                llm_config = getattr(agent, "llm_config", None)
                if llm_config:
                    return llm_config
                return self.llm_config

            elif isinstance(agent, dict):
                llm_config = agent.get("llm_config")
                if llm_config:
                    if isinstance(llm_config, dict):
                        return LLMConfig.from_dict(llm_config)
                    return llm_config
                return self.llm_config
            
            else:
                raise TypeError(f"Unsupported agent type: {type(agent)}")


class SimpleModelSelector(ModelSelector):
    """
    The simple model selector provides one LLMConfig for agents without tools and another for agents with tools.
    """

    def __init__(
        self, 
        llm_config_no_tools: LLMConfig, 
        llm_config_with_tools: LLMConfig,
        override: bool = True
    ):
        """
        Args:
            llm_config_no_tools: The LLMConfig to use for agents without tools.
            llm_config_with_tools: The LLMConfig to use for agents with tools.
            override: Whether to override the existing LLMConfig of agents.
        """
        self.llm_config_no_tools = llm_config_no_tools
        self.llm_config_with_tools = llm_config_with_tools
        self.override = override

    def get_model(self, agent: Union[Agent, Dict]) -> LLMConfig:
        self.validate_agent(agent)

        if isinstance(agent, Agent):
            llm_config = getattr(agent, "llm_config", None)
            if llm_config and not self.override:
                return llm_config

            tools = getattr(agent, "tools", None)
            if tools:
                return self.llm_config_with_tools
            return self.llm_config_no_tools

        elif isinstance(agent, dict):
            llm_config = agent.get("llm_config")
            if llm_config and not self.override:
                if isinstance(llm_config, dict):
                    return LLMConfig.from_dict(llm_config)
                return llm_config

            tools = agent.get("tools")
            tool_names = agent.get("tool_names")
            if tools or tool_names:
                return self.llm_config_with_tools
            return self.llm_config_no_tools

        else:
            raise TypeError(f"Unsupported agent type: {type(agent)}")


class ToolBasedModelSelector(SimpleModelSelector):
    """
    The tool-based model selector provides specific LLMConfigs based on the tools an agent possesses.
    """

    def __init__(
        self,
        llm_config_no_tools: LLMConfig,
        llm_config_with_tools: LLMConfig,
        tool_to_llm_config: Optional[Dict[str, LLMConfig]] = None,
        override: bool = True
    ):
        """
        Args:
            llm_config_no_tools: The LLMConfig to use for agents without tools.
            llm_config_with_tools: The LLMConfig to use for agents with tools.
            tool_to_llm_config: A dictionary mapping tool names to LLMConfigs.
            override: Whether to override the existing LLMConfig of agents.
        """
        super().__init__(llm_config_no_tools, llm_config_with_tools, override)
        self.tool_to_llm_config = tool_to_llm_config


    def get_model(self, agent: Union[Agent, Dict]) -> LLMConfig:
        self.validate_agent(agent)

        if self.tool_to_llm_config is None:
            return super().get_model(agent)

        if isinstance(agent, Agent):
            llm_config = getattr(agent, "llm_config", None)
            if llm_config and not self.override:
                return llm_config

            tools = getattr(agent, "tools", None)
            if tools:
                for tool in tools:
                    if tool.name in self.tool_to_llm_config:
                        return self.tool_to_llm_config[tool.name]
                return self.llm_config_with_tools
            return self.llm_config_no_tools

        elif isinstance(agent, dict):
            llm_config = agent.get("llm_config")
            if llm_config and not self.override:
                if isinstance(llm_config, dict):
                    return LLMConfig.from_dict(llm_config)
                return llm_config

            tools = agent.get("tools")
            tool_names = agent.get("tool_names")
            
            if tools:
                for tool in tools:
                    if isinstance(tool, dict):
                        tool_name = tool.get("name")
                    else:
                        tool_name = getattr(tool, "name", None)
                    
                    if tool_name in self.tool_to_llm_config:
                        return self.tool_to_llm_config[tool_name]
                return self.llm_config_with_tools
            
            elif tool_names:
                for tool_name in tool_names:
                    if tool_name in self.tool_to_llm_config:
                        return self.tool_to_llm_config[tool_name]
                return self.llm_config_with_tools
            
            return self.llm_config_no_tools

        else:
            raise TypeError(f"Unsupported agent type: {type(agent)}")