import unittest
from unittest.mock import MagicMock

from evoagentx.agents.agent import Agent
from evoagentx.models.model_configs import OpenAILLMConfig
from evoagentx.workflow.model_selector import (
    DefaultModelSelector,
    SimpleModelSelector,
    ToolBasedModelSelector,
)


class TestModelSelector(unittest.TestCase):

    def setUp(self):
        self.default_llm_config = OpenAILLMConfig(llm_type="OpenAILLM", model="gpt-4o")
        self.another_llm_config = OpenAILLMConfig(llm_type="OpenAILLM", model="gpt-4o-mini")
        self.tool_llm_config = OpenAILLMConfig(llm_type="OpenAILLM", model="gpt-4o-tool")

    def test_default_model_selector_override(self):
        selector = DefaultModelSelector(llm_config=self.default_llm_config, override=True)
        
        # Test with Agent instance having its own config
        agent = MagicMock(spec=Agent)
        agent.llm_config = self.another_llm_config
        
        result = selector.get_model(agent)
        self.assertEqual(result, self.default_llm_config)
        
        # Test with dict
        agent_dict = {"llm_config": self.another_llm_config}
        result = selector.get_model(agent_dict)
        self.assertEqual(result, self.default_llm_config)

    def test_default_model_selector_no_override(self):
        selector = DefaultModelSelector(llm_config=self.default_llm_config, override=False)
        
        # Test with Agent instance having its own config - should return agent's config
        agent = MagicMock(spec=Agent)
        agent.llm_config = self.another_llm_config
        result = selector.get_model(agent)
        self.assertEqual(result, self.another_llm_config)
        
        # Test with Agent instance NOT having its own config - should return default
        agent_no_config = MagicMock(spec=Agent)
        agent_no_config.llm_config = None
        result = selector.get_model(agent_no_config)
        self.assertEqual(result, self.default_llm_config)
        
        # Test with dict having its own config as object
        agent_dict = {"llm_config": self.another_llm_config}
        result = selector.get_model(agent_dict)
        self.assertEqual(result, self.another_llm_config)

        # Test with dict having its own config as dict (will be converted via LLMConfig.from_dict)
        agent_dict_raw = {"llm_config": self.another_llm_config.to_dict()}
        result = selector.get_model(agent_dict_raw)
        self.assertEqual(result.model, self.another_llm_config.model)
        self.assertEqual(result.llm_type, self.another_llm_config.llm_type)
        
    def test_simple_model_selector(self):
        selector = SimpleModelSelector(
            llm_config_no_tools=self.default_llm_config,
            llm_config_with_tools=self.tool_llm_config,
            override=True
        )
        
        # Agent with tools
        agent_with_tools = MagicMock(spec=Agent)
        agent_with_tools.tools = [MagicMock()]
        result = selector.get_model(agent_with_tools)
        self.assertEqual(result, self.tool_llm_config)
        
        # Agent without tools
        agent_no_tools = MagicMock(spec=Agent)
        agent_no_tools.tools = []
        result = selector.get_model(agent_no_tools)
        self.assertEqual(result, self.default_llm_config)

        # Dict with tool_names
        agent_dict = {"tool_names": ["search"]}
        result = selector.get_model(agent_dict)
        self.assertEqual(result, self.tool_llm_config)

    def test_simple_model_selector_no_override(self):
        selector = SimpleModelSelector(
            llm_config_no_tools=self.default_llm_config,
            llm_config_with_tools=self.tool_llm_config,
            override=False
        )
        
        # Agent with its own config - should return agent's config regardless of tools
        agent = MagicMock(spec=Agent)
        agent.llm_config = self.another_llm_config
        agent.tools = [MagicMock()]
        result = selector.get_model(agent)
        self.assertEqual(result, self.another_llm_config)

    def test_tool_based_model_selector(self):
        tool_mapping = {"coder": self.another_llm_config}
        selector = ToolBasedModelSelector(
            llm_config_no_tools=self.default_llm_config,
            llm_config_with_tools=self.tool_llm_config,
            tool_to_llm_config=tool_mapping,
            override=True
        )
        
        # Agent with 'coder' tool
        tool = MagicMock()
        tool.name = "coder"
        agent = MagicMock(spec=Agent)
        agent.tools = [tool]
        result = selector.get_model(agent)
        self.assertEqual(result, self.another_llm_config)
        
        # Agent with other tool - should fallback to llm_config_with_tools
        other_tool = MagicMock()
        other_tool.name = "random"
        agent.tools = [other_tool]
        result = selector.get_model(agent)
        self.assertEqual(result, self.tool_llm_config)
        
        # Agent with no tools - should fallback to llm_config_no_tools
        agent.tools = []
        result = selector.get_model(agent)
        self.assertEqual(result, self.default_llm_config)

    def test_tool_based_model_selector_dict(self):
        tool_mapping = {"coder": self.another_llm_config}
        selector = ToolBasedModelSelector(
            llm_config_no_tools=self.default_llm_config,
            llm_config_with_tools=self.tool_llm_config,
            tool_to_llm_config=tool_mapping,
            override=True
        )
        
        # Dict with 'coder' in tool_names
        agent_dict = {"tool_names": ["coder", "search"]}
        result = selector.get_model(agent_dict)
        self.assertEqual(result, self.another_llm_config)
        
        # Dict with 'coder' in tools (list of dicts)
        agent_dict_tools = {"tools": [{"name": "coder"}]}
        result = selector.get_model(agent_dict_tools)
        self.assertEqual(result, self.another_llm_config)
        
        # Dict with 'coder' in tools (list of objects)
        tool = MagicMock()
        tool.name = "coder"
        agent_dict_tools_obj = {"tools": [tool]}
        result = selector.get_model(agent_dict_tools_obj)
        self.assertEqual(result, self.another_llm_config)

    def test_invalid_agent_type(self):
        selector = DefaultModelSelector(llm_config=self.default_llm_config)
        with self.assertRaises(TypeError):
            selector.get_model("not an agent")

if __name__ == "__main__":
    unittest.main()
