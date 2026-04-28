import unittest
from typing import Dict, List, Optional

from evoagentx.agents.agent import Agent
from evoagentx.agents.agent_manager import AgentManager, AgentState
from evoagentx.agents.customize_agent import CustomizeAgent
from evoagentx.models.model_configs import LiteLLMConfig
from evoagentx.tools.tool import Tool, ToolMetadata, ToolResult


class TestModule(unittest.TestCase):

    def test_agent_manager(self):

        OPENAI_API_KEY = "xxxxx"
        llm_config = LiteLLMConfig(model="gpt-4o-mini", openai_key=OPENAI_API_KEY)

        class MockToolInAgentManager(Tool):
            name: str = "MockToolInAgentManager"
            description: str = "Mock tool description"
            inputs: Dict = {
                "tool_input": {
                    "type": "string", 
                    "description": "tool input description"
                }
            }
            required: Optional[List[str]] = None
            def __call__(self, tool_input: str) -> ToolResult:
                return ToolResult(
                    result="Mock tool result", 
                    metadata=ToolMetadata(
                        name="MockToolInAgentManager", 
                        args={"tool_input": tool_input}
                    )
                )
        
        agent = Agent(
            name="Bob",
            description="Bob is an engineer. He excels in writing and reviewing codes for different projects.", 
            system_prompt="You are an excellent engineer and you can solve diverse coding tasks.",
            llm_config=llm_config,
            actions = [
                {
                    "name": "WriteFileToDisk",
                    "description": "save several files to local storage.",
                    "tools": [MockToolInAgentManager()]
                }
            ]
        )

        # example 1
        agent_manager = AgentManager()
        agent_manager.add_agents(
            agents=[
                agent, 
                {
                    "class_name": "Agent", 
                    "name": "test_agent",
                    "description": "test_agent_description", 
                    "llm_config": llm_config
                }
            ]
        )

        self.assertEqual(agent_manager.size, 2)
        self.assertTrue(agent_manager.has_agent(agent_name="Bob"))
        
        num_agents = agent_manager.size
        agent_manager.add_agents(agents=[agent])
        self.assertEqual(agent_manager.size, num_agents)
        self.assertTrue(isinstance(agent_manager.get_agent("test_agent"), Agent))
        self.assertEqual(agent_manager.size, 2)

        agent_manager.add_agent(
            {
                "name": "custom_agent", 
                "description": "custom_agent_desc", 
                "prompt": "customize prompt", 
                "is_human": True
            }
        )
        self.assertEqual(agent_manager.size, 3)
        self.assertTrue(isinstance(agent_manager.get_agent("custom_agent"), CustomizeAgent))
        
        agent_manager.remove_agent(agent_name="test_agent")
        self.assertEqual(agent_manager.size, 2)
        self.assertTrue(agent_manager.has_agent("Bob"))
        self.assertTrue(agent_manager.has_agent("custom_agent"))

        self.assertEqual(agent_manager.get_agent_state("Bob"), AgentState.AVAILABLE)
        agent_manager.set_agent_state(agent_name="Bob", new_state=AgentState.RUNNING)
        self.assertEqual(agent_manager.get_agent_state("Bob"), AgentState.RUNNING)

        agent_manager.clear_agents()
        self.assertEqual(agent_manager.size, 0)


if __name__ == "__main__":
    unittest.main()
