import os 
import unittest
from evoagentx.models.litellm_model import LiteLLM
from evoagentx.models.model_configs import LiteLLMConfig
from evoagentx.agents.agent import Agent
from evoagentx.actions.action import Action


class TestModule(unittest.TestCase):

    def setUp(self):
        self.save_file = "tests/agents/saved_agent.json"

    def test_initialization(self):

        agent_data = {
            "name": "test_agent", 
            "description": "test_agent_description", 
            "llm_config": {
                "class_name": "LiteLLMConfig",
                "model": "gpt-4o-mini", 
                "openai_key": "xxxxx"
            }, 
            "actions": [
                {
                    "class_name": "Action", 
                    "name": "test_action_name", 
                    "description": "test_action_desc", 
                }
            ]
        }
        agent = Agent.from_dict(agent_data)
        self.assertEqual(agent.llm_config.model, "gpt-4o-mini")
        self.assertTrue(isinstance(agent.llm, LiteLLM))
        self.assertTrue(isinstance(agent.actions[0], Action))

        # test __eq__ & __hash__
        agent2 = Agent.from_dict(agent_data)
        agent_list = [agent]
        self.assertTrue(agent2 not in agent_list)
        self.assertTrue(agent2 != agent)
        agent2_id = agent2.agent_id
        agent2.agent_id = agent.agent_id
        self.assertTrue(agent2 in agent_list)
        self.assertTrue(agent2 == agent)
        agent2.agent_id = agent2_id

    def test_save_agent(self):

        llm_config = LiteLLMConfig(model="gpt-4o-mini", openai_key="xxxxx")
        
        agent = Agent(
            name="Bob",
            description="Bob is an engineer. He excels in writing and reviewing codes for different projects.", 
            system_prompt="You are an excellent engineer and you can solve diverse coding tasks.",
            llm_config=llm_config,
            actions = [
                {
                    "name": "WriteFileToDisk",
                    "description": "save several files to local storage.", 
                    "tools": [{"class_name": "Tool"}]
                }
            ]
        )
        agent.save_module(path=self.save_file, ignore=["llm_config"])
        loaded_agent = Agent.from_file(path=self.save_file, llm_config=llm_config)
        self.assertEqual(agent, loaded_agent)

    def tearDown(self):
        if os.path.exists(self.save_file):
            os.remove(self.save_file)

if __name__ == "__main__":
    unittest.main()