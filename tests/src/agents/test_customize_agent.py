import os 
import unittest
from unittest.mock import patch
from pydantic import Field 
from evoagentx.core.registry import register_parse_function 
from evoagentx.models.model_configs import LiteLLMConfig
from evoagentx.agents.customize_agent import CustomizeAgent
from evoagentx.actions.action import ActionOutput
from evoagentx.core.message import Message, MessageType
from evoagentx.core.module_utils import extract_code_blocks

class CodeWriterActionOutput(ActionOutput):
    code: str = Field(description="The generated Python code") 

@register_parse_function
def customize_parse_func(content: str) -> dict:
    return {"code": extract_code_blocks(content)[0]}


class TestModule(unittest.TestCase):

    def setUp(self):
        self.save_files = [
            "tests/agents/saved_customize_agent.json", 
            "tests/agents/saved_customize_agent_with_inputs.json", 
            "tests/agents/saved_customize_agent_with_outputs.json",  
            "tests/agents/saved_customize_agent_with_inputs_outputs.json",
            "tests/agents/saved_customize_agent_with_parser.json" 
        ]

    @patch("evoagentx.models.litellm_model.LiteLLM.single_generate")
    def test_simple_agent(self, mock_generate):
        mock_generate.return_value = "Hello, world!"
        llm_config = LiteLLMConfig(model="gpt-4o-mini", openai_key="xxxxx")

        simple_agent = CustomizeAgent(
            name = "Simple Agent",
            description = "A simple agent that prints hello world", 
            prompt = "You are a simple agent that prints hello world.",
            llm_config = llm_config
        )
        self.assertEqual(simple_agent.name, "Simple Agent") 
        self.assertEqual(simple_agent.prompt, "You are a simple agent that prints hello world.")
        self.assertEqual(simple_agent.customize_action_name, "SimpleAgentAction") 
        self.assertEqual(simple_agent.get_prompts()["SimpleAgentAction"]["prompt"], "You are a simple agent that prints hello world.")
        self.assertEqual(len(simple_agent.action.inputs_format.get_attrs()), 0) 
        self.assertEqual(len(simple_agent.action.outputs_format.get_attrs()), 0)

        # save and load the agent 
        simple_agent.save_module(self.save_files[0])
        new_agent: CustomizeAgent = CustomizeAgent.from_file(self.save_files[0], llm_config=llm_config)
        self.assertEqual(new_agent.name, "Simple Agent")
        self.assertEqual(len(new_agent.action.inputs_format.get_attrs()), 0) 
        self.assertEqual(len(new_agent.action.outputs_format.get_attrs()), 0)

        # execute
        msg = new_agent()
        self.assertTrue(isinstance(msg, Message))
        self.assertEqual(msg.msg_type, MessageType.UNKNOWN)
        self.assertEqual(msg.content.content, "Hello, world!")
    
    @patch("evoagentx.models.litellm_model.LiteLLM.single_generate")
    def test_agent_with_inputs_and_outputs(self, mock_generate):
        mock_generate.return_value = "```python\nprint('Hello, world!')```" 
        llm_config = LiteLLMConfig(model="gpt-4o-mini", openai_key="xxxxx")
        agent_with_inputs = CustomizeAgent(
            name = "CodeWriter",
            description = "Writes Python code based on requirements",
            prompt = "Write Python code that implements the following requirement: {requirement}",
            llm_config=llm_config,
            inputs = [
                {
                    "name": "requirement",
                    "type": "string",
                    "description": "The coding requirement",
                    "required": True
                }
            ]
        )
        self.assertEqual(len(agent_with_inputs.action.inputs_format.get_attrs()), 1)
        self.assertEqual(len(agent_with_inputs.action.outputs_format.get_attrs()), 0)

        agent_with_inputs.save_module(self.save_files[1])
        new_agent_with_inputs: CustomizeAgent = CustomizeAgent.from_file(self.save_files[1], llm_config=llm_config)
        self.assertEqual(len(new_agent_with_inputs.action.inputs_format.get_attrs()), 1)
        self.assertEqual(len(new_agent_with_inputs.action.outputs_format.get_attrs()), 0) 

        msg = new_agent_with_inputs(inputs={"requirement": "Write Python code that prints hello world"}, return_msg_type=MessageType.RESPONSE)
        self.assertEqual(msg.msg_type, MessageType.RESPONSE)
        self.assertEqual(msg.content.content, "```python\nprint('Hello, world!')```")

        agent_with_outputs = CustomizeAgent(
            name = "CodeWriter",
            description = "Writes Python code based on requirements",
            prompt = "Write Python code that implements the following requirement: Write Python code that prints hello world",
            llm_config=llm_config,
            outputs = [
                {
                    "name": "code",
                    "type": "string",
                    "description": "The generated Python code",
                    "required": True
                }
            ],
            parse_mode = "custom",
            parse_func = customize_parse_func,  
            title_format = "## {title}" 
        )
        self.assertEqual(len(agent_with_outputs.action.inputs_format.get_attrs()), 0)
        self.assertEqual(len(agent_with_outputs.action.outputs_format.get_attrs()), 1)

        agent_with_outputs.save_module(self.save_files[2]) 
        new_agent_with_outputs: CustomizeAgent = CustomizeAgent.from_file(self.save_files[2], llm_config=llm_config)
        self.assertEqual(len(new_agent_with_outputs.action.inputs_format.get_attrs()), 0)
        self.assertEqual(len(new_agent_with_outputs.action.outputs_format.get_attrs()), 1)
        self.assertEqual(new_agent_with_outputs.parse_func.__name__, "customize_parse_func")

        msg = new_agent_with_outputs(return_msg_type=MessageType.RESPONSE)
        self.assertEqual(msg.msg_type, MessageType.RESPONSE)
        self.assertEqual(msg.content.content, "```python\nprint('Hello, world!')```")
        self.assertEqual(msg.content.code, "print('Hello, world!')")

        agent_with_inputs_outputs = CustomizeAgent(
            name = "CodeWriter",
            description = "Writes Python code based on requirements",
            prompt = "Write Python code that implements the following requirement: {requirement}",
            llm_config=llm_config,
            inputs = [
                {
                    "name": "requirement",
                    "type": "string",
                    "description": "The coding requirement",
                    "required": True
                }
            ],
            outputs = [
                {
                    "name": "code",
                    "type": "string",
                    "description": "The generated Python code",
                    "required": True
                }
            ],
            parse_mode = "custom",
            parse_func = customize_parse_func
        )
        self.assertEqual(len(agent_with_inputs_outputs.action.inputs_format.get_attrs()), 1)
        self.assertEqual(len(agent_with_inputs_outputs.action.outputs_format.get_attrs()), 1)

        agent_with_inputs_outputs.save_module(self.save_files[3]) 
        new_agent_with_inputs_outputs: CustomizeAgent = CustomizeAgent.from_file(self.save_files[3], llm_config=llm_config)
        self.assertEqual(len(new_agent_with_inputs_outputs.action.inputs_format.get_attrs()), 1)
        self.assertEqual(len(new_agent_with_inputs_outputs.action.outputs_format.get_attrs()), 1)

        msg = new_agent_with_inputs_outputs(inputs={"requirement": "Write Python code that prints hello world"}, return_msg_type=MessageType.RESPONSE)
        self.assertEqual(msg.msg_type, MessageType.RESPONSE)
        self.assertEqual(msg.content.content, "```python\nprint('Hello, world!')```")
        self.assertEqual(msg.content.code, "print('Hello, world!')")

        agent_with_parser = CustomizeAgent(
            name = "CodeWriter",
            description = "Writes Python code based on requirements",
            prompt = "Write Python code that implements the following requirement: {requirement}",
            llm_config=llm_config,
            inputs = [
                {
                    "name": "requirement",
                    "type": "string",
                    "description": "The coding requirement",
                    "required": True
                }
            ],
            outputs = [
                {
                    "name": "code",
                    "type": "string",
                    "description": "The generated Python code",
                    "required": True
                },
                {
                    "name": "explanation",
                    "type": "string",
                    "description": "The explanation of the generated Python code",
                    "required": True
                }
            ], 
            output_parser = CodeWriterActionOutput,
            parse_mode = "custom",
            parse_func = customize_parse_func
        )

        self.assertEqual(agent_with_parser.action.outputs_format.__name__, "CodeWriterActionOutput")
        agent_with_parser.save_module(self.save_files[4])
        new_agent_with_parser: CustomizeAgent = CustomizeAgent.from_file(self.save_files[4], llm_config=llm_config)
        self.assertEqual(new_agent_with_parser.action.outputs_format.__name__, "CodeWriterActionOutput")

        msg = new_agent_with_parser(inputs={"requirement": "Write Python code that prints hello world"}, return_msg_type=MessageType.RESPONSE)
        self.assertEqual(msg.msg_type, MessageType.RESPONSE)
        self.assertEqual(msg.content.content, "```python\nprint('Hello, world!')```")
        self.assertEqual(msg.content.code, "print('Hello, world!')")

    def tearDown(self):
        for file in self.save_files:
            if os.path.exists(file):
                os.remove(file)

if __name__ == "__main__":
    unittest.main()