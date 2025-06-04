import os
import unittest
from unittest.mock import patch
from pydantic import BaseModel, Field
from evoagentx.core.registry import register_parse_function
from evoagentx.models.model_configs import LiteLLMConfig
from evoagentx.agents.customize_agent import CustomizeAgent
from evoagentx.actions.action import ActionOutput
from evoagentx.core.message import Message, MessageType
from evoagentx.core.module_utils import extract_code_blocks

# Define output format for code generation
class CodeWriterActionOutput(ActionOutput):
    code: str = Field(description="The generated Python code")

# Register custom parse function
@register_parse_function
def customize_parse_func(content: str) -> dict:
    code_blocks = extract_code_blocks(content)
    return {"code": code_blocks[0] if code_blocks else ""}

class TestCustomizeAgent(unittest.TestCase):
    def setUp(self):
        # Define paths for saving test agent files
        self.save_files = [
            "tests/agents/saved_simple_agent.json",
            "tests/agents/saved_agent_with_inputs.json",
            "tests/agents/saved_agent_with_outputs.json",
            "tests/agents/saved_agent_with_inputs_outputs.json",
            "tests/agents/saved_agent_with_parser.json"
        ]
        # Ensure test directory exists
        os.makedirs("tests/agents", exist_ok=True)

    @patch("evoagentx.models.litellm_model.LiteLLM.single_generate")
    def test_simple_agent(self, mock_generate):
        # Mock LLM output
        mock_generate.return_value = "Hello, world!"
        # Configure LiteLLMConfig for local Ollama model
        llm_config = LiteLLMConfig(
            llm_type="LiteLLM",
            model="ollama/llama3:8b",
            api_base="http://localhost:11434/v1",
            is_local=True,
            temperature=0.7,
            max_tokens=1000,
            output_response=True
        )

        # Create a simple agent
        simple_agent = CustomizeAgent(
            name="SimpleAgent",
            description="A simple agent that outputs a greeting",
            prompt="You are a simple agent that says hello world.",
            llm_config=llm_config
        )

        # Verify agent properties
        self.assertEqual(simple_agent.name, "SimpleAgent")
        self.assertEqual(simple_agent.prompt, "You are a simple agent that says hello world.")
        self.assertEqual(simple_agent.customize_action_name, "SimpleAgentAction")
        self.assertEqual(simple_agent.get_prompts()["SimpleAgentAction"]["prompt"], "You are a simple agent that says hello world.")
        self.assertEqual(len(simple_agent.action.inputs_format.get_attrs()), 0)
        self.assertEqual(len(simple_agent.action.outputs_format.get_attrs()), 0)

        # Save and load agent
        simple_agent.save_module(self.save_files[0])
        new_agent = CustomizeAgent.from_file(self.save_files[0], llm_config=llm_config)
        self.assertEqual(new_agent.name, "SimpleAgent")
        self.assertEqual(len(new_agent.action.inputs_format.get_attrs()), 0)
        self.assertEqual(len(new_agent.action.outputs_format.get_attrs()), 0)

        # Execute agent
        msg = new_agent()
        self.assertIsInstance(msg, Message)
        self.assertEqual(msg.msg_type, MessageType.UNKNOWN)
        self.assertEqual(msg.content.content, "Hello, world!")

    @patch("evoagentx.models.litellm_model.LiteLLM.single_generate")
    def test_agent_with_inputs_and_outputs(self, mock_generate):
        # Mock LLM output
        mock_generate.return_value = "```python\nprint('Hello, world!')```"
        # Configure LiteLLMConfig for local Ollama model
        llm_config = LiteLLMConfig(
            llm_type="LiteLLM",
            model="ollama/llama3:8b",
            api_base="http://localhost:11434/v1",
            is_local=True,
            temperature=0.7,
            max_tokens=1000,
            output_response=True
        )

        # Create agent with inputs
        agent_with_inputs = CustomizeAgent(
            name="CodeGenerator",
            description="Generates Python code based on requirements",
            prompt="Write Python code that implements the following requirement: {requirement}",
            llm_config=llm_config,
            inputs=[
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

        # Save and load agent
        agent_with_inputs.save_module(self.save_files[1])
        new_agent_with_inputs = CustomizeAgent.from_file(self.save_files[1], llm_config=llm_config)
        self.assertEqual(len(new_agent_with_inputs.action.inputs_format.get_attrs()), 1)
        self.assertEqual(len(new_agent_with_inputs.action.outputs_format.get_attrs()), 0)

        # Execute agent
        msg = agent_with_inputs(inputs={"requirement": "Write Python code that prints hello world"}, return_msg_type=MessageType.SUCCESS)
        self.assertEqual(msg.msg_type, MessageType.SUCCESS)
        self.assertEqual(msg.content.content, "```python\nprint('Hello, world!')```")

        # Create agent with outputs
        agent_with_outputs = CustomizeAgent(
            name="CodeGenerator",
            description="Generates Python code based on requirements",
            prompt="Write Python code that implements the following requirement: Write Python code that prints hello world",
            llm_config=llm_config,
            outputs=[
                {
                    "name": "code",
                    "type": "string",
                    "description": "The generated Python code",
                    "required": True
                }
            ],
            parse_mode="custom",
            parse_func=customize_parse_func,
            title_format="## {title}"
        )
        self.assertEqual(len(agent_with_outputs.action.inputs_format.get_attrs()), 0)
        self.assertEqual(len(agent_with_outputs.action.outputs_format.get_attrs()), 1)

        # Save and load agent
        agent_with_outputs.save_module(self.save_files[2])
        new_agent_with_outputs = CustomizeAgent.from_file(self.save_files[2], llm_config=llm_config)
        self.assertEqual(len(new_agent_with_outputs.action.inputs_format.get_attrs()), 0)
        self.assertEqual(len(new_agent_with_outputs.action.outputs_format.get_attrs()), 1)
        self.assertEqual(new_agent_with_outputs.parse_func.__name__, "customize_parse_func")

        # Execute agent
        msg = new_agent_with_outputs(return_msg_type=MessageType.SUCCESS)
        self.assertEqual(msg.msg_type, MessageType.SUCCESS)
        self.assertEqual(msg.content.content, "```python\nprint('Hello, world!')```")
        self.assertEqual(msg.content.code, "print('Hello, world!')")

        # Create agent with both inputs and outputs
        agent_with_inputs_outputs = CustomizeAgent(
            name="CodeGenerator",
            description="Generates Python code based on requirements",
            prompt="Write Python code that implements the following requirement: {requirement}",
            llm_config=llm_config,
            inputs=[
                {
                    "name": "requirement",
                    "type": "string",
                    "description": "The coding requirement",
                    "required": True
                }
            ],
            outputs=[
                {
                    "name": "code",
                    "type": "string",
                    "description": "The generated Python code",
                    "required": True
                }
            ],
            parse_mode="custom",
            parse_func=customize_parse_func
        )
        self.assertEqual(len(agent_with_inputs_outputs.action.inputs_format.get_attrs()), 1)
        self.assertEqual(len(agent_with_inputs_outputs.action.outputs_format.get_attrs()), 1)

        # Save and load agent
        agent_with_inputs_outputs.save_module(self.save_files[3])
        new_agent_with_inputs_outputs = CustomizeAgent.from_file(self.save_files[3], llm_config=llm_config)
        self.assertEqual(len(new_agent_with_inputs_outputs.action.inputs_format.get_attrs()), 1)
        self.assertEqual(len(new_agent_with_inputs_outputs.action.outputs_format.get_attrs()), 1)

        # Execute agent
        msg = new_agent_with_inputs_outputs(inputs={"requirement": "Write Python code that prints hello world"}, return_msg_type=MessageType.SUCCESS)
        self.assertEqual(msg.msg_type, MessageType.SUCCESS)
        self.assertEqual(msg.content.content, "```python\nprint('Hello, world!')```")
        self.assertEqual(msg.content.code, "print('Hello, world!')")

        # Create agent with parser
        agent_with_parser = CustomizeAgent(
            name="CodeGenerator",
            description="Generates Python code based on requirements",
            prompt="Write Python code that implements the following requirement: {requirement}",
            llm_config=llm_config,
            inputs=[
                {
                    "name": "requirement",
                    "type": "string",
                    "description": "The coding requirement",
                    "required": True
                }
            ],
            outputs=[
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
            output_parser=CodeWriterActionOutput,
            parse_mode="custom",
            parse_func=customize_parse_func
        )

        # Verify parser
        self.assertEqual(agent_with_parser.action.outputs_format.__name__, "CodeWriterActionOutput")
        
        # Save and load agent
        agent_with_parser.save_module(self.save_files[4])
        new_agent_with_parser = CustomizeAgent.from_file(self.save_files[4], llm_config=llm_config)
        self.assertEqual(new_agent_with_parser.action.outputs_format.__name__, "CodeWriterActionOutput")

        # Execute agent
        msg = new_agent_with_parser(inputs={"requirement": "Write Python code that prints hello world"}, return_msg_type=MessageType.SUCCESS)
        self.assertEqual(msg.msg_type, MessageType.SUCCESS)
        self.assertEqual(msg.content.content, "```python\nprint('Hello, world!')```")
        self.assertEqual(msg.content.code, "print('Hello, world!')")

    def tearDown(self):
        # Clean up test files
        for file in self.save_files:
            if os.path.exists(file):
                os.remove(file)

if __name__ == "__main__":
    unittest.main()
