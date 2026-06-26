"""Pure-logic tests for CustomizeAgent: construction, validation, property
setters, and config serialization. None of these require a real LLM call."""

import os
import unittest
from typing import Any, Dict, List, Optional

from pydantic import Field

from evoagentx.actions.action import ActionOutput
from evoagentx.agents.customize_agent import CustomizeAgent
from evoagentx.core.base_config import Parameter
from evoagentx.core.registry import register_parse_function
from evoagentx.models.model_configs import LiteLLMConfig
from evoagentx.prompts.template import ChatTemplate
from evoagentx.tools.tool import Tool


class CfgAddNumbersTool(Tool):
    name: str = "add_numbers"
    description: str = "Add two integers and return their sum."
    inputs: Dict[str, Dict[str, Any]] = {
        "a": {"type": "integer", "description": "First integer."},
        "b": {"type": "integer", "description": "Second integer."},
    }
    required: Optional[List[str]] = ["a", "b"]

    def __call__(self, a: int, b: int) -> Dict[str, int]:
        return {"sum": a + b}


@register_parse_function
def _cfg_parse_func(content: str) -> dict:
    return {"code": content}


class CodeOutput(ActionOutput):
    code: str = Field(description="The generated code")


def make_config() -> LiteLLMConfig:
    return LiteLLMConfig(model="gpt-4o-mini", openai_key="xxxxx")


class TestConfigSerialization(unittest.TestCase):
    """A1: get_config / get_customize_agent_info / round-trip parity."""

    def test_get_config_contains_expected_keys(self):
        agent = CustomizeAgent(
            name="CfgAgent",
            description="agent for config test",
            prompt="Do {task}",
            llm_config=make_config(),
            inputs=[{"name": "task", "type": "string", "description": "the task"}],
            outputs=[{"name": "result", "type": "string", "description": "the result"}],
            parse_mode="title",
            max_steps=7,
            max_tool_call_concurrency=3,
            custom_output_format=None,
        )

        info = agent.get_customize_agent_info()
        for key in [
            "class_name", "name", "description", "prompt", "prompt_template",
            "inputs", "outputs", "system_prompt", "output_parser", "parse_mode",
            "parse_func", "title_format", "tool_names", "custom_output_format",
            "max_steps", "max_tool_call_concurrency",
        ]:
            self.assertIn(key, info)

        self.assertEqual(info["class_name"], "CustomizeAgent")
        self.assertEqual(info["name"], "CfgAgent")
        self.assertEqual(info["prompt"], "Do {task}")
        self.assertEqual(info["max_steps"], 7)
        self.assertEqual(info["max_tool_call_concurrency"], 3)
        self.assertEqual(len(info["inputs"]), 1)
        self.assertEqual(len(info["outputs"]), 1)
        self.assertEqual(info["tool_names"], [])

        # get_config adds llm_config on top of the info dict
        config = agent.get_config()
        self.assertIn("llm_config", config)

    def test_get_config_round_trip_parity(self):
        agent = CustomizeAgent(
            name="RoundTrip",
            description="round trip parity",
            prompt="Implement {requirement}",
            llm_config=make_config(),
            inputs=[{"name": "requirement", "type": "string", "description": "req"}],
            outputs=[
                {"name": "code", "type": "string", "description": "the code"},
            ],
            output_parser=CodeOutput,
            parse_mode="custom",
            parse_func=_cfg_parse_func,
            max_steps=5,
        )

        config = agent.get_config()
        rebuilt = CustomizeAgent.from_dict(config, llm_config=make_config())

        self.assertEqual(rebuilt.get_customize_agent_info(), agent.get_customize_agent_info())

    def test_save_and_load_parity(self):
        path = "tests/agents/_tmp_cfg_agent.json"
        try:
            agent = CustomizeAgent(
                name="SaveLoad",
                description="save load",
                prompt="Echo {value}",
                llm_config=make_config(),
                inputs=[{"name": "value", "type": "string", "description": "v"}],
            )
            agent.save_module(path)
            loaded = CustomizeAgent.from_file(path, llm_config=make_config())
            self.assertEqual(loaded.get_customize_agent_info(), agent.get_customize_agent_info())
        finally:
            if os.path.exists(path):
                os.remove(path)

    def test_from_dict_rejects_class_name_mismatch(self):
        config = CustomizeAgent(
            name="X", description="d", prompt="p", llm_config=make_config(),
        ).get_config()
        config["class_name"] = "SomethingElse"
        with self.assertRaises(ValueError):
            CustomizeAgent.from_dict(config, llm_config=make_config())

    def test_from_dict_auto_corrects_parse_mode_for_object_output(self):
        # object/array outputs must be parsed as json; from_dict should flip parse_mode
        config = {
            "class_name": "CustomizeAgent",
            "name": "ObjAgent",
            "description": "d",
            "prompt": "p",
            "inputs": [],
            "outputs": [{
                "name": "data", "type": "object", "description": "obj",
                "json_schema": {"type": "object", "properties": {"k": {"type": "string"}}},
            }],
            "parse_mode": "title",
        }
        agent = CustomizeAgent.from_dict(config, llm_config=make_config())
        self.assertEqual(agent.parse_mode, "json")


class TestToolSerialization(unittest.TestCase):
    """A1 (tools): tool save/load round-trip via tool_names rehydration."""

    def _tool_agent(self):
        return CustomizeAgent(
            name="ToolCfgAgent",
            description="agent with a tool",
            prompt="Add the two numbers using the tool.",
            llm_config=make_config(),
            tools=[CfgAddNumbersTool()],
            max_steps=5,
        )

    def test_get_config_serializes_tool_names(self):
        agent = self._tool_agent()
        info = agent.get_customize_agent_info()
        self.assertEqual(info["tool_names"], ["add_numbers"])

    def test_round_trip_with_tools(self):
        agent = self._tool_agent()
        config = agent.get_config()

        # tools must be supplied so the names can be rehydrated to instances
        rebuilt = CustomizeAgent.from_dict(
            config, llm_config=make_config(), tools=[CfgAddNumbersTool()]
        )
        self.assertEqual([t.name for t in rebuilt.tools], ["add_numbers"])
        self.assertEqual(rebuilt.get_customize_agent_info(), agent.get_customize_agent_info())

    def test_save_and_load_with_tools(self):
        path = "tests/agents/_tmp_tool_agent.json"
        try:
            agent = self._tool_agent()
            agent.save_module(path)
            loaded = CustomizeAgent.from_file(
                path, llm_config=make_config(), tools=[CfgAddNumbersTool()]
            )
            self.assertEqual([t.name for t in loaded.tools], ["add_numbers"])
            self.assertEqual(loaded.get_customize_agent_info(), agent.get_customize_agent_info())
        finally:
            if os.path.exists(path):
                os.remove(path)

    def test_from_dict_with_tool_names_but_no_tools_raises(self):
        config = self._tool_agent().get_config()
        # tool_names present in config but no tools provided to resolve them
        with self.assertRaises(ValueError):
            CustomizeAgent.from_dict(config, llm_config=make_config())


class TestValidation(unittest.TestCase):
    """A2: validate_data / construction error paths."""

    def test_missing_prompt_and_template_raises(self):
        with self.assertRaises(ValueError):
            CustomizeAgent(name="N", description="d", llm_config=make_config())

    def test_input_not_in_prompt_raises(self):
        with self.assertRaises(KeyError):
            CustomizeAgent(
                name="N", description="d",
                prompt="No placeholder here",
                llm_config=make_config(),
                inputs=[{"name": "missing", "type": "string", "description": "x"}],
            )

    def test_prompt_and_template_together_prefers_template(self):
        agent = CustomizeAgent(
            name="N", description="d",
            prompt="ignored {x}",
            prompt_template=ChatTemplate(instruction="Do the task"),
            llm_config=make_config(),
        )
        # prompt is nulled, prompt_template wins
        self.assertIsNone(agent.prompt)
        self.assertIsNotNone(agent.prompt_template)

    def test_invalid_parse_mode_raises(self):
        with self.assertRaises(ValueError):
            CustomizeAgent(
                name="N", description="d", prompt="p",
                llm_config=make_config(), parse_mode="not_a_mode",
            )

    def test_custom_parse_mode_without_func_raises(self):
        with self.assertRaises(ValueError):
            CustomizeAgent(
                name="N", description="d", prompt="p",
                llm_config=make_config(), parse_mode="custom",
            )

    def test_object_output_auto_corrects_to_json(self):
        # Auto-correction only applies to prompt_template agents: the template injects the
        # JSON-schema output instruction, so json parsing is what the model is told to produce.
        agent = CustomizeAgent(
            name="N", description="d",
            prompt_template=ChatTemplate(instruction="Do the task"),
            llm_config=make_config(),
            outputs=[{
                "name": "data", "type": "object", "description": "obj",
                "json_schema": {"type": "object", "properties": {"k": {"type": "string"}}},
            }],
            parse_mode="title",
        )
        self.assertEqual(agent.parse_mode, "json")

    def test_object_output_without_schema_builds_minimal_schema(self):
        agent = CustomizeAgent(
            name="N", description="d",
            prompt_template=ChatTemplate(instruction="Do the task"),
            llm_config=make_config(),
            outputs=[{"name": "data", "type": "object", "description": "obj"}],
            parse_mode="title",
        )

        schema = agent.action.outputs_format.model_config.get("json_schema_extra")
        self.assertEqual(agent.parse_mode, "json")
        self.assertEqual(schema["properties"]["data"], {"type": "object", "description": "obj"})

    def test_raw_prompt_object_output_keeps_parse_mode(self):
        # A raw `prompt` is sent verbatim, so the model follows the format the prompt requests.
        # parse_mode must NOT be force-corrected to json (it would disagree with the prompt).
        agent = CustomizeAgent(
            name="N", description="d", prompt="p",
            llm_config=make_config(),
            outputs=[{"name": "data", "type": "object", "description": "obj"}],
            parse_mode="title",
        )
        self.assertEqual(agent.parse_mode, "title")

    def test_invalid_input_item_type_raises(self):
        with self.assertRaises(ValueError):
            CustomizeAgent(
                name="N", description="d", prompt="p {x}",
                llm_config=make_config(),
                inputs=["not a dict or Parameter"],
            )


class TestPropertySetters(unittest.TestCase):
    """A3: property setters propagate to the underlying action and validate."""

    def _agent(self, **overrides):
        kwargs = dict(
            name="Setter", description="d", prompt="Do {task}",
            llm_config=make_config(),
            inputs=[{"name": "task", "type": "string", "description": "t"}],
            outputs=[{"name": "result", "type": "string", "description": "r"}],
        )
        kwargs.update(overrides)
        return CustomizeAgent(**kwargs)

    def test_parse_mode_setter_rejects_non_json_for_object_outputs(self):
        # Setter enforcement only applies to prompt_template agents (see validate_data).
        agent = self._agent(
            prompt=None,
            prompt_template=ChatTemplate(instruction="Do the task"),
            outputs=[{
                "name": "data", "type": "object", "description": "obj",
                "json_schema": {"type": "object", "properties": {"k": {"type": "string"}}},
            }],
            parse_mode="json",
        )
        with self.assertRaises(ValueError):
            agent.parse_mode = "title"

    def test_parse_mode_setter_allows_non_json_for_raw_prompt_object_outputs(self):
        agent = self._agent(
            outputs=[{"name": "data", "type": "object", "description": "obj"}],
            parse_mode="title",
        )
        agent.parse_mode = "str"
        self.assertEqual(agent.parse_mode, "str")

    def test_parse_func_none_while_custom_raises(self):
        agent = self._agent(parse_mode="custom", parse_func=_cfg_parse_func)
        with self.assertRaises(ValueError):
            agent.parse_func = None

    def test_title_format_requires_placeholder(self):
        agent = self._agent()
        with self.assertRaises(ValueError):
            agent.title_format = "no placeholder"

    def test_title_format_setter_propagates_to_action(self):
        agent = self._agent()
        agent.title_format = "### {title}"
        self.assertEqual(agent.action.title_format, "### {title}")

    def test_inputs_setter_rebuilds_action_format(self):
        # prompt must contain every input placeholder, so provide both up front
        agent = self._agent(
            prompt="Process {task} and optionally {extra}",
            inputs=[{"name": "task", "type": "string", "description": "t"}],
        )
        self.assertEqual(len(agent.action.inputs_format.get_attrs()), 1)
        agent.inputs = [
            {"name": "task", "type": "string", "description": "t"},
            {"name": "extra", "type": "string", "description": "e"},
        ]
        self.assertEqual(len(agent.action.inputs_format.get_attrs()), 2)

    def test_outputs_setter_rebuilds_action_format(self):
        agent = self._agent()
        agent.outputs = [
            {"name": "result", "type": "string", "description": "r"},
            {"name": "explanation", "type": "string", "description": "e"},
        ]
        self.assertEqual(len(agent.action.outputs_format.get_attrs()), 2)

    def test_output_parser_must_subclass_action_output(self):
        class NotAnOutput:
            pass
        with self.assertRaises(ValueError):
            self._agent(output_parser=NotAnOutput)

    def test_output_parser_fields_must_be_subset_of_outputs(self):
        class ExtraFieldOutput(ActionOutput):
            result: str = Field(description="r")
            unknown: str = Field(description="not in outputs")
        with self.assertRaises(ValueError):
            self._agent(output_parser=ExtraFieldOutput)

    def test_parameter_objects_accepted_as_inputs(self):
        agent = self._agent(
            inputs=[Parameter(name="task", type="string", description="t")],
        )
        self.assertEqual(len(agent.inputs), 1)
        self.assertIsInstance(agent.inputs[0], Parameter)


if __name__ == "__main__":
    unittest.main()
