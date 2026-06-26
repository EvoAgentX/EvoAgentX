"""Unit tests for CustomizeAction internals: prompt/context preparation,
tool-call extraction, tool execution, the agent loop, and the sync `execute`
bridge. The agent loop tests mock `single_generate_async` so no real LLM is
called."""

import asyncio
import unittest
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, patch

from evoagentx.actions.customize_action import CustomizeAction
from evoagentx.agents.customize_agent import CustomizeAgent
from evoagentx.core.exception import NoAnswerError
from evoagentx.memory.context_manager import ContextManager
from evoagentx.models.litellm_model import LiteLLM
from evoagentx.models.model_configs import LiteLLMConfig
from evoagentx.prompts.template import ChatTemplate, StringTemplate
from evoagentx.tools.tool import Tool, ToolResult


def make_config() -> LiteLLMConfig:
    return LiteLLMConfig(model="gpt-4o-mini", openai_key="xxxxx")


def make_llm() -> LiteLLM:
    return LiteLLM(config=make_config())


class AddNumbersTool(Tool):
    name: str = "add_numbers"
    description: str = "Add two integers and return their sum."
    inputs: Dict[str, Dict[str, Any]] = {
        "a": {"type": "integer", "description": "First integer."},
        "b": {"type": "integer", "description": "Second integer."},
    }
    required: Optional[List[str]] = ["a", "b"]

    def __call__(self, a: int, b: int) -> Dict[str, int]:
        return {"sum": a + b}


class PromptOnlyTool(Tool):
    name: str = "prompt_only"
    description: str = "A legacy prompt-template tool that should be overridden."
    inputs: Dict[str, Dict[str, Any]] = {
        "text": {"type": "string", "description": "Text to echo."},
    }
    required: Optional[List[str]] = ["text"]

    def __call__(self, text: str) -> Dict[str, str]:
        return {"text": text}


def _user_messages(context: List[dict]) -> str:
    return "\n".join(m["content"] for m in context if m["role"] == "user" and isinstance(m["content"], str))


def _all_message_text(context: List[dict]) -> str:
    return "\n".join(m["content"] for m in context if isinstance(m.get("content"), str))


GUIDE_MARKER = "Tool Calling Guide"


class TestPrepareContext(unittest.IsolatedAsyncioTestCase):
    """A4: prepare_context tool-guide gating (validates the prepare_context fix)."""

    def _action(self, tools=None):
        agent = CustomizeAgent(
            name="PCAgent", description="d",
            prompt="Answer the question.",
            llm_config=make_config(),
            tools=tools,
        )
        return agent.action

    async def test_prompt_no_tools_omits_guide(self):
        action = self._action(tools=None)
        cm = ContextManager(llm=make_llm())  # default mode
        await action.prepare_context(llm=make_llm(), inputs={}, context_manager=cm)
        self.assertNotIn(GUIDE_MARKER, _user_messages(cm.context))

    async def test_prompt_with_tools_default_mode_includes_guide(self):
        action = self._action(tools=[AddNumbersTool()])
        cm = ContextManager(llm=make_llm())
        cm.mode = "default"  # simulate an LLM without native tool calling
        await action.prepare_context(llm=make_llm(), inputs={}, context_manager=cm)
        self.assertIn(GUIDE_MARKER, _user_messages(cm.context))

    async def test_prompt_with_tools_native_mode_omits_guide(self):
        action = self._action(tools=[AddNumbersTool()])
        cm = ContextManager(llm=make_llm())
        cm.mode = "native"  # native tool-calling path
        await action.prepare_context(llm=make_llm(), inputs={}, context_manager=cm)
        self.assertNotIn(GUIDE_MARKER, _user_messages(cm.context))

    async def test_string_template_with_tools_native_mode_omits_guide(self):
        agent = CustomizeAgent(
            name="StringTemplateTools", description="d",
            prompt_template=StringTemplate(instruction="Add two numbers using tools."),
            llm_config=make_config(),
            tools=[AddNumbersTool()],
        )
        cm = ContextManager(llm=make_llm())
        cm.mode = "native"
        await agent.action.prepare_context(llm=make_llm(), inputs={}, context_manager=cm)
        self.assertNotIn(GUIDE_MARKER, _all_message_text(cm.context))

    async def test_chat_template_with_tools_native_mode_omits_guide(self):
        agent = CustomizeAgent(
            name="ChatTemplateTools", description="d",
            prompt_template=ChatTemplate(instruction="Add two numbers using tools."),
            llm_config=make_config(),
            tools=[AddNumbersTool()],
        )
        cm = ContextManager(llm=make_llm())
        cm.mode = "native"
        await agent.action.prepare_context(llm=make_llm(), inputs={}, context_manager=cm)
        self.assertNotIn(GUIDE_MARKER, _all_message_text(cm.context))

    async def test_string_template_with_tools_default_mode_includes_guide(self):
        agent = CustomizeAgent(
            name="StringTemplateToolsDefault", description="d",
            prompt_template=StringTemplate(instruction="Add two numbers using tools."),
            llm_config=make_config(),
            tools=[AddNumbersTool()],
        )
        cm = ContextManager(llm=make_llm())
        cm.mode = "default"
        await agent.action.prepare_context(llm=make_llm(), inputs={}, context_manager=cm)
        self.assertIn(GUIDE_MARKER, _all_message_text(cm.context))

    async def test_action_tools_override_prompt_template_tools(self):
        template = StringTemplate(
            instruction="Use the available tool.",
            tools=[PromptOnlyTool()],
        )
        with patch("evoagentx.actions.customize_action.logger.warning") as mock_warning:
            agent = CustomizeAgent(
                name="OverrideTemplateTools", description="d",
                prompt_template=template,
                llm_config=make_config(),
                tools=[AddNumbersTool()],
            )

        warning_text = "\n".join(str(call.args[0]) for call in mock_warning.call_args_list)
        self.assertIn("`CustomizeAction.tools` will override `prompt_template.tools`", warning_text)

        cm = ContextManager(llm=make_llm())
        cm.mode = "default"
        await agent.action.prepare_context(llm=make_llm(), inputs={}, context_manager=cm)
        prompt_text = _all_message_text(cm.context)
        self.assertIn("add_numbers", prompt_text)
        self.assertNotIn("prompt_only", prompt_text)


class TestExtractHelpers(unittest.TestCase):
    """A4: tool-call / answer extraction helpers."""

    def _action(self):
        return CustomizeAgent(
            name="ExAgent", description="d", prompt="p", llm_config=make_config(),
        ).action

    def test_extract_single_tool_call(self):
        action = self._action()
        out = '<tool_call>\n[{"function_name": "add_numbers", "function_args": {"a": 1, "b": 2}}]\n</tool_call>'
        calls = action._extract_tool_calls(out)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["function_name"], "add_numbers")

    def test_extract_keeps_last_block_only(self):
        action = self._action()
        out = (
            '<tool_call>\n[{"function_name": "a", "function_args": {}}]\n</tool_call>'
            '<tool_call>\n[{"function_name": "b", "function_args": {}}]\n</tool_call>'
        )
        calls = action._extract_tool_calls(out)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["function_name"], "b")

    def test_extract_no_tool_call(self):
        action = self._action()
        self.assertEqual(action._extract_tool_calls("just text"), [])

    def test_extract_answer(self):
        self.assertEqual(CustomizeAction._extract_answer("<answer>hello</answer>"), "hello")
        self.assertIsNone(CustomizeAction._extract_answer("no answer tag"))

    def test_extract_no_answer(self):
        action = self._action()
        self.assertEqual(action._extract_no_answer("<no_answer>nope</no_answer>"), "nope")
        self.assertIsNone(action._extract_no_answer("no tag"))


class TestAddTools(unittest.TestCase):
    """A4: add_tools registration / dedup."""

    def _action(self):
        return CustomizeAgent(
            name="ToolReg", description="d", prompt="p", llm_config=make_config(),
        ).action

    def test_add_tool_registers_caller_and_schema(self):
        action = self._action()
        action.add_tools([AddNumbersTool()])
        self.assertIn("add_numbers", action.tools_caller)
        self.assertEqual(len(action.tool_schemas), 1)

    def test_duplicate_tool_overwrites(self):
        action = self._action()
        action.add_tools([AddNumbersTool()])
        action.add_tools([AddNumbersTool()])
        self.assertEqual(len(action.tools_caller), 1)

    def test_invalid_tool_type_raises(self):
        action = self._action()
        with self.assertRaises(ValueError):
            action.add_tools(["not a tool"])


class TestToolExecution(unittest.IsolatedAsyncioTestCase):
    """A5: _call_single_tool / _calling_tools."""

    def _action(self, tools=None):
        return CustomizeAgent(
            name="ExecAgent", description="d", prompt="p",
            llm_config=make_config(), tools=tools,
        ).action

    async def test_unknown_tool_returns_error(self):
        action = self._action(tools=[AddNumbersTool()])
        result = await action._call_single_tool({"function_name": "nope", "function_args": {}})
        self.assertIn("error", result.result)

    async def test_missing_tool_name_returns_error(self):
        action = self._action(tools=[AddNumbersTool()])
        result = await action._call_single_tool({"function_args": {}})
        self.assertIn("error", result.result)

    async def test_successful_tool_wrapped_in_tool_result(self):
        action = self._action(tools=[AddNumbersTool()])
        result = await action._call_single_tool(
            {"function_name": "add_numbers", "function_args": {"a": 2, "b": 3}}
        )
        self.assertIsInstance(result, ToolResult)
        self.assertEqual(result.result, {"sum": 5})

    async def test_calling_tools_preserves_order(self):
        action = self._action(tools=[AddNumbersTool()])
        args = [
            {"function_name": "add_numbers", "function_args": {"a": 1, "b": 1}},
            {"function_name": "add_numbers", "function_args": {"a": 10, "b": 10}},
        ]
        results = await action._calling_tools(args)
        self.assertEqual([r.result["sum"] for r in results], [2, 20])


class TestAgentLoop(unittest.TestCase):
    """A5: full loop driven by a mocked single_generate_async."""

    @patch("evoagentx.models.litellm_model.LiteLLM.single_generate_async", new_callable=AsyncMock)
    def test_tool_call_then_answer(self, mock_gen):
        mock_gen.side_effect = [
            '<tool_call>\n[{"function_name": "add_numbers", "function_args": {"a": 2, "b": 3}}]\n</tool_call>',
            "<answer>5</answer>",
        ]
        agent = CustomizeAgent(
            name="LoopAgent", description="d",
            prompt="Add 2 and 3 using the tool.",
            llm_config=make_config(),
            tools=[AddNumbersTool()],
            max_steps=5,
        )
        msg = agent()
        self.assertEqual(msg.content.content, "5")
        self.assertEqual(mock_gen.call_count, 2)

    @patch("evoagentx.models.litellm_model.LiteLLM.single_generate_async", new_callable=AsyncMock)
    def test_no_answer_raises(self, mock_gen):
        mock_gen.return_value = "<no_answer>cannot be done</no_answer>"
        agent = CustomizeAgent(
            name="NoAnsAgent", description="d", prompt="Do it.",
            llm_config=make_config(), max_steps=3,
        )
        with self.assertRaises(NoAnswerError):
            agent()

    @patch("evoagentx.models.litellm_model.LiteLLM.single_generate_async", new_callable=AsyncMock)
    def test_max_steps_exhausted_raises(self, mock_gen):
        mock_gen.return_value = "still thinking, no final answer yet"
        agent = CustomizeAgent(
            name="LoopForever", description="d", prompt="Do it.",
            llm_config=make_config(), max_steps=3,
        )
        with self.assertRaises(NoAnswerError):
            agent()


class TestExecuteSyncBridge(unittest.TestCase):
    """A6: execute() works both in a plain sync context and inside a running loop."""

    @patch("evoagentx.models.litellm_model.LiteLLM.single_generate_async", new_callable=AsyncMock)
    def test_execute_in_sync_context(self, mock_gen):
        mock_gen.return_value = "<answer>sync-ok</answer>"
        agent = CustomizeAgent(
            name="SyncBridge", description="d", prompt="p", llm_config=make_config(),
        )
        out = agent.action.execute(llm=agent.llm)
        self.assertEqual(out.content, "sync-ok")

    @patch("evoagentx.models.litellm_model.LiteLLM.single_generate_async", new_callable=AsyncMock)
    def test_execute_inside_running_loop(self, mock_gen):
        mock_gen.return_value = "<answer>loop-ok</answer>"
        agent = CustomizeAgent(
            name="LoopBridge", description="d", prompt="p", llm_config=make_config(),
        )

        async def driver():
            # sync execute() invoked while an event loop is already running
            return agent.action.execute(llm=agent.llm)

        out = asyncio.run(driver())
        self.assertEqual(out.content, "loop-ok")


if __name__ == "__main__":
    unittest.main()
