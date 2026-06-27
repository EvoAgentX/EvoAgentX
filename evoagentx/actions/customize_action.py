import asyncio
import json
import re
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Union

from pydantic import Field, PositiveInt

from ..core.exception import NoAnswerError
from ..core.logging import logger
from ..core.message import Message
from ..core.module_utils import parse_json_from_llm_output, parse_json_from_text
from ..memory.context_manager import ContextManager
from ..models import BaseLLM, LLMOutputParser
from ..prompts.customize_agent import (
    ANSWER_HINT,
    ANSWER_PROMPT,
    LAST_ATTEMPT_PROMPT,
    NO_TOOL_CALL_PROMPT,
    RETRY_TOOL_PROMPT,
)
from ..prompts.output_extraction import OUTPUT_EXTRACTION_PROMPT
from ..prompts.tool_calling import (
    TOOL_CALLING_RETRY_PROMPT,
    TOOL_CALLING_TEMPLATE,
)
from ..prompts.utils import DEFAULT_SYSTEM_PROMPT
from ..tools.tool import Tool, Toolkit, ToolMetadata, ToolResult
from ..utils.utils import compile_tool_schemas, pydantic_to_parameters
from .action import Action


class CustomizeAction(Action):

    parse_mode: Optional[str] = Field(default="title", description="the parse mode of the action, must be one of: ['title', 'str', 'json', 'xml', 'custom']")
    parse_func: Optional[Callable] = Field(default=None, exclude=True, description="the function to parse the LLM output. It receives the LLM output and returns a dict.")
    title_format: Optional[str] = Field(default="## {title}", exclude=True, description="the format of the title. It is used when the `parse_mode` is 'title'.")
    custom_output_format: Optional[str] = Field(default=None, exclude=True, description="the format of the output. It is used when the `prompt_template` is provided.")
    tools: Optional[List[Union[Tool, Toolkit]]] = Field(default=None, description="The tools that the action can use")
    conversation: Optional[Message] = Field(default=None, description="Current conversation state")
    max_steps: PositiveInt = Field(default=20, description="The maximum number of LLM calls allowed")
    max_tool_call_concurrency: PositiveInt = Field(default=5, description="The maximum number of tool calls that can be executed concurrently")

    def __init__(self, **kwargs):

        name = kwargs.pop("name", "CustomizeAction")
        description = kwargs.pop("description", "Customized action that can use tools to accomplish its task")
        tools = kwargs.pop("tools", None)

        super().__init__(name=name, description=description, **kwargs)

        # Validate that at least one of prompt or prompt_template is provided
        if not self.prompt and not self.prompt_template:
            raise ValueError("`prompt` or `prompt_template` is required when creating CustomizeAction action")
        # Prioritize template and give warning if both are provided
        if self.prompt and self.prompt_template:
            logger.warning("Both `prompt` and `prompt_template` are provided for CustomizeAction action. Prioritizing `prompt_template` and ignoring `prompt`.")
        if tools and self.prompt_template is not None and getattr(self.prompt_template, "tools", None):
            logger.warning(
                "Both `CustomizeAction.tools` and `prompt_template.tools` are provided. "
                "`CustomizeAction.tools` will override `prompt_template.tools`. "
                "`PromptTemplate.tools` is legacy and will be removed in a future release; "
                "prefer passing tools to `CustomizeAction`/`CustomizeAgent`, or to "
                "`PromptTemplate.format(..., tools=...)` when rendering prompt-based tool instructions."
            )

        self.tools_caller = dict()
        self.tools = []
        if tools:
            self.add_tools(tools)
        self.tool_schemas: List[dict] = compile_tool_schemas(self.tools)

    def prepare_extraction_prompt(self, llm_output_content: str) -> str:
        """Prepare extraction prompt for fallback extraction when parsing fails.

        Args:
            self: The action instance
            llm_output_content: Raw output content from LLM

        Returns:
            str: Formatted extraction prompt
        """
        ignore = ["class_name"]

        if not self.outputs_format._is_content_defined_in_subclass():
            ignore.append("content")

        output_params = pydantic_to_parameters(self.outputs_format, ignore=ignore)
        output_params = [param.to_dict(ignore=["class_name"]) for param in output_params]
        output_params_json = json.dumps(output_params, indent=4, ensure_ascii=False)
        prompt = OUTPUT_EXTRACTION_PROMPT.format(text=llm_output_content, output_description=output_params_json)
        return prompt

    def add_tools(self, tools: List[Union[Tool, Toolkit]]):
        if not tools:
            return

        duplicate = False
        # avoid duplication & type checks
        for tool in tools:
            new_tools: List[Tool] = []

            if isinstance(tool, Toolkit):
                new_tools = tool.get_tools()
            elif isinstance(tool, Tool):
                new_tools = [tool]
            else:
                raise ValueError(f"Invalid tool type: {type(tool)}")

            for new_tool in new_tools:
                if not isinstance(new_tool, Tool):
                    raise ValueError(f"Invalid tool type: {type(new_tool)}")

                if not callable(new_tool):
                    raise ValueError(f"Invalid tool '{new_tool.name}' in '{tool.name}': not callable.")

                if new_tool.name in self.tools_caller:
                    logger.warning(f"Duplicate tool '{new_tool.name}' detected. Overwriting previous tool.")
                    duplicate = True

                # update tools caller
                self.tools_caller[new_tool.name] = new_tool

            logger.info(f"Added '{tool.name}' to '{self.name}'")

            if duplicate:
                self.tools = [t for t in self.tools if t.name != tool.name]
                duplicate = False

            self.tools.append(tool)
            # update tool schemas
            self.tool_schemas = compile_tool_schemas(self.tools)

    def _extract_tool_calls(self, llm_output: str, llm: Optional[BaseLLM] = None) -> List[dict]:
        pattern = r"<tool_call>\s*(.*?)\s*</tool_call>"

        # Find all tool call blocks in the output
        matches = re.findall(pattern, llm_output, re.DOTALL)
        if not matches:
            return []

        # NOTE: This is a temporary workaround to address an issue where models
        # sometimes include an extra <tool_call> block in the response,
        # in addition to the native tool calls, which results in duplicated tool calls.
        matches = [matches[-1]]

        def _parse_tool_calls(text: str) -> List[dict]:
            text = text.strip()
            json_list = parse_json_from_text(text)
            if not json_list:
                logger.warning("No valid JSON found in tool call block")
                return []
            # Only use the first JSON string from each block
            parsed_tool_call = json.loads(json_list[0])
            if isinstance(parsed_tool_call, dict):
                return [parsed_tool_call]
            elif isinstance(parsed_tool_call, list):
                return parsed_tool_call
            else:
                logger.warning(f"Invalid tool call format: {parsed_tool_call}")
                return []

        parsed_tool_calls = []
        for match_content in matches:
            try:
                parsed_tool_calls.extend(_parse_tool_calls(match_content))
            except (json.JSONDecodeError, IndexError) as e:
                logger.warning(f"Failed to parse tool calls from LLM output: {e}")
                if llm is not None:
                    retry_prompt = TOOL_CALLING_RETRY_PROMPT.format(text=match_content)
                    try:
                        logger.info("Fixing tool call with LLM...")
                        fixed_output = llm.generate(prompt=retry_prompt).content
                        logger.info(f"Retrying with fixed tool call:\n{fixed_output}")
                        parsed_tool_calls.extend(_parse_tool_calls(fixed_output))
                    except Exception as retry_err:
                        logger.error(f"Retry failed: {retry_err}")

        # Guarantee every tool call carries an `id`. Native tool calls come back with
        # provider-issued ids, but a model may also emit a hand-written <tool_call>
        # block with no id. In native mode the id links the assistant `tool_calls`
        # message to its `role: tool` result, and providers reject a null/mismatched
        # id, so synthesize one when absent.
        for tool_call in parsed_tool_calls:
            if isinstance(tool_call, dict) and not tool_call.get("id"):
                tool_call["id"] = f"call_{uuid.uuid4().hex}"

        return parsed_tool_calls

    @staticmethod
    def _extract_answer(llm_output: str) -> Union[str, None]:
        pattern = r"<answer>\s*(.*?)\s*</answer>"
        matches = re.findall(pattern, llm_output, re.DOTALL)
        if matches:
            final_answer = matches[0].strip()
            return final_answer
        return None

    def _extract_no_answer(self, llm_output: str) -> Union[str, None]:
        pattern = r"<no_answer>\s*(.*?)\s*</no_answer>"
        matches = re.findall(pattern, llm_output, re.DOTALL)
        if matches:
            final_answer = matches[0].strip()
            return final_answer
        return None

    async def _async_extract_output(self, llm_output: Union[str, LLMOutputParser], llm: BaseLLM = None, **kwargs) -> LLMOutputParser:
        # Get the raw output content
        llm_output_content = getattr(llm_output, "content", str(llm_output))

        # Check if there are any defined output fields
        output_attrs = self.outputs_format.get_attrs()

        # If no output fields are defined, create a simple content-only output
        if not output_attrs:
            # Create output with just the content field
            output = self.outputs_format.parse(content=llm_output_content)
            return output

        # Use the action's parse_mode and parse_func for parsing
        try:
            # Use the outputs_format's parse method with the action's parse settings
            parsed_output = self.outputs_format.parse(
                content=llm_output_content,
                parse_mode=self.parse_mode,
                parse_func=getattr(self, 'parse_func', None),
                title_format=getattr(self, 'title_format', "## {title}")
            )
            return parsed_output

        except Exception as e:
            logger.info(f"Failed to parse with action's parse settings: {e}")
            logger.info("Falling back to using LLM to extract outputs...")

            # Fall back to extraction prompt if direct parsing fails
            extraction_prompt = self.prepare_extraction_prompt(llm_output_content)

            llm_extracted_output = await llm.async_generate(prompt=extraction_prompt)
            llm_extracted_data: dict = parse_json_from_llm_output(llm_extracted_output.content)
            output = self.outputs_format(**llm_extracted_data)
            return output

    async def _call_single_tool(self, function_param: dict, semaphore: Optional[asyncio.Semaphore] = None) -> ToolResult:
        # When called outside of `_calling_tools` (e.g. directly in tests), create a
        # loop-bound semaphore on the fly so concurrency limiting still applies.
        if semaphore is None:
            semaphore = asyncio.Semaphore(self.max_tool_call_concurrency)
        tool_call_id = function_param.get("id")
        function_name = function_param.get("function_name") or ""
        function_args = function_param.get("function_args") or {}

        metadata = ToolMetadata(
            tool_name=function_name,
            args=function_args
        )

        if not function_name:
            output = {"error": "No tool name provided"}
            return ToolResult(result=output, metadata=metadata, id=tool_call_id)

        tool = self.tools_caller.get(function_name, None)
        if tool is None:
            output = {"error": f"Tool '{function_name}' not found"}
            return ToolResult(result=output, metadata=metadata, id=tool_call_id)

        if not callable(tool):
            output = {"error": f"Tool '{function_name}' is not callable"}
            return ToolResult(result=output, metadata=metadata, id=tool_call_id)

        try:
            async with semaphore:
                tool_args_str = json.dumps(function_args, indent=4, ensure_ascii=False)
                logger.info(f"[Tool Call] Executing tool `{function_name}` with parameters:\n{tool_args_str}")

                if asyncio.iscoroutinefunction(tool.__call__):
                    result = await tool(**function_args)
                else:
                    result = await asyncio.to_thread(tool, **function_args)

                # Adapter: tools may return a `ToolResult` directly, or a raw value.
                # Wrap raw values into a `ToolResult` so the agent loop is uniform.
                if isinstance(result, ToolResult):
                    result.id = tool_call_id
                    return result
                return ToolResult(result=result, metadata=metadata, id=tool_call_id)

        except Exception as e:
            logger.exception(f"Error calling tool '{function_name}': {e}")
            return ToolResult(result={"error": str(e)}, metadata=metadata, id=tool_call_id)

    async def _calling_tools(self, tool_call_args: List[dict]) -> List[ToolResult]:
        # Create the semaphore inside the running event loop. `asyncio.Semaphore`
        # binds to the loop on first await, so a long-lived instance attribute would
        # be reused across the fresh loops that `execute()` spins up via
        # `asyncio.run()` / the thread-pool loop, raising "Semaphore is bound to a
        # different event loop". A per-call semaphore is loop-safe and still bounds
        # concurrency within a single tool-calling round.
        semaphore = asyncio.Semaphore(self.max_tool_call_concurrency)
        tasks = [
            self._call_single_tool(args, semaphore)
            for args in tool_call_args
        ]

        results = await asyncio.gather(*tasks)
        return results

    def execute(
        self,
        llm: Optional[BaseLLM] = None,
        inputs: Optional[dict] = None,
        sys_msg: Optional[str] = None,
        return_prompt: bool = False,
        **kwargs
    ):

        coro = self.async_execute(
            llm=llm,
            inputs=inputs,
            sys_msg=sys_msg,
            return_prompt=return_prompt,
            **kwargs
        )

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            # No event loop is running in this thread: drive the coroutine directly.
            return asyncio.run(coro)

        # We are already inside a running event loop (e.g. `execute` was called from
        # async code). `asyncio.run()` would raise `RuntimeError` here, so run the
        # coroutine to completion on a dedicated thread that owns its own event loop.
        with ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()

    async def async_execute(
        self,
        llm: Optional[BaseLLM] = None,
        inputs: Optional[dict] = None,
        sys_msg: Optional[str] = None,
        return_prompt: bool = False,
        context_manager: Optional[ContextManager] = None,
        **kwargs
    ):
        if llm is None:
            raise ValueError(f"LLM is required for CustomizeAction '{self.name}'.")

        inputs = inputs or {}
        self.inputs_format(**inputs)

        context_manager = await self.prepare_context(llm, inputs, sys_msg, context_manager)

        final_answer = None
        tool_calls = 0
        no_tool_call_answer = 0
        failed_tool_calls = 0
        iter = 0

        while True:
            if iter >= self.max_steps:
                logger.error(f"{self.name} exceeded maximum number of steps ({self.max_steps}).")
                logger.info(f"[Final Output] `{self.name}` failed to produce the requested output within the maximum number of allowed attempts.")
                raise NoAnswerError("Failed to produce the requested output within the maximum number of allowed attempts.")

            if iter == self.max_steps - 1:
                context_manager.add_user_prompt(LAST_ATTEMPT_PROMPT)

            # In native mode the tools schema is passed to the model directly; in
            # default mode tools are described in the prompt and we parse a textual
            # <tool_call> block instead.
            #
            # `enable_prompt_caching=True` opts this agent loop into provider prompt
            # caching: the loop re-sends a growing-but-shared prefix every iteration,
            # so cache reads from iteration 2 onward outweigh the first-call write
            # premium. The LLM layer decides what (if anything) to do per provider —
            # the action stays provider-agnostic. Non-OpenRouter providers ignore
            # the flag (it is filtered out before the request).
            llm_response = await llm.async_generate(
                messages=context_manager.context,
                tools=self.tool_schemas if context_manager.mode == "native" else None,
                enable_prompt_caching=True,
            )

            logger.info(f"[Raw LLM Response]: {llm_response.content}")
            iter += 1

            if no_tool_call_answer == 1 and "yes" in llm_response.content.lower():
                break

            tool_call_args = self._extract_tool_calls(llm_response.content, llm=llm)

            if not tool_call_args:
                context_manager.add_llm_response(llm_response.content)
                final_answer = CustomizeAction._extract_answer(llm_response.content)
                if final_answer is not None:
                    if self.tools and tool_calls == 0 and no_tool_call_answer == 0:
                        # if tools are provided but no tool call has been made,
                        # ask to confirm no tool is needed for final answer (only ask once)
                        context_manager.add_user_prompt(NO_TOOL_CALL_PROMPT)
                        no_tool_call_answer += 1
                        continue
                    break

                no_answer = self._extract_no_answer(llm_response.content)
                if no_answer is not None:
                    logger.error(f"{self.name} was unable to produce requested output: {no_answer}")
                    raise NoAnswerError(no_answer)

                context_manager.add_user_prompt(ANSWER_HINT)
                continue

            non_tool_call_response = llm_response.content.split("<tool_call>", 1)[0].strip() or None
            context_manager.add_llm_response(non_tool_call_response, tool_calls=tool_call_args)

            tool_results = await self._calling_tools(tool_call_args)
            tool_calls += 1

            context_manager.add_tool_results(tool_results)
            for result in tool_results:
                result_str = json.dumps(result.result, indent=4, ensure_ascii=False)
                logger.info(f"[Tool Call] Executed tool `{result.metadata.tool_name}` results:\n{result_str}")

            if failed_tool_calls == 0:
                # if this is the first time any tool has failed, ask agent to retry.
                for result in tool_results:
                    if isinstance(result.result, dict) and "error" in result.result:
                        failed_tool_calls += 1
                        context_manager.add_user_prompt(RETRY_TOOL_PROMPT)
                        break

        final_output = await self._async_extract_output(final_answer, llm=llm)
        logger.info(f"[Final Output] `{self.name}` final output:\n{final_output.to_str()}")

        if return_prompt:
            system_prompt = context_manager.get_system_prompt()

            user_prompt = ""
            for msg in context_manager.context:
                if msg["role"] == "user":
                    user_prompt = msg["content"]
                    break

            return final_output, f"<system_prompt>\n{system_prompt}\n</system_prompt>\n\n----\n\n<user_prompt>\n{user_prompt}\n</user_prompt>"
        return final_output

    async def prepare_context(
        self,
        llm: BaseLLM,
        inputs: Optional[dict] = None,
        sys_msg: Optional[str] = None,
        context_manager: Optional[ContextManager] = None,
    ) -> ContextManager:

        inputs = inputs or {}

        if context_manager is None:
            context_manager = ContextManager(llm=llm)
        elif len(context_manager.context) > 0:
            return context_manager

        if self.prompt_template is not None:
            context_manager.add_prompt_template(
                self.prompt_template,
                sys_msg=sys_msg,
                values=inputs,
                inputs_format=self.inputs_format,
                outputs_format=self.outputs_format,
                parse_mode=self.parse_mode,
                title_format=self.title_format,
                custom_output_format=self.custom_output_format,
                tools=self.tools
            )
        elif self.prompt is not None:
            sys_msg = sys_msg or DEFAULT_SYSTEM_PROMPT
            context_manager.add_system_prompt(sys_msg)
            prompt_inputs = {
                key: json.dumps(value, indent=2, ensure_ascii=False) if isinstance(value, (dict, list)) else value
                for key, value in inputs.items()
            }
            user_prompt = self.prompt.format(**prompt_inputs)
            # Only append the textual tool-calling guide when tools are actually
            # available AND we are not using native tool calling. Without tools,
            # the guide's web_search/code_execution examples can induce the model
            # to emit <tool_call> blocks for non-existent tools, causing loops or
            # failures. In native mode, tools are passed to the model directly, so
            # the textual guide would only duplicate the prompt.
            if self.tools and context_manager.mode != "native":
                user_prompt += "\n\n" + TOOL_CALLING_TEMPLATE.format(
                    tool_descriptions=json.dumps(self.tool_schemas, indent=4, ensure_ascii=False)
                )
            context_manager.add_user_prompt(user_prompt)

        context_manager.add_system_prompt(ANSWER_PROMPT)
        return context_manager
