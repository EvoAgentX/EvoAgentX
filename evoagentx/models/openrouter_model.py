import json
from typing import Dict, List, Optional, Union

from openai import AsyncOpenAI, OpenAI, Stream
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

from ..core.logging import logger
from ..core.registry import register_model
from ..prompts.tool_calling import TOOL_CALL_FORMAT
from ..utils.utils import format_tool_calls
from .base_model import BaseLLM
from .model_configs import OpenRouterConfig
from .model_utils import Cost, cost_manager


@register_model(config_cls=OpenRouterConfig, alias=["openrouter"])
class OpenRouterLLM(BaseLLM):

    def init_model(self):
        self._client = None
        self._async_client = None
        self._default_ignore_fields = ["llm_type", "openrouter_key", "openrouter_base", "openrouter_model_base", "output_response"]
    
    def supports_native_tool_calling(self) -> bool:
        # OpenRouter proxies the OpenAI tool-calling protocol for the models that
        # support it; native tool calling was the original behavior here.
        return True

    def _init_client(self, config: OpenRouterConfig):
        return OpenAI(api_key=config.openrouter_key, base_url=config.openrouter_base)

    def _init_async_client(self, config: OpenRouterConfig):
        return AsyncOpenAI(api_key=config.openrouter_key, base_url=config.openrouter_base)

    def ensure_client(self):
        if self._client is None or self._client.is_closed():
            self._client = self._init_client(self.config)
        return self._client

    def close_client(self):
        if self._client is not None and not self._client.is_closed():
            self._client.close()

    def ensure_async_client(self):
        if self._async_client is None or self._async_client.is_closed():
            self._async_client = self._init_async_client(self.config)
        return self._async_client

    async def close_async_client(self):
        if self._async_client is not None and not self._async_client.is_closed():
            await self._async_client.close()

    def formulate_messages(self, prompts: List[str], system_messages: Optional[List[str]] = None) -> List[List[dict]]:
        if system_messages:
            assert len(prompts) == len(system_messages), f"the number of prompts ({len(prompts)}) is different from the number of system_messages ({len(system_messages)})"
        else:
            system_messages = [None] * len(prompts)
        
        messages_list = [] 
        for prompt, system_message in zip(prompts, system_messages):
            messages = [] 
            if system_message:
                messages.append({"role": "system", "content": system_message})
            messages.append({"role": "user", "content": prompt})
            messages_list.append(messages)

        return messages_list

    def update_completion_params(self, params1: dict, params2: dict) -> dict:
        config_params: list = self.config.get_config_params()
        for key, value in params2.items():
            if key in self._default_ignore_fields:
                continue
            if key not in config_params:
                continue
            params1[key] = value
        return params1

    def get_completion_params(self, **kwargs):
        completion_params = self.config.get_set_params(ignore=self._default_ignore_fields)
        completion_params = self.update_completion_params(completion_params, kwargs)
        return completion_params
    
    def get_stream_output(self, response: Stream, output_response: bool=True) -> str:
        output = ""
        tool_calls_accum: Dict[int, dict] = {}
        usage_chunk = None
        for chunk in response:
            if chunk.usage is not None:
                usage_chunk = chunk
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta.content:
                if output_response:
                    print(delta.content, end="", flush=True)
                output += delta.content
            if delta.tool_calls:
                self._accumulate_tool_calls(delta.tool_calls, tool_calls_accum)
        if output_response:
            print("")
        if tool_calls_accum:
            formatted = self._format_streamed_tool_calls(tool_calls_accum)
            if formatted:
                tool_call_str = TOOL_CALL_FORMAT.format(tool_calls=json.dumps(formatted, indent=4, ensure_ascii=False))
                output += tool_call_str
                if output_response:
                    print(tool_call_str)
        if usage_chunk is not None:
            self._update_cost(usage_chunk)
        else:
            logger.warning("[OpenRouterLLM] No usage data in stream response; cost will not be recorded.")
        return output

    async def get_stream_output_async(self, response, output_response: bool = False) -> str:
        output = ""
        tool_calls_accum: Dict[int, dict] = {}
        usage_chunk = None
        async for chunk in response:
            if chunk.usage is not None:
                usage_chunk = chunk
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta.content:
                if output_response:
                    print(delta.content, end="", flush=True)
                output += delta.content
            if delta.tool_calls:
                self._accumulate_tool_calls(delta.tool_calls, tool_calls_accum)
        if output_response:
            print("")
        if tool_calls_accum:
            formatted = self._format_streamed_tool_calls(tool_calls_accum)
            if formatted:
                tool_call_str = TOOL_CALL_FORMAT.format(tool_calls=json.dumps(formatted, indent=4, ensure_ascii=False))
                output += tool_call_str
                if output_response:
                    print(tool_call_str)
        if usage_chunk is not None:
            self._update_cost(usage_chunk)
        else:
            logger.warning("[OpenRouterLLM] No usage data in stream response; cost will not be recorded.")
        return output

    def get_completion_output(self, response: ChatCompletion, output_response: bool=True) -> str:
        output = response.choices[0].message.content or ""
        tool_calls = getattr(response.choices[0].message, "tool_calls", None)
        if tool_calls:
            formatted = format_tool_calls(tool_calls)
            output += TOOL_CALL_FORMAT.format(tool_calls=json.dumps(formatted, indent=4, ensure_ascii=False))
        if output_response:
            print(output)
        self._update_cost(response)
        return output

    @staticmethod
    def _accumulate_tool_calls(delta_tool_calls, accum: Dict[int, dict]):
        for tc in delta_tool_calls:
            idx = tc.index
            if idx not in accum:
                accum[idx] = {"id": "", "function": {"name": "", "arguments": ""}}
            if tc.id:
                accum[idx]["id"] = tc.id
            if tc.function:
                if tc.function.name:
                    accum[idx]["function"]["name"] += tc.function.name
                if tc.function.arguments:
                    accum[idx]["function"]["arguments"] += tc.function.arguments

    @staticmethod
    def _format_streamed_tool_calls(accum: Dict[int, dict]) -> List[dict]:
        formatted = []
        for idx in sorted(accum.keys()):
            tc = accum[idx]
            try:
                args = json.loads(tc["function"]["arguments"])
            except Exception:
                logger.error(f"Failed to parse streaming tool call arguments for `{tc['function']['name']}`:\n{tc['function']['arguments']}")
                continue
            formatted.append({"id": tc["id"], "function_name": tc["function"]["name"], "function_args": args})
        return formatted

    def _update_cost(self, response: Union[ChatCompletion, ChatCompletionChunk]):
        usage = response.usage
        if usage is None:
            logger.warning(f"[OpenRouterLLM] usage is None in response (id={response.id}); cost will not be recorded.")
            return
        cost_value = getattr(usage, "cost", None)
        if cost_value is None:
            logger.warning(
                f"[OpenRouterLLM] usage.cost not present in response (id={response.id}); "
                "cost will be recorded as 0. Check OpenRouter dashboard for actual spend."
            )
            cost_value = 0.0
        cost = Cost(
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens,
            cost=cost_value,
        )
        cost_manager.update_cost(cost, model=self.config.model)

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))
    def single_generate(self, messages: List[dict], **kwargs) -> str:
        stream = kwargs.get("stream", self.config.stream)
        output_response = kwargs.get("output_response", self.config.output_response)

        try:
            client = self.ensure_client()
            completion_params = self.get_completion_params(**kwargs)
            response = client.chat.completions.create(messages=messages, **completion_params)
            if stream:
                output = self.get_stream_output(response, output_response=output_response)
            else:
                output: str = self.get_completion_output(response=response, output_response=output_response)
        except Exception as e:
            raise RuntimeError(f"Error during single_generate of OpenRouterLLM: {str(e)}")

        return output

    def batch_generate(self, batch_messages: List[List[dict]], **kwargs) -> List[str]:
        return [self.single_generate(messages=one_messages, **kwargs) for one_messages in batch_messages]

    async def single_generate_async(self, messages: List[dict], **kwargs) -> str:
        stream = kwargs.get("stream", self.config.stream)
        output_response = kwargs.get("output_response", self.config.output_response)

        try:
            async_client = self.ensure_async_client()
            completion_params = self.get_completion_params(**kwargs)
            response = await async_client.chat.completions.create(
                messages=messages, **completion_params
            )
            if stream:
                output = await self.get_stream_output_async(response, output_response=output_response)
            else:
                # The network I/O is already awaited above; the response is fully in memory here,
                # so this synchronous parsing/cost call does not block the event loop.
                output: str = self.get_completion_output(response=response, output_response=output_response)

        except Exception as e:
            raise RuntimeError(f"Error during single_generate_async of OpenRouterLLM: {str(e)}")

        return output
