from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
from openai import OpenAI, Stream 
from typing import Optional, List
from ..core.registry import register_model
from .model_configs import OpenAILLMConfig
from .base_model import BaseLLM


@register_model(config_cls=OpenAILLMConfig, alias=["openai_llm"])
class OpenAILLM(BaseLLM):

    def init_model(self):
        config: OpenAILLMConfig = self.config
        self._client = OpenAI(api_key=config.openai_key)
        self._default_ignore_fields = ["llm_type", "openai_key", "output_response"] # parameters in OpenAILLMConfig that are not OpenAI models' input parameters 

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
        for chunk in response:
            content = chunk.choices[0].delta.content
            if content:
                if output_response:
                    print(content, end="", flush=True)
                output += content
        if output_response:
            print("")
        return output

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))
    def single_generate(self, messages: List[dict], **kwargs) -> str:

        stream = kwargs["stream"] if "stream" in kwargs else self.config.stream
        output_response = kwargs["output_response"] if "output_response" in kwargs else self.config.output_response

        try:
            completion_params = self.get_completion_params(**kwargs)
            response = self._client.chat.completions.create(messages=messages, 
                                                            **completion_params)
            if stream:
                output = self.get_stream_output(response, output_response=output_response)
            else:
                output: str = response.choices[0].message.content
                if output_response:
                    print(output)
        except Exception as e:
            if "account balance is insufficient" in str(e):
                print("Warning: Account balance insufficient. Please recharge your account.")
                return ""
            raise RuntimeError(f"Error during single_generate of OpenAILLM: {str(e)}")
        
        return output
        
    def batch_generate(self, batch_messages: List[List[dict]], **kwargs) -> List[str]:
        return [self.single_generate(messages=one_messages, **kwargs) for one_messages in batch_messages]
    
