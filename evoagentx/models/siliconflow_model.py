from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
from typing import List

from .openai_model import OpenAILLM
from .model_configs import SiliconFlowConfig
from ..core.registry import register_model
from openai import OpenAI, Stream
# from loguru import logger
from .model_utils import Cost, cost_manager
from openai.types.chat import ChatCompletion
from .siliconflow_model_cost import model_cost

@register_model(config_cls=SiliconFlowConfig, alias=["siliconflow"])
class SiliconFlowLLM(OpenAILLM):

    def init_model(self):
        config: SiliconFlowConfig = self.config
        self._client = self._init_client(config) # OpenAI(api_key=config.siliconflow_key, base_url="https://api.siliconflow.cn/v1")
        self._default_ignore_fields = ["llm_type", "siliconflow_key", "output_response"] # parameters in SiliconFlowConfig that are not OpenAI models' input parameters 

    def _init_client(self, config: SiliconFlowConfig):
        client = OpenAI(api_key=config.siliconflow_key, base_url="https://api.siliconflow.cn/v1")
        return client

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))
    def single_generate(self, messages: List[dict], **kwargs) -> str:

        stream = kwargs["stream"] if "stream" in kwargs else self.config.stream
        output_response = kwargs["output_response"] if "output_response" in kwargs else self.config.output_response

        try:
            completion_params = self.get_completion_params(**kwargs)
            response = self._client.chat.completions.create(
                messages=messages, 
                **completion_params
            )
            if stream:
                output = self.get_stream_output(response, output_response=output_response)
                cost = self._completion_cost(self.response)  
            else:
                output: str = response.choices[0].message.content
                cost = self._completion_cost(response)
                if output_response:
                    print(output)
            self._update_cost(cost=cost)
        except Exception as e:
            if "account balance is insufficient" in str(e):
                print("Warning: Account balance insufficient. Please recharge your account.")
                return ""
            raise RuntimeError(f"Error during single_generate of OpenAILLM: {str(e)}")

        return output


    def _completion_cost(self, response: ChatCompletion) -> Cost:
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        return self._compute_cost(input_tokens=input_tokens, output_tokens=output_tokens)


    def _compute_cost(self, input_tokens: int, output_tokens: int) -> Cost:
        model: str = self.config.model
        # total_tokens = input_tokens + output_tokens
        if model not in model_cost:
            return Cost(input_tokens=input_tokens, output_tokens=output_tokens, input_cost=0.0, output_cost=0.0)
        
        if "token_cost" in model_cost[model]:
            # total_cost = total_tokens * model_cost[model]["token_cost"] / 1e6
            input_cost = input_tokens * model_cost[model]["token_cost"] / 1e6
            output_cost = output_tokens * model_cost[model]["token_cost"] / 1e6
        else:
            # total_cost = input_tokens * model_cost[model]["input_token_cost"] / 1e6 + output_tokens * model_cost[model]["output_token_cost"] / 1e6
            input_cost = input_tokens * model_cost[model]["input_token_cost"] / 1e6
            output_cost = output_tokens * model_cost[model]["output_token_cost"] / 1e6
        
        return Cost(input_tokens=input_tokens, output_tokens=output_tokens, input_cost=input_cost, output_cost=output_cost)


    def get_cost(self) -> dict:
        cost_info = {}
        try:
            tokens = self.response.usage
            if tokens.prompt_tokens == -1:
                cost_info["note"] = "Token counts not available in stream mode"
                cost_info["prompt_tokens"] = 0
                cost_info["completion_tokens"] = 0
                cost_info["total_tokens"] = 0
            else:
                cost_info["prompt_tokens"] = tokens.prompt_tokens
                cost_info["completion_tokens"] = tokens.completion_tokens
                cost_info["total_tokens"] = tokens.total_tokens
        except Exception as e:
            print(f"Error during get_cost of SiliconFlow: {str(e)}")
            cost_info["error"] = str(e)

        return cost_info

    def get_stream_output(self, response: Stream, output_response: bool=True) -> str:
        output = ""
        last_chunk = None
        for chunk in response:
            content = chunk.choices[0].delta.content
            if content:
                if output_response:
                    print(content, end="", flush=True)
                output += content
            last_chunk = chunk

        if output_response:
            print("")

        # Store usage information from the last chunk
        if hasattr(last_chunk, 'usage'):
            self.response = last_chunk
        else:
            # Create a placeholder response object for stream mode
            self.response = type('StreamResponse', (), {
                'usage': type('StreamUsage', (), {
                    'prompt_tokens': -1,
                    'completion_tokens': -1,
                    'total_tokens': -1
                })
            })

        return output

    def _update_cost(self, cost: Cost):
        cost_manager.update_cost(cost=cost, model=self.config.model)