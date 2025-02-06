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

@register_model(config_cls=SiliconFlowConfig, alias=["siliconflow"])
class SiliconFlow(OpenAILLM):
    
    def init_model(self):
        config: SiliconFlowConfig = self.config
        self._client = OpenAI(api_key=config.siliconflow_key,
                              base_url="https://api.siliconflow.cn/v1")
        # print(f"self._client: {self._client}")
        self._default_ignore_fields = ["llm_type", "siliconflow_key", "output_response"] # parameters in SiliconFlowConfig that are not OpenAI models' input parameters 
        
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
            else:
                output: str = response.choices[0].message.content
                self.response = response
                if output_response:
                    print(output)
        except Exception as e:
            if "account balance is insufficient" in str(e):
                print("Warning: Account balance insufficient. Please recharge your account.")
                return ""
            raise RuntimeError(f"Error during single_generate of OpenAILLM: {str(e)}")
        
        return output
    
    def get_cost(self) -> dict:
        cost_info = {}
        try:
            tokens = self.response.usage
            if tokens.prompt_tokens == -1:
                cost_info["note"] = "Token counts not available in stream mode"
                cost_info["prompt_tokens"] = "N/A"
                cost_info["completion_tokens"] = "N/A" 
                cost_info["total_tokens"] = "N/A"
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
