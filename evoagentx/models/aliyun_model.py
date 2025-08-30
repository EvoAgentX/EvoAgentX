import asyncio
from typing import Optional, List, Any
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
from dashscope import Generation  # aliyun DashScope SDK
import dashscope

from ..core.registry import register_model
from .model_configs import AliyunLLMConfig
from .base_model import BaseLLM
from .model_utils import Cost, cost_manager
from ..core.logging import logger
import os

@register_model(config_cls=AliyunLLMConfig, alias=["aliyun_llm"])
class AliyunLLM(BaseLLM):
    def init_model(self):
        """
        Initialize the DashScope Generation client.
        """
        config: AliyunLLMConfig = self.config
        if not config.aliyun_api_key:
            raise ValueError("Aliyun API key is required. You should set `aliyun_api_key` in AliyunLLMConfig")
        
        #  API key
        os.environ["DASHSCOPE_API_KEY"] = config.aliyun_api_key
        dashscope.api_key = config.aliyun_api_key
        
        # model
        self._client = Generation()
        self._default_ignore_fields = [
            "llm_type", "output_response", "aliyun_api_key", "aliyun_access_key_id",
            "aliyun_access_key_secret", "model_name"
        ]

    def formulate_messages(self, prompts: List[str], system_messages: Optional[List[str]] = None) -> List[List[dict]]:
        """
        Format messages for the Aliyun model.
        
        Args:
            prompts (List[str]): List of user prompts.
            system_messages (Optional[List[str]]): Optional list of system messages.
            
        Returns:
            List[List[dict]]: Formatted messages for the model.
        """
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
        """
        Update completion parameters with new values.
        
        Args:
            params1 (dict): Base parameters.
            params2 (dict): New parameters to update with.
            
        Returns:
            dict: Updated parameters.
        """
        config_params: list = self.config.get_config_params()
        for key, value in params2.items():
            if key in self._default_ignore_fields:
                continue
            if key not in config_params:
                continue
            params1[key] = value
        return params1

    def get_completion_params(self, **kwargs):
        """
        Get completion parameters for the model.
        
        Returns:
            dict: Parameters for model completion.
        """
        completion_params = self.config.get_set_params(ignore=self._default_ignore_fields)
        completion_params = self.update_completion_params(completion_params, kwargs)
        completion_params["model"] = self.config.model
        return completion_params

    def get_stream_output(self, response: Any, output_response: bool = True) -> str:
        """
        Process streaming response from the model.
        
        Args:
            response: The streaming response from the model.
            output_response (bool): Whether to print the response.
            
        Returns:
            str: The complete response text.
        """
        output = ""
        try:
            for chunk in response:
                if not hasattr(chunk, 'output') or chunk.output is None:
                    error_msg = getattr(chunk, 'message', 'Invalid chunk format from model')
                    raise ValueError(f"Model stream chunk error: {error_msg}")
                if hasattr(chunk.output, 'text'):
                    content = chunk.output.text
                elif hasattr(chunk.output, 'choices') and chunk.output.choices:
                    content = chunk.output.choices[0].message.content
                else:
                    continue
                if content:
                    if output_response:
                        print(content, end="", flush=True)
                    output += content
        except Exception as e:
            print(f"Error processing stream: {str(e)}")
            if not output:
                raise RuntimeError(f"Failed to process stream response: {str(e)}")
        if output_response:
            print("")
        return output

    async def get_stream_output_async(self, response: Any, output_response: bool = False) -> str:
        """
        Process streaming response asynchronously.
        
        Args:
            response: The streaming response from the model.
            output_response (bool): Whether to print the response.
            
        Returns:
            str: The complete response text.
        """
        output = ""
        try:
            async for chunk in response:
                if not hasattr(chunk, 'output') or chunk.output is None:
                    error_msg = getattr(chunk, 'message', 'Invalid chunk format from model')
                    raise ValueError(f"Model stream chunk error: {error_msg}")
                if hasattr(chunk.output, 'text'):
                    content = chunk.output.text
                elif hasattr(chunk.output, 'choices') and chunk.output.choices:
                    content = chunk.output.choices[0].message.content
                else:
                    continue
                if content:
                    if output_response:
                        print(content, end="", flush=True)
                    output += content
        except Exception as e:
            print(f"Error processing async stream: {str(e)}")
            if not output:
                raise RuntimeError(f"Failed to process async stream response: {str(e)}")
        if output_response:
            print("")
        return output

    def get_completion_output(self, response: Any, output_response: bool = True) -> str:
        """
        Process non-streaming response from the model.
        
        Args:
            response: The response from the model.
            output_response (bool): Whether to print the response.
            
        Returns:
            str: The complete response text.
        """
        try:
            if not hasattr(response, 'output') or response.output is None:
                error_msg = getattr(response, 'message', 'Invalid response format from model')
                raise ValueError(f"Model response error: {error_msg}")
            
            if hasattr(response.output, 'text'):
                output = response.output.text
            elif hasattr(response.output, 'choices') and response.output.choices:
                output = response.output.choices[0].message.content
            else:
                raise ValueError("Unexpected response format")
                
            if output_response:
                print(output)
            return output
        except Exception as e:
            raise RuntimeError(f"Error processing completion response: {str(e)}")

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))
    def single_generate(self, messages: List[dict], **kwargs) -> str:
        """
        Generate a single response from the model.
        
        Args:
            messages (List[dict]): The conversation history.
            **kwargs: Additional parameters for generation.
            
        Returns:
            str: The generated response.
        """
        stream = kwargs.get("stream", self.config.stream)
        output_response = kwargs.get("output_response", self.config.output_response)

        try:
            completion_params = self.get_completion_params(**kwargs)
            response = self._client.call(messages=messages, **completion_params)
            
            if response is None:
                raise RuntimeError("Received empty response from model")
                
            if stream:
                output = self.get_stream_output(response, output_response=output_response)
                cost = self._stream_cost(response)
            else:
                output = self.get_completion_output(response=response, output_response=output_response)
                cost = self._completion_cost(response)
            self._update_cost(cost=cost)
            return output
        except Exception as e:
            raise RuntimeError(f"Error during single_generate of AliyunLLM: {str(e)}")

    def batch_generate(self, batch_messages: List[List[dict]], **kwargs) -> List[str]:
        """
        Generate responses for a batch of messages.
        
        Args:
            batch_messages (List[List[dict]]): List of conversation histories.
            **kwargs: Additional parameters for generation.
            
        Returns:
            List[str]: List of generated responses.
        """
        if not isinstance(batch_messages, list) or not batch_messages:
            raise ValueError("batch_messages must be a non-empty list of message lists")
        return [self.single_generate(messages=one_messages, **kwargs) for one_messages in batch_messages]

    async def single_generate_async(self, messages: List[dict], **kwargs) -> str:
        """
        Asynchronously generate a single response.
        
        Args:
            messages (List[dict]): The conversation history.
            **kwargs: Additional parameters for the generation.
            
        Returns:
            str: The generated response.
        """
        stream = kwargs.get("stream", self.config.stream)
        output_response = kwargs.get("output_response", self.config.output_response)
        
        try:
            completion_params = self.get_completion_params(**kwargs)
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self._client.call(messages=messages, **completion_params)
            )
            
            if stream:
                output = await self.get_stream_output_async(response, output_response=output_response)
                cost = self._stream_cost(response)
            else:
                output = self.get_completion_output(response=response, output_response=output_response)
                cost = self._completion_cost(response)
            
            self._update_cost(cost=cost)
            return output
            
        except Exception as e:
            raise RuntimeError(f"Error during single_generate_async of AliyunLLM: {str(e)}")

    def _completion_cost(self, response: Any) -> Cost:
        """cost"""
        try:
            if not response:
                return Cost(input_tokens=0, output_tokens=0, input_cost=0.0, output_cost=0.0)
                
            # tokens number
            input_tokens = 0
            output_tokens = 0
            
            if hasattr(response, 'usage'):
                usage = response.usage
                if hasattr(usage, 'input_tokens'):
                    input_tokens = usage.input_tokens
                elif hasattr(usage, 'prompt_tokens'):
                    input_tokens = usage.prompt_tokens
                    
                if hasattr(usage, 'output_tokens'):
                    output_tokens = usage.output_tokens
                elif hasattr(usage, 'completion_tokens'):
                    output_tokens = usage.completion_tokens
            
            # 
            if input_tokens == 0 and output_tokens == 0 and hasattr(response, 'output'):
                if hasattr(response.output, 'text'):
                    output_tokens = len(response.output.text.split()) * 1.3  
                elif hasattr(response.output, 'choices') and response.output.choices:
                    output_tokens = len(response.output.choices[0].message.content.split()) * 1.3
            
            total_cost = self._estimate_cost(input_tokens, output_tokens)
            return Cost(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                input_cost=total_cost * 0.4,  # 
                output_cost=total_cost * 0.6   # 
            )
        except Exception as e:
            logger.warning(f"Error computing completion cost: {str(e)}")
            return Cost(input_tokens=0, output_tokens=0, input_cost=0.0, output_cost=0.0)

    def _stream_cost(self, response: Any) -> Cost:
        """cost"""
        try:
            if not response:
                return Cost(input_tokens=0, output_tokens=0, input_cost=0.0, output_cost=0.0)
                
            # 
            input_tokens = 0
            output_tokens = 0
            
            if hasattr(response, 'usage'):
                usage = response.usage
                if hasattr(usage, 'input_tokens'):
                    input_tokens = usage.input_tokens
                elif hasattr(usage, 'prompt_tokens'):
                    input_tokens = usage.prompt_tokens
                    
                if hasattr(usage, 'output_tokens'):
                    output_tokens = usage.output_tokens
                elif hasattr(usage, 'completion_tokens'):
                    output_tokens = usage.completion_tokens
            
            # 
            if input_tokens == 0 and output_tokens == 0 and hasattr(response, 'output'):
                if hasattr(response.output, 'text'):
                    output_tokens = len(response.output.text.split()) * 1.3  # 
                elif hasattr(response.output, 'choices') and response.output.choices:
                    output_tokens = len(response.output.choices[0].message.content.split()) * 1.3
            
            total_cost = self._estimate_cost(input_tokens, output_tokens)
            return Cost(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                input_cost=total_cost * 0.4,  # 
                output_cost=total_cost * 0.6   # 
            )
        except Exception as e:
            logger.warning(f"Error computing stream cost: {str(e)}")
            return Cost(input_tokens=0, output_tokens=0, input_cost=0.0, output_cost=0.0)

    def _estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """cost
        
        """
        model = self.config.model.lower()
        if "turbo" in model:
            input_cost = (input_tokens / 1000) * 0.0005
            output_cost = (output_tokens / 1000) * 0.001
        elif "max" in model:
            input_cost = (input_tokens / 1000) * 0.002
            output_cost = (output_tokens / 1000) * 0.004
        else:  # default
            input_cost = (input_tokens / 1000) * 0.001
            output_cost = (output_tokens / 1000) * 0.002
            
        return input_cost + output_cost

    def _update_cost(self, cost: Cost):
        """
        Update the cost manager with the new cost.
        
        Args:
            cost (Cost): The cost to update.
        """
        try:
            cost_manager.update_cost(cost=cost, model=self.config.model)
        except Exception as e:
            logger.warning(f"Error updating cost: {str(e)}")

