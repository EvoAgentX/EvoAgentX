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
    """SiliconFlow language model implementation that uses the SiliconFlow API.
    
    This class extends OpenAILLM to work with SiliconFlow's API, which is compatible
    with the OpenAI API format but has its own authentication, base URL, and pricing.
    It provides access to SiliconFlow-specific models while maintaining the same
    interface as other LLM implementations.
    
    Attributes:
        _client: The OpenAI client configured to use SiliconFlow's API.
        _default_ignore_fields: Parameters that should not be passed to the API.
        response: Temporary storage for the last response (used for cost calculation).
    """

    def init_model(self):
        """Initialize the SiliconFlow model and client.
        
        Sets up the OpenAI client with SiliconFlow's API key and base URL,
        and configures which configuration fields should be ignored when
        making API calls.
        """
        config: SiliconFlowConfig = self.config
        self._client = self._init_client(config) # OpenAI(api_key=config.siliconflow_key, base_url="https://api.siliconflow.cn/v1")
        self._default_ignore_fields = ["llm_type", "siliconflow_key", "output_response"] # parameters in SiliconFlowConfig that are not OpenAI models' input parameters 

    def _init_client(self, config: SiliconFlowConfig):
        """Initialize the client for SiliconFlow's API.
        
        Creates an OpenAI client instance configured to use SiliconFlow's API
        endpoint and authentication.
        
        Args:
            config: The SiliconFlow configuration object containing the API key.
            
        Returns:
            An initialized OpenAI client pointing to SiliconFlow's API.
        """
        client = OpenAI(api_key=config.siliconflow_key, base_url="https://api.siliconflow.cn/v1")
        return client

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))
    def single_generate(self, messages: List[dict], **kwargs) -> str:
        """Generate a single response using the SiliconFlow API.
        
        This method sends a request to the SiliconFlow API and processes the response.
        It includes retry logic with exponential backoff to handle transient API errors
        and specific handling for account balance issues.
        
        Args:
            messages: A list of message dictionaries to send to the API.
            **kwargs: Additional parameters for the API call.
            
        Returns:
            The generated text as a string, or an empty string if account balance is insufficient.
            
        Raises:
            RuntimeError: If an error occurs during generation that persists after retries
                         and is not related to account balance.
        """
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
        """Calculate the cost for a completion response.
        
        Extracts token usage from the API response and calculates the cost.
        
        Args:
            response: The ChatCompletion response from the API.
            
        Returns:
            A Cost object with token counts and monetary costs.
        """
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        return self._compute_cost(input_tokens=input_tokens, output_tokens=output_tokens)


    def _compute_cost(self, input_tokens: int, output_tokens: int) -> Cost:
        """Compute the monetary cost for a given number of tokens.
        
        Uses the SiliconFlow-specific pricing information to calculate costs.
        Different models may have different pricing structures (either uniform
        token cost or separate input/output token costs).
        
        Args:
            input_tokens: Number of tokens in the prompt.
            output_tokens: Number of tokens in the completion.
            
        Returns:
            A Cost object with token counts and monetary costs.
            Returns zero costs if the model is not found in the pricing data.
        """
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
        """Retrieve cost and token usage information from the last response.
        
        This method extracts token usage statistics from the stored response object.
        It handles cases where token counts are not available in stream mode.
        
        Returns:
            A dictionary containing token usage information, or error details if
            the information cannot be retrieved.
        """
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
        """Process stream response and return the complete output.
        
        Collects content from a streaming response and optionally prints it in real-time.
        This implementation also stores the last chunk for token usage information.
        
        Args:
            response: The stream response from the API.
            output_response: Whether to print the response in real-time.
            
        Returns:
            The complete output text as a string.
        """
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
        """Update the global cost tracking with the cost of this generation.
        
        Args:
            cost: A Cost object containing token counts and monetary costs.
        """
        cost_manager.update_cost(cost=cost, model=self.config.model)