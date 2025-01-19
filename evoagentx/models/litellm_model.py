from typing import Optional, Type, List
from ..core.registry import register_model
from .model_configs import LiteLLMConfig
from .base_model import BaseLLM, LLMOutputParser
from litellm import completion
import os
from litellm.utils import token_counter, cost_per_token
import litellm

@register_model(config_cls=LiteLLMConfig, alias=["litellm"])
class LiteLLM(BaseLLM):
    def __init__(self, config: LiteLLMConfig):
        """
        Initialize the LiteLLM instance with the provided config.
        """
        self.config = config
        self.init_model()

    def init_model(self):
        """
        Initialize the model based on the configuration.
        """
        # Check if llm_type is correct
        if self.config.llm_type != "LiteLLM":
            raise ValueError("llm_type must be 'LiteLLM'")
        
        # Set model and extract the company name
        self.model = self.config.model
        company = self.model.split("/")[0]

        # Set environment variables based on the company
        if company == "openai":
            if not self.config.openai_key:
                raise ValueError("OpenAI API key is required for OpenAI models")
            os.environ["OPENAI_API_KEY"] = self.config.openai_key
        elif company == "deepseek":
            if not self.config.deepseek_key:
                raise ValueError("DeepSeek API key is required for DeepSeek models")
            os.environ["DEEPSEEK_API_KEY"] = self.config.deepseek_key
        else:
            raise ValueError(f"Unsupported company: {company}")

    def single_generate(self, messages: List[dict], **kwargs) -> str:
        """
        Generate a single response using the completion function.

        :param messages: A list of dictionaries representing the conversation history.
        :param kwargs: Additional parameters to be passed to the `completion` function.
        :return: A string containing the model's response.
        """
        try:
            response = completion(
                model=self.model,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                top_p=self.config.top_p,
                n=self.config.n,
                stream=self.config.stream,
                timeout=self.config.timeout,
                **kwargs
            )
            # Extract and return the response content
            if isinstance(response, dict) and "choices" in response:
                return response["choices"][0]["message"]["content"]
            return response
        except Exception as e:
            raise RuntimeError(f"Error during single_generate: {str(e)}")

    def batch_generate(self, batch_messages: List[List[dict]], **kwargs) -> List[str]:
        """
        Generate responses for a batch of messages.

        :param batch_messages: A list of message lists, where each sublist represents a conversation.
        :param kwargs: Additional parameters to be passed to the `completion` function.
        :return: A list of responses for each conversation.
        """
        results = []
        for messages in batch_messages:
            response = self.single_generate(messages, **kwargs)
            results.append(response)
        return results
    
    def completion_cost(
        self,
        completion_response=None,
        prompt="",
        messages: List = [],
        completion="",
        total_time=0.0,
        call_type="completion",
        size=None,
        quality=None,
        n=None,
    ) -> float:
        """
        Calculate the cost of a given completion or other supported tasks.

        Parameters:
            completion_response (dict): The response received from a LiteLLM completion request.
            prompt (str): Input prompt text.
            messages (list): Conversation history.
            completion (str): Output text from the LLM.
            total_time (float): Total time used for request.
            call_type (str): Type of request (e.g., "completion", "image_generation").
            size (str): Image size for image generation.
            quality (str): Image quality for image generation.
            n (int): Number of generated images.

        Returns:
            float: The cost in USD.
        """
        try:
            # Default parameters
            prompt_tokens = 0
            completion_tokens = 0
            model = self.model  # Use the class model by default

            # Handle completion response
            if completion_response:
                prompt_tokens = completion_response.get("usage", {}).get("prompt_tokens", 0)
                completion_tokens = completion_response.get("usage", {}).get("completion_tokens", 0)
                model = completion_response.get("model", model)
                size = completion_response.get("_hidden_params", {}).get("optional_params", {}).get("size", size)
                quality = completion_response.get("_hidden_params", {}).get("optional_params", {}).get("quality", quality)
                n = completion_response.get("_hidden_params", {}).get("optional_params", {}).get("n", n)

            # Handle manual token counting
            else:
                if messages:
                    prompt_tokens = token_counter(model=model, messages=messages)
                elif prompt:
                    prompt_tokens = token_counter(model=model, text=prompt)
                completion_tokens = token_counter(model=model, text=completion)

            # Ensure model is valid
            if not model:
                raise ValueError("Model is not defined for cost calculation.")

            # Image generation cost calculation
            if call_type in ["image_generation", "aimage_generation"]:
                if size and "x" in size and "-x-" not in size:
                    size = size.replace("x", "-x-")
                height, width = map(int, size.split("-x-"))
                return (
                    litellm.model_cost[f"{size}/{model}"]["input_cost_per_pixel"]
                    * height * width * (n or 1)
                )

            # Regular completion cost calculation
            prompt_cost, completion_cost = cost_per_token(
                model=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                response_time_ms=total_time,
            )
            return prompt_cost + completion_cost
        except Exception as e:
            print(f"Error calculating cost: {e}")
            return 0.0
    
    def parse_generated_text(self, text: str, parser: Optional[Type[LLMOutputParser]]=None, **kwargs) -> LLMOutputParser:
        return LLMOutputParser.parse("")
    
    def parse_generated_texts(self, texts: List[str], parser: Optional[Type[LLMOutputParser]]=None, **kwargs) -> List[LLMOutputParser]:
        return [LLMOutputParser.parse("")]
