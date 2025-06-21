import os
import litellm
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
from litellm import completion, acompletion, token_counter, cost_per_token
from typing import List
from ..core.registry import register_model
from .model_configs import LiteLLMConfig
# from .base_model import BaseLLM, LLMOutputParser
from .openai_model import OpenAILLM
from .model_utils import infer_litellm_company_from_model

@register_model(config_cls=LiteLLMConfig, alias=["litellm"])
class LiteLLM(OpenAILLM):

    def init_model(self):
        """
        Initialize the model based on the configuration.
        """
        # Check if llm_type is correct
        if self.config.llm_type != "LiteLLM":
            raise ValueError("llm_type must be 'LiteLLM'")

        # Set model and extract the company name
        self.model = self.config.model
        self.api_base = self.config.api_base  # save api_base
        # company = self.model.split("/")[0] if "/" in self.model else "openai"
        company = infer_litellm_company_from_model(self.model)

        if self.config.is_local or company == "local":  # update support local model
            if not self.api_base:
                raise ValueError("api_base is required for local models in LiteLLMConfig")
            # local llm doesn't need API key
            litellm.api_base = self.api_base  # set litellm global api_base
        else:
            # Set environment variables based on the company
            if company == "openai":
                if not self.config.openai_key:
                    raise ValueError("OpenAI API key is required for OpenAI models. You should set `openai_key` in LiteLLMConfig")
                os.environ["OPENAI_API_KEY"] = self.config.openai_key
            elif company == "azure":
                if not self.config.azure_key or not self.config.azure_endpoint:
                    raise ValueError("Azure OpenAI key and endpoint are required for Azure models. You should set `azure_key` and `azure_endpoint` in LiteLLMConfig")
                os.environ["AZURE_API_KEY"] = self.config.azure_key
                os.environ["AZURE_API_BASE"] = self.config.azure_endpoint
                if self.config.api_version:
                    os.environ["AZURE_API_VERSION"] = self.config.api_version
            elif company == "deepseek":
                if not self.config.deepseek_key:
                    raise ValueError("DeepSeek API key is required for DeepSeek models. You should set `deepseek_key` in LiteLLMConfig")
                os.environ["DEEPSEEK_API_KEY"] = self.config.deepseek_key
            elif company == "anthropic":
                if not self.config.anthropic_key:
                    raise ValueError("Anthropic API key is required for Anthropic models. You should set `anthropic_key` in LiteLLMConfig")
                os.environ["ANTHROPIC_API_KEY"] = self.config.anthropic_key
            elif company == "gemini":
                if not self.config.gemini_key:
                    raise ValueError("Gemini API key is required for Gemini models. You should set `gemini_key` in LiteLLMConfig")
                os.environ["GEMINI_API_KEY"] = self.config.gemini_key 
            elif company == "meta_llama":
                if not self.config.meta_llama_key:
                    raise ValueError("Meta Llama API key is required for Meta Llama models. You should set `meta_llama_key` in LiteLLMConfig")
                os.environ["LLAMA_API_KEY"] = self.config.meta_llama_key
            elif company == "openrouter":
                if not self.config.openrouter_key:
                    raise ValueError("OpenRouter API key is required for OpenRouter models. You should set `openrouter_key` in LiteLLMConfig. You can also set `openrouter_base` in LiteLLMConfig to use a custom base URL [optional]")
                os.environ["OPENROUTER_API_KEY"] = self.config.openrouter_key
                os.environ["OPENROUTER_API_BASE"] = self.config.openrouter_base # [optional]
            elif company == "perplexity":
                if not self.config.perplexity_key:
                    raise ValueError("Perplexity API key is required for Perplexity models. You should set `perplexity_key` in LiteLLMConfig")
                os.environ["PERPLEXITYAI_API_KEY"] = self.config.perplexity_key
            elif company == "groq":
                if not self.config.groq_key:
                    raise ValueError("Groq API key is required for Groq models. You should set `groq_key` in LiteLLMConfig")
                os.environ["GROQ_API_KEY"] = self.config.groq_key
            else:
                raise ValueError(f"Unsupported company: {company}")

        self._default_ignore_fields = [
            "llm_type", "output_response", "openai_key", "deepseek_key", "anthropic_key", 
            "gemini_key", "meta_llama_key", "openrouter_key", "openrouter_base", "perplexity_key", 
            "groq_key", "api_base", "is_local", "azure_endpoint", "azure_key", "api_version"
        ] # parameters in LiteLLMConfig that are not LiteLLM models' input parameters 
    
    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))
    def single_generate(self, messages: List[dict], **kwargs) -> str:

        """
        Generate a single response using the completion function.

        Args: 
            messages (List[dict]): A list of dictionaries representing the conversation history.
            **kwargs (Any): Additional parameters to be passed to the `completion` function.
        
        Returns: 
            str: A string containing the model's response.
        """
        stream = kwargs["stream"] if "stream" in kwargs else self.config.stream
        output_response = kwargs["output_response"] if "output_response" in kwargs else self.config.output_response

        try:
            completion_params = self.get_completion_params(**kwargs)
            company = infer_litellm_company_from_model(self.model)
            if self.config.is_local or company == "local":  # update save api_base for local model
                completion_params["api_base"] = self.api_base
            elif company == "azure":  # Add Azure OpenAI specific parameters
                completion_params["api_base"] = self.config.azure_endpoint
                completion_params["api_version"] = self.config.api_version
                completion_params["api_key"] = self.config.azure_key
            response = completion(messages=messages, **completion_params)
            if stream:
                output = self.get_stream_output(response, output_response=output_response)
                cost = self._stream_cost(messages=messages, output=output)
            else:
                output: str = self.get_completion_output(response=response, output_response=output_response)
                cost = self._completion_cost(response=response)
            self._update_cost(cost=cost)

        except Exception as e:
            raise RuntimeError(f"Error during single_generate: {str(e)}")
        
        return output
    
    def batch_generate(self, batch_messages: List[List[dict]], **kwargs) -> List[str]:
        """
        Generate responses for a batch of messages.

        Args: 
            batch_messages (List[List[dict]]): A list of message lists, where each sublist represents a conversation.
            **kwargs (Any): Additional parameters to be passed to the `completion` function.
        
        Returns: 
            List[str]: A list of responses for each conversation.
        """
        results = []
        for messages in batch_messages:
            response = self.single_generate(messages, **kwargs)
            results.append(response)
        return results
    
    async def single_generate_async(self, messages: List[dict], **kwargs) -> str:
        """
        Generate a single response using the async completion function.

        Args: 
            messages (List[dict]): A list of dictionaries representing the conversation history.
            **kwargs (Any): Additional parameters to be passed to the `completion` function.
        
        Returns: 
            str: A string containing the model's response.
        """
        stream = kwargs["stream"] if "stream" in kwargs else self.config.stream
        output_response = kwargs["output_response"] if "output_response" in kwargs else self.config.output_response

        try:
            completion_params = self.get_completion_params(**kwargs)
            company = infer_litellm_company_from_model(self.model)
            if self.config.is_local or company == "local":  # add api base for local model
                completion_params["api_base"] = self.api_base
            elif company == "azure":  # Add Azure OpenAI specific parameters
                completion_params["api_base"] = self.config.azure_endpoint
                completion_params["api_version"] = self.config.api_version
                completion_params["api_key"] = self.config.azure_key
            response = await acompletion(messages=messages, **completion_params)
            if stream:
                if hasattr(response, "__aiter__"):
                    output = await self.get_stream_output_async(response, output_response=output_response)
                else:
                    output = self.get_stream_output(response, output_response=output_response)
                cost = self._stream_cost(messages=messages, output=output)
            else:
                output: str = self.get_completion_output(response=response, output_response=output_response)
                cost = self._completion_cost(response=response)
            self._update_cost(cost=cost)
        except Exception as e:
            raise RuntimeError(f"Error during single_generate_async: {str(e)}")
        
        return output

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
        
        Args:
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
        if self.config.is_local:  # local model 0 cost
            return 0.0
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
