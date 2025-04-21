import asyncio
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
from openai import OpenAI, Stream
from openai.types.chat import ChatCompletion
from typing import Optional, List
from litellm import token_counter, cost_per_token
from ..core.registry import register_model
from .model_configs import OpenAILLMConfig
from .base_model import BaseLLM
from .model_utils import Cost, cost_manager, get_openai_model_cost 


@register_model(config_cls=OpenAILLMConfig, alias=["openai_llm"])
class OpenAILLM(BaseLLM):
    """OpenAI language model implementation that wraps the OpenAI API.
    
    This class provides a standardized interface to OpenAI's language models,
    supporting both synchronous and asynchronous generation, streaming responses,
    cost tracking, and automatic retries with exponential backoff.
    
    Attributes:
        config: The configuration object of type OpenAILLMConfig.
        _client: The OpenAI client instance.
        _default_ignore_fields: Parameters that should not be passed to OpenAI API calls.
    """

    def init_model(self):
        """Initialize the OpenAI model and client.
        
        This method sets up the OpenAI client with the API key from the configuration
        and validates that the specified model is recognized by OpenAI.
        
        Raises:
            KeyError: If the configured model name is not a valid OpenAI model.
        """
        config: OpenAILLMConfig = self.config
        self._client = self._init_client(config) # OpenAI(api_key=config.openai_key)
        self._default_ignore_fields = ["llm_type", "output_response", "openai_key", "deepseek_key", "anthropic_key"] # parameters in OpenAILLMConfig that are not OpenAI models' input parameters 
        if self.config.model not in get_openai_model_cost():
            raise KeyError(f"'{self.config.model}' is not a valid OpenAI model name!")
    
    def _init_client(self, config: OpenAILLMConfig):
        """Initialize the OpenAI client.
        
        Args:
            config: The configuration object containing the OpenAI API key.
            
        Returns:
            An initialized OpenAI client instance.
        """
        client = OpenAI(api_key=config.openai_key)
        return client

    def formulate_messages(self, prompts: List[str], system_messages: Optional[List[str]] = None) -> List[List[dict]]:
        """Convert raw prompts into OpenAI chat format.
        
        Formats user prompts and optional system messages into the message format
        expected by OpenAI's chat completions API.
        
        Args:
            prompts: List of user prompt strings.
            system_messages: Optional list of system messages, one per prompt.
                             If provided, must match the length of prompts.
        
        Returns:
            A list of message lists, where each inner list contains properly
            formatted message dictionaries for the OpenAI API.
            
        Raises:
            AssertionError: If system_messages is provided but its length doesn't
                           match the length of prompts.
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
        """Update completion parameters from a secondary source.
        
        This method updates the primary parameters dictionary with values from 
        a secondary dictionary, but only for keys that are valid configuration
        parameters and not in the ignored fields list.
        
        Args:
            params1: The primary parameters dictionary to be updated.
            params2: The secondary parameters dictionary with update values.
            
        Returns:
            The updated parameters dictionary.
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
        """Prepare parameters for the completion API call.
        
        Extracts relevant parameters from the configuration and updates them
        with any provided keyword arguments.
        
        Args:
            **kwargs: Additional parameters to override configuration values.
            
        Returns:
            A dictionary of parameters to pass to the OpenAI API.
        """
        completion_params = self.config.get_set_params(ignore=self._default_ignore_fields)
        completion_params = self.update_completion_params(completion_params, kwargs)
        return completion_params
    
    def get_stream_output(self, response: Stream, output_response: bool=True) -> str:
        """Process stream response and return the complete output.
        
        Collects content from a streaming response and optionally prints it
        in real-time.
        
        Args:
            response: The stream response from OpenAI.
            output_response: Whether to print the response in real-time.
            
        Returns:
            The complete output text as a string.
        """
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
    
    async def get_stream_output_async(self, response, output_response: bool = False) -> str:
        """Process async stream response and return the complete output.
        
        Asynchronously collects content from a streaming response and
        optionally prints it in real-time.
        
        Args:
            response: The async stream response from OpenAI.
            output_response: Whether to print the response in real-time.
            
        Returns:
            The complete output text as a string.
        """
        output = ""
        async for chunk in response:
            content = chunk.choices[0].delta.content
            if content:
                if output_response:
                    print(content, end="", flush=True)
                output += content
        if output_response:
            print("")
        return output

    def get_completion_output(self, response: ChatCompletion, output_response: bool=True) -> str:
        """Extract and optionally display text from a completion response.
        
        Args:
            response: The ChatCompletion response from OpenAI.
            output_response: Whether to print the response.
            
        Returns:
            The output text as a string.
        """
        output = response.choices[0].message.content
        if output_response:
            print(output)
        return output

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))
    def single_generate(self, messages: List[dict], **kwargs) -> str:
        """Generate a single response using the OpenAI API.
        
        This method sends a request to the OpenAI API and processes the response.
        It includes retry logic with exponential backoff to handle transient API errors.
        
        Args:
            messages: A list of message dictionaries to send to the API.
            **kwargs: Additional parameters for the API call.
            
        Returns:
            The generated text as a string.
            
        Raises:
            RuntimeError: If an error occurs during generation that persists
                         after retries.
        """
        stream = kwargs["stream"] if "stream" in kwargs else self.config.stream
        output_response = kwargs["output_response"] if "output_response" in kwargs else self.config.output_response

        try:
            completion_params = self.get_completion_params(**kwargs)
            response = self._client.chat.completions.create(messages=messages, **completion_params)
            if stream:
                output = self.get_stream_output(response, output_response=output_response)
                cost = self._stream_cost(messages=messages, output=output)
            else:
                output: str = self.get_completion_output(response=response, output_response=output_response)
                cost = self._completion_cost(response) # calculate completion cost
            self._update_cost(cost=cost)
        except Exception as e:
            raise RuntimeError(f"Error during single_generate of OpenAILLM: {str(e)}")
        
        return output
        
    def batch_generate(self, batch_messages: List[List[dict]], **kwargs) -> List[str]:
        """Generate responses for a batch of message lists.
        
        Processes multiple conversation threads sequentially by calling
        single_generate for each one.
        
        Args:
            batch_messages: A list of message lists, where each message list 
                           represents a conversation to generate a response for.
            **kwargs: Additional parameters for the API calls.
            
        Returns:
            A list of generated text strings, one per input conversation.
        """
        return [self.single_generate(messages=one_messages, **kwargs) for one_messages in batch_messages]

    async def single_generate_async(self, messages: List[dict], **kwargs) -> str:
        """Asynchronously generate a single response using the OpenAI API.
        
        This method provides a non-blocking way to interact with the OpenAI API,
        useful for high-throughput scenarios or when integrating with async frameworks.
        
        Args:
            messages: A list of message dictionaries to send to the API.
            **kwargs: Additional parameters for the API call.
            
        Returns:
            The generated text as a string.
            
        Raises:
            RuntimeError: If an error occurs during asynchronous generation.
        """
        stream = kwargs.get("stream", self.config.stream)
        output_response = kwargs.get("output_response", self.config.output_response)

        try:
            # Create a completely new client instance to avoid thread-local storage issues
            # This is a more aggressive approach than using a lock
            # isolated_client = OpenAI(api_key=self.config.openai_key)
            isolated_client = self._init_client(self.config)
            completion_params = self.get_completion_params(**kwargs)

            # Use synchronous client in async context to avoid issues
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, 
                lambda: isolated_client.chat.completions.create(
                    messages=messages, 
                    **completion_params
                )
            )

            if stream:
                if hasattr(response, "__aiter__"):
                    output = await self.get_stream_output_async(response, output_response=output_response)
                else:
                    output = self.get_stream_output(response, output_response=output_response)
                cost = self._stream_cost(messages=messages, output=output)
            else:
                output: str = self.get_completion_output(response=response, output_response=output_response)
                cost = self._completion_cost(response) # calculate completion cost
            self._update_cost(cost=cost)
        
        except Exception as e:
            raise RuntimeError(f"Error during single_generate_async of OpenAILLM: {str(e)}")

        return output
    
    def _completion_cost(self, response: ChatCompletion) -> Cost:
        """Calculate the cost for a completion response.
        
        Uses the token usage information provided in the API response.
        
        Args:
            response: The ChatCompletion response from OpenAI.
            
        Returns:
            A Cost object with token counts and monetary costs.
        """
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        return self._compute_cost(input_tokens=input_tokens, output_tokens=output_tokens)

    def _stream_cost(self, messages: List[dict], output: str) -> Cost:
        """Calculate the cost for a streaming response.
        
        For streaming responses, we need to manually count tokens since 
        usage information is not available in the API response.
        
        Args:
            messages: The input messages sent to the API.
            output: The generated output text.
            
        Returns:
            A Cost object with token counts and monetary costs.
        """
        model: str = self.config.model
        input_tokens = token_counter(model=model, messages=messages)
        output_tokens = token_counter(model=model, text=output)
        return self._compute_cost(input_tokens=input_tokens, output_tokens=output_tokens)
    
    def _compute_cost(self, input_tokens: int, output_tokens: int) -> Cost:
        """Compute the monetary cost for a given number of tokens.
        
        Uses LiteLLM's cost_per_token function to calculate the costs
        based on the model's pricing.
        
        Args:
            input_tokens: Number of tokens in the prompt.
            output_tokens: Number of tokens in the completion.
            
        Returns:
            A Cost object with token counts and monetary costs.
        """
        # use LiteLLM to compute cost, require the model name to be a valid model name in LiteLLM.
        input_cost, output_cost = cost_per_token(
            model=self.config.model, 
            prompt_tokens=input_tokens, 
            completion_tokens=output_tokens, 
        )
        cost = Cost(input_tokens=input_tokens, output_tokens=output_tokens, input_cost=input_cost, output_cost=output_cost)
        return cost
    
    def _update_cost(self, cost: Cost):
        """Update the global cost tracking with the cost of this generation.
        
        Args:
            cost: A Cost object containing token counts and monetary costs.
        """
        cost_manager.update_cost(cost=cost, model=self.config.model)
    