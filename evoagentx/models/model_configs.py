from pydantic import BaseModel, Field
from typing import Optional, Union, List

# import torch 

from ..core.base_config import BaseConfig


#### LLM Configs
class LLMConfig(BaseConfig):
    """Base configuration class for all Large Language Model implementations.
    
    This abstract class defines the common parameters required by all LLM
    implementations in the framework. Specific model providers should extend
    this class with their own configuration parameters.
    
    Attributes:
        llm_type: The type of LLM implementation (e.g., "OpenAILLM", "LiteLLM").
        model: The specific model identifier to use (e.g., "gpt-4", "claude-2").
        output_response: Whether to output the raw LLM response for debugging.
    """
    llm_type: str
    model: str 
    output_response: bool = Field(default=False, description="Whether to output LLM response.")


class OpenAILLMConfig(LLMConfig):
    """Configuration for OpenAI models.
    
    This class provides configuration parameters specific to OpenAI LLM models,
    including API authentication and generation parameters for controlling
    the behavior of the model.
    
    Attributes:
        llm_type: Set to "OpenAILLM" by default.
        openai_key: The API key used to authenticate OpenAI requests.
        
        Generation parameters:
        temperature: Controls randomness in generation (higher = more random).
        max_tokens: Maximum number of tokens to generate (deprecated for o1 models).
        max_completion_tokens: Upper bound for completion tokens (for o1 models).
        top_p: Controls diversity via nucleus sampling.
        n: Number of chat completion choices to generate.
        stream: Whether to stream partial responses.
        
        Tools parameters:
        tools: List of tools the model may call.
        tool_choice: Controls which function is called by the model.
        
        And various other parameters for controlling model behavior.
    """
    llm_type: str = "OpenAILLM"
    openai_key: Optional[str] = Field(default=None, description="the API key used to authenticate OpenAI requests")

    # generation parameters
    temperature: Optional[float] = Field(default=None, description="the temperature used to scaling logits")
    max_tokens : Optional[int] = Field(default=None, description="maximum number of generated tokens. This value is now deprecated in favor of max_completion_tokens, and is not compatible with o1 series models.")
    max_completion_tokens: Optional[int] = Field(default=None, description="An upper bound for the number of tokens that can be generated for a completion, including visible output tokens and reasoning tokens. Commonly used in OpenAI's o1 series models.")
    top_p: Optional[float] = Field(default=None, description="Only sample from tokens with cumulative probability greater than top_p when generating text.")
    n: Optional[int] = Field(default=None, description="How many chat completion choices to generate for each input message.")
    stream: Optional[bool] = Field(default=None, description=" If set to true, it sends partial message deltas. Tokens will be sent as they become available, with the stream terminated by a [DONE] message.")
    stream_options: Optional[dict] = Field(default=None, description="Options for streaming response. Only set this when you set stream: true")
    timeout: Optional[Union[float, int]] = Field(default=None, description="Timeout in seconds for completion requests (Defaults to 600 seconds)")

    # tools 
    tools: Optional[List] = Field(default=None, description="A list of tools the model may call. Currently, only functions are supported as a tool. Use this to provide a list of functions the model may generate JSON inputs for.")
    tool_choice: Optional[str] = Field(default=None, description="Controls which (if any) function is called by the model. none means the model will not call a function and instead generates a message. auto means the model can pick between generating a message or calling a function. Specifying a particular function via {\"type\": \"function\", \"function\": {\"name\": \"my_function\"}} forces the model to call that function.")
    parallel_tool_calls: Optional[bool] = Field(default=None, description="Whether to enable parallel function calling during tool use. OpenAI default is true.")
    
    # reasoning parameters 
    reasoning_effort: Optional[str] = Field(default=None, description="Constrains effort on reasoning for reasoning models. Currently supported values are low, medium, and high. Reducing reasoning effort can result in faster responses and fewer tokens used on reasoning in a response.")

    # token probabilities
    logprobs: Optional[bool] = Field(default=None, description="Whether to return log probabilities of the output tokens or not. If true, returns the log probabilities of each output token returned in the content of message.")
    top_logprobs: Optional[int] = Field(default=None, description="An integer between 0 and 5 specifying the number of most likely tokens to return at each token position, each with an associated log probability. logprobs must be set to true if this parameter is used.")

    # predicted outputs 
    prediction: Optional[dict] = Field(default=None, description="Configuration for a Predicted Output, which can greatly improve response times when large parts of the model response are known ahead of time. This is most common when you are regenerating a file with only minor changes to most of the content.")

    # output format
    modalities: Optional[List] = Field(default=None, description="Output types that you would like the model to generate for this request. Most models are capable of generating text, which is the default: [\"text\"]")
    response_format: Optional[Union[BaseModel, dict]] = Field(default=None, description=" An object specifying the format that the model must output.")


class LiteLLMConfig(LLMConfig):
    """Configuration for LiteLLM models.
    
    LiteLLM is a unified interface for multiple LLM providers including OpenAI,
    Anthropic, and DeepSeek. This configuration class includes parameters for
    authentication with these different providers and common generation parameters.
    
    The model parameter should be specified with the format "provider/model_name"
    (e.g., "openai/gpt-4", "anthropic/claude-2").
    
    Attributes:
        llm_type: Set to "LiteLLM" by default.
        
        LLM provider keys:
        openai_key: The API key for OpenAI models.
        anthropic_key: The API key for Anthropic models.
        deepseek_key: The API key for DeepSeek models.
        
        Generation parameters:
        temperature: Controls randomness in generation.
        max_tokens: Maximum number of tokens to generate.
        max_completion_tokens: Upper bound for completion tokens (for o1 models).
        top_p: Controls diversity via nucleus sampling.
        n: Number of chat completion choices to generate.
        stream: Whether to stream partial responses.
        
        And various other parameters for controlling model behavior.
    """
    llm_type: str = "LiteLLM"

    # LLM keys
    openai_key: Optional[str] = Field(default=None, description="the API key used to authenticate OpenAI requests")
    anthropic_key: Optional[str] = Field(default=None, description="the API key used to authenticate Anthropic requests")
    deepseek_key: Optional[str] = Field(default=None, description="the API key used to authenticate Deepseek requests")

    # generation parameters 
    temperature: Optional[float] = Field(default=None, description="the temperature used to scaling logits")
    max_tokens : Optional[int] = Field(default=None, description="maximum number of generated tokens")
    max_completion_tokens: Optional[int] = Field(default=None, description="An upper bound for the number of tokens that can be generated for a completion, including visible output tokens and reasoning tokens. Commonly used in OpenAI's o1 series models.")
    top_p: Optional[float] = Field(default=None, description="Only sample from tokens with cumulative probability greater than top_p when generating text.")
    n: Optional[int] = Field(default=None, description="How many chat completion choices to generate for each input message.")
    stream: Optional[bool] = Field(default=None, description=" If set to true, it sends partial message deltas. Tokens will be sent as they become available, with the stream terminated by a [DONE] message.")
    stream_options: Optional[dict] = Field(default=None, description="Options for streaming response. Only set this when you set stream: true")
    timeout: Optional[Union[float, int]] = Field(default=None, description="Timeout in seconds for completion requests (Defaults to 600 seconds)")

    # tools
    tools: Optional[List] = Field(default=None, description="A list of tools the model may call. Currently, only functions are supported as a tool. Use this to provide a list of functions the model may generate JSON inputs for.")
    tool_choice: Optional[str] = Field(default=None, description="Controls which (if any) function is called by the model. none means the model will not call a function and instead generates a message. auto means the model can pick between generating a message or calling a function. Specifying a particular function via {\"type\": \"function\", \"function\": {\"name\": \"my_function\"}} forces the model to call that function.")
    parallel_tool_calls: Optional[bool] = Field(default=None, description="Whether to enable parallel function calling during tool use. OpenAI default is true.")

    # token probabilities
    logprobs: Optional[bool] = Field(default=None, description="Whether to return log probabilities of the output tokens or not. If true, returns the log probabilities of each output token returned in the content of message.")
    top_logprobs: Optional[int] = Field(default=None, description="An integer between 0 and 5 specifying the number of most likely tokens to return at each token position, each with an associated log probability. logprobs must be set to true if this parameter is used.")

    # output format
    response_format: Optional[Union[BaseModel, dict]] = Field(default=None, description=" An object specifying the format that the model must output.")

    def __str__(self):
        """String representation of the config.
        
        Returns:
            The model identifier as a string.
        """
        return self.model


class SiliconFlowConfig(LLMConfig):
    """Configuration for SiliconFlow models.
    
    SiliconFlow is an LLM provider with specific configuration parameters.
    This class includes parameters for authentication and generation control.
    
    Attributes:
        llm_type: Set to "SiliconFlowLLM" by default.
        siliconflow_key: The API key for SiliconFlow authentication.
        
        Generation parameters:
        temperature: Controls randomness in generation.
        max_tokens: Maximum number of tokens to generate.
        max_completion_tokens: Upper bound for completion tokens.
        top_p: Controls diversity via nucleus sampling.
        n: Number of chat completion choices to generate.
        stream: Whether to stream partial responses.
        
        And various other parameters for controlling model behavior.
    """
    # LLM keys
    llm_type: str = "SiliconFlowLLM"
    siliconflow_key: Optional[str] = Field(default=None, description="the API key used to authenticate SiliconFlow requests") 

    # generation parameters 
    temperature: Optional[float] = Field(default=None, description="the temperature used to scaling logits")
    max_tokens : Optional[int] = Field(default=None, description="maximum number of generated tokens")
    max_completion_tokens: Optional[int] = Field(default=None, description="An upper bound for the number of tokens that can be generated for a completion, including visible output tokens and reasoning tokens. Commonly used in OpenAI's o1 series models.")
    top_p: Optional[float] = Field(default=None, description="Only sample from tokens with cumulative probability greater than top_p when generating text.")
    n: Optional[int] = Field(default=None, description="How many chat completion choices to generate for each input message.")
    stream: Optional[bool] = Field(default=None, description=" If set to true, it sends partial message deltas. Tokens will be sent as they become available, with the stream terminated by a [DONE] message.")
    stream_options: Optional[dict] = Field(default=None, description="Options for streaming response. Only set this when you set stream: true")
    timeout: Optional[Union[float, int]] = Field(default=None, description="Timeout in seconds for completion requests (Defaults to 600 seconds)")

    # tools
    tools: Optional[List] = Field(default=None, description="A list of tools the model may call. Currently, only functions are supported as a tool. Use this to provide a list of functions the model may generate JSON inputs for.")
    tool_choice: Optional[str] = Field(default=None, description="Controls which (if any) function is called by the model. none means the model will not call a function and instead generates a message. auto means the model can pick between generating a message or calling a function. Specifying a particular function via {\"type\": \"function\", \"function\": {\"name\": \"my_function\"}} forces the model to call that function.")
    parallel_tool_calls: Optional[bool] = Field(default=None, description="Whether to enable parallel function calling during tool use. OpenAI default is true.")

    # token probabilities
    logprobs: Optional[bool] = Field(default=None, description="Whether to return log probabilities of the output tokens or not. If true, returns the log probabilities of each output token returned in the content of message.")
    top_logprobs: Optional[int] = Field(default=None, description="An integer between 0 and 5 specifying the number of most likely tokens to return at each token position, each with an associated log probability. logprobs must be set to true if this parameter is used.")

    # output format
    response_format: Optional[Union[BaseModel, dict]] = Field(default=None, description=" An object specifying the format that the model must output.")

    def __str__(self):
        """String representation of the config.
        
        Returns:
            The model identifier as a string.
        """
        return self.model


# def get_default_device():
#     return "cuda" if torch.cuda.is_available() else "cpu"

