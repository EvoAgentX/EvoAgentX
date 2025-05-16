from pydantic import BaseModel, Field
from typing import Optional, Union, List

# import torch 

from ..core.base_config import BaseConfig


#### LLM Configs
class LLMConfig(BaseConfig):

    llm_type: str
    model: str = Field(default="gpt-4o-mini", description="The model to use for the LLM")
    output_response: bool = Field(default=False, description="Whether to output LLM response.")

class OpenAILLMConfig(LLMConfig):

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
        return self.model


class SiliconFlowConfig(LLMConfig):

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
        return self.model


# def get_default_device():
#     return "cuda" if torch.cuda.is_available() else "cpu"

