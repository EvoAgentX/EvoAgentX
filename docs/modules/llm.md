# LLM

## Introduction

The LLM (Large Language Model) module provides a unified interface for interacting with various language model providers in the EvoAgentX framework. It abstracts away provider-specific implementation details, offering a consistent API for generating text, managing costs, and handling responses.

## Supported LLM Providers

EvoAgentX currently supports the following LLM providers:

### OpenAILLM

The primary implementation for accessing OpenAI's language models. It handles authentication, request formatting, and response parsing for models like GPT-4, GPT-3.5-Turbo, and other OpenAI models.

**Basic Usage:**

```python
from evoagentx.models import OpenAILLMConfig, OpenAILLM

# Configure the model
config = OpenAILLMConfig(
    model="gpt-4o-mini",  
    openai_key="your-api-key",
    temperature=0.7,
    max_tokens=1000
)

# Initialize the model
llm = OpenAILLM(config=config)

# Generate text
response = llm.generate(
    prompt="Explain quantum computing in simple terms.",
    system_message="You are a helpful assistant that explains complex topics simply."
)
```

### LiteLLM

LiteLLM is an adapter for the [LiteLLM project](https://github.com/BerriAI/litellm), which provides a unified Python SDK and proxy server for calling over 100 LLM APIs using the OpenAI API format. It supports providers such as Bedrock, Azure, OpenAI, VertexAI, Cohere, Anthropic, Sagemaker, HuggingFace, Replicate, and Groq. Thanks to this project, the `LiteLLM` model class in EvoAgentX can be used to seamlessly access a wide range of LLM providers through a single interface. 

**Basic Usage:**

To faciliate seamless integration with LiteLLM, you should specify the model name using the naming convention defied in the LiteLLM platform. For example, you need to specify `anthropic/claude-3-opus-20240229` for Claude 3.0 Opus. You can find a full list of supported providers and model names in their official documentation: [https://docs.litellm.ai/docs/providers](https://docs.litellm.ai/docs/providers).


```python
from evoagentx.models import LiteLLMConfig, LiteLLM

# Configure the model
config = LiteLLMConfig(
    model="anthropic/claude-3-opus-20240229", 
    anthropic_key="your-anthropic-api-key",
    temperature=0.7,
    max_tokens=1000
)

# Initialize the model
llm = LiteLLM(config=config)

# Generate text
response = llm.generate(
    prompt="Design a system for autonomous vehicles.",
    system_message="You are an expert in autonomous systems design."
)
```

### SiliconFlowLLM

SiliconFlowLLM is an adapter for models hosted on the [SiliconFlow platform](https://www.siliconflow.com/), which offers access to both open-source and proprietary models via an OpenAI-compatible API. It enables you to integrate models like Qwen, DeepSeek, or Mixtral by specifying their names using the SiliconFlow platform's naming conventions.

Thanks to SiliconFlow's unified interface, the `SiliconFlowLLM` model class in EvoAgentX allows seamless switching between a variety of powerful LLMs hosted on SiliconFlow using the same API format.

**Basic Usage:**

```python
from evoagentx.models import SiliconFlowConfig, SiliconFlowLLM

# Configure the model
config = SiliconFlowConfig(
    model="deepseek-ai/DeepSeek-V3",
    siliconflow_key="your-siliconflow-api-key",
    temperature=0.7,
    max_tokens=1000
)

# Initialize the model
llm = SiliconFlowLLM(config=config)

# Generate text
response = llm.generate(
    prompt="Write a poem about artificial intelligence.",
    system_message="You are a creative poet."
)
```

### OpenRouterLLM

OpenRouterLLM is an adapter for the [OpenRouter platform](https://openrouter.ai/), which provides access to a wide range of language models from various providers through a unified API. It supports models from providers like Anthropic, Google, Meta, Mistral AI, and more, all accessible through a single interface.

The `OpenRouterLLM` model class in EvoAgentX enables you to easily switch between different models hosted on OpenRouter while maintaining a consistent API format. This makes it simple to experiment with different models and find the best one for your specific use case.

**Basic Usage:**

```python
from evoagentx.models import OpenRouterConfig, OpenRouterLLM

# Configure the model
config = OpenRouterConfig(
    model="openai/gpt-4o-mini",  # or any other model supported by OpenRouter
    openrouter_key="your-openrouter-api-key",
    temperature=0.7,
    max_tokens=1000
)

# Initialize the model
llm = OpenRouterLLM(config=config)

# Generate text
response = llm.generate(
    prompt="Analyze the impact of artificial intelligence on healthcare.",
    system_message="You are an AI ethics expert specializing in healthcare applications."
)
```

### local models

Right now, we support users to call local LLMs, using methods similar to the LiteLLM methods. In the future, we will support calling native models for richer functionality, but at the moment, they still have the problem of workflow format unification.

**Basic Usage:**

```python
import json
from evoagentx.models.model_configs import LiteLLMConfig
from evoagentx.models import LiteLLM
from evoagentx.workflow import WorkFlowGenerator, WorkFlowGraph, WorkFlow
from evoagentx.agents import AgentManager

# configure local Qwen3 model
config = LiteLLMConfig(
    llm_type="LiteLLM",
    model="ollama/llama3",
    api_base="http://localhost:11434",
    is_local=True,
    temperature=0.7,
    max_tokens=1000,
    output_response=True
)

# test model
llm = LiteLLM(config)
response = llm.generate(prompt="What is Agentic Workflow?")
```

## Core Functions

All LLM implementations in EvoAgentX provide a consistent set of core functions for generating text and managing the generation process.

### Generate Function

The `generate` function is the primary method for producing text with language models:

```python
def generate(
    self,
    prompt: Optional[Union[str, List[str]]] = None,
    system_message: Optional[Union[str, List[str]]] = None,
    messages: Optional[Union[List[dict],List[List[dict]]]] = None,
    parser: Optional[Type[LLMOutputParser]] = None,
    parse_mode: Optional[str] = "json", 
    parse_func: Optional[Callable] = None,
    **kwargs
) -> Union[LLMOutputParser, List[LLMOutputParser]]:
    """
    Generate text based on the prompt and optional system message.

    Args:
        prompt: Input prompt(s) to the LLM.
        system_message: System message(s) for the LLM.
        messages: Chat message(s) for the LLM, already in the required format (either `prompt` or `messages` must be provided).
        parser: Parser class to use for processing the output into a structured format.
        parse_mode: The mode to use for parsing, must be the `parse_mode` supported by the `parser`. 
        parse_func: A function to apply to the parsed output.
        **kwargs: Additional generation configuration parameters.
        
    Returns:
        For single generation: An LLMOutputParser instance.
        For batch generation: A list of LLMOutputParser instances.
    """
```

#### Inputs 

In EvoAgentX, there are several ways to provide inputs to LLMs using the `generate` function:

**Method 1: Prompt and System Message**

1. **Prompt**: The specific query or instruction for which you want a response. 

2. **System Message** (optional): Instructions that guide the model's overall behavior and role. This sets the context for how the model should respond.

Together, these components are converted into a standardized message format that the language model can understand:

```python
# Simple example with prompt and system message
response = llm.generate(
    prompt="What are three ways to improve productivity?",
    system_message="You are a productivity expert providing concise, actionable advice."
)
```

Behind the scenes, this gets converted into messages with appropriate roles:

```python
messages = [
    {"role": "system", "content": "You are a productivity expert providing concise, actionable advice."},
    {"role": "user", "content": "What are three ways to improve productivity?"}
]
```

**Method 2: Using Messages Directly**

For more complex conversations or when you need precise control over the message format, you can use the `messages` parameter directly:

```python
# Using messages directly for a multi-turn conversation
response = llm.generate(
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, who are you?"},
        {"role": "assistant", "content": "I'm an AI assistant designed to help with various tasks."},
        {"role": "user", "content": "Can you help me with programming?"}
    ]
)
```

#### Batch Generation 

For batch processing, you can provide lists of prompts/system messages or list of messages. For example: 

```python
# Batch processing example
responses = llm.generate(
    prompt=["What is machine learning?", "Explain neural networks."],
    system_message=["You are a data scientist.", "You are an AI researcher."]
)
```

#### Output Parsing

The `generate` function provides flexible options for parsing and structuring the raw text output from language models:

- **parser**: Accepts a class (typically inheriting from `LLMOutputParser/ActionOutput`) that defines the structure for the parsed output. If not provided, the LLM output will not be parsed. In both cases, the raw LLM output can be accessed through the `.content` attribute of the returned object.   
- **parse_mode**: Determines how the raw LLM output is parsed into the structure defined by the parser, valid options are: `'str'`, `'json'`
