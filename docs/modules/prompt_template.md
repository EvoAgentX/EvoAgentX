# PromptTemplate

## Introduction

The `PromptTemplate` class provides a flexible and structured way to define prompts for language models. It supports various components like instructions, context, constraints, tools, and demonstrations, making it easier to create consistent and well-formatted prompts.

## Key Features

- **Structured Prompt Components**: Define prompts with clear sections for instruction, context, constraints, and more
- **Flexible Output Formats**: Support for multiple output parsing modes (JSON, XML, Title format)
- **Few-Shot Learning Support**: Easy integration of demonstrations for few-shot learning
- **Input/Output Validation**: Automatic validation of required inputs and outputs
- **Chat Format Support**: Special support for chat-based interactions via `ChatTemplate`

## Basic Usage

### Simple Template

The simplest way to create a `PromptTemplate` is with just an instruction:

```python
from evoagentx.prompts import StringTemplate

template = StringTemplate(
    instruction="Write a function that calculates the factorial of a number"
)

# Format the template into a prompt string
prompt = template.format()
```

### Template with Context and Constraints

You can add context and constraints to provide more guidance:

```python
template = StringTemplate(
    instruction="Write a function that calculates the factorial of a number",
    context="The factorial of a number n is the product of all positive integers less than or equal to n",
    constraints=[
        "Use a recursive implementation",
        "Include input validation",
        "Add docstring with examples"
    ]
)

prompt = template.format()
```

### Template with Demonstrations

You can add examples for few-shot learning:

```python
from evoagentx.models import OpenAILLMConfig
from evoagentx.actions import ActionInput, ActionOutput

template = StringTemplate(
    instruction="Write a function that implements the provided requirement",
    demonstrations=[
        {
            "requirement": "Write a function that returns the sum of two numbers",
            "code": "def add_numbers(a: int, b: int) -> int:\n    return a + b"
        },
        {
            "requirement": "Write a function that checks if a number is even",
            "code": "def is_even(n: int) -> bool:\n    return n % 2 == 0"
        }
    ]
)

class InputFormat(ActionInput):
    requirement: str

class OutputFormat(ActionOutput):
    code: str

prompt = template.format(
    values={"requirement": "Write a function that return the factorial of a number"}, 
    inputs_format=InputFormat,
    outputs_format=OutputFormat,
)
```

!!! note 
    `inputs_format` and `outputs_format` are required when using `demonstrations` to correctly map the inputs and outputs to the demonstrations. 

### Structured Output Formats

By default, the template automatically generate the output format based on the `outputs_format` and `parse_mode`. 

#### Title Format (Default)

```python
template = StringTemplate(
    instruction="Analyze the given text",
    inputs_format=TextAnalysisInput, # A Pydantic model with text field
    outputs_format=TextAnalysisOutput,  # A Pydantic model with summary and sentiment fields
    parse_mode="title"
)
```
The above `template` will generate output in format:
```
## summary
[Summary content]

## sentiment
[Sentiment analysis]
```

#### JSON Format

```python
template = StringTemplate(
    instruction="Analyze the given text",
    inputs_format=TextAnalysisInput,
    outputs_format=TextAnalysisOutput, 
    parse_mode="json"
)
```
The above `template` will generate output in format:
```
{
    "summary": "[Summary content]",
    "sentiment": "[Sentiment analysis]"
}
```

#### XML Format

```python
template = StringTemplate(
    instruction="Analyze the given text",
    inputs_format=TextAnalysisInput,
    outputs_format=TextAnalysisOutput,
    parse_mode="xml"
)
```
The above `template` will generate output in format:
```
<summary>
[Summary content]
</summary>
<sentiment>
[Sentiment analysis]
</sentiment>
```

!!! note
    1. For `parse_mode="str" or "custom"`, the model will follow the instruction to generate the response. 

    2. You can override the output format by setting `template.format(custom_output_format=...)`, see [Custom Output Format](#custom-output-format). 


## Chat Template

For chat-based interactions, you can use the `ChatTemplate` class:

```python
from evoagentx.prompts import ChatTemplate

template = ChatTemplate(
    instruction="You are a helpful coding assistant",
    context="You help users write Python code",
    constraints=["Always include comments", "Follow PEP 8 style guide"]
)

# Format will return a list of chat messages
messages = template.format(
    inputs_format=CodeInputs,
    outputs_format=CodeOutputs,
    values={"requirement": "Write a sorting function"}
)
```

The formatted output will be a list of messages suitable for chat-based models:

```python
[
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
]
```

## Best Practices

1. **Clear Instructions**: Make your instruction specific and unambiguous
2. **Relevant Context**: Include only context that's directly relevant to the task
3. **Specific Constraints**: List constraints that meaningfully guide the output
4. **Representative Demonstrations**: Choose examples that cover different cases
5. **Appropriate Output Format**: Choose the parsing mode that best fits your needs

## Advanced Features

### Custom Output Format

You can specify a custom output format:

```python
template = StringTemplate(
    instruction="Generate code documentation"
)

prompt = template.format(
    values={"code": "..."},
    custom_output_format="""
    Please provide your response in the following format:

    # USAGE
    [Code usage examples]

    # API
    [API documentation]

    # NOTES
    [Additional notes and warnings]
    """
)
```