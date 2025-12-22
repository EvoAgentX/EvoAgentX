# 提示模板

## 简介

`PromptTemplate` 类提供了一种灵活且结构化的方式来定义语言模型的提示。它支持各种组件，如指令、上下文、约束、工具和示例，使创建一致且格式良好的提示变得更加容易。

## 主要特性

- **结构化提示组件**：使用清晰的部分定义提示，包括指令、上下文、约束等
- **灵活的输出格式**：支持多种输出解析模式（JSON、XML、标题格式）
- **少样本学习支持**：易于集成示例进行少样本学习
- **输入/输出验证**：自动验证所需的输入和输出
- **聊天格式支持**：通过 `ChatTemplate` 特别支持基于聊天的交互

## 基本用法

### 简单模板

创建 `PromptTemplate` 的最简单方法是只使用指令：

```python
from evoagentx.prompts import StringTemplate

template = StringTemplate(
    instruction="Write a function that calculates the factorial of a number"
)

# 将模板格式化为提示字符串
prompt = template.format()
```

### 带有上下文和约束的模板

您可以添加上下文和约束来提供更多指导：

```python
template = StringTemplate(
    instruction="Write a function that calculates the factorial of a number",
    context="The factorial of a number n is the product of all positive integers less than or equal to n",
    constraints=[
        "Use recursion to implement",
        "Include input validation",
        "Add a docstring with examples"
    ]
)

prompt = template.format()
```

### 带有示例的模板

您可以添加示例进行少样本学习：

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
    values={"requirement": "Write a function that returns the factorial of a number"}, 
    inputs_format=InputFormat,
    outputs_format=OutputFormat,
)
```

!!! note 
    使用 `demonstrations` 时需要 `inputs_format` 和 `outputs_format` 来正确映射示例中的输入和输出。

### 结构化输出格式

默认情况下，模板会根据 `outputs_format` 和 `parse_mode` 自动生成输出格式。

#### 标题格式（默认）

```python
template = StringTemplate(
    instruction="Analyze the given text",
    inputs_format=TextAnalysisInput, # A Pydantic model with text field
    outputs_format=TextAnalysisOutput,  # A Pydantic model with summary and sentiment fields
    parse_mode="title"
)
```
上述 `template` 将生成如下格式的输出：
```
## summary
[摘要内容]

## sentiment
[情感分析]
```

#### JSON 格式

```python
template = StringTemplate(
    instruction="Analyze the given text",
    inputs_format=TextAnalysisInput,
    outputs_format=TextAnalysisOutput, 
    parse_mode="json"
)
```
上述 `template` 将生成如下格式的输出：
```
{
    "summary": "[摘要内容]",
    "sentiment": "[情感分析]"
}
```

#### XML 格式

```python
template = StringTemplate(
    instruction="Analyze the given text",
    inputs_format=TextAnalysisInput,
    outputs_format=TextAnalysisOutput,
    parse_mode="xml"
)
```
上述 `template` 将生成如下格式的输出：
```
<summary>
[摘要内容]
</summary>
<sentiment>
[情感分析]
</sentiment>
```

!!! note
    1. 对于 `parse_mode="str" 或 "custom"`，模型将遵循指令生成响应。

    2. 您可以通过设置 `template.format(custom_output_format=...)` 来覆盖输出格式，请参见[自定义输出格式](#自定义输出格式)。

## 聊天模板

对于基于聊天的交互，您可以使用 `ChatTemplate` 类：

```python
from evoagentx.prompts import ChatTemplate

template = ChatTemplate(
    instruction="You are a helpful programming assistant",
    context="You help users write Python code",
    constraints=["Always include comments", "Follow PEP 8 style guide"]
)

# format 将返回聊天消息列表
messages = template.format(
    inputs_format=CodeInputs,
    outputs_format=CodeOutputs,
    values={"requirement": "Write a sorting function"}
)
```

格式化的输出将是一个适合基于聊天的模型的消息列表：

```python
[
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
]
```

## 最佳实践

1. **清晰的指令**：使您的指令具体且明确
2. **相关的上下文**：仅包含与任务直接相关的上下文
3. **具体的约束**：列出有意义地指导输出的约束
4. **代表性的示例**：选择涵盖不同情况的示例
5. **适当的输出格式**：选择最适合您需求的解析模式

## 高级功能

### 自定义输出格式

您可以指定自定义输出格式：

```python
template = StringTemplate(
    instruction="Generate code documentation"
)

prompt = template.format(
    values={"code": "..."},
    custom_output_format="""
    Please provide your response in the following format:

    # Usage
    [Code usage examples]

    # API
    [API documentation]

    # Notes
    [Additional notes and warnings]
    """
)
``` 