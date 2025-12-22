# 自定义代理

## 简介

`CustomizeAgent` 类提供了一个灵活的框架，用于创建专门的 LLM 驱动的代理。它允许定义具有明确定义的输入、输出、自定义提示模板和可配置解析策略的代理，使其适合快速原型设计和部署特定领域的代理。

## 主要特性

- **无需自定义代码**：通过配置而不是编写自定义代理类来创建专门的代理
- **灵活的输入/输出定义**：明确定义代理接受的输入和产生的输出
- **可自定义的解析策略**：多种解析模式，用于从 LLM 响应中提取结构化数据
- **可重用组件**：保存和加载代理定义，以便在项目之间重用

## 基本用法

### 简单代理

创建 `CustomizeAgent` 的最简单方法是只使用名称、描述和提示：

```python
import os 
from dotenv import load_dotenv
from evoagentx.models import OpenAILLMConfig
from evoagentx.agents import CustomizeAgent

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 配置 LLM
openai_config = OpenAILLMConfig(model="gpt-4o-mini", openai_key=OPENAI_API_KEY)

# 创建一个简单代理
simple_agent = CustomizeAgent(
    name="SimpleAgent",
    description="A basic agent that responds to queries",
    prompt="Answer the following question: {question}",
    llm_config=openai_config,
    inputs=[
        {"name": "question", "type": "string", "description": "The question to answer"}
    ]
)

# 执行代理
response = simple_agent(inputs={"question": "What is a language model?"})
print(response.content.content)  # 访问原始响应内容
```

在这个例子中：
1. 由于提示需要输入，我们在 `inputs` 参数中指定了输入信息（包括其名称、类型和描述）。
2. 此外，当使用 `simple_agent(...)` 执行代理时，您应该在 `inputs` 参数中提供所有输入。

执行代理后的输出是一个 `Message` 对象，其中包含 `message.content.content` 中的原始 LLM 响应。

!!! note
    在 `CustomizeAgent(inputs=[...])` 中指定的所有输入名称都应该出现在 `prompt` 中。否则，将引发错误。

### 结构化输出

`CustomizeAgent` 最强大的功能之一是能够定义结构化输出。这允许您将非结构化的 LLM 响应转换为更容易以编程方式处理的明确定义的数据结构。

#### 基本结构化输出

以下是定义结构化输出的简单示例：

```python
from evoagentx.core.module_utils import extract_code_blocks

# 创建一个具有结构化输出的代码编写代理
code_writer = CustomizeAgent(
    name="CodeWriter",
    description="Writes Python code based on requirements",
    prompt="Write Python code that implements the following requirement: {requirement}",
    llm_config=openai_config,
    inputs=[
        {"name": "requirement", "type": "string", "description": "The coding requirement"}
    ],
    outputs=[
        {"name": "code", "type": "string", "description": "The generated Python code"}
    ],
    parse_mode="custom",  # 使用自定义解析函数
    parse_func=lambda content: {"code": extract_code_blocks(content)[0]}  # 提取第一个代码块
)

# 执行代理
message = code_writer(
    inputs={"requirement": "Write a function that returns the sum of two numbers"}
)
print(message.content.code)  # 直接访问解析后的代码
```

在这个例子中：
1. 我们在 `outputs` 参数中定义了一个名为 `code` 的输出字段。
2. 我们设置 `parse_mode="custom"` 来使用自定义解析函数。
3. `parse_func` 从 LLM 响应中提取第一个代码块。
4. 我们可以通过 `message.content.code` 直接访问解析后的代码。

您也可以通过 `message.content.content` 访问原始 LLM 响应。

!!! note 
    1. 如果在 `CustomizeAgent` 中设置了 `outputs` 参数，代理将尝试根据输出字段名称解析 LLM 响应。如果您不想解析 LLM 响应，则不应设置 `outputs` 参数。可以通过 `message.content.content` 访问原始 LLM 响应。

    2. CustomizeAgent 支持不同的解析模式，如 `['str', 'json', 'xml', 'title', 'custom']`。有关更多详细信息，请参阅[解析模式](#parsing-modes)部分。

#### 多个结构化输出

您可以定义多个输出字段来创建更复杂的结构化数据：

```python
# 生成代码和解释的代理
analyzer = CustomizeAgent(
    name="CodeAnalyzer",
    description="Generates and explains Python code",
    prompt="""
    Write Python code for: {requirement}
    
    Provide your response in the following format:
    
    ## code
    [Your code implementation here]
    
    ## explanation
    [A brief explanation of how the code works]
    
    ## complexity
    [Time and space complexity analysis]
    """,
    llm_config=openai_config,
    inputs=[
        {"name": "requirement", "type": "string", "description": "The coding requirement"}
    ],
    outputs=[
        {"name": "code", "type": "string", "description": "The generated Python code"},
        {"name": "explanation", "type": "string", "description": "Explanation of the code"},
        {"name": "complexity", "type": "string", "description": "Complexity analysis"}
    ],
    parse_mode="title"  # 使用默认的标题解析模式
)

# 执行代理
result = analyzer(inputs={"requirement": "Write a binary search algorithm"})

# 分别访问每个结构化输出
print("CODE:")
print(result.content.code)
print("\nEXPLANATION:")
print(result.content.explanation)
print("\nCOMPLEXITY:")
print(result.content.complexity)
```


## 提示模板用法

`CustomizeAgent` 还支持使用 `PromptTemplate` 进行更灵活的提示模板设计。有关提示模板及其高级功能的详细信息，请参阅[提示模板教程](./prompt_template.md)。

### 简单提示模板

以下是使用提示模板的基本示例：

```python
from evoagentx.prompts import StringTemplate

agent = CustomizeAgent(
    name="FirstAgent",
    description="A simple agent that prints hello world",
    prompt_template=StringTemplate(
        instruction="Print 'hello world'",
    ),
    llm_config=openai_config
)

message = agent()
print(message.content.content)
```

### 带有输入和输出的提示模板

您可以将提示模板与结构化输入和输出结合使用：

```python
code_writer = CustomizeAgent(
    name="CodeWriter",
    description="Writes Python code based on requirements",
    prompt_template=StringTemplate(
        instruction="Write Python code that implements the provided `requirement`",
        # 您可以选择添加示例：
        # demonstrations=[
        #     {
        #         "requirement": "Print 'hello world'",
        #         "code": "print('hello world')"
        #     }, 
        #     {
        #         "requirement": "Print 'Test Demonstration'",
        #         "code": "print('Test Demonstration')"
        #     }
        # ]
    ), # 不需要在提示模板的指令中明确指定输入占位符
    llm_config=openai_config,
    inputs=[
        {"name": "requirement", "type": "string", "description": "Coding requirement"}
    ],
    outputs=[
        {"name": "code", "type": "string", "description": "Generated Python code"},
    ],
    parse_mode="custom", 
    parse_func=lambda content: {"code": extract_code_blocks(content)[0]}
)

message = code_writer(
    inputs={"requirement": "Write a function that returns the sum of two numbers"}
)
print(message.content.code)
```

`PromptTemplate` 提供了一种更结构化的方式来定义提示，可以包括：
- 主要指令
- 可选的上下文，用于提供额外信息
- 可选的约束，LLM 应该遵循
- 可选的示例，用于少样本学习
- 可选的工具信息，LLM 可以使用
等。

!!! note
    1. 使用 `prompt_template` 时，您不需要在指令字符串中明确包含输入占位符，如 `{input_name}`。模板将自动处理输入的映射。

    2. 此外，您不需要在 `PromptTemplate` 的 `instruction` 字段中明确指定输出格式。模板将根据 `outputs` 参数和 `parse_mode` 参数自动制定输出格式。但是，`PromptTemplate` 也支持通过指定 `PromptTemplate.format(custom_output_format="...")` 来明确指定输出格式。

## 解析模式

CustomizeAgent 支持不同的方式来解析 LLM 输出：

### 1. 字符串模式 (`parse_mode="str"`)

使用原始 LLM 输出作为每个输出字段的值。适用于不需要结构化解析的简单代理。

```python
agent = CustomizeAgent(
    name="SimpleAgent",
    description="Returns raw output",
    prompt="Generate a greeting for {name}",
    inputs=[{"name": "name", "type": "string", "description": "The name to greet"}],
    outputs=[{"name": "greeting", "type": "string", "description": "The generated greeting"}],
    parse_mode="str",
    # 其他参数...
)
```

执行代理后，您可以通过 `message.content.content` 或 `message.content.greeting` 访问原始 LLM 响应。

### 2. 标题模式 (`parse_mode="title"`，默认)

提取与输出字段名称匹配的标题之间的内容。这是默认的解析模式。

```python
agent = CustomizeAgent(
    name="ReportGenerator",
    description="Generates a structured report",
    prompt="Create a report about {topic}",
    outputs=[
        {"name": "summary", "type": "string", "description": "Brief summary"},
        {"name": "analysis", "type": "string", "description": "Detailed analysis"}
    ],
    # 默认标题模式是 "## {title}"
    title_format="### {title}",  # 可选：自定义标题格式
    # 其他参数...
)
```

使用此配置，应该指示 LLM 将其响应格式化为：

```
### summary
Brief summary of the topic here.

### analysis
Detailed analysis of the topic here.
```

!!! note
    LLM 输出的章节标题应该与输出字段名称完全相同。否则，解析将失败。例如，在上面的例子中，如果 LLM 输出 `### Analysis`，这与输出字段名称 `analysis` 不同，解析将失败。

### 3. JSON 模式 (`parse_mode="json"`)

解析 LLM 输出的 JSON 字符串。JSON 字符串的键应该与输出字段名称完全相同。

```python
agent = CustomizeAgent(
    name="DataExtractor",
    description="Extracts structured data",
    prompt="Extract key information from this text: {text}",
    inputs=[
        {"name": "text", "type": "string", "description": "The text to extract information from"}
    ],
    outputs=[
        {"name": "people", "type": "string", "description": "Names of people mentioned"},
        {"name": "places", "type": "string", "description": "Locations mentioned"},
        {"name": "dates", "type": "string", "description": "Dates mentioned"}
    ],
    parse_mode="json",
    # 其他参数...
)
```

使用此模式时，LLM 应该输出一个有效的 JSON 字符串，其键与输出字段名称匹配。例如，您应该指示 LLM 输出：

```json
{
    "people": "extracted people",
    "places": "extracted places",
    "dates": "extracted dates"
}
```

如果 LLM 响应中有多个 JSON 字符串，将只使用第一个。

### 4. XML 模式 (`parse_mode="xml"`)

解析 LLM 输出的 XML 字符串。XML 字符串的键应该与输出字段名称完全相同。

```python
agent = CustomizeAgent(
    name="DataExtractor",
    description="Extracts structured data",
    prompt="Extract key information from this text: {text}",
    inputs=[
        {"name": "text", "type": "string", "description": "The text to extract information from"}
    ],
    outputs=[
        {"name": "people", "type": "string", "description": "Names of people mentioned"},
    ],
    parse_mode="xml",
    # 其他参数...
)
```

使用此模式时，LLM 应该生成包含与输出字段名称匹配的 XML 标签的文本。例如，您应该指示 LLM 输出：

```xml
The people mentioned in the text are: <people>John Doe and Jane Smith</people>.
```

如果 LLM 输出包含多个具有相同名称的 XML 标签，将只使用第一个。

### 5. 自定义解析 (`parse_mode="custom"`)

为了获得最大的灵活性，您可以定义自定义解析函数：

```python
from evoagentx.core.registry import register_parse_function

@register_parse_function  # 注册函数以便序列化
def extract_python_code(content: str) -> dict:
    """从 LLM 响应中提取 Python 代码"""
    code_blocks = extract_code_blocks(content)
    return {"code": code_blocks[0] if code_blocks else ""}

agent = CustomizeAgent(
    name="CodeExplainer",
    description="Generates and explains code",
    prompt="Write a Python function that {requirement}",
    inputs=[
        {"name": "requirement", "type": "string", "description": "The requirement to generate code for"}
    ],
    outputs=[
        {"name": "code", "type": "string", "description": "The generated code"},
    ],
    parse_mode="custom",
    parse_func=extract_python_code,
    # 其他参数...
)
```

!!! note 
    1. 解析函数应该有一个输入参数 `content`，它接收原始 LLM 响应作为输入，并返回一个字典，其键与输出字段名称匹配。

    2. 建议使用 `@register_parse_function` 装饰器来注册解析函数以便序列化，这样您就可以保存代理并在以后加载它。

## 保存和加载代理

您可以保存代理定义以便以后重用：

```python
# 保存代理配置。默认情况下，`llm_config` 不会被保存。
code_writer.save_module("./agents/code_writer.json")

# 从文件加载代理（需要再次提供 llm_config）
loaded_agent = CustomizeAgent.from_file(
    "./agents/code_writer.json", 
    llm_config=openai_config
)
```

## 高级示例：多步骤代码生成器

以下是一个更高级的示例，展示了如何创建一个具有多个结构化输出的专门代码生成代理：

```python
from pydantic import Field
from evoagentx.actions import ActionOutput
from evoagentx.core.registry import register_parse_function

class CodeGeneratorOutput(ActionOutput):
    code: str = Field(description="The generated Python code")
    documentation: str = Field(description="Documentation for the code")
    tests: str = Field(description="Unit tests for the code")

@register_parse_function
def parse_code_documentation_tests(content: str) -> dict:
    """将 LLM 输出解析为代码、文档和测试部分"""
    sections = content.split("## ")
    result = {"code": "", "documentation": "", "tests": ""}
    
    for section in sections:
        if not section.strip():
            continue
        
        lines = section.strip().split("\n")
        section_name = lines[0].lower()
        section_content = "\n".join(lines[1:]).strip()
        
        if "code" in section_name:
            # 从代码块中提取代码
            code_blocks = extract_code_blocks(section_content)
            result["code"] = code_blocks[0] if code_blocks else section_content
        elif "documentation" in section_name:
            result["documentation"] = section_content
        elif "test" in section_name:
            # 如果存在，从代码块中提取代码
            code_blocks = extract_code_blocks(section_content)
            result["tests"] = code_blocks[0] if code_blocks else section_content
    
    return result

# 创建高级代码生成器代理
advanced_generator = CustomizeAgent(
    name="AdvancedCodeGenerator",
    description="Generates complete code packages with documentation and tests",
    prompt="""
    Create a complete implementation based on this requirement:
    {requirement}
    
    Provide your response in the following format:
    
    ## Code
    [Include the Python code implementation here]
    
    ## Documentation
    [Include clear documentation explaining the code]
    
    ## Tests
    [Include unit tests that verify the code works correctly]
    """,
    llm_config=openai_config,
    inputs=[
        {"name": "requirement", "type": "string", "description": "The coding requirement"}
    ],
    outputs=[
        {"name": "code", "type": "string", "description": "The generated Python code"},
        {"name": "documentation", "type": "string", "description": "Documentation for the code"},
        {"name": "tests", "type": "string", "description": "Unit tests for the code"}
    ],
    output_parser=CodeGeneratorOutput,
    parse_mode="custom",
    parse_func=parse_code_documentation_tests,
    system_prompt="You are an expert Python developer specialized in writing clean, efficient code with comprehensive documentation and tests."
)

# 执行代理
result = advanced_generator(
    inputs={
        "requirement": "Create a function to validate if a string is a valid email address"
    }
)

# 访问结构化输出
print("CODE:")
print(result.content.code)
print("\nDOCUMENTATION:")
print(result.content.documentation)
print("\nTESTS:")
print(result.content.tests)
```

这个高级示例展示了如何创建一个专门的代理，它可以从单个 LLM 调用中产生多个结构化输出，提供包含实现、文档和测试的完整代码包。