# CustomizeAgent

## Introduction

The `CustomizeAgent` class provides a flexible framework for creating specialized LLM-powered agents. It enables the definition of agents with well-defined inputs, outputs, custom prompt templates, and configurable parsing strategies, making it suitable for rapid prototyping and deployment of domain-specific agents.

## Key Features

- **No Custom Code Required**: Create specialized agents through configuration rather than writing custom agent classes
- **Flexible Input/Output Definitions**: Define exactly what inputs your agent accepts and what outputs it produces
- **Customizable Parsing Strategies**: Multiple parsing modes to extract structured data from LLM responses
- **Reusable Components**: Save and load agent definitions for reuse across projects

## Basic Usage


### Simple Agent

The simplest way to create a `CustomizeAgent` is with just a name, description and prompt:

```python
import os 
from dotenv import load_dotenv
from evoagentx.models import OpenAILLMConfig
from evoagentx.agents import CustomizeAgent

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Configure LLM
openai_config = OpenAILLMConfig(model="gpt-4o-mini", openai_key=OPENAI_API_KEY)

# Create a simple agent
simple_agent = CustomizeAgent(
    name="SimpleAgent",
    description="A basic agent that responds to queries",
    prompt="Answer the following question: {question}",
    llm_config=openai_config,
    inputs=[
        {"name": "question", "type": "string", "description": "The question to answer"}
    ]
)

# Execute the agent
response = simple_agent(inputs={"question": "What is a language model?"})
print(response.content.content)  # Access the raw response content
```
In this example, 1. We specify the input information (including its name, type, and description) in the `inputs` parameter since the prompt requires an input. 2. Moreover, when executing the agent with `simple_agent(...)`, you should provide all the inputs in the `inputs` parameter. 

The output after executing the agent is a `Message` object, which contains the raw LLM response in `message.content.content`. 

!!! note
All the input names specified in the `CustomizeAgent(inputs=[...])` should appear in the `prompt`. Otherwise, an error will be raised.


### Structured Outputs 

One of the most powerful features of `CustomizeAgent` is the ability to define structured outputs. This allows you to transform unstructured LLM responses into well-defined data structures that are easier to work with programmatically.

#### Basic Structured Output

Here's a simple example of defining structured outputs:

```python
from evoagentx.core.module_utils import extract_code_blocks

# Create a CodeWriter agent with structured output
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
    parse_mode="custom",  # Use custom parsing function
    parse_func=lambda content: {"code": extract_code_blocks(content)[0]}  # Extract first code block
)

# Execute the agent
message = code_writer(
    inputs={"requirement": "Write a function that returns the sum of two numbers"}
)
print(message.content.code)  # Access the parsed code directly
```

In this example:
1. We define an output field named `code` in the `outputs` parameter.
2. We set `parse_mode="custom"` to use a custom parsing function.
3. The `parse_func` extracts the first code block from the LLM response.
4. We can directly access the parsed code with `message.content.code`.

You can also access the raw LLM response by `message.content.content`. 

!!! note 
1. If the `outputs` parameter is set in `CustomizeAgent`, the agent will try to parse the LLM response based on the output field names. If you don't want to parse the LLM response, you should not set the `outputs` parameter. The raw LLM response can be accessed by `message.content.content`. 

2. CustomizeAgent supports different parsing modes, such as `['str', 'json', 'xml', 'title', 'custom']. Please refer to the [Parsing Modes](#parsing-modes) section for more details. 

#### Multiple Structured Outputs

You can define multiple output fields to create more complex structured data:

```python
# Agent that generates both code and explanation
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
    parse_mode="title"  # Use default title parsing mode
)

# Execute the agent
result = analyzer(inputs={"requirement": "Write a binary search algorithm"})

# Access each structured output separately
print("CODE:")
print(result.content.code)
print("\nEXPLANATION:")
print(result.content.explanation)
print("\nCOMPLEXITY:")
print(result.content.complexity)
```

## Prompt Template Usage

The `CustomizeAgent` also supports using `PromptTemplate` for more flexible prompt templating. For detailed information about prompt templates and their advanced features, please refer to the [PromptTemplate Tutorial](./prompt_template.md).

### Simple Prompt Template

Here's a basic example using a prompt template:

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

### Prompt Template with Inputs and Outputs

You can combine prompt templates with structured inputs and outputs:

```python
code_writer = CustomizeAgent(
    name="CodeWriter",
    description="Writes Python code based on requirements",
    prompt_template=StringTemplate(
        instruction="Write Python code that implements the provided `requirement`",
        # You can optionally add demonstrations:
        # demonstrations=[
        #     {
        #         "requirement": "print 'hello world'",
        #         "code": "print('hello world')"
        #     }, 
        #     {
        #         "requirement": "print 'Test Demonstration'",
        #         "code": "print('Test Demonstration')"
        #     }
        # ]
    ), # no need to specify input placeholders in the instruction of the prompt template
    llm_config=openai_config,
    inputs=[
        {"name": "requirement", "type": "string", "description": "The coding requirement"}
    ],
    outputs=[
        {"name": "code", "type": "string", "description": "The generated Python code"},
    ],
    parse_mode="custom", 
    parse_func=lambda content: {"code": extract_code_blocks(content)[0]}
)

message = code_writer(
    inputs={"requirement": "Write a function that returns the sum of two numbers"}
)
print(message.content.code)
```

The `PromptTemplate` provides a more structured way to define prompts and can include:
- A main instruction
- Optional context that can be used to provide additional information
- Optional constraints that the LLM should follow 
- Optional demonstrations for few-shot learning
- Optional tools information that the LLM can use 
etc. 

!!! note
1. When using `prompt_template`, you don't need to explicitly include input placeholders in the instruction string like `{input_name}`. The template will automatically handle the mapping of inputs. 

2. Also, you don't need to explicitly specify the output format in the `instruction` field of the `PromptTemplate`. The template will automatically formulate the output format based on the `outputs` parameter and the `parse_mode` parameter. However, `PromptTemplate` also supports explicitly specifying the output format by specifying `PromptTemplate.format(custom_output_format="...")`. 


## Tools

You can extend a CustomizeAgent with external capabilities by attaching toolkits through the tools parameter. In practice, this means you can equip the agent with different toolkits depending on your use case.

Below we show two examples:File toolkit & MCP toolkit.

### File toolkit

The FileToolkit enables an agent to write and read files on disk. 
This is useful when you want the agent to persist its outputs (for example, generating code or reports).

```python
from evoagentx.agents import CustomizeAgent
from evoagentx.prompts import StringTemplate
from evoagentx.tools.file_tool import FileToolkit

agent = CustomizeAgent(
    name="CodeWriter",
    description="Writes Python code based on requirements",
    prompt_template=StringTemplate(
        instruction="Write Python code that implements the provided `requirement` and save it to the given `file_path`."
    ),
    llm_config=openai_config,
    inputs=[
        {"name": "requirement", "type": "string", "description": "The coding requirement"},
        {"name": "file_path", "type": "string", "description": "The path to save the code"}
    ],
    tools=[FileToolkit()]
)

message = agent(
    inputs={
        "requirement": "Write a function that returns the sum of two numbers",
        "file_path": "examples/output/test_code.py"
    }
)
print(message.content.content)
```

### MCP toolkit

The MCPToolkit loads tools defined in an MCP configuration file.
This allows the agent to dynamically access a set of tools exposed by MCP servers.

```python
from evoagentx.agents import CustomizeAgent
from evoagentx.prompts import StringTemplate
from evoagentx.tools.mcp import MCPToolkit

mcp_toolkit = MCPToolkit(config_path="examples/tools/sample_mcp.config")
tools = mcp_toolkit.get_toolkits()

agent = CustomizeAgent(
    name="MCPToolUser",
    description="Performs tasks using MCP tools",
    prompt_template=StringTemplate(
        instruction="Do some tasks using the tools"
    ),
    llm_config=openai_config,
    inputs=[
        {"name": "instruction", "type": "string", "description": "The instruction for the tool user"}
    ],
    outputs=[
        {"name": "result", "type": "string", "description": "The result of the task"},
        {"name": "tool_calls", "type": "string", "description": "The tool calls used to get the result (if any)"}
    ],
    tools=tools
)

message = agent(inputs={"instruction": "Summarize all your tools."})
print(message.content)
```

!!! note
- Multiple toolkits can be attached at the same time.
- If outputs are not defined, the raw model text is available at message.content.content.
- If outputs are defined, the message object will include fields corresponding to the defined outputs..


## Parsing Modes

CustomizeAgent supports different ways to parse the LLM output:

### 1. String Mode (`parse_mode="str"`)

Uses the raw LLM output as the value for each output field. Useful for simple agents where structured parsing isn't needed.

```python
agent = CustomizeAgent(
    name="SimpleAgent",
    description="Returns raw output",
    prompt="Generate a greeting for {name}",
    inputs=[{"name": "name", "type": "string", "description": "The name to greet"}],
    outputs=[{"name": "greeting", "type": "string", "description": "The generated greeting"}],
    parse_mode="str",
    # other parameters...
)
```

After executing the agent, you can access the raw LLM response by `message.content.content` or `message.content.greeting`.  

### 2. Title Mode (`parse_mode="title"`, default)

Extracts content between titles matching output field names. This is the default parsing mode.

```python
agent = CustomizeAgent(
    name="ReportGenerator",
    description="Generates a structured report",
    prompt="Create a report about {topic}",
    outputs=[
        {"name": "summary", "type": "string", "description": "Brief summary"},
        {"name": "analysis", "type": "string", "description": "Detailed analysis"}
    ],
    # Default title pattern is "## {title}"
    title_format="### {title}",  # Optional: customize title format
    # other parameters...
)
```
With this configuration, the LLM should be instructed to format its response like (only required when passing the complete prompt with `prompt` parameter to instantiate the `CustomizeAgent`. If using `prompt_template`, you don't need to specify this):

```
### summary
Brief summary of the topic here.

### analysis
Detailed analysis of the topic here.
```

!!! note
The section titles output by the LLM should be exactly the same as the output field names. Otherwise, the parsing will fail. For instance, in above example, if the LLM outputs `### Analysis`, which is different from the output field name `analysis`, the parsing will fail. 

### 3. JSON Mode (`parse_mode="json"`)

Parse the JSON string output by the LLM. The keys of the JSON string should be exactly the same as the output field names. 

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
    # other parameters...
)
```
When using this mode, the LLM should output a valid JSON string with keys matching the output field names. For instance, you should instruct the LLM to output (only required when passing the complete prompt with `prompt` parameter to instantiate the `CustomizeAgent`. If using `prompt_template`, you don't need to specify this):

```json
{
    "people": "extracted people",
    "places": "extracted places",
    "dates": "extracted dates"
}
```
If there are multiple JSON string in the LLM response, only the first one will be used. 

### 4. XML Mode (`parse_mode="xml"`)

Parse the XML string output by the LLM. The keys of the XML string should be exactly the same as the output field names.  

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
    # other parameters...
)
```

When using this mode, the LLM should generte texts containing xml tags with keys matching the output field names. For instance, you should instruct the LLM to output (only required when passing the complete prompt with `prompt` parameter to instantiate the `CustomizeAgent`. If using `prompt_template`, you don't need to specify this):

```xml
The people mentioned in the text are: <people>John Doe and Jane Smith</people>.
```

If the LLM output contains multiple xml tags with the same name, only the first one will be used. 

### 5. Custom Parsing (`parse_mode="custom"`)

For maximum flexibility, you can define a custom parsing function:

```python
from evoagentx.core.registry import register_parse_function

@register_parse_function  # Register the function for serialization
def extract_python_code(content: str) -> dict:
    """Extract Python code from LLM response"""
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
    # other parameters...
)
```

!!! note 
1. The parsing function should have an input parameter `content` that takes the raw LLM response as input, and return a dictionary with keys matching the output field names. 

2. It is recommended to use the `@register_parse_function` decorator to register the parsing function for serialization, so that you can save the agent and load it later. 


## Saving and Loading Agents

You can save agent definitions to reuse them later:

```python
# Save agent configuration. By default, the `llm_config` will not be saved. 
code_writer.save_module("./agents/code_writer.json")

# Load agent from file (requires providing llm_config again)
loaded_agent = CustomizeAgent.from_file(
    "./agents/code_writer.json", 
    llm_config=openai_config
)
```

## Advanced Example: Multi-Step Code Generator

Here's a more advanced example that demonstrates creating a specialized code generation agent with multiple structured outputs:

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
    """Parse LLM output into code, documentation, and tests sections"""
    sections = content.split("## ")
    result = {"code": "", "documentation": "", "tests": ""}
    
    for section in sections:
        if not section.strip():
            continue
        
        lines = section.strip().split("\n")
        section_name = lines[0].lower()
        section_content = "\n".join(lines[1:]).strip()
        
        if "code" in section_name:
            # Extract code from code blocks
            code_blocks = extract_code_blocks(section_content)
            result["code"] = code_blocks[0] if code_blocks else section_content
        elif "documentation" in section_name:
            result["documentation"] = section_content
        elif "test" in section_name:
            # Extract code from code blocks if present
            code_blocks = extract_code_blocks(section_content)
            result["tests"] = code_blocks[0] if code_blocks else section_content
    
    return result

# Create the advanced code generator agent
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

# Execute the agent
result = advanced_generator(
    inputs={
        "requirement": "Create a function to validate if a string is a valid email address"
    }
)

# Access the structured outputs
print("CODE:")
print(result.content.code)
print("\nDOCUMENTATION:")
print(result.content.documentation)
print("\nTESTS:")
print(result.content.tests)
```

This advanced example demonstrates how to create a specialized agent that produces multiple structured outputs from a single LLM call, providing a complete code package with implementation, documentation, and tests.
