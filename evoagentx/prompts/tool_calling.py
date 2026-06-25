TOOL_CALL_FORMAT = """
<tool_call>
{tool_calls}
</tool_call>
"""

TOOL_CALLING_HISTORY_PROMPT = """
<tool_result>
{results}
</tool_result>
"""

AGENT_GENERATION_TOOLS_PROMPT = """
### Tools
In the following **Tool Descriptions** section, you are offered with the following tools. A short description of each functionality is also provided for each tool.
You should assign tools to agent if you think it would be helpful for the agent to use the tool.

**Tool Descriptions**
{tool_descriptions}

"""

TOOL_CALLING_TEMPLATE = """
### Tool Calling Guide

The following tools are available:
{tool_descriptions}

#### Tool Calling Rules
- Only use the tools provided. Do not invent or use non-existent ones.
- Check the conversation history before calling a tool. If the information you need is already present, do not make another call.
- If a tool call fails, try a different tool or adjust the arguments. Do not repeat the failed call.
- Only use a tool when it is essential to complete the task, such as to retrieve external data.
- Each tool call must include `function_name` and `function_args`.
- The arguments in `function_args` must exactly match the tool's required parameters and their data types. Do not include any extra arguments.
- Each argument must be a valid JSON type (e.g., string, number, boolean, array, object).
- Output only the JSON format shown in the **Tool Calling Output Format** section. Do not include any additional text, explanations, or comments within the `<tool_call>` block. If no tools are needed, do not output this block at all.

#### Tool Calling Output Format
<tool_call>
[
    {{
        "function_name": "tool_name",
        "function_args": {{
            "param1": "value1",
            "param2": "value2"
        }}
    }},
    ...
]
</tool_call>

#### Tool Calling Examples
Example 1: Single tool call for web search.
<tool_call>
[
    {{
        "function_name": "web_search",
        "function_args": {{
            "query": "example search term",
            "num_results": 5
        }}
    }}
]
</tool_call>

Example 2: Multiple calls that don't depend on each other (e.g., search and code execution).
<tool_call>
[
    {{
        "function_name": "web_search",
        "function_args": {{
            "query": "latest tech news",
            "num_results": 5
        }}
    }},
    {{
        "function_name": "code_execution",
        "function_args": {{
            "code": "print('Hello world')"
        }}
    }}
]
</tool_call>
"""

TOOL_CALLING_RETRY_PROMPT = """
The following is an invalid JSON array. Please correct it and return only the valid JSON array.

**Invalid JSON Array**
```json
{text}
```
"""

def format_tool_descriptions(tools, default: str = "No tools provided.") -> str:
    """
    Args:
        tools: List of tools to format.
        default: Default description to use if no tools are provided.

    Returns:
        str: Formatted tool descriptions.

    Example output:
    - **Tool 1**: Description of tool 1
    - **Tool 2**: Description of tool 2
    - **Toolkit** is a toolkit that provides the following functionalities:
        * Description of tool 1 in toolkit
        * Description of tool 2 in toolkit
    """
    from ..tools import Tool, Toolkit

    if not tools:
        return default
        
    descriptions = []

    for tool in tools:
        if isinstance(tool, Tool):
            name = tool.name
            description = tool.description.replace("\n", "\n    ")
            descriptions.append(f"- **{name}**: {description}")
        elif isinstance(tool, Toolkit):
            name = tool.name
            description = "is a toolkit that provides the following functionalities:"

            for tool in tool.get_tools():
                tool_description = tool.description.replace("\n", "\n        ")
                description += f"\n    * {tool_description}"
            
            descriptions.append(f"- **{name}** {description}")

    descriptions = "\n".join(descriptions)
    return descriptions