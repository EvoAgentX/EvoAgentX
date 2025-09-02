TOOL_CALLING_HISTORY_PROMPT = """
<ToolResults>
Iteration {iteration_number} Results:
{results}
</ToolResults>
"""

AGENT_GENERATION_TOOLS_PROMPT = """
In the following Tools Description section, you are offered with the following tools. A short description of each functionality is also provided for each tool.
You should assign tools to agent if you think it would be helpful for the agent to use the tool.

**Tools Description**
{tools_description}

"""


TOOL_CALLING_TEMPLATE_OLD = """
### Tools Calling Instructions
You should try to use tools if you are given a list of tools.
You may have access to various tools that might help you accomplish your task.
Once you have completed all preparations, you SHOULD NOT call any tool and just generate the final answer.
If you need to use the tool, you should also include the ** very short ** thinking process before you call the tool and stop generating the output. 
In your short thinking process, you give short summary on ** everything you got in the history **, what is needed, and why you need to use the tool.
While you write the history summary, you should state information you got in each iteration.
You should STOP GENERATING responds RIGHT AFTER you give the tool calling instructions.
By checking the history, IF you get the information, you should **NOT** call any tool.
Do not generate any tool calling instructions if you have the information. 
Distinguish tool calls and tool calling arguments, only include "```ToolCalling" when you are calling the tool, otherwise you should pass arguments with out this catch phrase.
The tools in the Example Output does not really exist, you should use the tools in the Available Tools section.
Every tool call should contain the function name and function arguments. The function name should be the name of the tool you are calling. The function arguments in the next example are fake arguments, you should use the real arguments for the tool you are calling.
You should keep the tool call a list even if it is a single tool call.

** Example Output 1 **
Base on the goal, I found out that I need to use the following tools:
```ToolCalling
[{{
    "function_name": "search_repositories",
    "function_args": {{
        "query": "camel",
        "owner": "camel-ai",
        "repo": "camel",
        ...
    }}
}},{{
    "function_name": "search_jobs",
    "function_args": {{
        "query": "Data Scientist",
        "limit": 5
    }}
}},...]
```
** Example Output 2 **
To do this, I need to use the following tools call:
```ToolCalling
[{{
    "function_name": "search_repositories",
    "function_args": {{
        "command": "dir examples/output/invest/data_cache/",
        "timeout": 30
    }}
}}
```

** Example Output When Tool Calling not Needed **
Based on the information, ... 
There are the arguments I used for the tool call: [{{'function_name': 'read_file', 'function_args': {{'file_path': 'examples/output/jobs/test_pdf.pdf'}}}}, ...]// Normal output without ToolCalling & ignore the "Tools Calling Instructions" section


** Tool Calling Notes **
Remember, when you need to make a tool call, use ONLY the exact format specified above, as it will be parsed programmatically. The tool calls should be enclosed in triple backticks with the ToolCalling identifier, followed by JSON that specifies the tool name and parameters.
After using a tool, analyze its output and determine next steps. 

**Available Tools**
{tools_description}

** Tool Calling Key Points **
You should strictly follow the tool calling structure, even if it is a single tool call.
You should always check the history to determine if you have the information or the tool is not useful, if you have the information, you should not use the tool.
You should try to use tools to get the information you need
You should not call any tool if you completed the goal
The tool you called must exist in the available tools
You should never write comments in the call_tool function
If your next move cannot be completed by the tool, you should not call the tool
"""


TOOL_CALLING_TEMPLATE = """
### Tool Calling Guide

The following tools are available:
{tools_description}

#### Tool Calling Rules
- Only use the tools provided. Do not invent or use non-existent ones.
- Check the conversation history before calling a tool. If the information you need is already present, do not make another call.
- If a tool call fails, try a different tool or adjust the arguments. Do not repeat the failed call.
- Only use a tool when it is essential to complete the task, such as to retrieve external data.
- Each tool call must include `function_name` and `function_args`.
- The arguments in `function_args` must exactly match the tool's required parameters and their data types. Do not include any extra arguments.
- Each argument must be a valid JSON type (e.g., string, number, boolean, array, object).
- Output only the JSON format shown in the **Tool Calling Output Format** section. Do not include any additional text, explanations, or comments within the `<ToolCalling>` block. If no tools are needed, do not output this block at all.

#### Tool Calling Output Format
<ToolCalling>
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
</ToolCalling>

#### Tool Calling Examples
Example 1: Single tool call for web search.
<ToolCalling>
[
    {{
        "function_name": "web_search",
        "function_args": {{
            "query": "example search term",
            "num_results": 5
        }}
    }}
]
</ToolCalling>

Example 2: Multiple calls that don't depend on each other (e.g., search and code execution).
<ToolCalling>
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
</ToolCalling>
"""

TOOL_CALLING_RETRY_PROMPT = """
The following is an invalid JSON array. Please correct it and return only the valid JSON array.

**Invalid JSON Array**
```json
{text}
```
"""
