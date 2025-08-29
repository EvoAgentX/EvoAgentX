TOOL_CALLING_HISTORY_PROMPT = """
Iteration {iteration_number}:
Executed tool calls:
{tool_call_args}
Results:
{results}

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
# Tool Calling Guide

## When to Use Tools
- Use tools only when necessary to complete your task
- Check history first - if the information is already available, don't call tools
- If tools can't help with the task, proceed without them

## How to Call Tools
1. Include a brief explanation of:
   - What you know from the history
   - What information is needed
   - Why you're using the tool

2. Format your tool calls exactly as shown:
```ToolCalling
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
```

**Important Notes**
- All content within the `ToolCalling` code block must be valid JSON. Do not include any text, explanations or comments within these code blocks.
- Only use tools listed in Available Tools
- Stop generating after the tool call JSON
- Each tool call must include both function_name and function_args

## Available Tools
{tools_description}
"""

