OUTPUT_EXTRACTION_PROMPT = """
You are given the following text:
{text}

We need you to process this text and generate high-quality outputs for each of the following fields:
{output_description}

**Instructions:**
1. Read through the provided text carefully.
2. For each of the listed output fields, analyze the relevant information from the text and generate a well-formulated response.
3. You may summarize, process, restructure, or enhance the information as needed to provide the best possible answer.
4. Your analysis should be faithful to the content but can go beyond simple extraction - provide meaningful insights where appropriate.
5. Return your processed outputs in a single JSON object, where the JSON keys **exactly match** the output names given above.
6. If there is insufficient information for an output, provide your best reasonable inference or set its value to an empty string ("") or `null`.
7. Do not include any additional keys in the JSON.
8. Your final output should be valid JSON and should not include any explanatory text.

**Example JSON format:**
{{
  "<OUTPUT_NAME_1>": "Processed content here",
  "<OUTPUT_NAME_2>": "Processed content here",
  "<OUTPUT_NAME_3>": "Processed content here"
}}

Now, based on the text and the instructions above, provide your final JSON output.
"""



GOAL_BASED_TOOL_CALLING_PROMPT = """
You are an intelligent agent tasked with achieving a specific goal. You have access to various tools that can help you accomplish your task.

## Goal
{goal_prompt}

## Inputs
{inputs}

## History
{history}

## Tool Usage
If you are provided with tools, you should use them if you need to.
Once you have completed all preparations, you should not call any tool and just generate the final answer.
You should also include the very short thinking process in the output to explain why you need to use the tool, before you call the tool and stop generating the output. 
Tool call is only part of the output.

### Example Output
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

### Notes
Remember, when you need to make a tool call, use ONLY the exact format specified above, as it will be parsed programmatically. The tool calls should be enclosed in triple backticks with the ToolCalling identifier, followed by JSON that specifies the tool name and parameters.
After using a tool, analyze its output and determine next steps. You may need to:
- Use additional tools to complete the task
- Process and transform the information received
- Present the final output according to the specified format

### Available Tools
{tools_description}

### Additional Tool Calling Instructions
{additional_context}

### TOOL CALLING KEY POINTS
- You should check the history to determine if you have the information
- You should try to use tools to get the information you need
- You should not avoid using the tool if you have the information
- You should not call any tool if you completed the goal
- The tool you called must exist in the available tools
"""

TOOL_CALLING_TEMPLATE = """
### Tools Calling Instructions
You may have access to various tools that might help you accomplish your task.
If you are provided with tools, you should use them if you need to.
Once you have completed all preparations, you SHOULD NOT call any tool and just generate the final answer.
You should also include the VERY SHORT thinking process in the output to explain why you need to use the tool, before you call the tool and stop generating the output. 
Tool call is only part of the output.
You should STOP GENERATING responds RIGHT AFTER you give the tool calling instructions.
By checking the history, IF you get the information, you should IGNORE THIS TOOL CALLING INSTRUCTIONS SECTION.

** Example Output **
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
** Tool Calling Notes **
Remember, when you need to make a tool call, use ONLY the exact format specified above, as it will be parsed programmatically. The tool calls should be enclosed in triple backticks with the ToolCalling identifier, followed by JSON that specifies the tool name and parameters.
After using a tool, analyze its output and determine next steps. You may need to:
- Use additional tools to complete the task
- Process and transform the information received
- Present the final output according to the specified format

**Available Tools**
{tools_description}

**Additional Tool Calling Instructions**
{additional_context}

** Tool Calling Key Points **
- You dont have to use the tool.
- You should ALWAYS check the history to determine if you have the information
- You should try to use tools to get the information you need
- You should not avoid using the tool if you have the information by CHECKING the history
- You should not call any tool if you completed the goal
- The tool you called must exist in the available tools
- You should never write comments in the call_tool function
"""

