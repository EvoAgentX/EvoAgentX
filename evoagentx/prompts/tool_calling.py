OUTPUT_EXTRACTION_PROMPT = """
## Task
You are given a piece of unstructured text and a list of fields to extract. Each field includes:
- `name`: The name of the field.
- `description`: A description of the field.
- `type`: The type of the field.
- `required`: Whether the field is required.
- `json_schema`: The JSON schema for the field. If provided, your output must strictly follow the JSON schema.

Your task is to analyze the text carefully and generate a valid JSON object that includes all of the requested fields

## Instructions
1. Read through the provided text carefully.
2. For each of the listed fields, analyze the relevant information from the text and generate a well-formulated response.
3. You may summarize, process, restructure, or enhance the information as needed to provide the best possible answer.
4. Your analysis should be faithful to the content but can go beyond simple extraction - provide meaningful insights where appropriate.
5. Return your processed outputs in a single JSON object, where the JSON keys **exactly match** the field names provided in **Fields** section.
6. If there is insufficient information for an output follow the following rules:
    - If the field is not required, set it to `null`.
    - If the field is required and `type` is `string`, return an empty string `""`.
    - If the field is required and `type` is `array`, return an empty array `[]`.
    - If the field is required and `type` is `object`, return an empty object `{{}}`.
    - If the field is required and `type` is none of the above, provide your best reasonable inference.
7. Do not include any additional keys in the JSON.
8. Your final output should be valid JSON and should not include any explanatory text.

## Text
{text}

## Fields
```json
{output_description}
```

## Output
```json
"""


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
A sample output for tool argument looks like this following line (The example tools are not real tools): 
tools: ["File Tool", "Browser Tool"]

**Tools Description**
{tools_description}

"""


TOOL_CALLING_TEMPLATE = """
# Tool Calling Guide

You can call the following tools:
{tools_description}

## Rules
- ONLY use tools listed above. Do not invent or use non-existent tools.
- Check the conversation history before calling: If the needed information is already available (e.g., from previous tool results), do not call tools again. Summarize and use it directly.
- If a previous tool call failed (e.g., error in history), try a different tool or adjust arguments; do not repeat the same call.
- Call tools ONLY when necessary for the task (e.g., external data, computation). Otherwise, proceed to the final output without tools.
- Support multiple parallel calls: Use an array with multiple objects if needed.
- Each call MUST include "function_name" (exact match from tool's Action) and "function_args" (a dict with exact argument names and values).
- For arguments: Each parameter must be a valid JSON type (e.g., string, integer) and required/optional status as described. Do not add extra args.
- Output STRICTLY in the format below. NO explanations, comments, thoughts, or extra text outside the <ToolCalling> block. If no tools needed, do not output this block at all.

## Output format
Always return a JSON array of tool calls, like:

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

## Examples
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

Example 2: Multiple parallel calls (e.g., search and code execution).
<ToolCalling>
[
    {{
        "function_name": "web_search",
        "function_args": {{
            "query": "python tips"
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
The following output is supposed to be a JSON list of tool calls, but it's invalid.
Please fix it and return ONLY the valid JSON array:
--- Invalid Output ---
{text}
--- End ---
"""
