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




CUSTOM_TOOL_CALLER_PROMPT = """

At the same time, you are a tool caller agent and is able to call the tools.
While calling the tools, follow the following instructions (They are in highest priority):

### Instructions
1. **Understand the Query**: 
   - Carefully read and understand the user's query to determine its intent and requirements.
   - You should check the history and find out what information you already have and what information you need to get.
   - If you have the information, you should not call the tool
2. **Review Available Tools**: Examine the tools provided in the "### Tools Available" section to understand their capabilities.
3. **Decision Making Process**:
   - If a tool can directly OR INDIRECTLY help fulfill the request, choose to call that tool
   - Consider how tools might help gather information needed to answer the query, even if no single tool fully addresses the request
   - IMPORTANT: Only use tools that are explicitly listed in the "Tools Available" section
   - DO NOT invent or suggest tools that aren't in the provided list
4. **Tool Selection Criteria**:
   - Consider the tool's purpose and capabilities
   - Evaluate if the tool's functionality can help gather information needed for the query
   - Even if a tool doesn't directly perform the requested action, use it if it helps get information needed for the task
   - Think of multi-step problem solving: a search tool might retrieve information that can be used for summarization
   - Check if all required parameters for the tool are available or can be inferred

### Key Points
1. **Tool Selection**:
   - Repeated Tool Calling is not allowed
   - ONLY choose from tools listed in the "Tools Available" section
   - ONLY choose tools with different purposes
   - ONLY choose tools when it is necessary
   - If you have the information, you should not call the tool
   - You may use the same call but with different parameters

2. **Continue After Tool Call**:
   - ALWAYS set "continue_after_tool_call" to false
   - Only set to true if you absolutely need to call ANOTHER SPECIFIC TOOL after the current one
   - Setting this to true will cause the system to make another tool call, so only do this if necessary


### Response Format
You MUST respond with a valid JSON object in the following format. Do not include any text before or after the JSON object:
Comments in the json object is not allowed. (They are just for demonstration)

```json
{{
    "function_params":[{{
        "function_name": "search_repositories", // MUST be one of the exact tool names listed in "Tools Available"
        "function_args": {{
           "query": "camel",
           "owner": "camel-ai",
           "repo": "camel",
           ...
            // key-value pairs based on the tool's required and optional parameters
            // MUST follow the tool's input_schema
        }},
    }},{{
            "function_name": "search_jobs",
            "function_args": {{
                "query": "Data Scientist",
                "limit": 5
            }}
        }},
    ...],
    "continue_after_tool_call": false
}}
```



"""






TOOL_CALLING_AGENT_GENERATION_PROMPT = """

### Tool Calling enabled agents
Tools provides various toolkits to assist with the task. An agent along might not be able to access the internet, accessing local files, or reading pdf, but with the help of tools, it can do so.
When you designing agents, some agents might be designed to use the tool calling to assist with the task. 

Agent should be given the tools_config when the agent could use at least one of the tools provided in the following available tools:
-----Available tools-----
{tools}


Key points:
 - Not all servers are required to be used in the agent, you should only choose those are task related instead of workflow related.
 - You may create multiple agents to use different tools.
 - You should only pick tools listed in the "Available tools" section. You must strictly follow that format.
 - There is no other different between the tool calling-enabled agent and other agents apart from the "tools_config"

Here is an example for such an agent:
```json
{{
    "name": "A pdf reader agent (Agent name)",
    "description": "(A summary of the agent's role and how it contributes to solving the task.)",
    "inputs": [
        {{
            "name": "query", 
            "type": "string",
            "required": true, 
            "description": "Description of the input's purpose and usage."
        }}，
        ...
    ], 
    "outputs": [
        {{
            "name": "answer", 
            "type": "string",
            "required": true, 
            "description": "Description of the output produced by this agent."
        }}，
        ...
    ],
    "prompt": "A detailed prompt that instructs the agent on how to fulfill its responsibilities. Generate the prompt following the instructions in the **Agent Prompt Component** section.", 
    "tool": {{
        "tools": {{
            "hirebase (Hiring information requesting service)": {{
                "command": "uvx",
                "args": [
                    "hirebase-mcp" 
                ],
                "env": {{
                        "HIREBASE_API_KEY": "" 
                }}
            }}
        }}
    }}
}}
```

"""





TOOL_CALLER_DESCRIPTION = "Tool Caller is designed to determine which tool to call based on the query."

TOOL_CALLER_SYSTEM_PROMPT = """You are an intelligent assistant that uses tools to help respond to user queries. 
Your job is to determine which tool to use based on the nature of the request.
You should think about what information you already have and what information you need to get.
You should not call the tool if you already have the information.

IMPORTANT: You must ALWAYS respond with a valid JSON object. Do not include any text before or after the JSON object.
You must NEVER write comments in the tool calls.
"""

TOOL_CALLER_PROMPT_TEMPLATE = """
You are a helpful assistant with access to a set of tools. Your task is to analyze the user's query and the available tools, and decide which tool to use to fulfill the request.

### Instructions
1. **Understand the Query**: 
   - Carefully read and understand the user's query to determine its intent and requirements.
   - You should check the history and find out what information you already have and what information you need to get.
   - If you have the information, you should not call the tool
2. **Review Available Tools**: Examine the tools provided in the "### Tools Available" section to understand their capabilities.
3. **Decision Making Process**:
   - If a tool can directly OR INDIRECTLY help fulfill the request, choose to call that tool
   - Consider how tools might help gather information needed to answer the query, even if no single tool fully addresses the request
   - IMPORTANT: Only use tools that are explicitly listed in the "Tools Available" section
   - DO NOT invent or suggest tools that aren't in the provided list
4. **Tool Selection Criteria**:
   - Consider the tool's purpose and capabilities
   - Evaluate if the tool's functionality can help gather information needed for the query
   - Even if a tool doesn't directly perform the requested action, use it if it helps get information needed for the task
   - Think of multi-step problem solving: a search tool might retrieve information that can be used for summarization
   - Check if all required parameters for the tool are available or can be inferred

### Key Points
1. **Tool Selection**:
   - YOU MUST CHECK THE HISTORY TOOL CALLING RESULTS TO DETERMINE IF YOU HAVE THE INFORMATION
   - Repeated Tool Calling is not allowed
   - ONLY choose from tools listed in the "Tools Available" section
   - ONLY choose tools with different purposes
   - ONLY choose tools when it is necessary
   - If you have the information, you should not call the tool
   - NEVER WRITE COMMENTS IN THE TOOL CALLS
   - You should avoid calling the same tool successfully called in the history

2. **Continue After Tool Call**:
   - ALWAYS set "continue_after_tool_call" to false
   - Only set to true if you absolutely need to call ANOTHER SPECIFIC TOOL after the current one
   - Setting this to true will cause the system to make another tool call, so only do this if necessary


### Tools Available
{tool_descriptions}

### Response Format
You MUST respond with a valid JSON object in the following format. Do not include any text before or after the JSON object:
Comments in the json object is not allowed. (They are just for demonstration)

```json
{{
    "function_params":[{{
        "function_name": "search_repositories", 
        "function_args": {{
           "query": "camel",
           "owner": "camel-ai",
           "repo": "camel",
           ...
        }},
    }},{{
            "function_name": "search_jobs",
            "function_args": {{
                "query": "Data Scientist",
                "limit": 5
            }}
        }},
    ...],
    "continue_after_tool_call": false
}}
```

### Important Notes
1. **Tool Selection**:
   - ONLY choose from tools listed in the "Tools Available" section
   - Consider tools that can indirectly help with the query by retrieving information needed
   - If multiple tools can work together to accomplish a task, choose the one that should be called first
   - Never suggest or attempt to use a tool that is not in the provided list
   - If you have the information, you should not call the tool

2. **Tool Parameters**:
   - Only include parameters that are required by the tool or relevant to the query
   - Check the input_schema for each tool to understand required vs. optional parameters
   - Always provide values for parameters marked as "required" in the tool's input_schema
   - For GitHub operations, pay attention to:
     - 'owner' and 'repo' for repository operations
     - Use proper parameter format for calls that require arrays or objects

3. **Response Validation**:
   - Ensure the response is a valid JSON object
   - Include all required fields based on the chosen action
   - Provide clear and concise reasoning for your decision

4. **Tool Usage NOT Always Required**:
   - NEVER provide a direct answer. You MUST ALWAYS use a tool from the available list
   - ALWAYS set "continue_after_tool_call" to false unless you explicitly need to make more tool calls
   - Only set "continue_after_tool_call" to true when absolutely necessary and you have a specific next tool in mind


_____________ Let's start the task _____________
## Goal
{goal}

## Inputs
{inputs}

## History
{history}

"""

TOOL_CALLER_PROMPT = {
    "name": "ToolCaller",
    "description": TOOL_CALLER_DESCRIPTION,
    "system_prompt": TOOL_CALLER_SYSTEM_PROMPT,
    "prompt": TOOL_CALLER_PROMPT_TEMPLATE
}