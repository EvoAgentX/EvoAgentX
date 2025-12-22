
CONTEXT_EXTRACTION_DESC = "ContextExtraction is designed to extract necessary input data required to perform a specific action from a given context."

CONTEXT_EXTRACTION_SYSTEM_PROMPT = "You are an expert in extracting data required to perform an action. \
    Given the action's name, description, and input specifications, your role is to analyze the provided context \
        and accurately extract the required information."

# ### Instructions:
# 1. **Analyze Action Inputs**: Review the action's input specifications to understand the required input names, types, and descriptions.
# 2. **Break Down the Context**: Identify relevant information from the context that matches the input requirements. 
# 3. **Format Output**: Output the extracted input data in the provided JSON format. 

# ### Notes:
# 1. If the value of of an input is missing from the context, set it to `null`. 
# 2. For **required inputs**, ensure that a valid value is extracted from the context. 
# 3. For **optional inputs**, if no relevant value is found in the context, set the value to `null`. 
# 4. ONLY output the input data in JSON format and DO NOT include any other information or explanations.
# 5. The context is a list of messages with content and potential type, agent, action, goal, task, etc. You SHOULD focus on the content and try to extract the input data from the content.

CONTEXT_EXTRACTION_TEMPLATE = """
Given a context and action information, extract input data required to perform the action in the expected JSON format. 

### Instructions:
1. **Understand Action Inputs**: Carefully read the input specification for the action, including input names, types, and descriptions.
2. **Analyze Content Dictionaries**: The context consists of multiple messages, and each message's `Content` is a dictionary. Your task is to search across these dictionaries. You SHOULD focus on the **Content** field of each message and try to extract the input data from the content.
3. **Match Semantics, Not Just Names**: For each action input, find the most semantically similar key in the dictionaries, even if the key name is different. Focus on meaning, not just exact name match.
4. **Extract Full Value**: Once a matching key is found for an input, extract and use its **ENTIRE VALUE** as the value for that input.
5. **Format Output**: Return the results strictly in the expected JSON format.

### Notes:
1. If no matching value is found for an input, set it to `null`.
2. Required inputs must not be null unless they are truly missing.
3. Optional inputs can be `null` if no suitable match is found.
4. Do NOT include explanations or extra text — only output the JSON.
5. Focus exclusively on the `content` field of each message (which is a dictionary), ignore other metadata like `type`, `agent`, etc.

### Context:
{context}

### Action Details: 
Action Name: {action_name}
Action Description: {action_description}
Input Specifications: 
```json
{action_inputs}
```

The extracted input data is:
"""

CONTEXT_EXTRACTION = {
    "name": "ContextExtraction", 
    "description": CONTEXT_EXTRACTION_DESC,
    "system_prompt": CONTEXT_EXTRACTION_SYSTEM_PROMPT, 
    "prompt": CONTEXT_EXTRACTION_TEMPLATE
}