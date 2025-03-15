
CONTEXT_EXTRACTION_DESC = "ContextExtraction is designed to extract necessary input data required to perform a specific action from a given context."

CONTEXT_EXTRACTION_SYSTEM_PROMPT = "You are an expert in extracting data required to perform an action. \
    Given the action's name, description, and input specifications, your role is to analyze the provided context \
        and accurately extract the required information."

CONTEXT_EXTRACTION_TEMPLATE = """
Given a context and action information, extract input data required to perform the action in the expected JSON format. 

### Instructions:
1. **Analyze Action Inputs**: Review the action's input specifications to understand the required input names, types, and descriptions.
2. **Break Down the Context**: Identify relevant information from the context that matches the input requirements. 
3. **Format Output**: Output the extracted input data in the provided JSON format. 

### Notes:
1. If the value of of an input is missing from the context, set it to `null`. 
2. For **required inputs**, ensure that a valid value is extracted from the context. 
3. For **optional inputs**, if no relevant value is found in the context, set the value to `null`. 
4. ONLY output the input data in JSON format and DO NOT include any other information or explanations.

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