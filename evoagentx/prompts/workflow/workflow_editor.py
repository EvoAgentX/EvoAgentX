PROMPT_BASE = """
analyze the workflow and optimize it according to the user's advice and make it more reasonable and efficient, make sure to keep the original key of the json dict,and the original structure of the json dict:

current workflow structure:
{current_workflow_json}

user's advice:
{user_advice}

after the user's advice, please consider the following rules:
1. the description of the node is clear
2. the input and output parameters are reasonable
3. the dependency relationship between the nodes is correct
4. whether some nodes can be merged or split
"""

WORKFLOW_EDITOR_PROMPT = PROMPT_BASE + """
please return the optimized json structure, keep the original format.
"""

WORKFLOW_EDITOR_PROMPT_EXTRA = PROMPT_BASE + """

HERE ARE SOME EXAMPLES OF SOME AGENTS DEFINITIONS:

{pre_defined_agents_examples}

HERE ARE SOME EXAMPLES OF SOME TOOLS DEFINITIONS:

{pre_defined_tools_examples}

please return the optimized json structure, keep the original format.
"""