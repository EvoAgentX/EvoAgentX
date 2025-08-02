from evoagentx.prompts.app_prompts import MetaPromptTemplate
from evoagentx.app.components import QueryType, query_type_description_dict

unified_classification_decomposition_prompt_template = """You are an AI assistant specialized in analyzing user queries for workflow and frontend modifications.

Your task is to identify and extract all actionable operations from the user's query, then classify each operation into appropriate intent categories.

**FRONTEND MODIFICATIONS**: Changes to user interface, visual elements, interactions, styling, or frontend functionality.

**BACKEND WORKFLOW MODIFICATIONS**: Changes to the underlying workflow structure, including nodes, agents, and their configurations.

**User Query:** {user_query}

"""

class UnifiedClassificationDecompositionPromptTemplate(MetaPromptTemplate):
    def __init__(self, use_workflow_context: bool = False):
        super().__init__()
        if use_workflow_context:
            self.prompt_template_sub_part = {
                "part_1": unified_classification_decomposition_prompt_template,
                "categories": self.process_template_categories_part(),
                "workflow_context": self.process_template_workflow_context_part(),
                "instruction_part": self.process_template_instruction_part(),
                "output_part": self.process_template_output_part(),
            }
        else:
            self.prompt_template_sub_part = {
                "part_1": unified_classification_decomposition_prompt_template,
                "categories": self.process_template_categories_part(),
                "instruction_part": self.process_template_instruction_part(),
                "output_part": self.process_template_output_part(),
            }
        return
    
    def process_template_categories_part(self):
        categories_part = """Backend workflow categories include:\n"""
        for category in QueryType:
            categories_part += f"- {category.value}: {query_type_description_dict[category.value]}\n"
        categories_part += "\n"
        return categories_part

    def process_template_workflow_context_part(self):
        return """**Current Workflow Context(could be empty):**
{workflow_context}

"""

    def process_template_instruction_part(self):
        return """Instructions:
1. Break down the user query into individual, actionable operations.
2. Do not make false assumptions about the user's intent, if the user's intent is not clear, just keep it and I will process later.
3. Each operation should be specific, clear, and independently executable.
4. Classify each operation into one of the predefined categories.
5. If an operation relates to frontend modifications, classify it as "front_end".
6. Group operations by their category.

"""

    def process_template_output_part(self):
        return """Respond with a JSON object containing:
{{
    "classified_operations": [
        {{
            "category": "category_name",
            "atomic_query": "specific operation description"
        }},
        ...
    ],
    "total_operations": number,
    "has_frontend": boolean,
    "has_backend": boolean
}}

Only include operations that are actually present in the user query. If no operations are found, return an empty list for "classified_operations".

"""


query_rewriting_prompt_template = """You are rewriting a user query to be clearer and more actionable.

**Original Query:** {original_query}

**Intent Type:** {intent_type}

"""

class QueryRewritingPromptTemplate(MetaPromptTemplate):
    def __init__(self, use_workflow_context: bool = False):
        super().__init__()
        if use_workflow_context:
            self.prompt_template_sub_part = {
                "part_1": query_rewriting_prompt_template,
                "workflow_context": self.process_template_workflow_context_part(),
                "rest_part": self.process_rest_part(),
            }
        else:
            self.prompt_template_sub_part = {
                "part_1": query_rewriting_prompt_template,
                "rest_part": self.process_rest_part(),
            }
        return

    def process_template_workflow_context_part(self):
        return """**Workflow Context:** {workflow_context}\n"""

    def process_rest_part(self):
        return """Rewrite the query to be:
1. More specific and clear
2. Include necessary context
3. Use precise terminology
4. Maintain the original intent

Respond with just the rewritten query text.

"""


query_clarity_check_prompt_template = """You are an AI assistant specialized in evaluating the clarity and completeness of user queries for workflow and frontend modifications.

Your task is to determine whether a given atomic query contains sufficient information to be executed, or if it requires additional clarification from the user.

**Query Category:** {category}

**Atomic Query:** {atomic_query}

"""

class QueryClarityCheckPromptTemplate(MetaPromptTemplate):
    def __init__(self, use_workflow_context: bool = False):
        super().__init__()
        if use_workflow_context:
            self.prompt_template_sub_part = {
                "part_1": query_clarity_check_prompt_template,
                "workflow_context": self.process_template_workflow_context_part(),
                "instructions": self.process_template_instructions_part(),
                "output_part": self.process_template_output_part(),
            }
        else:
            self.prompt_template_sub_part = {
                "part_1": query_clarity_check_prompt_template,
                "instructions": self.process_template_instructions_part(),
                "output_part": self.process_template_output_part(),
            }
        return

    def process_template_workflow_context_part(self):
        return """**Current Workflow Context:**
{workflow_context}

"""

    def process_template_instructions_part(self):
        return """**Instructions:**
1. Analyze the atomic query to determine if it contains enough specific information to be executed
2. Consider the query category and what information is typically required for that type of operation
3. If using workflow context, check if the query references specific elements that exist in the current workflow
4. Evaluate for vague language, incomplete references, missing parameters, or ambiguous requirements

**Clarity Evaluation Criteria:**
- **Clear**: The query is specific, actionable, and contains all necessary information
- **Unclear**: The query is vague, missing key details, contains incomplete references, or requires additional specification

"""

    def process_template_output_part(self):
        return """**Response Format:**
Respond with a JSON object:
{{
    "is_clear": boolean,
    "reasoning": "Brief explanation of why the query is clear or unclear",
    "follow_up_questions": ["question1", "question2", ...]
}}

**Important:**
- If "is_clear" is true, provide an empty array for "follow_up_questions"
- If "is_clear" is false, provide {max_questions} specific follow-up questions to help clarify the query
- Follow-up questions should be direct, specific, and help gather the missing information needed to execute the operation
- Consider the workflow context when generating questions (reference specific nodes, agents, or components when relevant)

"""