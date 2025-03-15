
TASK_PLANNER_DESC = "TaskPlanner is an intelligent task planning agent designed to assist users in achieving their goals. \
    It specializes in breaking down complex tasks into clear, manageable sub-tasks and organizing them in the most efficient sequence." 

TASK_PLANNER_SYSTEM_PROMPT = "You are a highly skilled task planning expert. Your role is to analyze the user's goals, deconstruct complex tasks into actionable and manageable sub-tasks, and organize them in an optimal execution sequence."

TASK_PLANNER = {
    "name": "TaskPlanner", 
    "description": TASK_PLANNER_DESC,
    "system_prompt": TASK_PLANNER_SYSTEM_PROMPT,
}


TASK_PLANNING_ACTION_DESC = "This action analyzes a given task, breaks it down into manageable sub-tasks, and organizes them in the optimal order to help achieve the user's goal efficiently."

TASK_PLANNING_ACTION_PROMPT_OLD = """
Given a user's goal, analyze the task and create a workflow by breaking it down into actionable sub-tasks. Organize these sub-tasks in an optimal execution order for smooth and efficient completion.

### Instructions:
1. **Understand the Goal**: Identify the core objectives and outcomes the user wants to achieve. 
2. **Review the History**: Assess any previously generated task plan to identify gaps or areas needing refinement. 
3. **Consider Suggestions**: Consider user-provided suggestions to improve or optimize the workflow. 

4. **Define Sub-Tasks**: Break the task into logical, actionable sub-tasks based on the complexity of the goal.

4.1 **Principle for Breaking Task**:
- **Simplicity**: Each sub-task is designed to achieve a specific, clearly defined objective. Avoid overloading sub-tasks with multiple objectives. 
- **Modularity**: Ensure that each sub-task is self-contained, reusable, and contributes meaningfully to the overall solution. 
- **Consistency**: Sub-tasks must logically support the user's goal and maintain coherence across the workflow.
- **Optimize Complexity**: Adjust the number of sub-tasks according to task complexity. Highly complex tasks may require more detailed steps, while simpler tasks should remain concise.
- **Avoid Redundancy**: Ensure that there are no overlapping or unnecessary sub-tasks. 

4.2 **Sub-Task Format**: 
Each sub-task should follow the structure below:
```json
{{
    "name": "subtask_name",
    "description": "A clear and concise explanation of the goal of this sub-task.",
    "reason": "Why this sub-task is necessary and how it contributes to achieving user's goal.",
    "inputs": [
        {{
            "name": "the input's name", 
            "type": "string/int/float/other_type",
            "description": "Description of the input's purpose and usage."
        }},
        ...
    ], 
    "outputs": [
        {{
            "name": "the output's name", 
            "type": "string/int/float/other_type",
            "description": "Description of the output produced by this sub-task."
        }},
        ...
    ]
}}
```

### Special Instructions for Programming Tasks
- **Environment Setup and Deployment**: For programming-related tasks, **do not** include sub-tasks related to setting up environments or deployment unless explicitly requested.
- Focus on tasks such as requirements analysis, design, coding, debugging, and testing, etc. 
- Ensure that sub-tasks reflect standard coding practices like module design, coding, and testing.


### Notes:
- Provide clear and concise names for the sub-tasks, inputs, and outputs. 
- Maintain consistency in the flow of inputs and outputs between sub-tasks to ensure seamless integration. 
- The inputs of a sub-task can ONLY be chosen from the user's `goal` and any outputs from its preceding sub-tasks. 
- The inputs of a su-btask should contain SUFFICIENT information to effectivelly address the current sub-task.
- The inputs of a sub-task MUST include the user's input `goal`. 
- The first sub-task must have only one `input_name` "goal" with the following structure:
```json
"inputs": [
    {{
        "name": "goal",
        "type": "string",
        "description": "The user's goal in textual format."
    }}
]
```

### Output Format
Your final output should ALWAYS in the following format:

## Thought 
Provide a brief explanation of your reasoning for breaking down the task and the chosen task structure.  

## Goal
Restate the user's goal clearly and concisely.

## Plan
You MUST provide the workflow plan with detailed sub-tasks in the following JSON format. The description of each sub-task MUST STRICTLY follow the JSON format described in the **Sub-Task Format** section. If a sub-task doesn't require inputs or do not have ouputs, still include `inputs` and `outputs` in the definiton by setting them as empty list. 
```json
{{
    "sub_tasks": [
        {{
            "name": "subtask_name", 
            ...
        }}, 
        {{
            "name": "another_subtask_name", 
            ...
        }},
        ...
    ]
}}
```

-----
Let's begin. 

### History (previously generated task plan):
{history}

### Suggestions (idea of how to design the workflow or suggestions to refine the history plan):
{suggestion}

### User's Goal:
{goal}

Output:
"""

TASK_PLANNING_ACTION_INST = """
Given a user's goal, analyze the task and create a workflow by breaking it down into actionable sub-tasks. Organize these sub-tasks in an optimal execution order for smooth and efficient completion.

### Instructions:
1. **Understand the Goal**: Identify the core objectives and outcomes the user wants to achieve. 
2. **Review the History**: Assess any previously generated task plan to identify gaps or areas needing refinement. 
3. **Consider Suggestions**: Consider user-provided suggestions to improve or optimize the workflow. 

4. **Define Sub-Tasks**: Break the task into logical, actionable sub-tasks based on the complexity of the goal.

4.1 **Principle for Breaking Task**:
- **Simplicity**: Each sub-task is designed to achieve a specific, clearly defined objective. Avoid overloading sub-tasks with multiple objectives. 
- **Modularity**: Ensure that each sub-task is self-contained, reusable, and contributes meaningfully to the overall solution. 
- **Consistency**: Sub-tasks must logically support the user's goal and maintain coherence across the workflow.
- **Optimize Complexity**: Adjust the number of sub-tasks according to task complexity. Highly complex tasks may require more detailed steps, while simpler tasks should remain concise.
- **Avoid Redundancy**: Ensure that there are no overlapping or unnecessary sub-tasks. 
- **Consider Cycles**: Identify tasks that require iteration or feedback loops, and structure dependencies (by specifying inputs and outputs) accordingly. 

4.2 **Sub-Task Format**: 
Each sub-task should follow the structure below:
```json
{{
    "name": "subtask_name",
    "description": "A clear and concise explanation of the goal of this sub-task.",
    "reason": "Why this sub-task is necessary and how it contributes to achieving user's goal.",
    "inputs": [
        {{
            "name": "the input's name", 
            "type": "string/int/float/other_type",
            "required": true/false (only set to `false` when this input is the feedback from later sub-task, or the previous generated output for the current sub-task),
            "description": "Description of the input's purpose and usage."
        }},
        ...
    ], 
    "outputs": [
        {{
            "name": "the output's name", 
            "type": "string/int/float/other_type",
            "required": true (always set the `required` field of outputs as true), 
            "description": "Description of the output produced by this sub-task."
        }},
        ...
    ]
}}
```

### Special Instructions for Programming Tasks
- **Environment Setup and Deployment**: For programming-related tasks, **do not** include sub-tasks related to setting up environments or deployment unless explicitly requested.
- Focus on tasks such as requirements analysis, design, coding, debugging, and testing, etc. 
- Ensure that sub-tasks reflect standard coding practices like module design, coding, and testing.


### Notes:
- Provide clear and concise names for the sub-tasks, inputs, and outputs. 
- Maintain consistency in the flow of inputs and outputs between sub-tasks to ensure seamless integration. 
- The inputs of a sub-task can ONLY be chosen from the user's `goal` and any outputs from its preceding sub-tasks. 
- The inputs of a su-btask should contain SUFFICIENT information to effectivelly address the current sub-task.
- The inputs of a sub-task MUST include the user's input `goal`. 
- The first sub-task must have only one `input_name` "goal" with the following structure:
```json
"inputs": [
    {{
        "name": "goal",
        "type": "string",
        "required": true,
        "description": "The user's goal in textual format."
    }}
]
```
- If a sub-task require feedback from a later sub-task (for feedback or refinement), include the later sub-task's output and the current sub-task's output in the current sub-task's inputs and set `"required": false`. 
"""

TASK_PLANNING_ACTION_DEMOS = """
### Examples: 
Below are some generated workflows that follow the given instructions:

Example 1: 
### User's goal: 
Create a Python function that takes two numbers as input and returns their sum.
### Generated Workflow: 
{{
    "sub_tasks": [
        {{
            "name": "code_generation",
            "description": "Generate a Python function that takes two numbers as input and returns their sum.", 
            "reason": "This sub-task ensures that the function correctly implements the required summation logic.", 
            "inputs": [
                {{
                    "name": "goal",
                    "type": "string",
                    "required": true, 
                    "description": "The user's goal in textual format."
                }}
            ],
            "outputs": [
                {{
                    "name": "function_code", 
                    "type": "string", 
                    "required": true, 
                    "description": "The generated Python function code that takes two numbers and returns their sum."
                }}
            ]
        }}
    ]
}}


Example 2: 
### User's goal: 
Given two strings `s` and `t` of lengths `m` and `n` respectively, return the minimum window substring of `s` such that every character in `t` (including duplicates) is included in the window. If there is no such substring, return the empty string "".
Input: s = "ADOBECODEBANC", t = "ABC"
Output: "BANC"
Explanation: The minimum window substring "BANC" includes 'A', 'B', and 'C' from string t.
### Generated Workflow:
{{
    "sub_tasks": [
        {{
            "name": "task_parsing", 
            "description": "Analyze the problem statement and extract key requirements.", 
            "reason": "This step ensures that we understand the task and its constraints before proceeding with code implementation.", 
            "inputs": [
                {{
                    "name": "goal",
                    "type": "string", 
                    "required": true, 
                    "description": "The user's goal in textual format."
                }}
            ],
            "outputs": [
                {{
                    "name": "problem_analysis",
                    "type": "string",
                    "required": true,
                    "description": "A clear and detailed analysis of the problem, including input/output format and constraints."
                }}
            ]
        }}, 
        {{
            "name": "code_generation", 
            "description": "Generate a Python function that finds the minimum window substring.", 
            "reason": "This step ensures an initial solution is implemented based on the problem analysis.", 
            "inputs": [
                {{
                    "name": "problem_analysis",
                    "type": "string", 
                    "required": true, 
                    "description": "The clear and detailed analysis of the problem."
                }}
            ],
            "outputs": [
                {{
                    "name": "function_code",
                    "type": "string", 
                    "required": true, 
                    "description": "The generated Python function that finds the minimum window substring."
                }}
            ]
        }}, 
        {{
            "name": "code_review", 
            "description": "Review the generated function for correctness, efficiency, and edge case coverage.", 
            "reason": "Ensures the function meets correctness and performance expectations.", 
            "inputs": [
                {{
                    "name": "problem_analysis",
                    "type": "string", 
                    "required": true, 
                    "description": "The clear and detailed analysis of the problem."
                }},
                {{
                    "name": "function_code",
                    "type": "string", 
                    "required": true, 
                    "description": "The generated Python function."
                }}
            ],
            "outputs": [
                {{
                    "name": "code_feedback",
                    "type": "string", 
                    "required": true, 
                    "description": "Feedback, suggestions, and identified issues in the initial code."
                }}
            ]
        }}, 
        {{
            "name": "code_refinement", 
            "description": "Refine the implementation based on review feedback.", 
            "reason": "Improves code quality by addressing identified issues and optimizing performance.", 
            "inputs": [
                {{
                    "name": "function_code",
                    "type": "string", 
                    "required": true, 
                    "description": "The generated Python function."
                }},
                {{
                    "name": "code_feedback", 
                    "type": "string", 
                    "required": true, 
                    "description": "Feedback or suggestions to refine the generated code."
                }}
            ],
            "outputs": [
                {{
                    "name": "optimized_function_code",
                    "type": "string", 
                    "required": true, 
                    "description": "The final, optimized python code."
                }}
            ]
        }}
    ]
}}

Example 3 (include loops):
### User's goal: 
Create a Tetris game using Python
### Generated Workflow: 
{{
    "sub_tasks": [
        {{
            "name": "requirement_analysis", 
            "description": "Define the specifications and technical requirements for the Tetris game.", 
            "reason": "Understanding the game mechanics, required features, and development constraints ensures a well-structured implementation.", 
            "inputs": [
                {{
                    "name": "goal", 
                    "type": "string", 
                    "required": true, 
                    "description": "The user's goal in textual format."
                }}
            ],
            "outputs": [
                {{
                    "name": "game_requirements", 
                    "type": "string", 
                    "required": true, 
                    "description": "A detailed specification of the game's features, mechanics, and technical constraints."
                }}
            ]
        }},
        {{
            "name": "code_generation", 
            "description": "Generate Python code for the Tetris game based on the defined requirements.", 
            "reason": "This step involves writing the initial implementation of the game. If previous code exists, the task will refine it based on debugging feedback to improve functionality and correctness.", 
            "inputs": [
                {{
                    "name": "game_requirements", 
                    "type": "string", 
                    "required": true, 
                    "description": "Specifications and constraints for the Tetris game."
                }},
                {{
                    "name": "generated_code", 
                    "type": "string", 
                    "required": false, 
                    "description": "Previously generated Tetris code. If provided, the task will refine it instead of starting from scratch." 
                }},
                {{
                    "name": "debugging_feedback", 
                    "type": "string",
                    "required": false, 
                    "description": "Feedback and identified issues from the debugging phase. If provided, the task will improve implementation by addressing detected problems."
                }}
            ],
            "outputs": [
                {{
                    "name": "generated_code", 
                    "type": "string", 
                    "required": true, 
                    "description": "The implementation of the Tetris game. If previous generated code and debugging feedback exists, it is the improved implementation."
                }}
            ]
        }},
        {{
            "name": "code_debugging", 
            "description": "Test and debug the generated code to identify and fix any issues.", 
            "reason": "Ensuring the game functions correctly by catching errors and iterating on the implementation.", 
            "inputs": [
                {{
                    "name": "generated_code", 
                    "type": "string", 
                    "required": true, 
                    "description": "The python code for the Tetris game."
                }}
            ],
            "outputs": [
                {{
                    "name": "debugging_feedback", 
                    "type": "string", 
                    "required": true, 
                    "description": "Identified issues and necessary fixes for the generated code."
                }}
            ]
        }},
        {{
            "name": "documentation", 
            "description": "Create documentation for the Tetris game, including user instructions and developer notes.", 
            "reason": "Providing clear explanations of the game's functionality and code structure helps future development and maintenance.", 
            "inputs": [
                {{
                    "name": "generated_code", 
                    "type": "string", 
                    "required": true, 
                    "description": "The final implementation of the Tetris game."
                }}
            ],
            "outputs": [
                {{
                    "name": "documentation", 
                    "type": "string", 
                    "required": true, 
                    "description": "Written documentation for users and developers."
                }}
            ]
        }}
    ]
}}
"""

TASK_PLANNING_OUTPUT_FORMAT = """
### Output Format
Your final output should ALWAYS in the following format:

## Thought 
Provide a brief explanation of your reasoning for breaking down the task and the chosen task structure.  

## Goal
Restate the user's goal clearly and concisely.

## Plan
You MUST provide the workflow plan with detailed sub-tasks in the following JSON format. The description of each sub-task MUST STRICTLY follow the JSON format described in the **Sub-Task Format** section. If a sub-task doesn't require inputs or do not have ouputs, still include `inputs` and `outputs` in the definiton by setting them as empty list. 
```json
{{
    "sub_tasks": [
        {{
            "name": "subtask_name", 
            ...
        }}, 
        {{
            "name": "another_subtask_name", 
            ...
        }},
        ...
    ]
}}
```

-----
Let's begin. 

### History (previously generated task plan):
{history}

### Suggestions (idea of how to design the workflow or suggestions to refine the history plan):
{suggestion}

### User's Goal:
{goal}

Output:
"""

TASK_PLANNING_ACTION_PROMPT = TASK_PLANNING_ACTION_INST + TASK_PLANNING_ACTION_DEMOS + TASK_PLANNING_OUTPUT_FORMAT

TASK_PLANNING_ACTION = {
    "name": "TaskPlanning", 
    "description": TASK_PLANNING_ACTION_DESC, 
    "prompt": TASK_PLANNING_ACTION_PROMPT, 
}
