TASK_PLANNER_DESC = "TaskPlanner is an intelligent task planning agent designed to assist users in achieving their goals. \
    It specializes in breaking down complex tasks into clear, manageable sub-tasks and organizing them in the most efficient sequence." 

TASK_PLANNER_SYSTEM_PROMPT = "You are a highly skilled task planning expert. Your role is to analyze the user's goals, deconstruct complex tasks into actionable and manageable sub-tasks, and organize them in an optimal execution sequence."

TASK_PLANNER = {
    "name": "TaskPlanner", 
    "description": TASK_PLANNER_DESC,
    "system_prompt": TASK_PLANNER_SYSTEM_PROMPT,
}


TASK_PLANNING_ACTION_DESC = "This action analyzes a given task, breaks it down into manageable sub-tasks, and organizes them in the optimal order to help achieve the user's goal efficiently."


TASK_PLANNING_ACTION_INST = """
Your Task: Given a user's goal, break it down into clear, manageable sub-tasks that are easy to follow and efficient to execute. 

### Instructions:
1. **Understand the Goal**: Identify the core objective the user is trying to achieve. 
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
- **Complete Code Generation**: For programming-related tasks, ensure that the final sub-task outputs a complete and working solution.
- **IMPORTANT - Include Full Requirements**: For EVERY code generation tasks, in addition to the outputs from previous sub-tasks, the overall goal (and analysed requirements if any) MUST be included as inputs. This ensures each code generation step maintains full context of what's being built, even when split across multiple steps.

### Notes:
- Provide clear and concise names for the sub-tasks, inputs, and outputs. 
- Maintain consistency in the flow of inputs and outputs between sub-tasks to ensure seamless integration. 
- The inputs of a sub-task can ONLY be chosen from the workflow's inputs and any outputs from its preceding sub-tasks. 
- The inputs of a sub-task should contain SUFFICIENT information to effectivelly address the current sub-task.
- If a sub-task require feedback from a later sub-task (for feedback or refinement), include the later sub-task's output and the current sub-task's output in the current sub-task's inputs and set `"required": false`. 
- You will be provided with the inputs and outputs requirements of the workflow in the "### Workflow Inputs" and "### Workflow Outputs" sections. You must incorporate all inputs in "### Workflow Inputs". All outputs in "### Workflow Outputs" must be the direct results of the sub-tasks.
- You MUST NOT alter the provided workflow inputs or outputs in any way. Use them exactly as they are in your sub-tasks.
"""

TASK_PLANNING_ACTION_DEMOS = """
### Examples: 
Below are some generated workflows that follow the given instructions:

{examples}
"""

TASK_PLANNING_EXAMPLES = """
Example 1:
### User's goal:
Given the name of a popular movie, return the 3 most recent reviews for that movie from trusted review sites (like Rotten Tomatoes, IMDb, or Metacritic). The reviews should include the review title, author, and a short summary.

For instance:
Input: movie_name = "The Dark Knight"
Output:
```json
[
    {
        "review_title": "A Dark Masterpiece",
        "author": "John Doe",
        "summary": "Christopher Nolan's direction and Heath Ledger's performance make 'The Dark Knight' a must-watch."
    },
    {
        "review_title": "A Cinematic Triumph",
        "author": "Jane Smith",
        "summary": "A captivating sequel that pushes the boundaries of superhero cinema."
    },
    {
        "review_title": "A Grim Tale of Justice and Vengeance",
        "author": "Samantha Green",
        "summary": "*The Dark Knight* perfectly balances action and philosophy, with Christian Bale and Heath Ledger giving powerhouse performances that elevate the film to iconic status."
    }
]
```

### Workflow Inputs:
```json
[
    {
        "name": "movie_name",
        "type": "string",
        "required": true,
        "description": "The name of the movie for which the 3 most recent reviews are needed."
    }
]
```

### Workflow Outputs:
```json
[
    {
        "name": "movie_reviews",
        "type": "array",
        "required": true,
        "description": "An array of the 3 most recent reviews for the given movie, including the review title, author, and summary."
    }
]
```

### Generated Workflow:
```json
{
    "sub_tasks": [
        {
            "name": "task_search",
            "description": "Perform a web search for recent reviews of the specified movie on trusted review sites like Rotten Tomatoes, IMDb, and Metacritic.",
            "reason": "The task gathers the most relevant and up-to-date reviews from reputable sources.",
            "inputs": [
                {
                    "name": "movie_name",
                    "type": "string",
                    "required": true,
                    "description": "The name of the movie to search for."
                }
            ],
            "outputs": [
                {
                    "name": "search_results",
                    "type": "array",
                    "required": true,
                    "description": "A list of search results that includes recent reviews from trusted sources."
                }
            ]
        },
        {
            "name": "task_extract_reviews",
            "description": "Extract the 3 most recent reviews from the search results, focusing on the review title, author, and summary.",
            "reason": "This task processes the search results to format them according to the user's needs.",
            "inputs": [
                {
                    "name": "search_results",
                    "type": "array",
                    "required": true,
                    "description": "The search results containing relevant movie reviews."
                }
            ],
            "outputs": [
                {
                    "name": "movie_reviews",
                    "type": "array",
                    "required": true,
                    "description": "The 3 most recent reviews for the movie, including the review title, author, and summary."
                }
            ]
        }
    ]
}
```
"""

TASK_PLANNING_EXAMPLE_TEMPLATE = """
Example {example_id}:
### User's Goal:
{goal}

### Workflow Inputs:
```json
{workflow_inputs}
```

### Workflow Outputs:
```json
{workflow_outputs}
```

### Generated Workflow:
```json
{workflow_plan}
```

"""


TASK_PLANNING_OUTPUT_FORMAT = """
### Output Format
Your final output should ALWAYS in the following format:

## Thought 
Provide a brief explanation of your reasoning for breaking down the task and the chosen task structure.  

## Goal
Restate the user's goal clearly and concisely.

## Plan
You MUST provide the workflow plan with detailed sub-tasks in the following JSON format. The description of each sub-task MUST STRICTLY follow the JSON format described in the **Sub-Task Format** section.
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

### Workflow Inputs:
{workflow_inputs}

### Workflow Outputs:
{workflow_outputs}

### Output:
"""

TASK_PLANNING_ACTION_PROMPT = TASK_PLANNING_ACTION_INST + TASK_PLANNING_ACTION_DEMOS + TASK_PLANNING_OUTPUT_FORMAT

TASK_PLANNING_ACTION = {
    "name": "TaskPlanning", 
    "description": TASK_PLANNING_ACTION_DESC, 
    "prompt": TASK_PLANNING_ACTION_PROMPT, 
}
