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

## Instructions
1. **Understand the Goal**: Accurately interpret the core objective the user is trying to accomplish.
2. **Review the History**: Examine any previous task plans or partial attempts to identify areas for refinement or improvement.
3. **Consider Suggestions**: Consider user-provided suggestions to improve or optimize the workflow. 
4. **Define Sub-Tasks**: Decompose the goal into logical, actionable sub-tasks based on the complexity of the goal.

## Critical Requirements
- **Inputs Rules**:
    * A sub-task's inputs can only come from:
        1. Inputs from the **Workflow Inputs** section, and
        2. Outputs from other sub-tasks.
    * Each input in the **Workflow Inputs** section MUST be used by at least one sub-task.
    * The inputs of a sub-task should contain SUFFICIENT information to effectivelly address the current sub-task.
- **Outputs Rules**:
    * Each sub-task must produce outputs that are:
        1. Used by other sub-tasks, or
        2. One or more of the outputs in the **Workflow Outputs** section.
    * Do not generate outputs that are unused or irrelevant to the workflow.
- **Strictly Adhere to Provided Workflow Inputs and Outputs**: You MUST NOT alter the inputs and outputs in **Workflow Inputs** and **Workflow Outputs** in any way when using them in sub-tasks. Use them exactly as they are and pay close attention to the `name`, `type` and `required` fields in the sub-tasks that use them to make sure they are consistent with the ones given in **Workflow Inputs** and **Workflow Outputs**.

## Task Structuring Principles
- **Simplicity**: Each sub-task is designed to achieve a specific, clearly defined objective. Avoid overloading sub-tasks with multiple objectives. 
- **Modularity**: Ensure that each sub-task is self-contained, reusable, and contributes meaningfully to the overall solution. 
- **Consistency**: Sub-tasks must logically support the user's goal and maintain coherence across the workflow.
- **Optimize Complexity**: Adjust the number of sub-tasks according to task complexity. Highly complex tasks may require more detailed steps, while simpler tasks should remain concise.
- **Avoid Redundancy**: Ensure that there are no overlapping or unnecessary sub-tasks. 
- **Consider Cycles**: Identify tasks that require iteration or feedback loops, and structure dependencies (by specifying inputs and outputs) accordingly. 

## Sub-Task Format
Each sub-task must follow this exact structure:
```json
{{
    "name": "subtask_name",
    "description": "A clear and concise explanation of what this sub-task achieves.",
    "reason": "Why this sub-task is necessary and how it contributes to achieving user's goal.",
    "inputs": [
        {{
            "name": "the input's name", 
            "type": "string/int/float/other_type",
            "required": true/false,
            "description": "Description of the input's purpose and usage."
        }},
        ...
    ], 
    "outputs": [
        {{
            "name": "the output's name", 
            "type": "string/int/float/other_type",
            "required": true/false, 
            "description": "Description of the output produced by this sub-task."
        }},
        ...
    ]
}}
```

## Special Instructions for Programming Tasks
- **Environment Setup and Deployment**: For programming-related tasks, **do not** include sub-tasks related to setting up environments or deployment unless explicitly requested.
- **Complete Code Generation**: For programming-related tasks, ensure that the final sub-task outputs a complete and working solution.
- **IMPORTANT - Include Full Requirements**: For EVERY code generation tasks, in addition to the outputs from previous sub-tasks, the overall goal (and analysed requirements if any) MUST be included as inputs. This ensures each code generation step maintains full context of what's being built, even when split across multiple steps.
"""

TASK_PLANNING_ACTION_DEMOS = """
## Examples: 
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
## Output Format
Your final output should ALWAYS in the following format:

### Thought

Describe your reasoning in a step-by-step fashion, following these phases:

1. **Goal Interpretation**  
   Begin by interpreting the user's goal. What is the user ultimately trying to achieve? Paraphrase or reframe the goal if needed to make it more actionable or concrete.

2. **High-Level Task Segmentation**  
   Think about the **major steps** needed to achieve the goal.  
   - What are the key stages or phases of the process?  
   - What logical components naturally break apart into separate steps?  
   - If a subtask is doing too much, can it be split?  
   - If two subtasks are too small or meaningless on their own, should they be merged?

3. **Information & Data Flow**  
   Consider how data flows through the process.  
   - Which subtasks produce outputs that are needed by later steps?  
   - Which inputs from the workflow are needed, and at what stage?
   - Which subtasks depend on others being completed first?

5. **Validation Against Goal & Outputs**  
   Cross-check that your subtask design, when executed, will:
   - Satisfy all outputs requirements in the **Workflow Outputs** section.
   - Use all inputs in the **Workflow Inputs** section.
   - Align with the user's end-goal.

6. **Mermaid Graph**
    Create a Mermaid diagram that visualizes the relationships and execution flow between the subtasks.

Keep this section focused on **how** you're thinking and making design choices. The specifics of each subtask will come later.


### Goal
Restate the user's goal clearly and concisely.

### Inputs to Include
List the exact names of all inputs from the **Workflow Inputs** section.
Even if an input has `"required": false`, it must be included in the plan.

### Outputs to Include
List the exact names of all outputs from the **Workflow Outputs** section that the workflow will produce.
Even if an output has `"required": false`, it must be included in the plan.

### Subtask Details
Now break down each subtask with full technical detail:
For each subtask, include:
- **Description**: What it does
- **Reason**: Why this sub-task is necessary and how it contributes to achieving user's goal.
- **Inputs**: List of inputs (can include workflow inputs or outputs from other subtasks)
- **Outputs**: List of outputs this subtask generates
- **Depends on**: Any subtasks that must be completed before this one

### Workflow Plan
Create the final workflow plan in the following JSON format. This must match the details from the previous section.
Each sub-task MUST STRICTLY follow the JSON format described in the **Sub-Task Format** section.
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
