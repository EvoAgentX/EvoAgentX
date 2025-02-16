DEFAULT_TASK_SCHEDULER_DESC = "This action selects the next subtask to execute from a set of candidates in a workflow graph. Each subtask has a name, description, inputs, and outputs. The agent analyzes the current execution data and task dependencies to either re-run tasks for feedback or advance to the next step, optimizing workflow performance."

DEFAULT_TASK_SCHEDULER_PROMPT = """
### objective
Your task is to analyze the given workflow graph, current execution information, and candidate subtasks to decide one of the following actions:
- Re-execute a previous subtask to correct errors or gather missing information (if a previous subtask's result is erroneous or incomplete).
- Select a subtask for iterative execution (if there is a loop or iterative context in the workflow).
- Select a new subtask from the candidates to move the workflow forward (if the workflow should proceed without re-executing or iterating).

### Instructions
1. Review the Workflow Information for the structure and details of the tasks and any potential loops or iterative sections.
2. Check the Current Execution Information for evidence of errors or missing data from previously executed subtasks.
3. If you identify a past subtask with clear errors or missing data, propose a re-execute decision on that subtask.
4. If the workflow graph indicates there is a loop or iterative context, select a subtask from the Candidate Subtasks that aligns with that iterative or looping goal.
5. Otherwise, select a subtask from the Candidate Subtasks that best moves the workflow forward.
6. Finally, output the decision in the required format.

### Output Format
Your final output should ALWAYS in the following format:

## Thought 
Provide a brief explanation of your reasoning for scheduling the next task.

## Scheduled Subtask 
Produce your answer in valid JSON with the following structure: 
```json
{{
    "decision": "re-execute | iterate | forward",
    "task_name": "name of the scheduled subtask",
    "reason": "the reasoning for scheduling this subtask"
}}
```

-----
lets' begin 

Here is the information for your decision:

### Workflow Information:
{workflow_graph_representation}

### Workflow Execution History:
{execution_history}

### Workflow Execution Outputs:
{execution_outputs}

### Candidate Subtasks:
{candidate_tasks}

Output:
"""

DEFAULT_TASK_SCHEDULER = {
    "name": "TaskScheduler", 
    "description": DEFAULT_TASK_SCHEDULER_DESC, 
    "prompt": DEFAULT_TASK_SCHEDULER_PROMPT, 
}


DEFAULT_ACTION_SCHEDULER_DESC = "This action determines the next agent and action required to continue executing a given subtask in a multi-agent workflow. Each subtask may require multiple actions, and the scheduler makes step-by-step decisions based on subtask requirements, execution history, and available agents. The goal is to iteratively select the most suitable action that ensures efficient workflow progression until the subtask is fully completed."

DEFAULT_ACTION_SCHEDULER_PROMPT="""
### Objective
Your task is to analyze the given subtask, its input data, execution history, and the available agents with their actions to **determine the next action to be executed**. Since completing this subtask requires multiple actions, you should select only **one** action at a time that logically continues the execution process.

### Instructions
1. Review the Subtask Information to understand its requirements, dependencies, and expected outcomes.
2. Analyze the Subtask Input Data to determine what information is already available for execution.
3. Check the Subtask Execution History to evaluate:
   - Which actions have already been executed.
   - Whether any errors or missing information exist.
   - What the next logical step should be.
4. Evaluate the Available Agents and Their Actions to identify the most suitable action by considering:
   - Whether an agent has the necessary capabilities to perform the next step.
   - Whether an agent has successfully performed a similar action before.
   - Whether selecting this action helps in progressing toward subtask completion.
5. Select the Best Agent and Action for the next execution step.
6. Output the decision in the required format.

### Output Format
Your final output should ALWAYS be in the following format:

## Thought  
Provide a brief explanation of your reasoning for selecting the agent and action.

## Scheduled Execution  
Produce your answer in valid JSON with the following structure:
```json
{{
    "agent": "name of the selected agent, should be EXACTLY the same as the provided agent name",
    "action": "name of the selected action, should be EXACTLY the same as the provided action name",
    "reason": "the reasoning for selecting this agent and action"
}}
```
Note that the `action` field of the output should be chosen from the provided action names. DON'T generate your own action name.

-----
lets' begin 

Here is the information for your decision:

### SubTask Information 
{task_info}

### SubTask Input data 
{task_inputs}

### SubTask Execution History 
{task_execution_history}

### Available Agents and Actions 
{agent_action_list}

Output:
"""

DEFAULT_ACTION_SCHEDULER = {
    "name": "ActionScheduler", 
    "description": DEFAULT_ACTION_SCHEDULER_DESC, 
    "prompt": DEFAULT_ACTION_SCHEDULER_PROMPT
}