AGENT_GENERATOR_DESC = "AgentGenerator is an intelligent agent assignment system designed to support task execution. \
    Its primary function is to generate or assign suitable agents for each sub-task within a workflow. \
        By focusing on one sub-task at a time, AgentGenerator ensures that the specific requirements and \
            objectives of that sub-task are met by assigning the most appropriate agents." 

AGENT_GENERATOR_SYSTEM_PROMPT = "You are an expert in agent generation and assignment. Your role is to analyze a single sub-task within a workflow, \
    evaluate its requirements and objectives, and generate or assign agents that can efficiently complete it. Ensure that each agent matches the sub-task's needs to facilitate successful execution."

AGENT_GENERATOR = {
    "name": "AgentGenerator", 
    "description": AGENT_GENERATOR_DESC,
    "system_prompt": AGENT_GENERATOR_SYSTEM_PROMPT,
}


AGENT_GENERATION_ACTION_DESC = "This action focuses on a single sub-task within a workflow. It analyzes the sub-task's requirements and objectives, \
    then generates or assigns one or more suitable agents capable of efficiently completing the task."


AGENT_GENERATION_ACTION_PROMPT = """
You are tasked with generating agents to complete a sub-task within a workflow. Analyze the sub-task's requirements and objectives, and assign or create suitable agents to ensure its successful execution.

### Instructions
1. **Understand the workflow**: Familiar yourself with the overall workflow, including its goal, to understand the relationship between the sub-tasks and and how outputs from one may serve as inputs for another. 
2. **Analyse the Sub-Task**: Review the sub-task details in the following format to fully understand its objective, requirements, and expected inputs and outputs.
```json
{{
    "name": "subtask_name",
    "description": "A clear and concise explanation of the goal of this sub-task.",
    "reason": "Why this sub-task is necessary and how it contributes to achieving user's goal.",
    "inputs": [
        {{
            "name": "the input's name", 
            "type": "string/int/float/other_type",
            "required": true/false (`false` means the input is the feedback from later sub-task, or the previous output for the current sub-task), 
            "description": "Description of the input's purpose and usage."
        }},
        ...
    ], 
    "outputs": [
        {{
            "name": "the output's name", 
            "type": "string/int/float/other_type",
            "required": true (the `required` field of outputs are always true), 
            "description": "Description of the output produced by this sub-task."
        }},
        ...
    ]
}}
```
3. **Identify Required Capabilities**: 
- Determine the skills, resources, or capabilities needed to perform the sub-task effectively. 
- If necessary, identify tools provided in the "### Tools" section that the agents can use to complete their tasks efficiently.
4. **Agent Selection**:
- Review the prebuilt agents and their descriptions in the "### Prebuilt Agents" section.
- Select one or more agents (if provided) that can fulfill part or all of the sub-task's requirements. 
- If the provided agents are not relevant, you may choose not to select any. Similarly, you may select only the agents that are directly applicable to the sub-task.
- You MUST provide the inputs and outputs for each selected agent.
```json
{{
    "name": "the name of the selected agent", 
    "inputs": [
        {{
            "name": "the input's name", 
            "type": "string/int/float/other_type",
            "required": true/false (`false` means the input is the feedback from later sub-task, or the previous output for the current sub-task), 
            "description": "Description of the input's purpose and usage."
        }},
        ...
    ], 
    "outputs": [
        {{
            "name": "the output's name", 
            "type": "string/int/float/other_type",
            "required": true (the `required` field of outputs are always true), 
            "description": "Description of the output produced by this sub-task."
        }},
        ...
    ]
}}
```
5. **Agent Generation**: If the selected predefined agents cannot fully address all aspects of the sub-task, create additional agents to handle the remaining functionality. Follow these principles when creating new agents:
5.1 **Agent Structure**: Each generated agent MUST be defined in the following JSON format:
```json
{{
    "name": "A concise identifier of the agent",
    "description": "A summary of the agent's role and how it contributes to solving the task.",
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
            "description": "Description of the output produced by this agent."
        }},
        ...
    ],
    "prompt_template": {{
        "class_name": "ChatTemplate",
        "instruction": "(Required) A detailed prompt that instructs the agent on how to fulfill its responsibilities.",
        "constraints": "(Optional) Constraints that the agent must follow. A list of strings", 
    }},
    "tool_names": "(Optional) The tools the agent may use, selected from the tools listed in the '### Tools' section. If no tool is required or no tools are provided, set this field to `null`, otherwise set as a list of str.",
}}
```
5.3 **Determine the Number of Agents**: Decide how many agents are needed based on the task's complexity and requirements. 
- **Sequential WorkFlow**: Agents should work sequentially, where the outputs of an agent can serve as inputs for the following agents. 
- **Distinct Responsibilities**: Ensure each agent has a distinct, non-overlapping responsibility. 
5.4 **Validation**: Make sure the result can be correctly parsed as a JSON object. Pay special attention to the JSON string within the `prompt` field of an agent.


### Notes:
- Use concise, meaningful names for agents, inputs, and outputs. 
- Ensure that ALL `inputs` defined in the sub-task are used by at least one created agent. 
- Ensure that ALL `outputs` defined in the sub-task can be derived from the `outputs` of the created agents.
- Ensure that the generated agent's input and output strictly follow the input and output names defined in the sub-task description. Do not replace the task's expected input and output with internal function parameters or return values.  

### Output Format
Your final output should ALWAYS in the following format:

## Thought 
Briefly explain the reasoning process for the selection of predefined agents and the generation of new agents.

## Objective
Restate the objectives and requirements of the sub-task. 

## Selected or Generated Agents
- You MUST output the selected and generated agents in the following JSON format. Even if there are not selected or generated agents, still include the `selected_agents` and `generated_agents` fields by setting them as empty list. 
- The description of each **generated** agent MUST STRICTLY follow the JSON format described in the **Agent Structure** section. If a generated agent doesn't require inputs or do not have ouputs, still include `inputs` and `outputs` in the definiton by setting them as empty list. 
```json
{{
    "selected_agents": [

        "name": "the name of the selected agent", 
        ... (other selected agent fields)
    ],
    "generated_agents": [
        {{
            "name": "the name of the generated agent", 
            ... (other generated agent fields)
        }}
    ]
}}
```

### Examples
{examples}

----- 
Let's begin. 

### History (previously selected or generated agents):
{history}

### Suggestions (suggestions to refine the selected or generated agents):
{suggestion}

### Prebuilt Agents
{prebuilt_agents}

### Tools
{tools}

### User's Goal:
{goal}

### Workflow:
{workflow}

### Sub-Task:
{task}

### Output: 
"""

AGENT_GENERATION_ACTION = {
    "name": "AgentGeneration", 
    "description": AGENT_GENERATION_ACTION_DESC, 
    "prompt": AGENT_GENERATION_ACTION_PROMPT, 
}
