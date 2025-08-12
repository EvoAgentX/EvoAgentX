import json
from typing import Dict, List, Optional, Union

from pydantic import Field

from ..agents.agent import Agent
from ..core.base_config import Parameter
from ..core.logging import logger
from ..core.module import BaseModule
from ..models.base_model import BaseLLM
from ..prompts.agent_generator import AGENT_GENERATION_ACTION
from ..prompts.template import PromptTemplate
from ..prompts.tool_calling import AGENT_GENERATION_TOOLS_PROMPT
from ..tools.tool import Tool, Toolkit
from .action import Action, ActionInput, ActionOutput


class AgentGenerationInput(ActionInput):
    """
    Input specification for the agent generation action.
    """
    goal: str = Field(description="A detailed statement of the workflow's goal, explaining the objectives the entire workflow aims to achieve")
    workflow: str = Field(description="An overview of the entire workflow, detailing all sub-tasks with their respective names, descriptions, inputs, and outputs")
    task: str = Field(description="A detailed JSON representation of the sub-task requiring agent generation. It should include the task's name, description, inputs, and outputs.")

    history: Optional[str] = Field(default=None, description="Optional field containing previously selected or generated agents.")
    suggestion: Optional[str] = Field(default=None, description="Optional suggestions to refine the generated agents.")
    prebuilt_agents: Optional[List[Agent]] = Field(default=None, description="Optional list containing predefined agents that can be selected for the sub-task.")
    tools: Optional[List[Union[Toolkit, Tool]]] = Field(default=None, description="Optional list containing the tools that agents can use.")
    examples: Optional[List[Dict]] = Field(default=None, description="Optional list containing examples of agent generation.")


class GeneratedAgent(BaseModule):
    """
    Representation of a generated agent.
    """
    name: str 
    description: str 
    inputs: List[Parameter]
    outputs: List[Parameter]
    prompt_template: PromptTemplate
    tool_names: Optional[List[str]] = None


class SelectedAgent(BaseModule):
    """
    Representation of a selected agent.
    """
    name: str 
    inputs: List[Parameter]
    outputs: List[Parameter]


class AgentGenerationOutput(ActionOutput):
    selected_agents: List[SelectedAgent] = Field(description="A list of selected agent's names")
    generated_agents: List[GeneratedAgent] = Field(description="A list of generated agents to address a sub-task")


class AgentGeneration(Action):
    """
    Action for generating agent specifications for workflow tasks.
    
    This action analyzes task requirements and generates appropriate agent
    specifications, including their prompts, inputs, and outputs. It can either
    select from existing agents or create new ones tailored to the task.
    """

    def __init__(self, **kwargs):
        name = kwargs.pop("name") if "name" in kwargs else AGENT_GENERATION_ACTION["name"]
        description = kwargs.pop("description") if "description" in kwargs else AGENT_GENERATION_ACTION["description"]
        prompt = kwargs.pop("prompt") if "prompt" in kwargs else AGENT_GENERATION_ACTION["prompt"]
        # inputs_format = kwargs.pop("inputs_format") if "inputs_format" in kwargs else AgentGenerationInput
        # outputs_format = kwargs.pop("outputs_format") if "outputs_format" in kwargs else AgentGenerationOutput
        inputs_format = kwargs.pop("inputs_format", None) or AgentGenerationInput
        outputs_format = kwargs.pop("outputs_format", None) or AgentGenerationOutput 
        tools = kwargs.pop("tools", None)
        super().__init__(name=name, description=description, prompt=prompt, inputs_format=inputs_format, outputs_format=outputs_format, **kwargs)
        self.tools = tools
    
    def execute(self, llm: Optional[BaseLLM] = None, inputs: Optional[dict] = None, sys_msg: Optional[str]=None, return_prompt: bool = False, **kwargs) -> AgentGenerationOutput:
        """Execute the agent generation process.
        
        This method uses the provided language model to generate agent specifications
        based on the workflow context and task requirements.
        
        Args:
            llm: The language model to use for generation.
            inputs: Input data containing workflow and task information.
            sys_msg: Optional system message for the language model.
            return_prompt: Whether to return both the generated agents and the prompt used.
            **kwargs: Additional keyword arguments.
            
        Returns:
            If return_prompt is False (default): The generated agents output.
            If return_prompt is True: A tuple of (generated agents, prompt used).
            
        Raises:
            ValueError: If the inputs are None or empty.
        """
        if not inputs:
            logger.error("AgentGeneration action received invalid `inputs`: None or empty.")
            raise ValueError('The `inputs` to AgentGeneration action is None or empty.')
        
        inputs_format: AgentGenerationInput = self.inputs_format
        outputs_format: AgentGenerationOutput = self.outputs_format

        prompt_params_names = inputs_format.get_attrs()
        prompt_params_values = dict()

        for param in prompt_params_names:
            if param in inputs and inputs[param] is not None:
                prompt_params_values[param] = inputs[param]
            else:
                prompt_params_values[param] = "None"

        if isinstance(prompt_params_values["examples"], list):
            prompt_params_values["examples"] = self.format_agent_examples(
                prompt_params_values["examples"]
            )

        if isinstance(prompt_params_values["prebuilt_agents"], list):
            prompt_params_values["prebuilt_agents"] = self.format_prebuilt_agents(
                prompt_params_values["prebuilt_agents"]
            )

        if isinstance(self.tools, list) and len(self.tools) > 0:
            tool_description = self.format_tools(self.tools)
            prompt_params_values["tools"] = AGENT_GENERATION_TOOLS_PROMPT.format(tools_description=tool_description)
        else:
            prompt_params_values["tools"] = "None"
        
        prompt = self.prompt.format(**prompt_params_values)
        agents = llm.generate(
            prompt = prompt, 
            system_message = sys_msg, 
            parser=outputs_format,
            parse_mode="json"
        )
        
        if return_prompt:
            return agents, prompt
        
        return agents


    async def async_execute(self, llm: Optional[BaseLLM] = None, inputs: Optional[dict] = None, sys_msg: Optional[str]=None, return_prompt: bool = False, **kwargs) -> AgentGenerationOutput:
        """Execute the agent generation process asynchronously.
        
        This method uses the provided language model to generate agent specifications
        based on the workflow context and task requirements.
        
        Args:
            llm: The language model to use for generation.
            inputs: Input data containing workflow and task information.
            sys_msg: Optional system message for the language model.
            return_prompt: Whether to return both the generated agents and the prompt used.
            **kwargs: Additional keyword arguments.
            
        Returns:
            If return_prompt is False (default): The generated agents output.
            If return_prompt is True: A tuple of (generated agents, prompt used).
        """
        if not inputs:
            logger.error("AgentGeneration action received invalid `inputs`: None or empty.")
            raise ValueError('The `inputs` to AgentGeneration action is None or empty.')
        
        inputs_format: AgentGenerationInput = self.inputs_format
        outputs_format: AgentGenerationOutput = self.outputs_format

        prompt_params_names = inputs_format.get_attrs()
        prompt_params_values = dict()

        for param in prompt_params_names:
            if param in inputs and inputs[param] is not None:
                prompt_params_values[param] = inputs[param]
            else:
                prompt_params_values[param] = "None"

        if isinstance(prompt_params_values["examples"], list):
            prompt_params_values["examples"] = self.format_agent_examples(
                prompt_params_values["examples"]
            )

        if isinstance(prompt_params_values["prebuilt_agents"], list):
            prompt_params_values["prebuilt_agents"] = self.format_prebuilt_agents(
                prompt_params_values["prebuilt_agents"]
            )

        if isinstance(self.tools, list) and len(self.tools) > 0:
            tool_description = self.format_tools(self.tools)
            prompt_params_values["tools"] = AGENT_GENERATION_TOOLS_PROMPT.format(tools_description=tool_description)
        else:
            prompt_params_values["tools"] = "None"
        
        prompt = self.prompt.format(**prompt_params_values)
        agents = await llm.async_generate(
            prompt = prompt, 
            system_message = sys_msg, 
            parser=outputs_format,
            parse_mode="json"
        )
        
        if return_prompt:
            return agents, prompt
        
        return agents

    @staticmethod
    def format_agent_examples(examples: List[Dict]) -> str:
        """
        Args:
            examples (List[Dict]): A list of dictionaries where each dictionary must have
                a "subtask" key and an "agents" key.

        Example output:
        Example 1:
        **Sub-Task**:
        ```json
        {subtask}
        ```

        **Output**:
        ```json
        {
            "selected_agents": [...],
            "generated_agents": [...]
        }
        ```
        """
        if len(examples) == 0:
            return ""

        prompt = []

        for i, example in enumerate(examples):
            subtask = json.dumps(example["subtask"], indent=4)

            output = {
                "selected_agents": [],
                "generated_agents": example["agents"]
            }
            output = json.dumps(output, indent=4)
            
            prompt.append(f"Example {i+1}:\n**Sub-Task**:\n```json\n{subtask}\n```\n\n**Output**:\n```json\n{output}\n```\n\n")

        return "\n".join(prompt)

    @staticmethod
    def format_prebuilt_agents(agents: List[Agent]) -> str:
        """
        Example output:
        - **Agent 1**: Description of agent 1
        - **Agent 2**: Description of agent 2
        """
        if len(agents) == 0:
            return ""
        
        prompt = []

        for agent in agents:
            name = agent.name
            description = agent.description
            prompt.append(f"- **{name}**: {description}")
        
        return "\n".join(prompt)

    @staticmethod
    def format_tools(tools: List[Union[Tool, Toolkit]]) -> str:
        """
        Example output:
        - **Tool 1**: Description of tool 1
        - **Tool 2**: Description of tool 2
        - **Toolkit** is a toolkit that provides the following functionalities:
            * Description of tool 1 in toolkit
            * Description of tool 2 in toolkit
        """
        if len(tools) == 0:
            return ""
        
        prompt = []

        for tool in tools:
            if isinstance(tool, Tool):
                name = tool.name
                description = tool.description
                prompt.append(f"- **{name}**: {description}")
            elif isinstance(tool, Toolkit):
                name = tool.name
                description = "is a toolkit that provides the following functionalities:"

                for tool in tool.get_tools():
                    description += f"\n  * {tool.description}"
                
                prompt.append(f"- **{name}** {description}")

        return "\n".join(prompt)

