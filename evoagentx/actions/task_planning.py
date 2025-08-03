import json
from typing import Dict, List, Optional

from pydantic import Field

from ..core.logging import logger
from ..models.base_model import BaseLLM
from ..prompts.task_planner import TASK_PLANNING_ACTION, TASK_PLANNING_EXAMPLES
from ..workflow.workflow_graph import WorkFlowNode
from .action import Action, ActionInput, ActionOutput


class TaskPlanningInput(ActionInput):
    """
    Input specification for the task planning action.
    """
    goal: str = Field(description="A clear and detailed description of the user's goal, specifying what needs to be achieved.")
    history: Optional[str] = Field(default=None, description="Optional field containing previously generated task plan.")
    suggestion: Optional[str] = Field(default=None, description="Optional suggestions or ideas to guide the planning process.")
    examples: Optional[List[Dict]] = Field(default=None, description="Task planning examples")


class TaskPlanningOutput(ActionOutput):
    """
    Output structure for the task planning action.
    """
    sub_tasks: List[WorkFlowNode] = Field(description="A list of sub-tasks that collectively achieve user's goal.")
    

class TaskPlanning(Action):
    """
    Action for planning a series of tasks to achieve a goal.
    """

    def __init__(self, **kwargs):

        name = kwargs.pop("name") if "name" in kwargs else TASK_PLANNING_ACTION["name"]
        description = kwargs.pop("description") if "description" in kwargs else TASK_PLANNING_ACTION["description"]
        prompt = kwargs.pop("prompt") if "prompt" in kwargs else TASK_PLANNING_ACTION["prompt"]
        # inputs_format = kwargs.pop("inputs_format") if "inputs_format" in kwargs else TaskPlanningInput
        # outputs_format = kwargs.pop("outputs_format") if "outputs_format" in kwargs else TaskPlanningOutput
        inputs_format = kwargs.pop("inputs_format", None) or TaskPlanningInput
        outputs_format = kwargs.pop("outputs_format", None) or TaskPlanningOutput
        super().__init__(name=name, description=description, prompt=prompt, inputs_format=inputs_format, outputs_format=outputs_format, **kwargs)
    
    def execute(self, llm: Optional[BaseLLM] = None, inputs: Optional[dict] = None, sys_msg: Optional[str]=None, return_prompt: bool = False, **kwargs) -> TaskPlanningOutput:
        """Execute the task planning process.
        
        This method uses the provided language model to generate a structured
        plan of sub-tasks based on the user's goal and any additional context.
        
        Args:
            llm: The language model to use for planning.
            inputs: Input data containing the goal and optional context.
            sys_msg: Optional system message for the language model.
            return_prompt: Whether to return both the task plan and the prompt used.
            **kwargs: Additional keyword arguments.
            
        Returns:
            If return_prompt is False (default): The generated task plan.
            If return_prompt is True: A tuple of (task plan, prompt used).
            
        Raises:
            ValueError: If the inputs are None or empty.
        """
        if not inputs:
            logger.error("TaskPlanning action received invalid `inputs`: None or empty.")
            raise ValueError('The `inputs` to TaskPlanning action is None or empty.')
        
        prompt_params_names = self.inputs_format.get_attrs()

        prompt_params_values = dict()
        for param in prompt_params_names:
            if param in inputs and inputs[param] is not None:
                prompt_params_values[param] = inputs[param]
            else:
                prompt_params_values[param] = ""
        
        if isinstance(prompt_params_values["examples"], list) and len(prompt_params_values["examples"]) > 0:
            prompt_params_values["examples"] = self.format_task_planning_examples(
                prompt_params_values["examples"]
            )
        else:
            prompt_params_values["examples"] = TASK_PLANNING_EXAMPLES

        prompt = self.prompt.format(**prompt_params_values)
        task_plan = llm.generate(
            prompt = prompt, 
            system_message = sys_msg, 
            parser=self.outputs_format,
            parse_mode="json"
        )
        
        if return_prompt:
            return task_plan, prompt
        
        return task_plan

    @staticmethod
    def format_task_planning_examples(examples: List[Dict]) -> str:
        """
        Returns a formatted string containing all the examples that can be used in a prompt.

        Args:
            exmaples (List[Dict]): A list of dictionaries where each dictionary is
                an example and must contain `goal` and `sub_tasks` keys.

        Returns: A string with the following format
            Example 1:
            ### User's goal:
            {goal}
            ### Generated Workflow:
            ```json
            {
                "sub_tasks": [
                    {
                        ...
                    },
                    ...
                ]
            }
            ```
            
            Example 2:
            ...
        """
        if len(examples) == 0:
            return ""

        prompt = []

        for i, example in enumerate(examples):
            goal = example.pop("goal")
            tasks_str = json.dumps(example, indent=4)
            example_prompt = f"Example {i+1}:\n### User's goal:\n{goal}\n### Generated Workflow: \n```json\n{tasks_str}\n```\n"
            prompt.append(example_prompt)

        return "\n".join(prompt) 
