from .customize_action import CustomizeAction
from ..prompts.web_agent import WEB_AGENT_ACTION_PROMPT, WEB_AGENT_SUMMARIZATION_PROMPT
from ..models.base_model import BaseLLM
from ..prompts.template import StringTemplate
from pydantic import Field
from datetime import datetime
from typing import Optional
from ..actions.action import ActionOutput
import json

class WebOperationOutput(ActionOutput):
    new_action_record: str = Field(description="The new action record you have drafted.")
    new_information: str = Field(description="The new information you have extracted.")
    new_links: str = Field(description="The new links you have extracted.")
    thinking: str = Field(description="The thinking process you have drafted.")
    decision: str = Field(description="The decision you have drafted.")
    
class WebSearchAction(CustomizeAction):
    
    def __init__(self, search_budget = 10, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.search_budget = search_budget
        self.searching_memory = {
            "environment_information": self._get_environment_information(),
            "links": ["google.com"],
            "collected_information": [],
            "action_records": [],
            "current_state": None,
            "left_budget": self.search_budget
        }

    def _update_searching_memory(self, new_links: list, new_information: list, new_action_record: str):
        self.searching_memory["links"].append(new_links)
        self.searching_memory["collected_information"].append(new_information)
        self.searching_memory["action_records"].append(new_action_record)
    
    def _get_environment_information(self):
        return {"current_date": datetime.now().strftime("%Y-%m-%d"), "current_time": datetime.now().strftime("%H:%M:%S")}
    
    def _generate_operation_prompt(self, task_inputs: dict):
        input_data = {
            "task_inputs": task_inputs,
            "environment_information": self.searching_memory["environment_information"],
            "links": self.searching_memory["links"],
            "collected_information": self.searching_memory["collected_information"],
            "action_records": self.searching_memory["action_records"],
            "current_state": self.searching_memory["current_state"],
            "left_budget": self.searching_memory["left_budget"],
        }
        if self.prompt:
            input_data["goal_description"] = self.prompt
        else:
            input_data["goal_description"] = self.prompt_template.get_instruction()
        template = StringTemplate(
            instruction=WEB_AGENT_ACTION_PROMPT.format(**input_data),
            parse_mode="title",
            title_format=self.title_format,
            tools=self.tools
        )
        return template.format(
            values=task_inputs,
            inputs_format=self.inputs_format,
            outputs_format=self.outputs_format,
            parse_mode="title",
            title_format=self.title_format,
            custom_output_format=self.custom_output_format,
            tools=self.tools
        )
    
    def execute(self, llm: Optional[BaseLLM] = None, inputs: Optional[dict] = None, sys_msg: Optional[str]=None, return_prompt: bool = False, **kwargs):
        """Execute the task planning process.
        
        This method uses the provided language model to generate a structured
        plan of sub-tasks based on the user's goal and any additional context.
        
        Args:
            llm: The language model to use for planning.
            inputs: Input data containing the goal and optional context.
            sys_msg: Optional system message for the language model.
            return_prompt: Whether to return both the task plan and the prompt used.
            **kwargs: Additional keyword arguments.
        """
        
        iteration_count = 0
        
        while True:
            iteration_count += 1
            operation_prompt = self._generate_operation_prompt(inputs)
            
            if not self.searching_memory["left_budget"]:
                break
            
            operation_result = llm.generate(
                prompt=operation_prompt,
                system_message=sys_msg
            )
            try:
                new_action_record = WebOperationOutput.parse(operation_result.content).new_action_record
                new_information = WebOperationOutput.parse(operation_result.content).new_information
                new_links = WebOperationOutput.parse(operation_result.content).new_links
            except Exception:
                new_action_record = ""
                new_information = ""
                new_links = ""
            self._update_searching_memory(new_links, new_information, new_action_record)
            
            tool_call_args = self._extract_tool_calls(operation_result.content)
            
            if not tool_call_args:
                break
            
            print(f"\n=== Tool Call Arguments (Iteration {iteration_count}) ===")
            print(json.dumps(tool_call_args, indent=2))
            
            results = self._calling_tools(tool_call_args)
            
            print(f"\n=== Tool Call Results (Iteration {iteration_count}) ===")
            print(json.dumps(results, indent=2))
            print("=" * 50)
            
            self.searching_memory["current_state"] = results
            
            self.searching_memory["left_budget"] -= 1
            
        # Get the appropriate prompt for return
        current_prompt = self._generate_operation_prompt(inputs or {})
        
        web_search_process = {
            "goal_description": self.prompt if self.prompt else self.prompt_template.get_instruction(),
            "task_inputs": inputs,
            "environment_information": self.searching_memory["environment_information"],
            "links": self.searching_memory["links"],
            "collected_information": self.searching_memory["collected_information"],
            "action_records": self.searching_memory["action_records"],
            "current_state": self.searching_memory["current_state"],
        }
        extraction_prompt = StringTemplate(instruction = WEB_AGENT_SUMMARIZATION_PROMPT.format(**web_search_process), parse_mode="title", title_format=self.title_format, output_format=self.outputs_format)
        llm_extracted_output = llm.generate(extraction_prompt.format())
        output = {"output": llm_extracted_output.content}
        
        if return_prompt:
            return output, current_prompt
        return output
    
    async def async_execute(self, llm: Optional[BaseLLM] = None, inputs: Optional[dict] = None, sys_msg: Optional[str]=None, return_prompt: bool = False, time_out = 0, **kwargs):
        # Allow empty inputs if the action has no required input attributes
        iteration_count = 0
        
        while True:
            iteration_count += 1
            operation_prompt = self._generate_operation_prompt(inputs)
            
            if not self.searching_memory["left_budget"]:
                break
            
            operation_result = await llm.async_generate(
                prompt=operation_prompt,
                system_message=sys_msg
            )
            try:
                new_action_record = WebOperationOutput.parse(operation_result.content).new_action_record
                new_information = WebOperationOutput.parse(operation_result.content).new_information
                new_links = WebOperationOutput.parse(operation_result.content).new_links
            except Exception:
                new_action_record = ""
                new_information = ""
                new_links = ""
            self._update_searching_memory(new_links, new_information, new_action_record)
            
            tool_call_args = self._extract_tool_calls(operation_result.content)
            
            if not tool_call_args:
                break
            
            print(f"\n=== Tool Call Arguments (Iteration {iteration_count}) ===")
            print(json.dumps(tool_call_args, indent=2))
            
            results = self._calling_tools(tool_call_args)
            
            print(f"\n=== Tool Call Results (Iteration {iteration_count}) ===")
            print(json.dumps(results, indent=2))
            print("=" * 50)
            
            self.searching_memory["current_state"] = results
            
            self.searching_memory["left_budget"] -= 1
            
        # Get the appropriate prompt for return
        current_prompt = self._generate_operation_prompt(inputs or {})
        
        web_search_process = {
            "goal_description": self.prompt if self.prompt else self.prompt_template.get_instruction(),
            "task_inputs": inputs,
            "environment_information": self.searching_memory["environment_information"],
            "links": self.searching_memory["links"],
            "collected_information": self.searching_memory["collected_information"],
            "action_records": self.searching_memory["action_records"],
            "current_state": self.searching_memory["current_state"],
        }
        extraction_prompt = StringTemplate(instruction = WEB_AGENT_SUMMARIZATION_PROMPT.format(**web_search_process), parse_mode="title", title_format=self.title_format, output_format=self.outputs_format)
        llm_extracted_output = await llm.async_generate(extraction_prompt.format())
        output = {"output": llm_extracted_output.content}
        
        if return_prompt:
            return output, current_prompt
        return output
        