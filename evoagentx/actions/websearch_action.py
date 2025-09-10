from .customize_action import CustomizeAction
from ..prompts.web_agent import WEB_AGENT_OPERATION_PROMPT, WEB_AGENT_RESULT_EXTRACTION_PROMPT
from ..models.base_model import BaseLLM
from ..prompts.template import PromptTemplate, StringTemplate
from pydantic import Field
from typing import Optional
from ..actions.action import ActionOutput
from ..core.logging import logger
import json

class WebOperationOutput(ActionOutput):
    thinking: str = Field(description="The thinking process you have drafted.")
    decision: str = Field(description="The decision you have drafted.")

class WebResultExtractionOutput(ActionOutput):
    new_action_record: str = Field(description="The new action record you have drafted.")
    new_information: str = Field(description="The new information you have extracted.")
    new_links: str = Field(description="The new links you have extracted.")

class WebSearchAction(CustomizeAction):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.searching_memory = {
            "environment_information": None,
            "links": ["google.com"],
            "collected_information": [],
            "action_records": [],
            "current_state": None,
        }

    def _update_searching_memory(self, new_links: list, new_information: list, new_action_record: str):
        self.searching_memory["links"].append(new_links)
        self.searching_memory["collected_information"].append(new_information)
        self.searching_memory["action_records"].append(new_action_record)
    
    def _get_environment_information(self):
        return {"current_date": datetime.now().strftime("%Y-%m-%d"), "current_time": datetime.now().strftime("%H:%M:%S")}
    
    def _generate_operation_prompt(self, task_inputs: dict):
        if self.prompt:
            input_data = {
                "task_inputs": task_inputs,
                "goal_description": self.prompt,
                "environment_information": self.searching_memory["environment_information"],
                "links": self.searching_memory["links"],
                "collected_information": self.searching_memory["collected_information"],
                "action_records": self.searching_memory["action_records"],
                "current_state": self.searching_memory["current_state"],
            }
        else:
            input_data = {
                "task_inputs": task_inputs,
                "goal_description": self.prompt_template.get_instruction(),
                "environment_information": self.searching_memory["environment_information"],
                "links": self.searching_memory["links"],
                "collected_information": self.searching_memory["collected_information"],
                "action_records": self.searching_memory["action_records"],
                "current_state": self.searching_memory["current_state"],
            }
        template = StringTemplate(
            instruction=WEB_AGENT_OPERATION_PROMPT.format(**input_data),
            parse_mode="title",
            title_format=self.title_format,
            tools=self.tools
        )
        return template.format(
            values=input_data,
            inputs_format=self.inputs_format,
            outputs_format=self.outputs_format,
            parse_mode="title",
            title_format=self.title_format,
            custom_output_format=self.custom_output_format,
            tools=self.tools
        )
    
    def _generate_result_extraction_prompt(self, task_inputs: dict):
        if self.prompt:
            input_data = {
                "task_inputs": task_inputs,
                "result": result,
                "goal_description": self.prompt,
                "environment_information": self.searching_memory["environment_information"],
                "links": self.searching_memory["links"],
                "collected_information": self.searching_memory["collected_information"],
                "action_records": self.searching_memory["action_records"],
                "current_state": self.searching_memory["current_state"],
            }
        else:
            input_data = {
                "task_inputs": task_inputs,
                "goal_description": self.prompt_template.get_instruction(),
                "environment_information": self.searching_memory["environment_information"],
                "links": self.searching_memory["links"],
                "collected_information": self.searching_memory["collected_information"],
                "action_records": self.searching_memory["action_records"],
                "current_state": self.searching_memory["current_state"],
            }
        template = StringTemplate(
            instruction=WEB_AGENT_RESULT_EXTRACTION_PROMPT.format(**input_data),
            parse_mode="title",
            title_format=self.title_format,
            tools=self.tools
        )
        
        return template.format(
            values=input_data,
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
        
        while True:
            operation_prompt = self._generate_operation_prompt(inputs)
            operation_result = llm.generate(
                prompt=operation_prompt,
                system_message=sys_msg
            )
            try:
                thinking = WebOperationOutput.parse(operation_result.content).thinking
                decision = WebOperationOutput.parse(operation_result.content).decision
            except Exception:
                thinking = operation_result.content
                decision = ""
            
            from pdb import set_trace; set_trace()
            
            tool_call_args = self._extract_tool_calls(operation_result.content)
            if not tool_call_args:
                break
            
            logger.info("Extracted tool call args:")
            logger.info(json.dumps(tool_call_args, indent=4))
            
            results = self._calling_tools(tool_call_args)
            
            logger.info("Tool call results:")
            logger.info(json.dumps(results, indent=4))
            
            self.searching_memory["current_state"] = results
            
            result_extraction_prompt = self._generate_result_extraction_prompt(inputs)
            result_extraction_result = llm.generate(
                prompt=result_extraction_prompt,
                system_message=sys_msg
            )
            try:
                new_action_record = WebResultExtractionOutput.parse(result_extraction_result.content).new_action_record
                new_information = WebResultExtractionOutput.parse(result_extraction_result.content).new_information
                new_links = WebResultExtractionOutput.parse(result_extraction_result.content).new_links
            except Exception:
                new_action_record = result_extraction_result.content
                new_information = ""
                new_links = ""
            
            self._update_searching_memory(new_links, new_information, new_action_record)
            
        # Get the appropriate prompt for return
        current_prompt = self._generate_operation_prompt(inputs or {})
        content_to_extract = f"{self.searching_memory}"
        extraction_prompt = self.prepare_extraction_prompt(content_to_extract)
        llm_extracted_output: LLMOutputParser = llm.generate(prompt=extraction_prompt)
        llm_extracted_data: dict = parse_json_from_llm_output(llm_extracted_output.content)
        output = self.outputs_format.from_dict(llm_extracted_data)
        
        if return_prompt:
            return output, current_prompt
        return output
        