from .customize_action import CustomizeAction
from ..prompts.web_agent import WEB_AGENT_ACTION_PROMPT
from ..models.base_model import BaseLLM
from ..prompts.template import PromptTemplate, StringTemplate
from pydantic import Field
from datetime import datetime
from typing import Optional
from ..models.base_model import LLMOutputParser
from ..core.module_utils import parse_json_from_llm_output
from ..actions.action import ActionOutput
from ..core.logging import logger
import json
import time

class WebOperationOutput(ActionOutput):
    new_action_record: str = Field(description="The new action record you have drafted.")
    new_information: str = Field(description="The new information you have extracted.")
    new_links: str = Field(description="The new links you have extracted.")
    thinking: str = Field(description="The thinking process you have drafted.")
    decision: str = Field(description="The decision you have drafted.")
    
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
            instruction=WEB_AGENT_ACTION_PROMPT.format(**input_data),
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
        
        # Initialize timing variables
        total_start_time = time.time()
        iteration_count = 0
        total_prompt_generation_time = 0
        total_llm_generation_time = 0
        total_tool_calls_time = 0
        total_tool_extraction_time = 0
        total_tool_execution_time = 0
        total_extraction_time = 0
        
        logger.info("🚀 Starting WebSearchAction execution")
        
        while True:
            iteration_count += 1
            iteration_start_time = time.time()
            
            # Time prompt generation
            prompt_start_time = time.time()
            operation_prompt = self._generate_operation_prompt(inputs)
            prompt_generation_time = time.time() - prompt_start_time
            total_prompt_generation_time += prompt_generation_time
            
            logger.info(f"📝 Iteration {iteration_count} - Prompt generation: {prompt_generation_time:.2f}s")
            
            # Time LLM generation
            llm_start_time = time.time()
            operation_result = llm.generate(
                prompt=operation_prompt,
                system_message=sys_msg
            )
            llm_generation_time = time.time() - llm_start_time
            total_llm_generation_time += llm_generation_time
            
            logger.info(f"🤖 Iteration {iteration_count} - LLM generation: {llm_generation_time:.2f}s")
            try:
                new_action_record = WebOperationOutput.parse(operation_result.content).new_action_record
                new_information = WebOperationOutput.parse(operation_result.content).new_information
                new_links = WebOperationOutput.parse(operation_result.content).new_links
                thinking = WebOperationOutput.parse(operation_result.content).thinking
                decision = WebOperationOutput.parse(operation_result.content).decision
            except Exception:
                new_action_record = ""
                new_information = ""
                new_links = ""
                thinking = operation_result.content
                decision = ""
            self._update_searching_memory(new_links, new_information, new_action_record)
            
            # Time tool calls - detailed breakdown
            tool_calls_start_time = time.time()
            
            # Time tool call extraction
            extraction_start_time = time.time()
            tool_call_args = self._extract_tool_calls(operation_result.content)
            extraction_time = time.time() - extraction_start_time
            
            if not tool_call_args:
                tool_calls_time = time.time() - tool_calls_start_time
                total_tool_calls_time += tool_calls_time
                total_tool_extraction_time += extraction_time
                logger.info(f"🔧 Iteration {iteration_count} - Tool extraction: {extraction_time:.2f}s")
                logger.info(f"🔧 Iteration {iteration_count} - Tool calls (no calls): {tool_calls_time:.2f}s")
                break
            
            total_tool_extraction_time += extraction_time
            logger.info(f"🔧 Iteration {iteration_count} - Tool extraction: {extraction_time:.2f}s")
            logger.info("Extracted tool call args:")
            logger.info(json.dumps(tool_call_args, indent=4))
            
            # Time individual tool executions
            tool_execution_start_time = time.time()
            results = self._calling_tools(tool_call_args)
            tool_execution_time = time.time() - tool_execution_start_time
            total_tool_execution_time += tool_execution_time
            
            tool_calls_time = time.time() - tool_calls_start_time
            total_tool_calls_time += tool_calls_time
            
            logger.info(f"🔧 Iteration {iteration_count} - Tool execution: {tool_execution_time:.2f}s")
            logger.info(f"🔧 Iteration {iteration_count} - Total tool calls: {tool_calls_time:.2f}s")
            logger.info("Tool call results:")
            logger.info(json.dumps(results, indent=4))
            
            self.searching_memory["current_state"] = results
            
            # Log iteration total time
            iteration_total_time = time.time() - iteration_start_time
            logger.info(f"⏱️  Iteration {iteration_count} total time: {iteration_total_time:.2f}s")
            
            
        # Time final extraction step
        extraction_start_time = time.time()
        logger.info("📊 Starting final extraction step")
        
        # Get the appropriate prompt for return
        current_prompt = self._generate_operation_prompt(inputs or {})
        content_to_extract = f"{self.searching_memory['collected_information']}"
        extraction_prompt = self.prepare_extraction_prompt(content_to_extract)
        llm_extracted_output: LLMOutputParser = llm.generate(prompt=extraction_prompt)
        llm_extracted_data: dict = parse_json_from_llm_output(llm_extracted_output.content)
        output = self.outputs_format.from_dict(llm_extracted_data)
        
        extraction_time = time.time() - extraction_start_time
        total_extraction_time = extraction_time
        
        # Calculate total execution time
        total_execution_time = time.time() - total_start_time
        
        # Log comprehensive timing summary
        logger.info("=" * 60)
        logger.info("📈 WebSearchAction Timing Summary")
        logger.info("=" * 60)
        logger.info(f"🔄 Total iterations: {iteration_count}")
        logger.info(f"📝 Total prompt generation time: {total_prompt_generation_time:.2f}s ({total_prompt_generation_time/total_execution_time*100:.1f}%)")
        logger.info(f"🤖 Total LLM generation time: {total_llm_generation_time:.2f}s ({total_llm_generation_time/total_execution_time*100:.1f}%)")
        logger.info(f"🔧 Total tool calls time: {total_tool_calls_time:.2f}s ({total_tool_calls_time/total_execution_time*100:.1f}%)")
        logger.info(f"   ├─ Tool extraction time: {total_tool_extraction_time:.2f}s ({total_tool_extraction_time/total_execution_time*100:.1f}%)")
        logger.info(f"   └─ Tool execution time: {total_tool_execution_time:.2f}s ({total_tool_execution_time/total_execution_time*100:.1f}%)")
        logger.info(f"📊 Final extraction time: {total_extraction_time:.2f}s ({total_extraction_time/total_execution_time*100:.1f}%)")
        logger.info(f"⏱️  Total execution time: {total_execution_time:.2f}s")
        logger.info("=" * 60)
        
        if return_prompt:
            return output, current_prompt
        return output
        