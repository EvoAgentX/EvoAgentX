from pydantic import Field, create_model
from typing import Optional, Any, Callable, Type, List, Union
import re
import json

from ..core.logging import logger
from ..models.base_model import BaseLLM
from .action import Action, ActionInput, ActionOutput
from ..core.message import Message
from ..prompts.tool_calling import GOAL_BASED_TOOL_CALLING_PROMPT, OUTPUT_EXTRACTION_PROMPT
from ..tools.tool import Tool
from ..utils.utils import generate_dynamic_class_name
from ..core.registry import MODULE_REGISTRY
from ..models.base_model import LLMOutputParser
from ..core.module_utils import parse_json_from_llm_output
from ..prompts.template import StringTemplate

class CustomizeAction(Action):
    max_tool_try: int = 2
    tools_schema: Optional[dict] = None
    tools_caller: Optional[dict[str, Callable]] = None
    tool_calling_instructions: Optional[list] = None
    conversation: Optional[Message] = None
    inputs: dict = {}
    outputs: dict = {}
    output_parser: Optional[Type[ActionOutput]] = None
    prompt: str = ""
    prompt_template: Optional[StringTemplate] = None
    parse_mode: str = "title"
    title_format: str = "## {title}"
    custom_output_format: Optional[str] = None
    execution_history: list[Any] = []
    customize_prompting: bool = False
    llm: BaseLLM = None
    
    def __init__(self, **kwargs):
        name = kwargs.pop("name", "tool_calling")
        description = kwargs.pop("description", "Call a tool function and return the results")
        inputs_format = kwargs.pop("inputs")
        outputs_format = kwargs.pop("outputs")
        output_parser = kwargs.pop("output_parser", None)
        
        # Handle template-related parameters
        prompt = kwargs.pop("prompt", "")
        prompt_template = kwargs.pop("prompt_template", None)
        parse_mode = kwargs.pop("parse_mode", "title")
        title_format = kwargs.pop("title_format", "## {title}")
        custom_output_format = kwargs.pop("custom_output_format", None)
        
        super().__init__(name=name, description=description, **kwargs)
        
        # Validate that at least one of prompt or prompt_template is provided
        if not prompt and not prompt_template:
            raise ValueError("`prompt` or `prompt_template` is required when creating CustomizeAction action")
        # Prioritize template and give warning if both are provided
        if prompt and prompt_template:
            logger.warning("Both `prompt` and `prompt_template` are provided for CustomizeAction action." " Prioritizing `prompt_template` and ignoring `prompt`.")
            self.prompt = prompt_template.get_instruction()
            self.prompt_template = prompt_template
            self.execution_history = [prompt_template.get_history()]
        elif prompt_template:
            self.prompt_template = prompt_template
            self.prompt = prompt_template.get_instruction()
            self.execution_history = [prompt_template.get_history()]
        else:
            self.prompt_template = StringTemplate(
                instruction = prompt
            )
            self.prompt = prompt
        
        self._generate_inputs_outputs_info(inputs_format, outputs_format, output_parser, name)
        self.parse_mode = parse_mode
        self.title_format = title_format
        self.custom_output_format = custom_output_format
        # Allow max_tool_try to be configured through kwargs
        self.max_tool_try = kwargs.pop("max_tool_try", 2)
        self.customize_prompting = kwargs.pop("customize_prompting", False)
    
    
    def prepare_action_prompt(self, inputs: Optional[dict] = None, system_prompt: Optional[str] = None, 
                            execution_history: Optional[list] = None) -> Union[str, List[dict]]:
        """Prepare prompt for action execution.
        
        This helper function transforms the input dictionary into a formatted prompt
        for the language model, handling different prompting modes.
        
        Args:
            inputs: Dictionary of input parameters
            system_prompt: Optional system prompt to include
            execution_history: History of tool executions for goal-based tool calling
            
        Returns:
            Union[str, List[dict]]: Formatted prompt ready for LLM (string or chat messages)
            
        Raises:
            TypeError: If an input value type is not supported
            ValueError: If neither prompt nor prompt_template is available
        """
        # Process inputs into prompt parameter values
        if inputs is None:
            inputs = {}
            
        prompt_params_names = self.inputs_format.get_attrs()
        prompt_params_values = {}
        for param in prompt_params_names:
            value = inputs.get(param, "")
            if isinstance(value, str):
                prompt_params_values[param] = value
            elif isinstance(value, (dict, list)):
                prompt_params_values[param] = json.dumps(value, indent=4)
            else:
                raise TypeError(f"The input type {type(value)} is invalid! Valid types: [str, dict, list].")

        # Handle different prompting modes based on customize_prompting flag
        if self.customize_prompting:
            # Use custom prompting with template formatting (no tool calling)
            if self.prompt_template is not None:
                return self.prompt_template.format(
                    system_prompt=system_prompt
                )
            else:
                # Simple string prompt formatting
                return self.prompt.format(**prompt_params_values) if prompt_params_values else self.prompt
        else:
            # Use goal-based tool calling mode
            
            if self.prompt_template is not None:
                # Set history if provided and format with tools for goal-based tool calling
                if execution_history is not None:
                    self.prompt_template.set_history(execution_history)
                return self.prompt_template.format(
                    system_prompt=system_prompt,
                    values=prompt_params_values,
                    inputs_format=self.inputs_format,
                    outputs_format=self.outputs_format,
                    parse_mode=self.parse_mode,
                    title_format=self.title_format,
                    custom_output_format=self.custom_output_format,
                    tools=self.tools
                )
            else:
                # Use GOAL_BASED_TOOL_CALLING_PROMPT for simple prompt
                return GOAL_BASED_TOOL_CALLING_PROMPT.format(
                    goal_prompt=self.prompt,
                    inputs=prompt_params_values,
                    history=execution_history or [],
                    tools_description=self.tools_schema or {},
                    additional_context=self.tool_calling_instructions or []
                )
        
        # This should never be reached due to validation in __init__
        raise ValueError("`prompt` or `prompt_template` is required when creating a CustomizeAction.")

    def prepare_extraction_prompt(self, llm_output_content: str) -> str:
        """Prepare extraction prompt for fallback extraction when parsing fails.
        
        Args:
            self: The action instance
            llm_output_content: Raw output content from LLM
            
        Returns:
            str: Formatted extraction prompt
        """
        attr_descriptions: dict = self.outputs_format.get_attr_descriptions()
        output_description_list = [] 
        for i, (name, desc) in enumerate(attr_descriptions.items()):
            output_description_list.append(f"{i+1}. {name}\nDescription: {desc}")
        output_description = "\n\n".join(output_description_list)
        return OUTPUT_EXTRACTION_PROMPT.format(text=llm_output_content, output_description=output_description)
    
    
    def _generate_inputs_outputs_info(self, inputs: Union[List[dict], Type], outputs: Union[List[dict], Type], output_parser: ActionOutput = None, name: str = None):
        # Handle inputs - check if it's already a processed class or raw list
        if isinstance(inputs, type) and hasattr(inputs, 'get_attrs'):
            # Already processed input class
            action_input_type = inputs
        else:
            # Raw list of dictionaries - create the action input type
            action_input_fields = {}
            for field in inputs:
                required = field.get("required", True)
                if required:
                    action_input_fields[field["name"]] = (str, Field(description=field["description"]))
                else:
                    action_input_fields[field["name"]] = (Optional[str], Field(default=None, description=field["description"]))
            
            action_input_type = create_model(
                self._get_unique_class_name(
                    generate_dynamic_class_name(name+" action_input")
                ),
                **action_input_fields, 
                __base__=ActionInput
            )
        
        # Handle outputs - check if it's already a processed class or raw list
        if isinstance(outputs, type) and hasattr(outputs, 'get_attrs'):
            # Already processed output class
            action_output_type = outputs
        else:
            # Raw list of dictionaries or using output_parser
            if output_parser is None:
                action_output_fields = {}
                for field in outputs:
                    required = field.get("required", True)
                    if required:
                        action_output_fields[field["name"]] = (Union[str, dict, list], Field(description=field["description"]))
                    else:
                        action_output_fields[field["name"]] = (Optional[Union[str, dict, list]], Field(default=None, description=field["description"]))
                
                action_output_type = create_model(
                    self._get_unique_class_name(
                        generate_dynamic_class_name(name+" action_output")
                    ),
                    **action_output_fields, 
                    __base__=ActionOutput,
                    # get_content_data=customize_get_content_data,
                    # to_str=customize_to_str
                )
            else:
                # self._check_output_parser(outputs, output_parser)
                action_output_type = output_parser
        
        self.inputs_format = action_input_type
        self.outputs_format = action_output_type
    
    def _get_unique_class_name(self, candidate_name: str) -> str:
        """
        Get a unique class name by checking if it already exists in the registry.
        If it does, append "Vx" to make it unique.
        """
        if not MODULE_REGISTRY.has_module(candidate_name):
            return candidate_name 
        
        i = 1 
        while True:
            unique_name = f"{candidate_name}V{i}"
            if not MODULE_REGISTRY.has_module(unique_name):
                break
            i += 1 
        return unique_name 
    
    def add_tools(self, tools: List[Tool]):
        if not self.tools_schema:
            self.tools_schema = {}
            self.tools_caller = {}
            self.tool_calling_instructions = []
            self.tools = []
        if not tools:
            return
        self.tools += tools
        tools_schemas = [tool.get_tool_schemas() for tool in tools]
        tools_schemas = [j for i in tools_schemas for j in i]
        tools_callers = [tool.get_tools() for tool in tools]
        tools_callers = [j for i in tools_callers for j in i]
        tools_names = [i["function"]["name"] for i in tools_schemas]
        self.tool_calling_instructions += [tool.get_tool_prompt() for tool in tools]
        for tool_schema, tool_caller, tool_name in zip(tools_schemas, tools_callers, tools_names):
            self.tools_schema[tool_name] = tool_schema
            self.tools_caller[tool_name] = tool_caller
        self.prompt_template.set_tools(self.tools)
    
    def _extract_tool_calls(self, llm_output: str):
        if match := re.search(r"```(?:ToolCalling)?\s*\n(.*?)\n```", llm_output, re.DOTALL):
            json_str = match.group(1)
            return json.loads(json_str)
        return None
    
    def _extract_output(self, llm_output: Any, llm: BaseLLM = None, **kwargs):
        # Get the raw output content
        if hasattr(llm_output, 'content'):
            llm_output_content = llm_output.content
        else:
            llm_output_content = str(llm_output)
        
        # Check if there are any defined output fields
        output_attrs = self.outputs_format.get_attrs()
        
        # If no output fields are defined, create a simple content-only output
        if not output_attrs:
            # Create output with just the content field
            output = self.outputs_format(content=llm_output_content)
            print("Created simple content output for agent with no defined outputs:")
            print(output)
            return output
        
        # Use the action's parse_mode and parse_func for parsing
        try:
            # Use the outputs_format's parse method with the action's parse settings
            parsed_output = self.outputs_format.parse(
                content=llm_output_content,
                parse_mode=self.parse_mode,
                parse_func=getattr(self, 'parse_func', None),
                title_format=getattr(self, 'title_format', "## {title}")
            )
            
            print("Successfully parsed output using action's parse settings:")
            print(parsed_output)
            return parsed_output
            
        except Exception as e:
            print(f"Failed to parse with action's parse settings: {e}")
            print("Falling back to extraction prompt...")
            
            # Fall back to extraction prompt if direct parsing fails
            extraction_prompt = self.prepare_extraction_prompt(llm_output_content)
                
            llm_extracted_output: LLMOutputParser = llm.generate(prompt=extraction_prompt, history=kwargs.get("history", []) + [llm_output_content])
            llm_extracted_data: dict = parse_json_from_llm_output(llm_extracted_output.content)
            output = self.outputs_format.from_dict(llm_extracted_data)
            
            print("Extracted output using fallback:")
            print(output)
            
            return output
    
    def _calling_tools(self, tool_call_args) -> dict:
        ## ___________ Call the tools ___________
        errors = []
        results  =[]
        for function_param in tool_call_args:
            try:
                function_name = function_param.get("function_name")
                function_args = function_param.get("function_args") or {}
        
                # Check if we have a valid function to call
                if not function_name:
                    errors.append("No function name provided")
                    break
                
                # Try to get the callable from our tools_caller dictionary
                callable_fn = None
                if self.tools_caller and function_name in self.tools_caller:
                    callable_fn = self.tools_caller[function_name]
                elif callable(function_name):
                    callable_fn = function_name
                        
                if not callable_fn:
                    errors.append(f"Function '{function_name}' not found or not callable")
                    break
                
                try:
                    # Determine if the function is async or not
                    print("_____________________ Start Function Calling _____________________")
                    print(f"Executing function calling: {function_name} with {function_args}")
                    result = callable_fn(**function_args)
                
                except Exception as e:
                    logger.error(f"Error executing tool {function_name}: {e}")
                    errors.append(f"Error executing tool {function_name}: {str(e)}")
                    break
            
                results.append(result)
            except Exception as e:
                logger.error(f"Error executing tool: {e}")
                errors.append(f"Error executing tool: {str(e)}")
        

        ## 3. Add the tool call results to the query and continue the conversation
        results = {"result": results, "error": errors}
        return results
    
    def _get_current_prompt(self, prompt_params_values: dict = None) -> str:
        """Get the current prompt for return, formatted appropriately."""
        return self.prepare_action_prompt(inputs=prompt_params_values or {})
    
    def execute(self, llm: Optional[BaseLLM] = None, inputs: Optional[dict] = None, system_prompt = None, return_prompt: bool = False, time_out = 0, **kwargs):
        # Allow empty inputs if the action has no required input attributes
        input_attributes: dict = self.inputs_format.get_attr_descriptions()
        if not inputs and input_attributes:
            logger.error("CustomizeAction action received invalid `inputs`: None or empty.")
            raise ValueError('The `inputs` to CustomizeAction action is None or empty.')
        
        # Set inputs to empty dict if None and no inputs are required
        if inputs is None:
            inputs = {}

        self.execution_history = []
        final_llm_response = None
        
        ## 1. Generate tool call args
        prompt_params_values = {k: inputs.get(k, "") for k in input_attributes.keys()}
        
        while True:
            ### Generate response from LLM
            if time_out > self.max_tool_try:
                # Get the appropriate prompt for return
                current_prompt = self._get_current_prompt(prompt_params_values)
                # Use the final LLM response if available, otherwise fall back to execution history
                content_to_extract = final_llm_response if final_llm_response is not None else "{content}".format(content = self.execution_history)
                if return_prompt:
                    return self._extract_output(content_to_extract, llm = llm), current_prompt
                return self._extract_output(content_to_extract, llm = llm) 
            time_out += 1
            
            # Use goal-based tool calling prompt
            prompt = self.prepare_action_prompt(
                inputs=prompt_params_values,
                execution_history=self.execution_history,
                system_prompt = system_prompt
            )
            
            # Handle both string prompts and chat message lists
            if isinstance(prompt, str):
                llm_response = llm.generate(
                    prompt=prompt
                )
            else:
                llm_response = llm.generate(
                    messages=prompt
                )
            
            # Store the final LLM response
            final_llm_response = llm_response
            
            tool_call_args = self._extract_tool_calls(llm_response.content)
            if not tool_call_args:
                break
            
            print("Extracted tool call args:")
            print(tool_call_args)
            
            results = self._calling_tools(tool_call_args)
            
            print("results:")
            print(results)
            
            self.execution_history.append({"tool_call_args": tool_call_args, "results": results})
        
        # Get the appropriate prompt for return
        current_prompt = self._get_current_prompt(prompt_params_values)
        # Use the final LLM response if available, otherwise fall back to execution history
        content_to_extract = final_llm_response if final_llm_response is not None else "{content}".format(content = self.execution_history)
        if return_prompt:
            return self._extract_output(content_to_extract, llm = llm), current_prompt
        return self._extract_output(content_to_extract, llm = llm)
        

    async def async_execute(self, llm: Optional[BaseLLM] = None, inputs: Optional[dict] = None, system_prompt = None, return_prompt: bool = False, time_out = 0, **kwargs) -> ActionOutput:
        return self.execute(llm, inputs, system_prompt, return_prompt, time_out, **kwargs)