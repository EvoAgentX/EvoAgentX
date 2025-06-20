from pydantic import Field
from typing import Optional, Any, Callable, List, Union, Dict
import re
import json

from ..core.logging import logger
from ..models.base_model import BaseLLM
from .action import Action
from ..core.message import Message
from ..prompts.tool_calling import OUTPUT_EXTRACTION_PROMPT, TOOL_CALLING_TEMPLATE, TOOL_CALLING_HISTORY_PROMPT
from ..tools.tool import Tool
from ..core.registry import MODULE_REGISTRY
from ..models.base_model import LLMOutputParser
from ..core.module_utils import parse_json_from_llm_output, parse_json_from_text

class CustomizeAction(Action):

    parse_mode: Optional[str] = Field(default="title", description="the parse mode of the action, must be one of: ['title', 'str', 'json', 'xml', 'custom']")
    parse_func: Optional[Callable] = Field(default=None, exclude=True, description="the function to parse the LLM output. It receives the LLM output and returns a dict.")
    title_format: Optional[str] = Field(default="## {title}", exclude=True, description="the format of the title. It is used when the `parse_mode` is 'title'.")
    custom_output_format: Optional[str] = Field(default=None, exclude=True, description="the format of the output. It is used when the `prompt_template` is provided.")

    tools_schema: Optional[Dict[str, Any]] = Field(default=None, description="Schema definitions for available tools")
    tools_caller: Optional[Dict[str, Callable]] = Field(default=None, description="Mapping of tool names to their callable functions")
    tool_calling_instructions: Optional[List[str]] = Field(default=None, description="Additional instructions for tool calling")
    conversation: Optional[Message] = Field(default=None, description="Current conversation state")

    max_tool_try: int = Field(default=2, description="Maximum number of tool calling attempts allowed")
    
    def __init__(self, **kwargs):

        name = kwargs.pop("name", "CustomizeAction")
        description = kwargs.pop("description", "Customized action that can use tools to accomplish its task")

        super().__init__(name=name, description=description, **kwargs)
        
        # Validate that at least one of prompt or prompt_template is provided
        if not self.prompt and not self.prompt_template:
            raise ValueError("`prompt` or `prompt_template` is required when creating CustomizeAction action")
        # Prioritize template and give warning if both are provided
        if self.prompt and self.prompt_template:
            logger.warning("Both `prompt` and `prompt_template` are provided for CustomizeAction action. Prioritizing `prompt_template` and ignoring `prompt`.")
    
    def prepare_action_prompt(
        self, 
        inputs: Optional[dict] = None, 
        system_prompt: Optional[str] = None, 
        **kwargs
    ) -> Union[str, List[dict]]:
        """Prepare prompt for action execution.
        
        This helper function transforms the input dictionary into a formatted prompt
        for the language model, handling different prompting modes.
        
        Args:
            inputs: Dictionary of input parameters
            system_prompt: Optional system prompt to include
            
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
        
        if self.prompt:
            prompt = self.prompt.format(**prompt_params_values) if prompt_params_values else self.prompt
            if self.tools:
                prompt += "\n\n" + TOOL_CALLING_TEMPLATE.format(tools_description = self.tools_schema, additional_context = self.tool_calling_instructions)
            return prompt
        else:
            # Use goal-based tool calling mode
            if self.tools:
                self.prompt_template.set_tools(self.tools)
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
    
    def _extract_tool_calls(self, llm_output: str) -> List[dict]:
        # if match := re.search(r"```(?:ToolCalling)?\s*\n(.*?)\n```", llm_output, re.DOTALL):
        #     json_list = parse_json_from_text(match.group(1))
        #     return json.loads(json_list[0] if json_list else "{}")

        # Improved regex pattern to match ```ToolCalling blocks more accurately
        # This pattern handles:
        # - Optional whitespace after ToolCalling
        # - Content capture with proper handling of newlines
        # - Optional whitespace before closing ```
        pattern = r"```ToolCalling\s*\n(.*?)\n\s*```"
        
        # Find all ToolCalling blocks in the output
        matches = re.findall(pattern, llm_output, re.DOTALL)

        if not matches:
            return []
        
        parsed_tool_calls = []
        for match_content in matches:
            try:
                json_content = match_content.strip()
                json_list = parse_json_from_text(json_content)
                if not json_list:
                    logger.warning("No valid JSON found in ToolCalling block")
                    continue
                # Only use the first JSON string from each block
                parsed_tool_call = json.loads(json_list[0])
                if isinstance(parsed_tool_call, dict):
                    parsed_tool_calls.append(parsed_tool_call)
                elif isinstance(parsed_tool_call, list):
                    parsed_tool_calls.extend(parsed_tool_call)
                else:
                    logger.warning(f"Invalid tool call format: {parsed_tool_call}")
                    continue
            except (json.JSONDecodeError, IndexError) as e:
                logger.warning(f"Failed to parse tool calls from LLM output: {e}")
                continue

        return parsed_tool_calls
    
    def _extract_output(self, llm_output: Any, llm: BaseLLM = None, **kwargs):

        # Get the raw output content
        llm_output_content = getattr(llm_output, "content", str(llm_output))
        
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
    
    async def _async_extract_output(self, llm_output: Any, llm: BaseLLM = None, **kwargs):
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
                
            llm_extracted_output = await llm.async_generate(prompt=extraction_prompt, history=kwargs.get("history", []) + [llm_output_content])
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
    
    def execute(self, llm: Optional[BaseLLM] = None, inputs: Optional[dict] = None, sys_msg: Optional[str]=None, return_prompt: bool = False, time_out = 0, **kwargs):
        # Allow empty inputs if the action has no required input attributes
        input_attributes: dict = self.inputs_format.get_attr_descriptions()
        if not inputs and input_attributes:
            logger.error("CustomizeAction action received invalid `inputs`: None or empty.")
            raise ValueError('The `inputs` to CustomizeAction action is None or empty.')
        # Set inputs to empty dict if None and no inputs are required
        if inputs is None:
            inputs = {}
        final_llm_response = None
        
        if self.prompt_template:
            conversation = [{"role": "system", "content": self.prepare_action_prompt(inputs=inputs, system_prompt=sys_msg)}]
        else:
            conversation = [{"role": "system", "content": sys_msg}, {"role": "user", "content": self.prepare_action_prompt(inputs=inputs, system_prompt=sys_msg)}]
        
        
        ## 1. get all the input parameters
        prompt_params_values = {k: inputs.get(k, "") for k in input_attributes.keys()}
        while True:
            ### Generate response from LLM
            if time_out > self.max_tool_try:
                # Get the appropriate prompt for return
                current_prompt = self.prepare_action_prompt(inputs=prompt_params_values or {})
                # Use the final LLM response if available, otherwise fall back to execution history
                content_to_extract = final_llm_response if final_llm_response is not None else "{content}".format(content = conversation)
                if return_prompt:
                    return self._extract_output(content_to_extract, llm = llm), current_prompt
                return self._extract_output(content_to_extract, llm = llm) 
            time_out += 1
            
            # Handle both string prompts and chat message lists
            llm_response = llm.generate(messages=conversation)
            conversation.append({"role": "assistant", "content": llm_response.content})
            
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
            
            conversation.append({"role": "assistant", "content": TOOL_CALLING_HISTORY_PROMPT.format(
                iteration_number=time_out,
                tool_call_args=f"{tool_call_args}",
                results=f"{results}"
            )})
        
        # Get the appropriate prompt for return
        current_prompt = self.prepare_action_prompt(inputs=prompt_params_values or {})
        # Use the final LLM response if available, otherwise fall back to execution history
        content_to_extract = final_llm_response if final_llm_response is not None else "{content}".format(content = conversation)
        if return_prompt:
            return self._extract_output(content_to_extract, llm = llm), current_prompt
        return self._extract_output(content_to_extract, llm = llm)
        

    async def async_execute(self, llm: Optional[BaseLLM] = None, inputs: Optional[dict] = None, sys_msg: Optional[str]=None, return_prompt: bool = False, time_out = 0, **kwargs):
        # Allow empty inputs if the action has no required input attributes
        input_attributes: dict = self.inputs_format.get_attr_descriptions()
        if not inputs and input_attributes:
            logger.error("CustomizeAction action received invalid `inputs`: None or empty.")
            raise ValueError('The `inputs` to CustomizeAction action is None or empty.')
        # Set inputs to empty dict if None and no inputs are required
        if inputs is None:
            inputs = {}
        final_llm_response = None
        
        if self.prompt_template:
            conversation = [{"role": "system", "content": self.prepare_action_prompt(inputs=inputs, system_prompt=sys_msg)}]
        else:
            conversation = [{"role": "system", "content": sys_msg}, {"role": "user", "content": self.prepare_action_prompt(inputs=inputs, system_prompt=sys_msg)}]
        
        
        ## 1. get all the input parameters
        prompt_params_values = {k: inputs.get(k, "") for k in input_attributes.keys()}
        while True:
            ### Generate response from LLM
            if time_out > self.max_tool_try:
                # Get the appropriate prompt for return
                current_prompt = self.prepare_action_prompt(inputs=prompt_params_values or {})
                # Use the final LLM response if available, otherwise fall back to execution history
                content_to_extract = final_llm_response if final_llm_response is not None else "{content}".format(content = conversation)
                if return_prompt:
                    return await self._async_extract_output(content_to_extract, llm = llm), current_prompt
                return await self._async_extract_output(content_to_extract, llm = llm) 
            time_out += 1
            
            # Handle both string prompts and chat message lists
            llm_response = await llm.async_generate(messages=conversation)
            conversation.append({"role": "assistant", "content": llm_response.content})
            
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
            
            conversation.append({"role": "assistant", "content": TOOL_CALLING_HISTORY_PROMPT.format(
                iteration_number=time_out,
                tool_call_args=f"{tool_call_args}",
                results=f"{results}"
            )})
        
        # Get the appropriate prompt for return
        current_prompt = self.prepare_action_prompt(inputs=prompt_params_values or {})
        # Use the final LLM response if available, otherwise fall back to execution history
        content_to_extract = final_llm_response if final_llm_response is not None else "{content}".format(content = conversation)
        if return_prompt:
            return await self._async_extract_output(content_to_extract, llm = llm), current_prompt
        return await self._async_extract_output(content_to_extract, llm = llm)