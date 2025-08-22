import json
import re
from collections.abc import Callable
from typing import Any, List, Optional, Union

from pydantic import Field

from ..core.logging import logger
from ..core.message import Message
from ..core.module_utils import parse_json_from_llm_output, parse_json_from_text
from ..models.base_model import BaseLLM, LLMOutputParser
from ..prompts.template import ChatTemplate, StringTemplate
from ..prompts.tool_calling import (
    OUTPUT_EXTRACTION_PROMPT,
    TOOL_CALLING_HISTORY_PROMPT,
    TOOL_CALLING_TEMPLATE,
)
from ..tools.tool import Toolkit
from ..utils.utils import fix_json_booleans
from .action import Action


class CustomizeAction(Action):

    parse_mode: Optional[str] = Field(default="title", description="the parse mode of the action, must be one of: ['title', 'str', 'json', 'xml', 'custom']")
    parse_func: Optional[Callable] = Field(default=None, exclude=True, description="the function to parse the LLM output. It receives the LLM output and returns a dict.")
    title_format: Optional[str] = Field(default="## {title}", exclude=True, description="the format of the title. It is used when the `parse_mode` is 'title'.")
    custom_output_format: Optional[str] = Field(default=None, exclude=True, description="the format of the output. It is used when the `prompt_template` is provided.")

    tools: Optional[List[Toolkit]] = Field(default=None, description="The tools that the action can use")
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
        if self.tools:
            self.tools_caller = {}
            self.add_tools(self.tools)
    
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
            elif isinstance(value, (int, float)):
                prompt_params_values[param] = str(value)
            else:
<<<<<<< HEAD
                try:
                    prompt_params_values[param] = str(value)
                except:
                    raise TypeError(f"Invalid input type {type(value)}. Expected a type that can be converted to a string.")
=======
                raise TypeError(f"The input type {type(value)} is invalid! Valid types: [str, dict, list, int, float].")
        
>>>>>>> new_workflow
        if self.prompt:
            prompt = self.prompt.format(**prompt_params_values) if prompt_params_values else self.prompt
            if self.tools:
                tools_schemas = [j["function"] for i in [tool.get_tool_schemas() for tool in self.tools] for j in i]
                prompt += "\n\n" + TOOL_CALLING_TEMPLATE.format(tools_description = tools_schemas)
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
    
    
    def add_tools(self, tools: Union[Toolkit, List[Toolkit]]):
        if not tools:
            return
        if isinstance(tools,Toolkit):
            tools = [tools]
        if not all(isinstance(tool, Toolkit) for tool in tools):
            raise TypeError("`tools` must be a Toolkit or list of Toolkit instances.")
        if not self.tools:
            self.tools_caller = {}
            self.tools = []
        # self.tools += tools
        # tools_callers = [tool.get_tools() for tool in tools]
        # tools_callers = [j for i in tools_callers for j in i]
        # for tool_caller in tools_callers:
        #     self.tools_caller[tool_caller.name] = tool_caller

        # avoid duplication & type checks 
        for toolkit in tools:
            try:
                tool_callers = toolkit.get_tools()
                if not isinstance(tool_callers, list):
                    logger.warning(f"Expected list of tool functions from '{toolkit.name}.get_tools()', got {type(tool_callers)}.")
                    continue 
                
                # add tool callers to the tools_caller dictionary
                valid_tools_count = 0 
                valid_tools_names, valid_tool_callers = [], []
                for tool_caller in tool_callers:
                    tool_caller_name = getattr(tool_caller, "name", None)
                    if not tool_caller_name or not callable(tool_caller):
                        logger.warning(f"Invalid tool function in '{toolkit.name}': missing name or not callable.")
                        continue
                    if tool_caller_name in self.tools_caller:
                        logger.warning(f"Duplicate tool function '{tool_caller_name}' detected. Overwriting previous function.")
                    # self.tools_caller[tool_caller_name] = tool_caller
                    valid_tools_count += 1
                    valid_tools_names.append(tool_caller_name)
                    valid_tool_callers.append(tool_caller)

                if valid_tools_count == 0:
                    logger.info(f"No valid tools found in toolkit '{toolkit.name}'. Skipping.")
                    continue 

                if valid_tools_count > 0 and all(name in self.tools_caller for name in valid_tools_names):
                    logger.info(f"All tools from toolkit '{toolkit.name}' are already added. Skipping.")
                    continue
                
                if valid_tools_count > 0:
                    self.tools_caller.update({name: caller for name, caller in zip(valid_tools_names, valid_tool_callers)}) 
                
                # only add toolkit if at least one valid tool is added and toolkit is not already added 
                existing_toolkit_names = {tkt.name for tkt in self.tools}
                if valid_tools_count > 0 and toolkit.name not in existing_toolkit_names:
                    self.tools.append(toolkit)
                if valid_tools_count > 0:
                    logger.info(f"Added toolkit '{toolkit.name}' with {valid_tools_count} valid tools in {self.name}: {valid_tools_names}.")
            
            except Exception as e:
                logger.error(f"Failed to load tools from toolkit '{toolkit.name}': {e}")
    
    # def _extract_tool_calls(self, llm_output: str) -> List[dict]:
    #     pattern = r"```ToolCalling\s*\n(.*?)\n\s*```"
        
    #     # Find all ToolCalling blocks in the output
    #     matches = re.findall(pattern, llm_output, re.DOTALL)

    #     if not matches:
    #         return []
        
    #     parsed_tool_calls = []
    #     for match_content in matches:
    #         try:
    #             json_content = match_content.strip()
    #             json_list = parse_json_from_text(json_content)
    #             if not json_list:
    #                 logger.warning("No valid JSON found in ToolCalling block")
    #                 continue
    #             # Only use the first JSON string from each block
    #             parsed_tool_call = json.loads(json_list[0])
    #             if isinstance(parsed_tool_call, dict):
    #                 parsed_tool_calls.append(parsed_tool_call)
    #             elif isinstance(parsed_tool_call, list):
    #                 parsed_tool_calls.extend(parsed_tool_call)
    #             else:
    #                 logger.warning(f"Invalid tool call format: {parsed_tool_call}")
    #                 continue
    #         except (json.JSONDecodeError, IndexError) as e:
    #             logger.warning(f"Failed to parse tool calls from LLM output: {e}")
    #             continue

    #     return parsed_tool_calls

    #smolagent风格：
    def _extract_tool_calls(self, llm_output: str) -> List[dict]:
        """
        Parse tool calls from LLM output (smolagents style).
        Expecting a JSON array of objects with keys: "name", "arguments".
        """
        # content = llm_output.strip()
        # # Quick heuristic: tool calls must start with '[' or '{'
        # if not (content.startswith("[") or content.startswith("{")):
        #     return []
        # try:
        #     tool_calls = json.loads(content)
        # except json.JSONDecodeError:
        #     logger.warning(f"LLM output is not valid JSON: {llm_output}")
        #     return []
        # # Check if the parsed result is a list
        # if not isinstance(tool_calls, list):
        #     logger.warning(f"Tool calls must be a JSON array, got: {type(tool_calls)}")
        #     return []       
        # parsed = []
        # for call in tool_calls:
        #     if not isinstance(call, dict):
        #         logger.warning(f"Invalid tool call entry: {call}")
        #         continue
        #     if "name" not in call or "arguments" not in call:
        #         logger.warning(f"Missing 'name' or 'arguments' in tool call: {call}")
        #         continue
        #     parsed.append(call)
        # return parsed
       
        # First, try to extract from <ToolCalling> tags
        pattern = r"<ToolCalling>\s*(.*?)\s*</ToolCalling>"
        matches = re.findall(pattern, llm_output, re.DOTALL)
        
        parsed = []
        for match_content in matches:
            try:
                # Parse JSON from the extracted content
                json_content = match_content.strip()
<<<<<<< HEAD
                json_list = parse_json_from_text(json_content)
                if not json_list:
                    logger.warning("No valid JSON found in ToolCalling block")
                    continue
                tool_call_json = fix_json_booleans(json_list[0])
                # Only use the first JSON string from each block
                parsed_tool_call = json.loads(tool_call_json)
                if isinstance(parsed_tool_call, dict):
                    parsed_tool_calls.append(parsed_tool_call)
                elif isinstance(parsed_tool_call, list):
                    parsed_tool_calls.extend(parsed_tool_call)
                else:
                    logger.warning(f"Invalid tool call format: {parsed_tool_call}")
                    continue
            except (json.JSONDecodeError, IndexError) as e:
                logger.warning(f"Failed to parse tool calls from LLM output: {e}")
=======
                tool_calls = json.loads(json_content)
                
                if not isinstance(tool_calls, list):
                    tool_calls = [tool_calls]
                
                for call in tool_calls:
                    if not isinstance(call, dict):
                        logger.warning(f"Invalid tool call entry: {call}")
                        continue
                    
                    # Support both new and legacy field names
                    if "tool_name" in call and "arguments" in call:
                        parsed.append({
                            "name": call["tool_name"],
                            "arguments": call.get("arguments", {})
                        })
                    elif "name" in call and "arguments" in call:
                        parsed.append({
                            "name": call["name"],
                            "arguments": call.get("arguments", {})
                        })
                    else:
                        logger.warning(f"Missing required fields in tool call: {call}")
                        continue
                        
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Failed to parse tool calls: {e}")
>>>>>>> new_workflow
                continue
        
        return parsed

    def _extract_output(self, llm_output: Any, llm: BaseLLM = None, **kwargs):

        # Get the raw output content
        llm_output_content = getattr(llm_output, "content", str(llm_output))
        
        # Check if there are any defined output fields
        output_attrs = self.outputs_format.get_attrs()
        
        # If no output fields are defined, create a simple content-only output
        if not output_attrs:
            # Create output with just the content field
            output = self.outputs_format.parse(content=llm_output_content)
            # print("Created simple content output for agent with no defined outputs:")
            # print(output)
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
            
            # print("Successfully parsed output using action's parse settings:")
            # print(parsed_output)
            return parsed_output
            
        except Exception as e:
            logger.info(f"Failed to parse with action's parse settings: {e}")
            logger.info("Falling back to using LLM to extract outputs...")
            
            # Fall back to extraction prompt if direct parsing fails
            extraction_prompt = self.prepare_extraction_prompt(llm_output_content)
                
            llm_extracted_output: LLMOutputParser = llm.generate(prompt=extraction_prompt)
            llm_extracted_data: dict = parse_json_from_llm_output(llm_extracted_output.content)
            output = self.outputs_format.from_dict(llm_extracted_data)
            
            # print("Extracted output using fallback:")
            # print(output)
            return output
    
    async def _async_extract_output(self, llm_output: Any, llm: BaseLLM = None, **kwargs):
        
        # Get the raw output content
        llm_output_content = getattr(llm_output, "content", str(llm_output))
        
        # Check if there are any defined output fields
        output_attrs = self.outputs_format.get_attrs()
        
        # If no output fields are defined, create a simple content-only output
        if not output_attrs:
            # Create output with just the content field
            output = self.outputs_format.parse(content=llm_output_content)
            # print("Created simple content output for agent with no defined outputs:")
            # print(output)
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
            
            # print("Successfully parsed output using action's parse settings:")
            # print(parsed_output)
            return parsed_output
            
        except Exception as e:
            logger.info(f"Failed to parse with action's parse settings: {e}")
            logger.info("Falling back to using LLM to extract outputs...")
            
            # Fall back to extraction prompt if direct parsing fails
            extraction_prompt = self.prepare_extraction_prompt(llm_output_content)
                
            llm_extracted_output = await llm.async_generate(prompt=extraction_prompt)
            llm_extracted_data: dict = parse_json_from_llm_output(llm_extracted_output.content)
            output = self.outputs_format.from_dict(llm_extracted_data)
            
            # print("Extracted output using fallback:")
            # print(output)
            return output
    
    # def _calling_tools(self, tool_call_args) -> dict:
    #     ## ___________ Call the tools ___________
    #     errors = []
    #     results  =[]
    #     for function_param in tool_call_args:
    #         try:
    #             function_name = function_param.get("function_name")
    #             function_args = function_param.get("function_args") or {}
        
    #             # Check if we have a valid function to call
    #             if not function_name:
    #                 errors.append("No function name provided")
    #                 break
                
    #             # Try to get the callable from our tools_caller dictionary
    #             callable_fn = None
    #             if self.tools_caller and function_name in self.tools_caller:
    #                 callable_fn = self.tools_caller[function_name]
    #             elif callable(function_name):
    #                 callable_fn = function_name
                        
    #             if not callable_fn:
    #                 errors.append(f"Function '{function_name}' not found or not callable")
    #                 break
                
    #             try:
    #                 # Determine if the function is async or not
    #                 print("_____________________ Start Function Calling _____________________")
    #                 print(f"Executing function calling: {function_name} with parameters: {function_args}")
    #                 result = callable_fn(**function_args)
                
    #             except Exception as e:
    #                 logger.error(f"Error executing tool {function_name}: {e}")
    #                 errors.append(f"Error executing tool {function_name}: {str(e)}")
    #                 break
            
    #             results.append(result)
    #         except Exception as e:
    #             logger.error(f"Error executing tool: {e}")
    #             errors.append(f"Error executing tool: {str(e)}")
        

    #     ## 3. Add the tool call results to the query and continue the conversation
    #     results = {"result": results, "error": errors}
    #     return results

    #smolagent风格：
    def _calling_tools(self, tool_call_args: List[dict]) -> dict:
        """
        Call the tools with the given arguments.
        """
        # errors, results = [], []
        # for call in tool_call_args:
        #     try:
        #         tool_name = call["name"]
        #         args = call.get("arguments", {})

        #         if tool_name not in self.tools_caller:
        #             errors.append(f"Unknown tool: {tool_name}")
        #             continue

        #         tool_fn = self.tools_caller[tool_name]
        #         print("_____________________ Start Function Calling _____________________")
        #         print(f"Executing tool: {tool_name} with arguments: {args}")
        #         result = tool_fn(**args)
        #         results.append(result)
        #     except Exception as e:
        #         logger.error(f"Error executing tool {call}: {e}")
        #         errors.append(f"Error executing {call}: {str(e)}")

        # return {"result": results, "error": errors}
        errors, results = [], []
        for call in tool_call_args:
            try:
                tool_name = call["name"]
                args = call.get("arguments", {})

                if tool_name not in self.tools_caller:
                    errors.append(f"Unknown tool: {tool_name}")
                    continue

                tool_fn = self.tools_caller[tool_name]
                print("_____________________ Start Function Calling _____________________")
                print(f"Executing tool: {tool_name} with arguments: {args}")
                result = tool_fn(**args)
                results.append(result)
            except Exception as e:
                logger.error(f"Error executing tool {call}: {e}")
                errors.append(f"Error executing {call}: {str(e)}")

        return {"result": results, "error": errors}
    
    def execute(self, llm: Optional[BaseLLM] = None, inputs: Optional[dict] = None, sys_msg: Optional[str]=None, return_prompt: bool = False, time_out = 0, **kwargs):
        input_attributes: dict = self.inputs_format.get_attr_descriptions()
        required_inputs = self.inputs_format.get_required_input_names()

        if inputs is None:
            if len(required_inputs)>0:
                logger.error(f"CustomizeAction '{self.name}' requires the following inputs: {required_inputs} but received 'None' instead.")
                raise ValueError(f"CustomizeAction '{self.name}' requires the following inputs: {required_inputs} but received 'None' instead.")
            
            # Set inputs to empty dict if None and no inputs are required
            inputs = {}
        
        elif len(required_inputs)>0:
            # Check if all required inputs are provided
            for required_input in required_inputs:
                if required_input not in inputs:
                    logger.error(f"Required input '{required_input}' not found in inputs provided to CustomizeAction '{self.name}'.")
                    raise ValueError(f"Required input '{required_input}' not found in inputs provided to CustomizeAction '{self.name}'.")
   
        final_llm_response = None
        
        if self.prompt_template:

            if isinstance(self.prompt_template, ChatTemplate):
                # must determine whether prompt_template is ChatTemplate first since ChatTemplate is a subclass of StringTemplate
                conversation = self.prepare_action_prompt(inputs=inputs, system_prompt=sys_msg)
            elif isinstance(self.prompt_template, StringTemplate):
                conversation = [{"role": "system", "content": self.prepare_action_prompt(inputs=inputs, system_prompt=sys_msg)}]
            else:
                raise ValueError(f"`prompt_template` must be a StringTemplate or ChatTemplate instance, but got {type(self.prompt_template)}")
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
            
            # logger.info(f"这是此时LLM的输出: {final_llm_response}")
            tool_call_args = self._extract_tool_calls(llm_response.content)
            if not tool_call_args:
                break
            
            logger.info("Extracted tool call args:")
            logger.info(json.dumps(tool_call_args, indent=4))
            
            results = self._calling_tools(tool_call_args)
            
            logger.info("Tool call results:")
            logger.info(json.dumps(results, indent=4))
            
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
        input_attributes: dict = self.inputs_format.get_attr_descriptions()
        required_inputs = self.inputs_format.get_required_input_names()

        if inputs is None:
            if len(required_inputs)>0:
                logger.error(f"CustomizeAction '{self.name}' requires the following inputs: {required_inputs} but received 'None' instead.")
                raise ValueError(f"CustomizeAction '{self.name}' requires the following inputs: {required_inputs} but received 'None' instead.")
            
            # Set inputs to empty dict if None and no inputs are required
            inputs = {}
        
        elif len(required_inputs)>0:
            # Check if all required inputs are provided
            for required_input in required_inputs:
                if required_input not in inputs:
                    logger.error(f"Required input '{required_input}' not found in inputs provided to CustomizeAction '{self.name}'.")
                    raise ValueError(f"Required input '{required_input}' not found in inputs provided to CustomizeAction '{self.name}'.")
        
        final_llm_response = None
        
        if self.prompt_template:
            if isinstance(self.prompt_template, ChatTemplate):
                # must determine whether prompt_template is ChatTemplate first since ChatTemplate is a subclass of StringTemplate
                conversation = self.prepare_action_prompt(inputs=inputs, system_prompt=sys_msg)
            elif isinstance(self.prompt_template, StringTemplate):
                conversation = [{"role": "system", "content": self.prepare_action_prompt(inputs=inputs, system_prompt=sys_msg)}]
            else:
                raise ValueError(f"`prompt_template` must be a StringTemplate or ChatTemplate instance, but got {type(self.prompt_template)}")
        else:
            conversation = [{"role": "system", "content": sys_msg}, {"role": "user", "content": self.prepare_action_prompt(inputs=inputs, system_prompt=sys_msg)}]
        
        logger.info(f"此时conversation的内容为: {conversation}")
        ## 1. get all the input parameters
        prompt_params_values = {k: inputs.get(k, "") for k in input_attributes.keys()}
        while True:
            ### Generate response from LLM
            if time_out > self.max_tool_try:
                # Get the appropriate prompt for return
                current_prompt = self.prepare_action_prompt(inputs=prompt_params_values or {})
                # Use the final LLM response if available, otherwise fall back to execution history
                content_to_extract = final_llm_response if final_llm_response is not None else "{content}".format(content = conversation)
                logger.info(f"此时提取出来的conten为: {content_to_extract}")
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
            
            logger.info("Extracted tool call args:")
            logger.info(json.dumps(tool_call_args, indent=4))
            
            results = self._calling_tools(tool_call_args)
            
            logger.info("Tool call results:")
            try:
                logger.info(json.dumps(results, indent=4))
            except Exception:
                logger.info(str(results))
            
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