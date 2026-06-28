import json
from copy import deepcopy
from typing import Any, List, Optional, Tuple, Type, Union

import regex
from pydantic import Field
from pydantic_core import PydanticUndefined

from ..core.logging import logger
from ..core.module import BaseModule
from ..models.base_model import PARSER_VALID_MODE, LLMOutputParser
from ..tools import Tool, Toolkit
from ..utils.utils import compile_tool_schemas, extract_type, python_to_json_type
from .output_format import (
    JSON_SCHEMA_OUTPUT_FORMAT,
    JSON_SCHEMA_OUTPUT_FORMAT_JSON,
)
from .tool_calling import TOOL_CALLING_TEMPLATE
from .utils import JSONSchemaExampleGenerator


class PromptTemplate(BaseModule):

    instruction: str = Field(description="The instruction that the LLM will follow.")
    context: Optional[str] = Field(default=None, description="Additional context that can help the LLM understand the instruction.")
    constraints: Optional[Union[List[str], str]] = Field(default=None, description="Constraints that the LLM must follow.")
    tools: Optional[List[Union[Tool, Toolkit]]] = Field(
        default=None,
        description=(
            "Legacy prompt-owned tools used only for rendering prompt-based tool "
            "instructions. This field will be removed in a future release; prefer "
            "passing tools to `format(..., tools=...)` when rendering tool instructions."
        ),
    )
    demonstrations: Optional[List[dict]] = Field(default=None, description="Examples of how to use the instruction.")
    history: Optional[List[Any]] = Field(default=None, description="History of the conversation between the user and the LLM.")

    def get_field_names(self) -> List[str]:
        return [name for name, _ in type(self).model_fields.items() if name != "class_name"]
    
    def get(self, key: str) -> Any:
        fields = self.get_field_names()
        if key not in fields:
            raise ValueError(f"Invalid key `{key}` for `{self.__class__.__name__}`. Valid keys are: {fields}")
        return getattr(self, key)
    
    def set(self, key: str, value: Any):
        fields = self.get_field_names()
        if key not in fields:
            raise ValueError(f"Invalid key `{key}` for `{self.__class__.__name__}`. Valid keys are: {fields}")
        setattr(self, key, value)

    def get_instruction(self) -> str:
        return self.instruction

    def get_demonstrations(self) -> List[Any]:
        return self.demonstrations
    
    def get_context(self) -> Optional[str]:
        return self.context
    
    def get_history(self) -> Optional[List[Any]]:
        return self.history
    
    def get_constraints(self) -> Optional[Union[List[str], str]]:
        return self.constraints
    
    def get_tools(self) -> Optional[List[str]]:
        return self.tools
    
    def set_instruction(self, instruction: str):
        self.set("instruction", instruction)

    def set_demonstrations(self, demonstrations: List[Any]):
        self.set("demonstrations", demonstrations)

    def set_context(self, context: str):
        self.set("context", context)

    def set_history(self, history: List[Any]):
        self.set("history", history)

    def set_constraints(self, constraints: Union[List[str], str]):
        self.set("constraints", constraints)

    def set_tools(self, tools: List[Toolkit]):
        self.set("tools", tools)

    def get_required_inputs_or_outputs(self, format: Type[LLMOutputParser]) -> List[str]:
        """
        Get the required fields of the format.
        """
        required_fields = []
        attrs = format.get_attrs()
        for field_name, field_info in format.model_fields.items():
            if field_name not in attrs:
                continue
            field_default = field_info.default
            # A field is required if it doesn't have a default value
            if field_default is PydanticUndefined:
                required_fields.append(field_name)
        return required_fields

    def clear_placeholders(self, text: str) -> str:
        """
        Find all {xx} placeholders in the text, and replace them with `xx`,
        adding backticks only if not already present.
        """
        # Step 1: Find all unique {xx} placeholders (single braces only)
        matches = set(regex.findall(r"(?<!\{)\{([^\{\},\s]+)\}(?!\})", text))

        for field in matches:
            # Pattern: only single-brace {field}, not {{field}} or {{{field}}}
            pattern = r"(?<!\{)\{" + regex.escape(field) + r"\}(?!\})"

            def replacer(match):
                start, end = match.start(), match.end()
                before = text[start - 1] if start > 0 else ""
                after = text[end] if end < len(text) else ""

                replacement = field
                if before != "`":
                    replacement = "`" + replacement
                if after != "`":
                    replacement = replacement + "`"

                return replacement

            text = regex.sub(pattern, replacer, text)

        return text
    
    def check_required_inputs(self, inputs_format: Type[LLMOutputParser], values: dict):
        if inputs_format is None: 
            return 
        required_inputs = self.get_required_inputs_or_outputs(inputs_format)
        missing_required_inputs = [field for field in required_inputs if field not in values]
        if missing_required_inputs:
            logger.warning(f"Missing required inputs (without default values) for `{inputs_format.__name__}`: {missing_required_inputs}, will set them to empty strings.")
    
    def render_input_example(
        self,
        inputs_format: Optional[Type[LLMOutputParser]] = None,
        values: Optional[dict] = None,
        missing_field_value: str = ""
    ) -> str:
        
        if inputs_format is None and values is None:
            return ""

        if inputs_format is not None:
            fields = inputs_format.get_attrs()
            if values is None:
                values = {}
            field_values = {field: values.get(field, missing_field_value) for field in fields}
        else:
            field_values = values

        template = "**`{name}`** {description}\n{value}\n\n"
        inputs_str = ""

        for name, value in field_values.items():
            description = ""

            if inputs_format is not None:
                field_description = inputs_format.model_fields[name].description
                if field_description is not None:
                    description = f"({field_description})"
            
            if isinstance(value, (dict, list)):
                value = json.dumps(value, indent=2, ensure_ascii=False)
            inputs_str += template.format(name=name, description=description, value=value)
        return inputs_str

    def render_inputs(
        self,
        inputs_format: Optional[Type[LLMOutputParser]] = None,
        values: Optional[dict] = None
    ) -> str:

        if (inputs_format is None and values is None) or (inputs_format is not None and len(inputs_format.get_attrs()) == 0):
            return "" 
        # Check if all required fields are provided
        self.check_required_inputs(inputs_format, values)
        input_str = "### Inputs\n"
        input_example = self.render_input_example(inputs_format, values, missing_field_value="Not provided")
        
        if len(input_example) == 0:
            input_str += "No inputs provided.\n"
        else:
            input_str += "These are the input values provided by the user:\n\n" + input_example
        
        return input_str
    
    def get_output_template(self, outputs_format: Type[LLMOutputParser], parse_mode: str="title", title_format: str="## {title}") -> Tuple[str, List[str]]:
        
        if outputs_format is None:
            raise ValueError("`outputs_format` is required in `get_output_format`.")
        valid_modes = ["json", "xml", "title"]
        if parse_mode not in valid_modes:
            raise ValueError(f"Invalid parse mode `{parse_mode}` for `{self.__class__.__name__}.get_output_template`. Valid modes are: {valid_modes}.")
        
        fields = outputs_format.get_attrs()
        required_fields = self.get_required_inputs_or_outputs(outputs_format)
        if parse_mode == "json":
            json_template = "{{\n"
            for field in fields: 
                json_template += f"    \"{field}\""
                json_template += f": {{{field}}},\n" if field in required_fields else f" (Optional): {{{field}}},\n"
            json_template = json_template.rstrip(",\n") + "\n}}"
            output_template, output_keys = json_template, fields
        elif parse_mode == "xml":
            xml_template = ""
            for field in fields:
                xml_template += f"<{field}>\n" if field in required_fields else f"<{field}> (Optional)\n" 
                xml_template += f"{{{field}}}\n</{field}>\n"
            xml_template = xml_template.rstrip("\n")
            output_template, output_keys = xml_template, fields
        elif parse_mode == "title":
            title_template = ""
            for field in fields:
                title_section = title_format.format(title=field)
                title_section += "\n" if field in required_fields else " (Optional)\n"
                title_section += f"{{{field}}}\n\n"
                title_template += title_section
            title_template = title_template.rstrip("\n")
            output_template, output_keys = title_template, fields
        
        return output_template, output_keys

    def render_instruction(self) -> str:
        # clear the potential placeholders in the instruction. we will use the input section to specify the inputs. 
        instruction_str = self.clear_placeholders(self.instruction)
        return f"### Instruction\nThis is the main task instruction you must follow:\n{instruction_str}\n"
    
    def render_context(self) -> str:
        if not self.context:
            return ""
        return f"### Context\nHere is some additional background information to help you understand the task:\n{self.context}\n"

    def render_tools(self, tools: Optional[List[Union[Tool, Toolkit]]] = None) -> str:
        if not self.tools and not tools:
            return ""
        
        tools = tools or self.tools
        tool_schemas = compile_tool_schemas(tools)
        tool_schemas_str = json.dumps(tool_schemas, indent=4, ensure_ascii=False)
        return TOOL_CALLING_TEMPLATE.format(tool_descriptions=tool_schemas_str)
    
    def render_constraints(self) -> str:
        if not self.constraints:
            return ""
        if isinstance(self.constraints, list):
            constraints_str = "\n".join(f"- {c}" for c in self.constraints)
        else:
            constraints_str = self.constraints
        return f"### Constraints\nYou must follow these rules or constraints when generating your output:\n{constraints_str}\n"
    
    def _render_system_message(self, system_prompt: Optional[str] = None, tools: Optional[List[Union[Tool, Toolkit]]] = None) -> str:
        """
        Render the system message by combining system prompt, instruction, context, tools and constraints.
        """
        prompt_pieces = []
        if system_prompt:
            prompt_pieces.append(system_prompt + "\n")
        prompt_pieces.append(self.render_instruction())
        if self.context:
            prompt_pieces.append(self.render_context())
        if self.tools or tools:
            prompt_pieces.append(self.render_tools(tools))
        if self.constraints:
            prompt_pieces.append(self.render_constraints())

        return "\n".join(prompt_pieces)
    
    def render_outputs(self, outputs_format: Type[LLMOutputParser], parse_mode: str="title", title_format: str="## {title}") -> str:

        if outputs_format is None or parse_mode in [None, "str", "custom"] or len(outputs_format.get_attrs()) == 0:
            return "### Outputs Format\nPlease generate a response that best fits the task instruction.\n"
        
        required_outputs = self.get_required_inputs_or_outputs(outputs_format)
        output_template, output_keys = self.get_output_template(outputs_format, parse_mode=parse_mode, title_format=title_format)
        fields_str = ", ".join(f'"{field}"' for field in required_outputs)
        
        example_values = {} 
        for key in output_keys:
            field_info = outputs_format.model_fields.get(key)
            if field_info:
                try:
                    field_type = python_to_json_type[extract_type(field_info.annotation)]
                except KeyError:
                    field_type = "string"

                if field_info.description:
                    template = "[Insert here: {description} {type_note}]"

                    if field_type != "string":
                        type_note = f"(Must be a valid JSON `{field_type}`. DO NOT add comments.)"
                    else:
                        type_note = "(Must be a string)"
                    example_values[key] = template.format(description=field_info.description, type_note=type_note)
                else:
                    template = "[Insert your response for `{field}` here {type_note}]"
                    if field_type != "string":
                        type_note = f"(Must be a valid JSON `{field_type}`. DO NOT add comments.)"
                    else:
                        type_note = ""
                    example_values[key] = template.format(field=key, type_note=type_note)
            else:
                example_values[key] = f"[Insert your response for '{key}' here]"        

        output_example = output_template.format(**example_values)

        json_schema = outputs_format.model_config.get("json_schema_extra")
        if json_schema is not None:
            return self._format_json_schema_output_prompt(json_schema=json_schema, parse_mode=parse_mode, example=output_example)

        output_str = "### Outputs Format\nYou MUST strictly follow the following rules when generating your output.\n\n"
        
        if parse_mode == "json":
            output_str += f"Your final response must be a valid JSON object with these fields: {fields_str}.\n\nExample:\n"
        elif parse_mode == "xml":
            output_str += f"Your final response must contain the following XML elements: {fields_str}.\n\nExample:\n"
        elif parse_mode == "title":
            output_str += f"Your final response must contain these sectioned titles: {fields_str}.\n\nExample:\n"
        
        output_str += output_example

        if "(Optional)" in output_template:
            output_str += "\n\nNote: For optional fields, you can omit them in your output if they are not necessary."
        output_str += "\n"
        return output_str

    def _format_json_schema_output_prompt(self, json_schema: dict, parse_mode: str, example: str) -> str:
        json_schema_str = json.dumps(json_schema, ensure_ascii=False, indent=4)
        if parse_mode == "json":
            example_generator = JSONSchemaExampleGenerator()
            example = example_generator.generate(json_schema, to_str=True)
            return JSON_SCHEMA_OUTPUT_FORMAT_JSON.format(json_schema=json_schema_str, example=example)
        elif parse_mode == "xml":
            return JSON_SCHEMA_OUTPUT_FORMAT.format(json_schema=json_schema_str, parse_mode="XML element", example=example)
        elif parse_mode == "title":
            return JSON_SCHEMA_OUTPUT_FORMAT.format(json_schema=json_schema_str, parse_mode="sectioned title", example=example)

    def format(
        self,
        inputs_format: Optional[Type[LLMOutputParser]] = None,
        outputs_format: Optional[Type[LLMOutputParser]] = None,
        values: Optional[dict] = None, 
        parse_mode: Optional[str] = "title", 
        title_format: Optional[str] = "## {title}",
        output_format: Optional[str] = None, 
        **kwargs
    ) -> str:
        raise NotImplementedError(f"`format` method is not implemented for `{self.__class__.__name__}`.") 

    def get_config(self) -> dict:
        return self.to_dict()
    
    def copy(self, **kwargs) -> "PromptTemplate":
        """
        Create a deep-copied new PromptTemplate, optionally overriding fields with provided kwargs.
        """
        config = self.get_config()
        new_config = deepcopy(config)
        new_config = {k: kwargs.get(k, v) for k, v in new_config.items()}
        return self.__class__.from_dict(new_config)


class StringTemplate(PromptTemplate):

    def render_demonstrations(
        self, 
        inputs_format: Type[LLMOutputParser], 
        outputs_format: Type[LLMOutputParser], 
        parse_mode: str, 
        title_format: str = None, 
        custom_output_format: str = None, 
        **kwargs
    ) -> str:
        
        if not self.demonstrations:
            return "" 
        
        if inputs_format is None or outputs_format is None:
            raise ValueError("`inputs_format` and `outputs_format` are required in `render_demonstrations`.")
        if len(inputs_format.get_attrs()) == 0 or len(outputs_format.get_attrs()) == 0:
            raise ValueError("`inputs_format` and `outputs_format` must have at least one attribute.")
        
        demo_str_list = [] 
        for i, demo in enumerate(self.demonstrations):
            demo_str = f"Example {i+1}:\n"
            
            demo_str += "### Inputs\n"
            input_fields = inputs_format.get_attrs()
            input_values = {field: demo.get(field, "Not provided") for field in input_fields}
            demo_str += self.render_input_example(inputs_format, input_values, missing_field_value="Not provided")
            demo_str += "\n\n"

            demo_str += "### Outputs\n"
            output_fields = outputs_format.get_attrs()
            output_values = {field: demo.get(field, "Not provided") for field in output_fields}
            if custom_output_format is not None or parse_mode in [None, "str", "custom"]:
                output_str = "\n".join(f"{field}:\n{value}" for field, value in output_values.items())
            else:
                output_template, output_keys = self.get_output_template(outputs_format, parse_mode=parse_mode, title_format=title_format)
                output_str = output_template.format(**output_values)
                output_str = output_str.replace("(Optional)", "")
            demo_str += output_str
            demo_str_list.append(demo_str)
        
        result = "### Examples\n" + "\n\n".join(demo_str_list) + "\n\n=== End of Examples ===\n"
        return result

    def render_history(self) -> str:
        result = "### History\n{history}".format(history=self.history)
        return result

    def format(
        self, 
        system_prompt: Optional[str] = None, 
        values: Optional[dict] = None, 
        inputs_format: Optional[Type[LLMOutputParser]] = None, 
        outputs_format: Optional[Type[LLMOutputParser]] = None, 
        parse_mode: Optional[str] = "title", 
        title_format: Optional[str] = "## {title}", 
        custom_output_format: Optional[str] = None, 
        tools: Optional[List[Union[Tool, Toolkit]]] = None,
        **kwargs
    ) -> str:
        """
        Format the prompt template.

        Convert the prompt template into a prompt string. 
        It will sequentially concatenate the following sections (if provided): instruction, context, tools, constraints, demonstrations, history, inputs and outputs.

        Args: 
            values (Optional[dict]): The values to be used to render the inputs. 
            inputs_format (Optional[Type[LLMOutputParser]]): Define the input variables. If provided, it will be used to extract inputs (specified in `inputs_format`) from `values` and use them to render the inputs section. 
                Otherwise, will use all fields in `values` (if provided) directly to render the inputs section. 
            outputs_format (Optional[Type[LLMOutputParser]]): Define the output variables. If provided, it will be used to construct the output format based on `parse_mode`. 
                Otherwise, a default output format will be used. 
            parse_mode (Optional[str]): The mode to parse the outputs, chosen from ["json", "xml", "title", "str", "custom"]. It will be used to construct the output format if `outputs_format` is provided. 
                Moreover, if `parse_mode` is "title", `title_format` will be used to format the title of the outputs. 
            title_format (Optional[str]): The format to format the title of the outputs. Default is "## {title}". Only used when `parse_mode` is "title".
            custom_output_format (Optional[str]): User-specified output format. If provided, it will be directly used in the `Outputs Format` section of the prompt. Otherwise, the output format will be constructed from `outputs_format` and `parse_mode`. 
            tools (Optional[List[Union[Tool, Toolkit]]]): Tools that the LLM can use. Overrides `self.tools`.
            **kwargs: Additional keyword arguments. 
        
        Returns: 
            str: The formatted prompt string.
        """

        if parse_mode not in PARSER_VALID_MODE:
            raise ValueError(f"Invalid parse mode `{parse_mode}` for `{self.__class__.__name__}.format`. Valid modes are: {PARSER_VALID_MODE}.")

        prompt_pieces = []
        prompt_pieces.append(self._render_system_message(system_prompt, tools))

        if self.demonstrations:
            prompt_pieces.append(
                self.render_demonstrations(
                    inputs_format=inputs_format, 
                    outputs_format=outputs_format, 
                    parse_mode=parse_mode, 
                    title_format=title_format, 
                    custom_output_format=custom_output_format
                )
            )
        if self.history:
            prompt_pieces.append(self.render_history())
        
        if inputs_format or values:
            prompt_pieces.append("-"*20)
            prompt_pieces.append(self.render_inputs(inputs_format, values))
        
        # define the output format
        if custom_output_format:
            prompt_pieces.append(f"### Outputs Format\n{custom_output_format}")
        else:
            prompt_pieces.append(self.render_outputs(outputs_format, parse_mode, title_format))
        
        prompt_pieces = [piece for piece in prompt_pieces if piece]
        prompt = "\n".join(prompt_pieces)
        return prompt.strip()
    

class ChatTemplate(StringTemplate):

    def _create_message(self, role: str, content: str) -> dict:
        """Create a message dictionary with role and content."""
        return {"role": role, "content": content}
    
    def render_demonstrations(
        self,
        inputs_format: Type[LLMOutputParser],
        outputs_format: Type[LLMOutputParser],
        parse_mode: str,
        title_format: str = None,
        custom_output_format: str = None,
        **kwargs
    ) -> List[dict]:
        """
        Render demonstrations as alternating user and assistant messages.
        """

        if not self.demonstrations:
            return []
        
        if inputs_format is None or outputs_format is None:
            raise ValueError("`inputs_format` and `outputs_format` are required in `render_demonstrations`.")
        if len(inputs_format.get_attrs()) == 0 or len(outputs_format.get_attrs()) == 0:
            raise ValueError("`inputs_format` and `outputs_format` must have at least one attribute.")
        
        messages = []
        for demo in self.demonstrations:
            # Render user message (input)
            input_fields = inputs_format.get_attrs()
            input_values = {field: demo.get(field, "Not provided") for field in input_fields}
            user_content = self.render_input_example(inputs_format, input_values, missing_field_value="Not provided")
            messages.append(self._create_message("user", user_content))
            
            # Render assistant message (output)
            output_fields = outputs_format.get_attrs() 
            output_values = {field: demo.get(field, "Not provided") for field in output_fields}
            if custom_output_format is not None or parse_mode in [None, "str", "custom"]:
                assistant_content = "\n".join(f"{field}:\n{value}" for field, value in output_values.items())
            else:
                output_template, output_keys = self.get_output_template(outputs_format, parse_mode=parse_mode, title_format=title_format)
                assistant_content = output_template.format(**output_values)
                assistant_content = assistant_content.replace("(Optional)", "")
            messages.append(self._create_message("assistant", assistant_content))

        return messages
    
    def render_history(self) -> List[dict]:
        """
        Render conversation history as a list of chat messages.

        Unlike demonstrations (synthetic few-shot examples that are re-rendered
        through `inputs_format`/`outputs_format`), history represents actual prior
        conversation turns and is passed through directly as chat messages without
        reformatting. Each history item is expected to be a message dict with
        "role" and "content" keys. Items that are not in this shape are wrapped into
        a single "user" message so that no history content is silently dropped.
        """
        if not self.history:
            return []

        valid_roles = {"system", "user", "assistant", "tool"}
        messages = []
        for item in self.history:
            if isinstance(item, dict) and "role" in item and "content" in item:
                role = item["role"] if item["role"] in valid_roles else "user"
                messages.append(self._create_message(role, item["content"]))
            else:
                messages.append(self._create_message("user", str(item)))
        return messages
    
    def render_current_user_message(
        self, 
        values: Optional[dict], 
        inputs_format: Optional[Type[LLMOutputParser]],
    ) -> str:
        
        """Render the current user input message."""
        input_pieces = []
        if inputs_format or values:
            input_pieces.append(self.render_inputs(inputs_format, values))

        input_pieces = [piece for piece in input_pieces if piece]
        user_message = "\n".join(input_pieces)
        return user_message.strip()
    
    def format(
        self, 
        system_prompt: Optional[str] = None, 
        values: Optional[dict] = None, 
        inputs_format: Optional[Type[LLMOutputParser]] = None, 
        outputs_format: Optional[Type[LLMOutputParser]] = None, 
        parse_mode: Optional[str] = "title", 
        title_format: Optional[str] = "## {title}", 
        custom_output_format: Optional[str] = None,
        tools: Optional[List[Union[Tool, Toolkit]]] = None,
        **kwargs
    ) -> List[dict]:
        """
        Format the prompt template into a list of chat messages.
        
        The messages will be formatted in the following order:
        1. System message (containing system prompt, instruction, context, tools, and constraints)
        2. Few-shot examples (if provided in demonstrations)
        3. Conversation history (if provided)
        4. Current user input (with input values and output format requirements)
        
        Args:
            system_prompt (Optional[str]): Additional system prompt to prepend to the template.
            values (Optional[dict]): The values to be used to render the inputs.
            inputs_format (Optional[Type[LLMOutputParser]]): Define the input variables.
            outputs_format (Optional[Type[LLMOutputParser]]): Define the output variables.
            parse_mode (Optional[str]): The mode to parse the outputs.
            title_format (Optional[str]): The format to format the title of the outputs.
            custom_output_format (Optional[str]): User-specified output format.
            tools (Optional[List[Union[Tool, Toolkit]]]): Tools that the LLM can use. Overrides `self.tools`.
            **kwargs: Additional keyword arguments.
            
        Returns:
            List[dict]: A list of chat messages in the format:
            [
                {"role": "system", "content": system_message},
                # Begin few-shot examples
                {"role": "user", "content": few_shot_example_1_input},
                {"role": "assistant", "content": few_shot_example_1_output},
                ...
                # End few-shot examples
                {"role": "user", "content": current_input},
            ]
        """
        if parse_mode not in PARSER_VALID_MODE:
            raise ValueError(f"Invalid parse mode `{parse_mode}` for `{self.__class__.__name__}.prompt`. Valid modes are: {PARSER_VALID_MODE}.")
            
        messages = []
        
        # Add system message
        system_content = self._render_system_message(system_prompt, tools)

        if custom_output_format:
            system_content += f"\n### Outputs Format\n{custom_output_format}"
        else:
            system_content += "\n" + self.render_outputs(outputs_format, parse_mode, title_format)

        messages.append(self._create_message("system", system_content))
        
        # Add few-shot examples
        if self.demonstrations:
            messages.extend(
                self.render_demonstrations(
                    inputs_format=inputs_format, 
                    outputs_format=outputs_format, 
                    parse_mode=parse_mode, 
                    title_format=title_format, 
                    custom_output_format=custom_output_format
                )
            )
        
        # Add conversation history
        if self.history:
            messages.extend(self.render_history())

        # Add current user input & output format requirements
        current_input = self.render_current_user_message(
            values=values,
            inputs_format=inputs_format
        )
        messages.append(self._create_message("user", current_input))
        
        return messages
        

class MiproPromptTemplate(ChatTemplate):

    def render_demonstrations(self, inputs_format: LLMOutputParser, outputs_format: LLMOutputParser, parse_mode: str, title_format: str = None, custom_output_format: str = None, **kwargs) -> List[dict]:

        import dspy
        if self.demonstrations:
            demo = self.demonstrations[0]
            if isinstance(demo, dspy.Example):
                self.demonstrations = [demo.toDict() for demo in self.demonstrations]
        return super().render_demonstrations(inputs_format, outputs_format, parse_mode, title_format, custom_output_format, **kwargs)
