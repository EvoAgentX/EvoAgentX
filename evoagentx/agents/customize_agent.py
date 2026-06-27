import inspect
import json
from collections.abc import Callable
from copy import deepcopy
from typing import Any, Dict, List, Literal, Optional, Type, Union

from pydantic import ConfigDict, Field, create_model

from ..actions.action import Action, ActionInput, ActionOutput
from ..actions.customize_action import CustomizeAction
from ..core.base_config import Parameter
from ..core.logging import logger
from ..core.message import Message, MessageType
from ..core.registry import MODULE_REGISTRY, PARSE_FUNCTION_REGISTRY
from ..models.base_model import PARSER_VALID_MODE
from ..models.model_configs import LLMConfig
from ..prompts.template import PromptTemplate
from ..prompts.utils import DEFAULT_SYSTEM_PROMPT
from ..tools.tool import Tool, Toolkit
from ..utils.utils import (
    add_llm_config_to_agent_dict,
    generate_dynamic_class_name,
    get_unique_class_name,
    make_parent_folder,
    string_to_json_schema_type,
    string_to_python_type,
    to_params,
    tool_names_to_tools,
)
from .agent import Agent


COMPLEX_PARAM_TYPES = {"object", "array", "dict", "list"}


class CustomizeAgent(Agent):

    """
    CustomizeAgent provides a flexible framework for creating specialized LLM-powered agents without
    writing custom code. It enables the creation of agents with well-defined inputs and outputs,
    custom prompt templates, and configurable parsing strategies.

    Attributes:
        name (str): The name of the agent.
        description (str): A description of the agent's purpose and capabilities.
        prompt_template (PromptTemplate, optional): The prompt template that will be used for the agent's primary action.
        prompt (str, optional): The prompt template that will be used for the agent's primary action.
            Should contain placeholders in the format `{input_name}` for each input parameter.
        llm_config (LLMConfig, optional): Configuration for the language model.
        inputs (List[Union[dict, Parameter]], optional): List of input specifications as dicts or Parameter objects. Each dict (e.g., `{"name": str, "type": str, "description": str, ["required": bool, "json_schema": dict]}`) contains:
            - name (str): Name of the input parameter
            - type (str): Type of the input
            - description (str): Description of what the input represents
            - required (bool, optional): Whether this input is required (default: True)
            - json_schema (dict, optional): The json schema of the input, recommended when type is `object` or `array`.
        outputs (List[Union[dict, Parameter]], optional): List of output specifications as dicts or Parameter objects. Each dict (e.g., `{"name": str, "type": str, "description": str, ["required": bool, "json_schema": dict]}`) contains:
            - name (str): Name of the output field
            - type (str): Type of the output
            - description (str): Description of what the output represents
            - required (bool, optional): Whether this output is required (default: True)
            - json_schema (dict, optional): The json schema of the output, recommended when type is `object` or `array`.
        system_prompt (str, optional): The system prompt for the LLM. Defaults to DEFAULT_SYSTEM_PROMPT.
        output_parser (Type[ActionOutput], optional): A custom class for parsing the LLM's output.
            Must be a subclass of ActionOutput.
        parse_mode (str, optional): Mode for parsing LLM output. Options are:
            - "title": Parse outputs using section titles (default)
            - "str": Parse as plain text
            - "json": Parse as JSON
            - "xml": Parse as XML
            - "custom": Use a custom parsing function
        parse_func (Callable, optional): Custom function for parsing LLM output when parse_mode is "custom".
            Must accept a "content" parameter and return a dictionary.
        title_format (str, optional): Format string for title parsing mode with {title} placeholder.
            Default is "## {title}".
        tools (list[Toolkit], optional): List of tools to be used by the agent.
        custom_output_format (str, optional): Specify the output format. Only used when `prompt_template` is used.
            If not provided, the output format will be constructed from the `outputs` specification and `parse_mode`.
    """
    def __init__(
        self,
        name: str,
        description: str,
        prompt: Optional[str] = None,
        prompt_template: Optional[PromptTemplate] = None,
        llm_config: Optional[LLMConfig] = None,
        inputs: Optional[List[Union[dict, Parameter]]] = None,
        outputs: Optional[List[Union[dict, Parameter]]] = None,
        system_prompt: Optional[str] = None,
        output_parser: Optional[Type[ActionOutput]] = None,
        parse_mode: Optional[str] = "title",
        parse_func: Optional[Callable] = None,
        title_format: Optional[str] = None,
        tools: Optional[List[Union[Toolkit, Tool]]] = None,
        custom_output_format: Optional[str] = None,
        max_steps: int = 20,
        max_tool_call_concurrency: int = 5,
        **kwargs
    ):
        system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        inputs = inputs or []
        outputs = outputs or []

        if prompt is not None and prompt_template is not None:
            logger.warning("Both `prompt` and `prompt_template` are provided in `CustomizeAgent`. `prompt_template` will be used.")
            prompt = None

        if isinstance(parse_func, str):
            if not PARSE_FUNCTION_REGISTRY.has_function(parse_func):
                raise ValueError(f"parse function `{parse_func}` is not registered! To instantiate a CustomizeAgent from a file, you should use decorator `@register_parse_function` to register the parse function.")
            parse_func = PARSE_FUNCTION_REGISTRY.get_function(parse_func)

        if isinstance(output_parser, str):
            output_parser = MODULE_REGISTRY.get_module(output_parser)

        # set default title format
        if parse_mode == "title" and title_format is None:
            title_format = "## {title}"

        # validate the data and normalize inputs/outputs to Parameter objects
        valid_inputs, valid_outputs, parse_mode = self.validate_data(
            prompt=prompt,
            prompt_template=prompt_template,
            inputs=inputs,
            outputs=outputs,
            output_parser=output_parser,
            parse_mode=parse_mode,
            parse_func=parse_func,
            title_format=title_format
        )

        customize_action = CustomizeAgent.create_customize_action(
            name=name,
            desc=description,
            prompt=prompt,
            prompt_template=prompt_template,
            inputs=valid_inputs,
            outputs=valid_outputs,
            parse_mode=parse_mode,
            parse_func=parse_func,
            output_parser=output_parser,
            title_format=title_format,
            custom_output_format=custom_output_format,
            tools=tools,
            max_steps=max_steps,
            max_tool_call_concurrency=max_tool_call_concurrency,
        )
        super().__init__(
            name=name,
            description=description,
            llm_config=llm_config,
            system_prompt=system_prompt,
            actions=[customize_action],
            **kwargs
        )

        # Set backing attributes after super().__init__ so Pydantic doesn't wipe them.
        # parse_func must be stored before parse_mode because the parse_mode setter reads it.
        self._inputs  = valid_inputs
        self._outputs = valid_outputs
        self.output_parser = output_parser
        self._parse_func = parse_func
        self.parse_mode = parse_mode
        self._title_format = title_format
        self.custom_output_format = custom_output_format

    def _add_tools(self, tools: List[Toolkit]):
        self.action.add_tools(tools)

    @property
    def customize_action_name(self) -> str:
        """
        Get the name of the primary custom action for this agent.
        
        Returns:
            The name of the primary custom action
        """
        for action in self.actions:
            if action.name != self.cext_action_name:
                return action.name
        raise ValueError("Couldn't find the customize action name!")

    @property
    def action(self) -> Action:
        """
        Get the primary custom action for this agent.
        
        Returns:
            The primary custom action
        """
        return self.get_action(self.customize_action_name) 
    
    @property
    def inputs(self) -> List[Parameter]:
        return self._inputs

    @inputs.setter
    def inputs(self, inputs: List[Union[dict, Parameter]]):
        valid_inputs, valid_outputs, parse_mode = self.validate_data(
            prompt=self.prompt,
            prompt_template=self.prompt_template,
            inputs=inputs,
            outputs=self.outputs,
            output_parser=self.output_parser,
            parse_mode=self.parse_mode,
            parse_func=self.parse_func,
            title_format=self.title_format
        )
        self._inputs = valid_inputs
        self.action.inputs_format = CustomizeAgent.create_action_input(valid_inputs, self.name)
        self.parse_mode = parse_mode

    @property
    def outputs(self) -> List[Parameter]:
        return self._outputs

    @outputs.setter
    def outputs(self, outputs: List[Union[dict, Parameter]]):
        valid_inputs, valid_outputs, parse_mode = self.validate_data(
            prompt=self.prompt,
            prompt_template=self.prompt_template,
            inputs=self.inputs,
            outputs=outputs,
            output_parser=self.output_parser,
            parse_mode=self.parse_mode,
            parse_func=self.parse_func,
            title_format=self.title_format
        )
        self._outputs = valid_outputs
        self.action.outputs_format = CustomizeAgent.create_action_output(valid_outputs, self.name)
        self.parse_mode = parse_mode
    
    @property
    def prompt(self) -> str:
        """
        Get the prompt for the primary custom action.
        
        Returns:
            The prompt for the primary custom action
        """
        return self.action.prompt
    
    @property
    def prompt_template(self) -> PromptTemplate:
        """
        Get the prompt template for the primary custom action.
        
        Returns:
            The prompt template for the primary custom action
        """
        return self.action.prompt_template
    
    @property
    def tools(self) -> List[Union[Tool, Toolkit]]:
        return self.action.tools
    
    @property
    def parse_mode(self) -> str:
        return self._parse_mode
    
    @parse_mode.setter
    def parse_mode(self, parse_mode: str):
        if parse_mode not in PARSER_VALID_MODE:
            raise ValueError(f"'{parse_mode}' is an invalid value for `parse_mode`. Available choices: {PARSER_VALID_MODE}.")
        # Only enforce json for prompt_template-based agents (see validate_data): a raw `prompt`
        # owns its output format, so the user is free to pair complex outputs with any parse_mode.
        if self.prompt_template is not None and CustomizeAgent._outputs_require_json_mode(self.outputs, self.parse_func) and parse_mode != "json":
            raise ValueError(
                f"Cannot set parse_mode='{parse_mode}': current outputs contain object/array types or json_schema. "
                f"Set parse_mode='json', or provide a custom parse_func first."
            )
        if parse_mode == "custom" and self.parse_func is None:
            raise ValueError("`parse_func` must be set before switching parse_mode to 'custom'.")
        self._parse_mode = parse_mode
        self.action.parse_mode = parse_mode

    @property
    def parse_func(self) -> Optional[Callable]:
        return self._parse_func
    
    @parse_func.setter
    def parse_func(self, parse_func: Optional[Callable]):
        if parse_func is None:
            if self.parse_mode == "custom":
                raise ValueError("Cannot set parse_func to None while parse_mode is 'custom'. Change parse_mode first.")
        else:
            CustomizeAgent._validate_parse_func(parse_func)
        self._parse_func = parse_func
        self.action.parse_func = parse_func
    
    @property
    def title_format(self) -> Optional[str]:
        return self._title_format
    
    @title_format.setter
    def title_format(self, title_format: Optional[str]):
        CustomizeAgent._validate_title_format(title_format, self.parse_mode)
        self._title_format = title_format
        self.action.title_format = title_format
    
    @staticmethod
    def _outputs_require_json_mode(outputs: List[Parameter], parse_func: Optional[Callable]) -> bool:
        """Return True when outputs force parse_mode='json' (only relevant if no parse_func is supplied)."""
        if parse_func is not None:
            return False
        return (
            any(p.type in COMPLEX_PARAM_TYPES for p in outputs)
            or CustomizeAgent.contain_json_schema(outputs)
        )
    
    @staticmethod
    def _validate_parse_func(parse_func: Optional[Callable]) -> None:
        """Raise ValueError / emit a warning if parse_func is not a valid parsing callable."""
        if parse_func is None:
            return
        if not callable(parse_func):
            raise ValueError("`parse_func` must be a callable function with an input argument `content`.")
        signature = inspect.signature(parse_func)
        if "content" not in signature.parameters:
            raise ValueError("`parse_func` must have an input argument `content`.")
        if not PARSE_FUNCTION_REGISTRY.has_function(parse_func.__name__):
            logger.warning(
                f"parse function `{parse_func.__name__}` is not registered. This can cause issues when loading "
                f"the agent from a file. It is recommended to register the parse function using "
                f"`register_parse_function`:\n"
                f"from evoagentx.core.registry import register_parse_function\n"
                f"@register_parse_function\n"
                f"def {parse_func.__name__}(content: str) -> dict:\n"
                r"    return {'output_name': output_value}"
            )

    @staticmethod
    def _validate_title_format(title_format: Optional[str], parse_mode: Optional[str]) -> None:
        """Raise ValueError / emit a warning if title_format is invalid or parse_mode is incompatible."""
        if title_format is None:
            return
        if r"{title}" not in title_format:
            raise ValueError(r"`title_format` must contain the placeholder `{title}`.")
        if parse_mode is not None and parse_mode != "title":
            logger.warning(
                f"`title_format` will not be used because `parse_mode` is '{parse_mode}', not 'title'. "
                f"Set `parse_mode='title'` to use title formatting."
            )

    def _check_params_types(self, params: List[Union[dict, Parameter]], param_name: str) -> List[Parameter]:
        """
        Converts `params` into a list of `Parameter` objects and at the same time validates them.

        Args:
            params: A list of `dict` or `Parameter` objects to convert and validate
            param_name: The name of the parameter (used for error messages)

        Returns:
            A list of `Parameter` objects
        """
        if not params:
            return

        # check if params is a list of dict
        if not isinstance(params, list):
            raise ValueError(f"`{param_name}` must be a list of dict or Parameter objects.")

        valid_params = []
        for param in params:
            if isinstance(param, dict):
                try:
                    valid_params.append(Parameter(**param))
                except Exception as e:
                    raise ValueError(
                        f"`{param}` is an invalid {param_name} item. \n"
                        f"Expected format: `{{'name': str, 'type': str, 'description': str, ['required': bool, 'json_schema': dict]}}`. \n"
                        f"Details: {e}"
                    )
            elif isinstance(param, Parameter):
                valid_params.append(param)
            else:
                raise ValueError(f"`{param}` is an invalid {param_name} item. \nExpected type: `dict` or `Parameter`.")

        return valid_params

    def validate_data(
        self,
        prompt: str,
        prompt_template: PromptTemplate,
        inputs: List[Union[dict, Parameter]],
        outputs: List[Union[dict, Parameter]],
        output_parser: Type[ActionOutput],
        parse_mode: str,
        parse_func: Callable,
        title_format: str,
    ) -> tuple:
        """Validate and normalize agent configuration, auto-correcting parse_mode where needed.

        Converts `inputs` and `outputs` to `Parameter` objects, validates all
        parsing-related options, and (for `prompt_template`-based agents only)
        auto-corrects `parse_mode` to `"json"` when the output schema contains
        `object`/`array` types or a `json_schema` without a custom parse function.
        Raw-`prompt` agents keep their `parse_mode`, since the prompt is sent verbatim
        and dictates its own output format.

        Returns:
            A tuple containing:
            - `valid_inputs` (List[Parameter]): normalized input parameters.
            - `valid_outputs` (List[Parameter]): normalized output parameters.
            - `parse_mode` (str): validated (and possibly auto-corrected) parse mode.
        """
        # Normalize inputs and outputs to Parameter objects first so all subsequent code
        # can safely use attribute access regardless of what the caller passed in.
        valid_inputs  = self._check_params_types(inputs,  "inputs")  or []
        valid_outputs = self._check_params_types(outputs, "outputs") or []

        # check if the prompt is provided
        if prompt is None and prompt_template is None:
            raise ValueError("`prompt` or `prompt_template` is required when creating a CustomizeAgent.")

        # check if all the inputs are in the prompt (only used when prompt_template is not provided)
        if prompt_template is None and valid_inputs:
            all_input_names = [input_item.name for input_item in valid_inputs]
            inputs_names_not_in_prompt = [name for name in all_input_names if f'{{{name}}}' not in prompt]
            if inputs_names_not_in_prompt:
                raise KeyError(f"The following inputs are not found in the prompt: {inputs_names_not_in_prompt}.")

        # check if the output_parser is valid
        if output_parser is not None:
            self._check_output_parser(valid_outputs, output_parser)

        # check the parse_mode value itself
        if parse_mode not in PARSER_VALID_MODE:
            raise ValueError(f"'{parse_mode}' is an invalid value for `parse_mode`. Available choices: {PARSER_VALID_MODE}.")

        # Auto-correct parse_mode to "json" when outputs require it and no custom parse_func is provided.
        # Only do this for prompt_template-based agents: the template renders the output-format section
        # (and injects the JSON schema) so the model is actually instructed to emit JSON. A raw `prompt`
        # is sent verbatim, so the model follows whatever format the prompt itself specifies; forcing
        # json here would make the parser disagree with the prompt's requested format.
        if prompt_template is not None and CustomizeAgent._outputs_require_json_mode(valid_outputs, parse_func) and parse_mode != "json":
            logger.warning(
                f"parse_mode='{parse_mode}' is not compatible with the current outputs (object/array types or "
                f"json_schema). Auto-correcting to parse_mode='json'. To suppress this warning, explicitly set "
                f"parse_mode='json'."
            )
            return valid_inputs, valid_outputs, "json"

        if parse_mode == "custom" and parse_func is None:
            raise ValueError("`parse_func` (a callable function with an input argument `content`) must be provided when `parse_mode` is 'custom'.")

        CustomizeAgent._validate_parse_func(parse_func)
        CustomizeAgent._validate_title_format(title_format, parse_mode)

        return valid_inputs, valid_outputs, parse_mode

    @staticmethod
    def create_customize_action(
        name: str,
        desc: str,
        prompt: str,
        prompt_template: PromptTemplate,
        inputs: List[Union[dict, Parameter]],
        outputs: List[Union[dict, Parameter]],
        parse_mode: str,
        parse_func: Optional[Callable] = None,
        output_parser: Optional[ActionOutput] = None,
        title_format: Optional[str] = "## {title}",
        custom_output_format: Optional[str] = None,
        tools: Optional[List[Union[Tool, Toolkit]]] = None,
        max_steps: int = 20,
        max_tool_call_concurrency: int = 5,
        **kwargs
    ) -> Action:
        """Create a custom action based on the provided specifications.

        This method dynamically generates an Action class and instance with:
        - Input parameters defined by the inputs specification
        - Output format defined by the outputs specification
        - Custom execution logic using the customize_action_execute function
        - If tools is provided, returns a CustomizeAction action instead

        Args:
            name: Base name for the action
            desc: Description of the action
            prompt: Prompt template for the action
            prompt_template: Prompt template for the action
            inputs: List of input field specifications
            outputs: List of output field specifications
            parse_mode: Mode to use for parsing LLM output
            parse_func: Optional custom parsing function
            output_parser: Optional custom output parser class
            tools: Optional list of tools
            max_steps: Maximum number of steps the agent can take
            max_tool_call_concurrency: Maximum number of concurrent tool calls

        Returns:
            A newly created Action instance
        """
        assert prompt is not None or prompt_template is not None, "must provide `prompt` or `prompt_template` when creating CustomizeAgent"

        inputs: List[Parameter] = to_params(inputs)
        outputs: List[Parameter] = to_params(outputs)

        action_input_type = CustomizeAgent.create_action_input(inputs, name)

        if output_parser is None:
            action_output_type = CustomizeAgent.create_action_output(outputs, name)
        else:
            action_output_type = output_parser

        action_cls_name = get_unique_class_name(
            generate_dynamic_class_name(name + " action")
        )

        customize_action = CustomizeAction(
            name=action_cls_name,
            description=desc,
            prompt=prompt,
            prompt_template=prompt_template,
            inputs_format=action_input_type,
            outputs_format=action_output_type,
            parse_mode=parse_mode,
            parse_func=parse_func,
            title_format=title_format,
            custom_output_format=custom_output_format,
            tools=tools,
            max_steps=max_steps,
            max_tool_call_concurrency=max_tool_call_concurrency,
        )

        return customize_action
    
    @staticmethod
    def _prepare_action_info(params: List[Parameter]) -> Dict:
        """Returns the fields for ActionInput/ActionOutput."""
        action_fields = {}
        for field in params:
            required = field.required if field.required is not None else True
            try:
                field_type = string_to_python_type[field.type]
            except KeyError:
                logger.warning(f'Could not find Python type for "{field.type}" (field: "{field.name}"), falling back to `Any`.')
                field_type = Any

            json_schema = field.json_schema

            if required:
                action_fields[field.name] = (field_type, Field(description=field.description, json_schema_extra=json_schema))
            else:
                action_fields[field.name] = (Optional[field_type], Field(default=None, description=field.description, json_schema_extra=json_schema))

        return action_fields

    @staticmethod
    def _create_action_parser(params: List[Union[dict, Parameter]], action_name: str, type: Literal["input", "output"]) -> Type[Union[ActionInput, ActionOutput]]:
        params: List[Parameter] = to_params(params)

        action_parser_type = ActionInput if type == "input" else ActionOutput
        action_fields = CustomizeAgent._prepare_action_info(params)

        if CustomizeAgent._requires_model_json_schema(params):
            json_schema = CustomizeAgent.create_json_schema(params)
        else:
            json_schema = None

        action_parser_class = create_model(
            get_unique_class_name(
                generate_dynamic_class_name(action_name + " action_input_output")
            ),
            **action_fields,
            __base__=action_parser_type,
            __config__=ConfigDict(
                json_schema_extra=json_schema
            )
        )
        return action_parser_class

    @staticmethod
    def create_action_input(inputs: List[Union[dict, Parameter]], action_name: str) -> Type[ActionInput]:
        return CustomizeAgent._create_action_parser(inputs, action_name, "input")

    @staticmethod
    def create_action_output(outputs: List[Union[dict, Parameter]], action_name: str) -> Type[ActionOutput]:
        return CustomizeAgent._create_action_parser(outputs, action_name, "output")

    @staticmethod
    def create_json_schema(params: List[Union[dict, Parameter]]) -> Optional[dict]:
        params: List[Parameter] = to_params(params)

        if not params:
            return None

        properties = {}
        required_params = []
        for param in params:
            param_name = param.name
            param_type = string_to_json_schema_type[param.type]
            param_description = param.description
            param_required = param.required if param.required is not None else True
            param_json_schema = param.json_schema
            if not param_json_schema:
                param_json_schema = {
                    "type": param_type,
                    "description": param_description
                }
            properties[param_name] = param_json_schema
            if param_required:
                required_params.append(param_name)

        json_schema = {
            "type": "object",
            "properties": properties,
            "required": required_params
        }

        return json_schema
    
    @staticmethod
    def contain_json_schema(params: List[Union[dict, Parameter]]) -> bool:
        params: List[Parameter] = to_params(params)
        return any(param.json_schema for param in params)

    @staticmethod
    def _requires_model_json_schema(params: List[Union[dict, Parameter]]) -> bool:
        params: List[Parameter] = to_params(params)
        return any(param.json_schema or param.type in COMPLEX_PARAM_TYPES for param in params)

    def _check_output_parser(self, outputs: List[Parameter], output_parser: Type[ActionOutput]):

        if output_parser is not None:
            if not isinstance(output_parser, type):
                raise TypeError(f"output_parser must be a class, but got {type(output_parser).__name__}")
            if not issubclass(output_parser, ActionOutput):
                raise ValueError(f"`output_parser` must be a class and a subclass of `ActionOutput`, but got `{output_parser.__name__}`.")

        # check if the output parser is compatible with the outputs
        output_parser_fields = output_parser.get_attrs()
        all_output_names = [output_item.name for output_item in outputs]
        for field in output_parser_fields:
            if field not in all_output_names:
                raise ValueError(
                    f"The output parser `{output_parser.__name__}` is not compatible with the `outputs`.\n"
                    f"The output parser fields: {output_parser_fields}.\n"
                    f"The outputs: {all_output_names}.\n"
                    f"All the fields in the output parser must be present in the outputs."
                )
    
    def __call__(self, inputs: dict = None, return_msg_type: MessageType = MessageType.UNKNOWN, **kwargs) -> Message:
        """
        Call the customize action.

        Args:
            inputs (dict): The inputs to the customize action.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            ActionOutput: The output of the customize action.
        """
        # return self.execute(action_name=self.customize_action_name, action_input_data=inputs, **kwargs) 
        inputs = inputs or {} 
        return super().__call__(action_name=self.customize_action_name, action_input_data=inputs, return_msg_type=return_msg_type, **kwargs)
    
    def get_customize_agent_info(self) -> dict:
        """
        Get the information of the customize agent.
        """
        customize_action = self.get_action(self.customize_action_name)
        
        config = {
            "class_name": "CustomizeAgent",
            "name": self.name,
            "description": self.description,
            "prompt": customize_action.prompt,
            "prompt_template": customize_action.prompt_template.to_dict() if customize_action.prompt_template is not None else None, 
            # "llm_config": self.llm_config.to_dict(exclude_none=True),
            "inputs": [p.to_dict(ignore=["class_name"]) for p in self.inputs] if self.inputs else [],
            "outputs": [p.to_dict(ignore=["class_name"]) for p in self.outputs] if self.outputs else [],
            "system_prompt": self.system_prompt,
            "output_parser": self.output_parser.__name__ if self.output_parser is not None else None,
            "parse_mode": self.parse_mode,
            "parse_func": self.parse_func.__name__ if self.parse_func is not None else None,
            "title_format": self.title_format,
            "tool_names": [tool.name for tool in customize_action.tools] if customize_action.tools else [],
            "custom_output_format": self.custom_output_format,
            "max_steps": customize_action.max_steps,
            "max_tool_call_concurrency": customize_action.max_tool_call_concurrency,
        }
        return config
    
    def save_module(self, path: str, ignore: List[str] = [], **kwargs)-> str:
        """Save the customize agent's configuration to a JSON file.
        
        Args:
            path: File path where the configuration should be saved
            ignore: List of keys to exclude from the saved configuration
            **kwargs (Any): Additional parameters for the save operation
            
        Returns:
            The path where the configuration was saved
        """
        config = self.get_customize_agent_info()

        for ignore_key in ignore:
            config.pop(ignore_key, None)
        
        # Save to JSON file
        make_parent_folder(path)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4, ensure_ascii=False)

        return path
    
    def get_config(self) -> dict:
        """
        Get a dictionary containing all necessary configuration to recreate this agent.
        
        Returns:
            dict: A configuration dictionary that can be used to initialize a new Agent instance
            with the same properties as this one.
        """
        config = self.get_customize_agent_info()
        config["llm_config"] = self.llm_config.to_dict()
        return config
    
    @classmethod
    def from_dict(
        cls, 
        data: Dict[str, Any], 
        llm_config: Optional[LLMConfig] = None, 
        tools: Optional[List[Union[Toolkit, Tool]]] = None, 
        **kwargs
    ) -> 'CustomizeAgent':

        agent_data = deepcopy(data)

        class_name = agent_data.pop("class_name", None)
        if class_name is not None and class_name != "CustomizeAgent":
            raise ValueError(f"Expected class name 'CustomizeAgent', but got '{class_name}'")

        agent_data = add_llm_config_to_agent_dict(agent_data, llm_config)
        tool_names = agent_data.pop("tool_names", None)
        
        if tool_names:
            agent_data["tools"] = tool_names_to_tools(tool_names, tools)
        
        parse_mode = agent_data.get("parse_mode")

        # if parse_mode is not 'json', check if there are outputs in format 'object' or 'array'
        if parse_mode != "json":
            agent_outputs = agent_data.get("outputs")

            if agent_outputs is not None:
                for output in agent_outputs:
                    if output["type"] == "object" or output["type"] == "array":
                        agent_data["parse_mode"] = "json"
                        logger.warning(f"`parse_mode` is set to 'json' for '{agent_data['name']}' because it has outputs in format 'object' or 'array'")
                        break

        return cls(**agent_data, **kwargs)
