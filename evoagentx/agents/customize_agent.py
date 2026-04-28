import json
import inspect
from copy import deepcopy
from pydantic import ConfigDict, create_model, Field
from typing import Optional, Callable, Type, List, Any, Union, Dict, Literal

from .agent import Agent
from ..core.logging import logger
from ..core.registry import MODULE_REGISTRY, PARSE_FUNCTION_REGISTRY
from ..core.message import Message, MessageType
from ..core.base_config import Parameter
from ..models.model_configs import LLMConfig 
from ..models.base_model import PARSER_VALID_MODE
from ..prompts.utils import DEFAULT_SYSTEM_PROMPT
from ..prompts.template import PromptTemplate
from ..actions.action import Action, ActionOutput
from ..utils.utils import add_llm_config_to_agent_dict, generate_dynamic_class_name, get_unique_class_name, make_parent_folder, string_to_json_schema_type, string_to_python_type, tool_names_to_tools
from ..actions.customize_action import CustomizeAction
from ..actions.action import ActionInput
from ..tools.tool import Toolkit, Tool


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
        inputs (List[dict], optional): List of input specifications, where each dict (e.g., `{"name": str, "type": str, "description": str, ["required": bool]}`) contains:
            - name (str): Name of the input parameter
            - type (str): Type of the input
            - description (str): Description of what the input represents
            - required (bool, optional): Whether this input is required (default: True)
        outputs (List[dict], optional): List of output specifications, where each dict (e.g., `{"name": str, "type": str, "description": str, ["required": bool]}`) contains:
            - name (str): Name of the output field
            - type (str): Type of the output
            - description (str): Description of what the output represents
            - required (bool, optional): Whether this output is required (default: True)
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
        max_tool_calls (int, optional): Maximum number of tool calls. Defaults to 5. 
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
        max_tool_calls: Optional[int] = 5,
        custom_output_format: Optional[str] = None, 
        **kwargs
    ):
        system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        inputs = inputs or [] 
        outputs = outputs or [] 
        inputs = [item.to_dict(ignore=["class_name"]) if isinstance(item, Parameter) else item for item in inputs]
        outputs = [item.to_dict(ignore=["class_name"]) if isinstance(item, Parameter) else item for item in outputs]
        if tools is not None:
            raw_tool_map = {tool.name: tool for tool in tools}
            tools = [tool if isinstance(tool, Toolkit) else Toolkit(name=tool.name, tools=[tool]) for tool in tools]
        else:
            raw_tool_map = None

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

        # validate the data 
        self.validate_data(
            prompt = prompt, 
            prompt_template = prompt_template, 
            inputs = inputs, 
            outputs = outputs, 
            output_parser = output_parser, 
            parse_mode = parse_mode, 
            parse_func = parse_func, 
            title_format = title_format
        )

        customize_action = self.create_customize_action(
            name=name, 
            desc=description, 
            prompt=prompt, 
            prompt_template=prompt_template, 
            inputs=inputs, 
            outputs=outputs, 
            parse_mode=parse_mode, 
            parse_func=parse_func,
            output_parser=output_parser,
            title_format=title_format,
            custom_output_format=custom_output_format ,
            tools=tools,
            max_tool_calls=max_tool_calls
        )
        super().__init__(
            name=name, 
            description=description, 
            llm_config=llm_config, 
            system_prompt=system_prompt, 
            actions=[customize_action], 
            **kwargs
        )
        self.inputs = inputs
        self.outputs = outputs
        self._raw_tool_map = raw_tool_map
        self.output_parser = output_parser 
        self.parse_mode = parse_mode 
        self.parse_func = parse_func 
        self.title_format = title_format
        self.tools = tools
        self.max_tool_calls = max_tool_calls
        self.custom_output_format = custom_output_format

    def _add_tools(self, tools: List[Toolkit]):
        self.get_action(self.customize_action_name).add_tools(tools)

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

    def _check_params_types(self, params: List[dict], param_name: str):

        if not params:
            return 
        
        # check if params is a list of dict  
        if not isinstance(params, list):
            raise ValueError(f"`{param_name}` must be a list of dict.")
        for param in params:
            if not isinstance(param, dict):
                raise ValueError(f"`{param_name}` must be a list of dict.")
            try:
                Parameter.from_dict(param)
            except Exception as e:
                raise ValueError(
                    f"`{param}` is an invalid {param_name} item. \n"
                    f"Expected format: `{{'name': str, 'type': str, 'description': str, ['required': bool, 'json_schema': dict]}}`. \n"
                    f"Details: {e}"
                )
    
    def validate_data(self, prompt: str, prompt_template: PromptTemplate, inputs: List[dict], outputs: List[dict], output_parser: Type[ActionOutput], parse_mode: str, parse_func: Callable, title_format: str):

        # check if the prompt is provided
        if prompt is None and prompt_template is None:
            raise ValueError("`prompt` or `prompt_template` is required when creating a CustomizeAgent.")
        
        # check if all the inputs are in the prompt (only used when prompt_template is not provided)
        if prompt_template is None and inputs:
            all_input_names = [input_item["name"] for input_item in inputs]
            inputs_names_not_in_prompt = [name for name in all_input_names if f'{{{name}}}' not in prompt]
            if inputs_names_not_in_prompt:
                raise KeyError(f"The following inputs are not found in the prompt: {inputs_names_not_in_prompt}.") 

        # check the inputs and outputs types
        self._check_params_types(inputs, "inputs")
        self._check_params_types(outputs, "outputs")
        
        # check if the output_parser is valid 
        if output_parser is not None:
            self._check_output_parser(outputs, output_parser)
        
        # check the parse_mode, parse_func, and title_format
        if parse_mode not in PARSER_VALID_MODE:
            raise ValueError(f"'{parse_mode}' is an invalid value for `parse_mode`. Available choices: {PARSER_VALID_MODE}.")

        # `parse_mode` must be "json" when there are object or array parameters in the outputs 
        if any(output_item.get("type") in {"object", "array"} for output_item in outputs):
            if parse_mode != "json":
                raise ValueError("`parse_mode` must be 'json' when there are object or array parameters in the outputs. ") 
        
        if parse_mode == "custom":
            if parse_func is None:
                raise ValueError("`parse_func` (a callable function with an input argument `content`) must be provided when `parse_mode` is 'custom'.")
        
        if parse_func is not None:
            if not callable(parse_func):
                raise ValueError("`parse_func` must be a callable function with an input argument `content`.")
            signature = inspect.signature(parse_func)
            if "content" not in signature.parameters:
                raise ValueError("`parse_func` must have an input argument `content`.")
            if not PARSE_FUNCTION_REGISTRY.has_function(parse_func.__name__):
                logger.warning(
                    f"parse function `{parse_func.__name__}` is not registered. This can cause issues when loading the agent from a file. "
                    f"It is recommended to register the parse function using `register_parse_function`:\n"
                    f"from evoagentx.core.registry import register_parse_function\n"
                    f"@register_parse_function\n"
                    f"def {parse_func.__name__}(content: str) -> dict:\n"
                    r"    return {'output_name': output_value}" 
                )

        if title_format is not None:
            if parse_mode != "title":
                logger.warning(f"`title_format` will not be used because `parse_mode` is '{parse_mode}', not 'title'. Set `parse_mode='title'` to use title formatting.")
            if r'{title}' not in title_format:
                raise ValueError(r"`title_format` must contain the placeholder `{title}`.")
            
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
        tools: Optional[List[Toolkit]] = None,
        max_tool_calls: Optional[int] = 5
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
            
        Returns:
            A newly created Action instance
        """
        assert prompt is not None or prompt_template is not None, "must provide `prompt` or `prompt_template` when creating CustomizeAgent"

        inputs = [item.to_dict(ignore=["class_name"]) if isinstance(item, Parameter) else item for item in inputs]
        outputs = [item.to_dict(ignore=["class_name"]) if isinstance(item, Parameter) else item for item in outputs]

        action_input_type = CustomizeAgent.create_action_input(inputs, name)
        
        # create the action output type
        if output_parser is None:
            action_output_type = CustomizeAgent.create_action_output(outputs, name)
        else:
            # self._check_output_parser(outputs, output_parser)
            action_output_type = output_parser
        
        action_cls_name = get_unique_class_name(
            generate_dynamic_class_name(name+" action")
        )

        # Create CustomizeAction-based action with parsing properties only
        customize_action_cls = create_model(
            action_cls_name,
            __base__=CustomizeAction
        )

        customize_action = customize_action_cls(
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
            max_tool_try=max_tool_calls,
            tools=tools
        )

        return customize_action

    @staticmethod
    def _prepare_action_info(params: List[Dict]) -> Dict:
        """
        Returns the fields for ActionInput/ActionOutput.
        """
        action_fields = {}
        for field in params:
            required = field.get("required", True)
            try:
                field_type = string_to_python_type[field["type"]]
            except KeyError:
                logger.warning(f'Could not find Python type for "{field["type"]}" (field: "{field["name"]}"), falling back to `Any`.')
                field_type = Any

            json_schema = field.get("json_schema", None)
            
            if required:
                action_fields[field["name"]] = (field_type, Field(description=field["description"], json_schema_extra=json_schema))
            else:
                action_fields[field["name"]] = (Optional[field_type], Field(default=None, description=field["description"], json_schema_extra=json_schema))

        return action_fields

    @staticmethod
    def _create_action_parser(params: List[Union[dict, Parameter]], action_name: str, type: Literal["input", "output"]) -> Type[Union[ActionInput, ActionOutput]]:
        params = [item.to_dict(ignore=["class_name"]) if isinstance(item, Parameter) else item for item in params]

        action_parser_type = ActionInput if type == "input" else ActionOutput
        action_fields = CustomizeAgent._prepare_action_info(params)

        if CustomizeAgent.contain_json_schema(params):
            json_schema = CustomizeAgent.create_json_schema(params)
        else:
            json_schema = None

        action_parser_class = create_model(
            get_unique_class_name(
                generate_dynamic_class_name(action_name+" action_input_output")
            ),
            **action_fields, 
            __base__=action_parser_type,
            __config__=ConfigDict(
                json_schema_extra=json_schema
            )
        )
        action_parser_class._evoagentx_param_specs = params
        return action_parser_class

    @staticmethod
    def create_action_input(inputs: List[Union[dict, Parameter]], action_name: str) -> Type[ActionInput]:
        return CustomizeAgent._create_action_parser(inputs, action_name, "input")

    @staticmethod
    def create_action_output(outputs: List[Union[dict, Parameter]], action_name: str) -> Type[ActionOutput]:
        return CustomizeAgent._create_action_parser(outputs, action_name, "output")
    
    @staticmethod
    def create_json_schema(params: List[dict]) -> Optional[dict]:

        if not params:
            return None 

        properties = {}
        required_params = [] 
        for param in params:
            param_name = param["name"]
            param_type = string_to_json_schema_type[param["type"]]
            param_description = param["description"]
            param_required = param.get("required", True)
            param_json_schema = param.get("json_schema", None)
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
    def contain_json_schema(params: List[dict]) -> bool:
        for param in params:
            if param.get("json_schema", None):
                return True
        return False
    
    def _check_output_parser(self, outputs: List[dict], output_parser: Type[ActionOutput]):

        if output_parser is not None:
            if not isinstance(output_parser, type):
                raise TypeError(f"output_parser must be a class, but got {type(output_parser).__name__}")
            if not issubclass(output_parser, ActionOutput):
                raise ValueError(f"`output_parser` must be a class and a subclass of `ActionOutput`, but got `{output_parser.__name__}`.")
        
        # check if the output parser is compatible with the outputs
        output_parser_fields = output_parser.get_attrs()
        all_output_names = [output_item["name"] for output_item in outputs]
        for field in output_parser_fields:
            if field not in all_output_names:
                raise ValueError(
                    f"The output parser `{output_parser.__name__}` is not compatible with the `outputs`.\n"
                    f"The output parser fields: {output_parser_fields}.\n"
                    f"The outputs: {all_output_names}.\n"
                    f"All the fields in the output parser must be present in the outputs." 
                )
    
    def _store_inputs_outputs_info(self, inputs: List[dict], outputs: List[dict], tool_map: Dict[str, Union[Toolkit, Tool]]):

        self._action_input_types, self._action_input_required = {}, {} 
        for field in inputs:
            required = field.get("required", True)
            self._action_input_types[field["name"]] = field["type"]
            self._action_input_required[field["name"]] = required
        self._action_output_types, self._action_output_required = {}, {}
        for field in outputs:
            required = field.get("required", True)
            self._action_output_types[field["name"]] = field["type"]
            self._action_output_required[field["name"]] = required
        self._raw_tool_map = tool_map

    def update_inputs(self, inputs: List[Union[dict, Parameter]]):
        """
        Update the inputs format of the customize agent.

        Args:
            inputs (List[Union[dict, Parameter]]): The new inputs format of the customize agent.
        """
        inputs = [item.to_dict(ignore=["class_name"]) if isinstance(item, Parameter) else item for item in inputs]
        self.inputs = inputs
        new_action_input = CustomizeAgent.create_action_input(inputs, self.name)
        self.action.inputs_format = new_action_input


    def update_outputs(self, outputs: List[Union[dict, Parameter]]):
        """
        Update the outputs format of the customize agent.

        Args:
            outputs (List[Union[dict, Parameter]]): The new outputs format of the customize agent.
        """
        outputs = [item.to_dict(ignore=["class_name"]) if isinstance(item, Parameter) else item for item in outputs]
        self.outputs = outputs
        new_action_output = CustomizeAgent.create_action_output(outputs, self.name)
        self.action.outputs_format = new_action_output
    
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
            "inputs": self.inputs,
            "outputs": self.outputs,
            "system_prompt": self.system_prompt,
            "output_parser": self.output_parser.__name__ if self.output_parser is not None else None,
            "parse_mode": self.parse_mode,
            "parse_func": self.parse_func.__name__ if self.parse_func is not None else None,
            "title_format": self.title_format,
            "tool_names": [tool.name for tool in customize_action.tools] if customize_action.tools else [],
            "max_tool_calls": self.max_tool_calls,
            "custom_output_format": self.custom_output_format
        }
        return config
    
    @classmethod
    def load_module(cls, path: str, llm_config: LLMConfig = None, tools: List[Union[Toolkit, Tool]] = None, **kwargs) -> "CustomizeAgent":
        """
        load the agent from local storage. Must provide `llm_config` when loading the agent from local storage. 
            If tools is provided, tool_names must also be provided. 

        Args:
            path: The path of the file
            llm_config: The LLMConfig instance
            tool_names: List of tool names to be used by the agent. If provided,
            tool_dict: Dictionary mapping tool names to Tool instances. Required when tool_names is provided.

        Returns:
            CustomizeAgent: The loaded agent instance
        """
        match_dict = {}
        agent = super().load_module(path=path, llm_config=llm_config, **kwargs)
        if tools:
            match_dict = {tool.name:tool for tool in tools}
        if agent.get("tool_names", None):
            assert tools is not None, "must provide `tools: List[Union[Toolkit, Tool]]` when using `load_module` or `from_file` to load the agent from local storage and `tool_names` is not None or empty"
            added_tools = [match_dict[tool_name] for tool_name in agent["tool_names"]]
            agent["tools"] = [tool if isinstance(tool, Toolkit) else Toolkit(name=tool.name, tools=[tool]) for tool in added_tools]
        return agent 
    
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
        
        if tool_names and "tools" not in agent_data:
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
    
    def get_config(self) -> dict:
        """
        Get a dictionary containing all necessary configuration to recreate this agent.
        
        Returns:
            dict: A configuration dictionary that can be used to initialize a new Agent instance
            with the same properties as this one.
        """
        config = self.get_customize_agent_info()
        config["llm_config"] = self.llm_config.to_dict()
        tool_names = config.pop("tool_names", None)
        if tool_names:
            config["tools"] = [self._raw_tool_map[name] for name in tool_names]
        return config
    
