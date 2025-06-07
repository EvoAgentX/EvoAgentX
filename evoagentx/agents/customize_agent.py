import json
import inspect
from pydantic import create_model, Field
from typing import Union, Optional, Callable, Type, List

from .agent import Agent
from ..core.logging import logger
from ..core.registry import MODULE_REGISTRY, PARSE_FUNCTION_REGISTRY
from ..core.message import Message, MessageType
from ..models.model_configs import LLMConfig 
from ..models.base_model import PARSER_VALID_MODE
from ..prompts.utils import DEFAULT_SYSTEM_PROMPT
from ..prompts.template import PromptTemplate
from ..actions.action import Action, ActionOutput
from ..utils.utils import generate_dynamic_class_name, make_parent_folder
from ..actions.customize_action import CustomizeAction
from ..tools.tool import Tool
from ..actions.action import ActionInput


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
        tool_names (list[str], optional): List of tool names to be used by the agent. If provided,
            tool_dict must also be provided. Both parameters are required when using tools.
        tool_dict (dict, optional): Dictionary mapping tool names to Tool instances. Required when
            tool_names is provided.
        max_tool_calls (int, optional): Maximum number of tool calls. Defaults to 5. 
        customize_prompting (bool, optional): Whether to use customize prompting. Defaults to False.
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
        inputs: Optional[List[dict]] = None, 
        outputs: Optional[List[dict]] = None, 
        system_prompt: Optional[str] = None,
        output_parser: Optional[Type[ActionOutput]] = None, 
        parse_mode: Optional[str] = "title", 
        parse_func: Optional[Callable] = None, 
        title_format: Optional[str] = None, 
        tool_names: Optional[list[str]] = None,
        tool_dict: Optional[dict] = None,
        max_tool_calls: Optional[int] = 5,
        customize_prompting: Optional[bool] = False,
        custom_output_format: Optional[str] = None, 
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

        # validate the data 
        self.validate_data(
            prompt = prompt, 
            prompt_template = prompt_template, 
            inputs = inputs, 
            outputs = outputs, 
            output_parser = output_parser, 
            parse_mode = parse_mode, 
            parse_func = parse_func, 
            title_format = title_format,
            tool_names = tool_names, 
            tool_dict = tool_dict
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
            tool_names=tool_names,
            customize_prompting=customize_prompting,
            tool_dict=tool_dict,
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
        self._store_inputs_outputs_info(inputs, outputs)
        self.output_parser = output_parser 
        self.parse_mode = parse_mode 
        self.parse_func = parse_func 
        self.title_format = title_format
        self.tool_names = tool_names
        self.tool_dict = tool_dict 
        self.max_tool_calls = max_tool_calls
        self.customize_prompting = customize_prompting
        self.custom_output_format = custom_output_format

    def _add_tools(self, tools: list[Tool]):
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
    
    def validate_data(self, prompt: str, prompt_template: PromptTemplate, inputs: List[dict], outputs: List[dict], output_parser: Type[ActionOutput], parse_mode: str, parse_func: Callable, title_format: str, tool_names: List[str], tool_dict: dict):

        # check if the prompt is provided
        if prompt is None and prompt_template is None:
            raise ValueError("`prompt` or `prompt_template` is required when creating a CustomizeAgent.")
        
        # check if all the inputs are in the prompt (only used when prompt_template is not provided)
        if prompt_template is None and inputs:
            all_input_names = [input_item["name"] for input_item in inputs]
            inputs_names_not_in_prompt = [name for name in all_input_names if f'{{{name}}}' not in prompt]
            if inputs_names_not_in_prompt:
                raise KeyError(f"The following inputs are not found in the prompt: {inputs_names_not_in_prompt}.") 
        
        # check if the output_parser is valid 
        if output_parser is not None:
            self._check_output_parser(outputs, output_parser)
        
        # check the parse_mode, parse_func, and title_format
        if parse_mode not in PARSER_VALID_MODE:
            raise ValueError(f"'{parse_mode}' is an invalid value for `parse_mode`. Available choices: {PARSER_VALID_MODE}.")
        
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
        
        # check if the tool_names are provided in the tool_dict
        if tool_names:
            if tool_dict is None:
                raise ValueError("`tool_dict: Dict[str, Callable]` must be provided when `tool_names` is provided.")
            not_provided_tool_names = [name for name in tool_names if name not in tool_dict]
            if not_provided_tool_names:
                logger.warning(f"The following tool names are not provided in `tool_dict`: {not_provided_tool_names}. This can cause unexpected issues when executing the agent.")
    
    def create_customize_action(
        self, 
        name: str, 
        desc: str, 
        prompt: str, 
        prompt_template: PromptTemplate, 
        inputs: List[dict], 
        outputs: List[dict], 
        parse_mode: str, 
        parse_func: Optional[Callable] = None,
        output_parser: Optional[ActionOutput] = None,
        title_format: Optional[str] = "## {title}",
        custom_output_format: Optional[str] = None,
        tool_names: Optional[list[str]] = None,
        customize_prompting: Optional[bool] = False,
        tool_dict: Optional[dict] = None,
        max_tool_calls: Optional[int] = 5
    ) -> Action:
        """Create a custom action based on the provided specifications.
        
        This method dynamically generates an Action class and instance with:
        - Input parameters defined by the inputs specification
        - Output format defined by the outputs specification
        - Custom execution logic using the customize_action_execute function
        - If tool_names is provided, returns a CustomizeAction action instead
        
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
            tool_names: Optional list of tool names
            customize_prompting: Whether to use customize prompting
            
        Returns:
            A newly created Action instance
        """
        assert prompt is not None or prompt_template is not None, "must provide `prompt` or `prompt_template` when creating CustomizeAgent"

        # create the action input type
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
        
        # create the action output type
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
        
        action_cls_name = self._get_unique_class_name(
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
            max_tool_try=max_tool_calls,  # Default value, can be made configurable
            customize_prompting = customize_prompting,
        )

        # Use dict to deduplicate tools by id to avoid adding the same tool multiple times
        tools_by_id = {}
        if tool_names:
            for tool_name in tool_names:
                if tool_name in tool_dict:
                    tool = tool_dict[tool_name]
                    tools_by_id[id(tool)] = tool
        tools = list(tools_by_id.values())
        customize_action.add_tools(tools)
        
        return customize_action
    
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
    
    def _store_inputs_outputs_info(self, inputs: List[dict], outputs: List[dict]):

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
        action_input_params = customize_action.inputs_format.get_attrs()
        action_output_params = customize_action.outputs_format.get_attrs()
        
        config = {
            "class_name": "CustomizeAgent",
            "name": self.name,
            "description": self.description,
            "prompt": customize_action.prompt,
            "prompt_template": customize_action.prompt_template.to_dict() if customize_action.prompt_template is not None else None, 
            # "llm_config": self.llm_config.to_dict(exclude_none=True),
            "inputs": [
                {
                    "name": field,
                    "type": self._action_input_types[field],
                    "description": field_info.description,
                    "required": self._action_input_required[field]
                }
                for field, field_info in customize_action.inputs_format.model_fields.items() if field in action_input_params
            ],
            "outputs": [
                {
                    "name": field,
                    "type": self._action_output_types[field],
                    "description": field_info.description,
                    "required": self._action_output_required[field]
                }
                for field, field_info in customize_action.outputs_format.model_fields.items() if field in action_output_params
            ],
            "system_prompt": self.system_prompt,
            "output_parser": self.output_parser.__name__ if self.output_parser is not None else None,
            "parse_mode": self.parse_mode,
            "parse_func": self.parse_func.__name__ if self.parse_func is not None else None,
            "title_format": self.title_format,
            "tool_names": self.tool_names,
            "max_tool_calls": self.max_tool_calls,
            "customize_prompting": self.customize_prompting, 
            "custom_output_format": self.custom_output_format
        }
        return config
    
    @classmethod
    def load_module(cls, path: str, llm_config: LLMConfig = None, tool_dict: dict = None, **kwargs) -> "CustomizeAgent":
        """
        load the agent from local storage. Must provide `llm_config` when loading the agent from local storage. 
            If tool_names is provided, tool_dict must also be provided. 

        Args:
            path: The path of the file
            llm_config: The LLMConfig instance
            tool_names: List of tool names to be used by the agent. If provided,
            tool_dict: Dictionary mapping tool names to Tool instances. Required when tool_names is provided.

        Returns:
            CustomizeAgent: The loaded agent instance
        """
        agent = super().load_module(path=path, llm_config=llm_config, **kwargs)
        if agent["tool_names"]:
            if tool_dict is None:
                raise ValueError("must provide `tool_dict` when using `load_module` or `from_file` to load the agent from local storage since `tool_names` is provided.")
            agent["tool_dict"] = tool_dict 
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
    
    def get_config(self) -> dict:
        """
        Get a dictionary containing all necessary configuration to recreate this agent.
        
        Returns:
            dict: A configuration dictionary that can be used to initialize a new Agent instance
            with the same properties as this one.
        """
        config = self.get_customize_agent_info()
        config["llm_config"] = self.llm_config.to_dict()
        config["tool_dict"] = self.tool_dict
        return config 
    