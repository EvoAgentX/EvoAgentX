import json
from pydantic_core import PydanticUndefined
from typing import Optional, Type, Tuple, Union, List

from ..core.module import BaseModule
from ..core.module_utils import get_type_name
# from ..core.base_config import Parameter
from ..core.parser import Parser
from ..core.message import Message
from ..models.base_model import BaseLLM, LLMOutputParser
from ..tools.tool import Tool 
from ..prompts.context_extraction import CONTEXT_EXTRACTION


class ActionInput(LLMOutputParser):
    """Input specification and parsing for actions.
    
    This class defines the input requirements for actions and provides methods
    to generate structured input specifications. It inherits from LLMOutputParser 
    to allow parsing of LLM outputs into structured inputs for actions.
    
    Notes:
        Parameters in ActionInput should be defined in Pydantic Field format.
        For optional variables, use format: 
        var: Optional[int] = Field(default=None, description="xxx")
        Remember to add `default=None` for optional parameters.
    """

    @classmethod
    def get_input_specification(cls, ignore_fields: List[str] = []) -> str:
        """Generate a JSON specification of the input requirements.
        
        Examines the class fields and produces a structured specification of
        the input parameters, including their types, descriptions, and whether
        they are required.
        
        Args:
            ignore_fields (List[str]): List of field names to exclude from the specification.
            
        Returns:
            A JSON string containing the input specification, or an empty string
            if no fields are defined or all are ignored.
        """
        fields_info = {}
        attrs = cls.get_attrs()
        for field_name, field_info in cls.model_fields.items():
            if field_name in ignore_fields:
                continue
            if field_name not in attrs:
                continue
            field_type = get_type_name(field_info.annotation)
            field_desc = field_info.description if field_info.description is not None else None
            # field_required = field_info.is_required()
            field_default = str(field_info.default) if field_info.default is not PydanticUndefined else None
            field_required = True if field_default is None else False
            description = field_type + ", "
            if field_desc is not None:
                description += (field_desc.strip() + ", ") 
            description += ("required" if field_required else "optional")
            if field_default is not None:
                description += (", Default value: " + field_default)
            fields_info[field_name] = description
        
        if len(fields_info) == 0:
            return "" 
        fields_info_str = json.dumps(fields_info, indent=4)
        return fields_info_str


class ActionOutput(LLMOutputParser):
    """Output representation for actions.
    
    This class handles the structured output of actions, providing methods
    to convert the output to structured data. It inherits from LLMOutputParser
    to support parsing of LLM outputs into structured action results.
    """
    
    def to_str(self) -> str:
        """Convert the output to a formatted JSON string.
        
        Returns:
            A pretty-printed JSON string representation of the structured data.
        """
        return json.dumps(self.get_structured_data(), indent=4)
    

class Action(BaseModule):
    """Base class for all actions in the EvoAgentX framework.
    
    Actions represent discrete operations that can be performed by agents.
    They define inputs, outputs, and execution behavior, and can optionally
    use tools to accomplish their tasks.
    
    Attributes:
        name (str): Unique identifier for the action.
        description (str): Human-readable description of what the action does.
        prompt (Optional[str]): Optional prompt template for this action.
        tools (Optional[List[Tool]]): Optional list of tools that can be used by this action.
        inputs_format (Optional[Type[ActionInput]]): Optional class defining the expected input structure.
        outputs_format (Optional[Type[Parser]]): Optional class defining the expected output structure.
    """

    name: str
    description: str
    prompt: Optional[str] = None
    tools: Optional[List[Tool]] = None # specify the possible tool for the action
    inputs_format: Optional[Type[ActionInput]] = None # specify the input format of the action
    outputs_format: Optional[Type[Parser]] = None  # specify the possible structured output format

    def init_module(self):
        """Initialize the action module.
        
        This method is called after the action is instantiated.
        Subclasses can override this to perform custom initialization.
        """
        pass 

    def execute(self, llm: Optional[BaseLLM] = None, inputs: Optional[dict] = None, sys_msg: Optional[str]=None, return_prompt: bool = False, **kwargs) -> Optional[Union[Parser, Tuple[Parser, str]]]:
        """Execute the action to produce a result.
        
        This is the main entry point for executing an action. Subclasses must
        implement this method to define the action's behavior.

        Args:
            llm (Optional[BaseLLM]): The LLM used to execute the action.
            inputs (Optional[dict]): Input data for the action execution. The input data should be a dictionary that matches the input format of the provided prompt. 
                For example, if the prompt contains a variable `{input_var}`, the `inputs` dictionary should have a key `input_var`, otherwise the variable will be set to empty string. 
            sys_msg (Optional[str]): Optional system message for the LLM.
            return_prompt (bool): Whether to return the complete prompt passed to the LLM.
            **kwargs: Additional keyword arguments for the execution.
        
        Returns:
            If `return_prompt` is False, the method returns a Parser object containing the structured result of the action.
            If `return_prompt` is True, the method returns a tuple containing the Parser object and the complete prompt passed to the LLM.
        """
        pass


class ContextExtraction(Action):
    """Action for extracting structured inputs from context.
    
    This action analyzes a conversation context to extract relevant information
    that can be used as inputs for other actions. It uses the LLM to interpret
    unstructured contextual information and format it according to the target
    action's input requirements.
    """

    def __init__(self, **kwargs):
        name = kwargs.pop("name") if "name" in kwargs else CONTEXT_EXTRACTION["name"]
        description = kwargs.pop("description") if "description" in kwargs else CONTEXT_EXTRACTION["description"]
        super().__init__(name=name, description=description, **kwargs)

    def get_context_from_messages(self, messages: List[Message]) -> str:
        str_context = "\n\n".join([str(msg) for msg in messages])
        return str_context 
    
    def execute(self, llm: Optional[BaseLLM] = None, action: Action = None, context: List[Message] = None, **kwargs) -> Union[dict, None]:
        """Extract structured inputs for an action from conversation context.
        
        This method uses the LLM to analyze the conversation context and extract
        information that matches the input requirements of the target action.
        
        Args:
            llm: The language model to use for extraction.
            action: The target action whose input requirements (`inputs_format`) define what to extract.
            context: List of messages providing the conversation context.
            **kwargs: Additional keyword arguments.
            
        Returns:
            A dictionary containing the extracted inputs for the target action,
            or None if extraction is not possible (e.g., if the action doesn't
            require inputs or if context is missing).
        """
        if action is None or context is None:
            return None
        
        action_inputs_cls: Type[ActionInput] = action.inputs_format
        if action_inputs_cls is None:
            # the action does not require inputs
            return None
        
        action_inputs_desc = action_inputs_cls.get_input_specification()
        str_context = self.get_context_from_messages(messages=context)

        if not action_inputs_desc or not str_context:
            return None
        
        prompt = CONTEXT_EXTRACTION["prompt"].format(
            context=str_context,
            action_name=action.name, 
            action_description=action.description,
            action_inputs=action_inputs_desc
        )

        action_inputs = llm.generate(
            prompt=prompt, 
            system_message=CONTEXT_EXTRACTION["system_prompt"],
            parser=action_inputs_cls
        )
        action_inputs_data = action_inputs.get_structured_data()

        return action_inputs_data