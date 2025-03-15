import json
from pydantic_core import PydanticUndefined
from typing import Optional, Type, Union, List

from ..core.module import BaseModule
from ..core.module_utils import get_type_name
# from ..core.base_config import Parameter
from ..core.parser import Parser
from ..core.message import Message
from ..models.base_model import BaseLLM, LLMOutputParser
from ..tools.tool import Tool 
from ..prompts.context_extraction import CONTEXT_EXTRACTION


class ActionInput(LLMOutputParser):

    # parameters in ActionInput should be defined in Pydantic Field format
    # for optional variable, use var: Optional[int] = Field(default=None, description="xxx") to define. Remember to add `default=None`

    @classmethod
    def get_input_specification(cls, ignore_fields: List[str] = []) -> str:

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
    
    def to_str(self) -> str:
        return json.dumps(self.get_structured_data(), indent=4)
    

class Action(BaseModule):

    name: str
    description: str
    prompt: Optional[str] = None
    tools: Optional[List[Tool]] = None # specify the possible tool for the action
    inputs_format: Optional[Type[ActionInput]] = None # specify the input format of the action
    outputs_format: Optional[Type[Parser]] = None  # specify the possible structured output format

    def init_module(self):
        pass 

    def execute(self, llm: Optional[BaseLLM] = None, inputs: Optional[dict] = None, sys_msg: Optional[str]=None, return_prompt: bool = False, **kwargs) -> Optional[Parser]:
        """
        The main entrance for executing an action. 

        Args:
            llm (BaseLLM, optional): the LLM used to execute the action.
            msgs (List[Message], optional): the context for executing the action.
            sys_msg (str, optional): The system message for the llm.
        
        Returns:
            Parser: returns a Parser object (a structured output of LLM's generated text  or other structured object).
        """
        pass


class ContextExtraction(Action):

    def __init__(self, **kwargs):
        name = kwargs.pop("name") if "name" in kwargs else CONTEXT_EXTRACTION["name"]
        description = kwargs.pop("description") if "description" in kwargs else CONTEXT_EXTRACTION["description"]
        super().__init__(name=name, description=description, **kwargs)

    def get_context_from_messages(self, messages: List[Message]) -> str:
        str_context = "\n\n".join([str(msg) for msg in messages])
        return str_context 
    
    def execute(self, llm: Optional[BaseLLM] = None, action: Action = None, context: List[Message] = None, **kwargs) -> Union[dict, None]:

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