from .customize_agent import CustomizeAgent
from ..actions.websearch_action import WebSearchAction
from typing import Optional, Callable, List, Any
from pydantic import create_model, Field
from ..prompts.template import PromptTemplate
from ..actions.action import ActionOutput
from ..tools.tool import Toolkit
from ..actions.action import Action, ActionInput
from ..utils.utils import generate_dynamic_class_name



class WebSearchAgent(CustomizeAgent):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    
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
                    action_output_fields[field["name"]] = (Any, Field(description=field["description"]))
                else:
                    action_output_fields[field["name"]] = (Optional[Any], Field(default=None, description=field["description"]))
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
            generate_dynamic_class_name(name+" web search action")
        )

        # Create CustomizeAction-based action with parsing properties only
        customize_action_cls = create_model(
            action_cls_name,
            __base__=WebSearchAction
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
