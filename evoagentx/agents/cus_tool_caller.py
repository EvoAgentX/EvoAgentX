from typing import Dict, Optional, Callable
from .customize_agent import CustomizeAgent
from ..actions.tool_calling import ToolCalling
from ..prompts.tool_caller import CUSTOM_TOOL_CALLER_PROMPT
from ..tools.tool import Tool

class CusToolCaller(CustomizeAgent):
    """
    An agent that determines whether to call a tool or provide a direct answer
    based on analyzing the user query and available tools.
    
    This agent has two main actions:
    1. ToolCalling - Executes the selected tool if required
    """
    tools_schema: Optional[Dict[str, dict]] = None
    tools_caller: Optional[Dict[str, Callable]] = None
    tool_calling_action: Optional[ToolCalling] = None
    _initialized: bool = False
    ori_prompt: str = ""
    tool_calling_prompt: str = ""
    
    def __init__(self, **kwargs):
        # Extract our specific parameters before passing to parent
        max_tool_try = kwargs.pop("max_tool_try", 2)
        # Create actions with the max_tool_try value
        # Don't add to kwargs - let CustomizeAgent handle action creation
        inputs = kwargs.get("inputs", None)
        outputs = kwargs.get("outputs", None)
        tool_actions = [ToolCalling(max_tool_try=max_tool_try, inputs = inputs, outputs = outputs, ori_prompt = kwargs.get("prompt", ""))]
        tools = kwargs.get("tools", None)
        
        # Initialize as CustomizeAgent - all other parameters should be passed by the caller
        super().__init__(**kwargs)
        self.outputs_format = self.actions[0].outputs_format
        self.actions = []
        self._action_map = {}
        self.init_module()
        self.tools = tools
        
        # Now add our tool actions (after CustomizeAgent has created its action)
        self.ori_prompt = self.system_prompt + kwargs.get("prompt", "")
        for action in tool_actions:
            self.add_action(action)
        
        # Store values as instance attributes after parent initialization
        self.max_tool_try = max_tool_try
        # Store the tool calling action for easy access
        self.tool_calling_action = self._action_map[self.tool_calling_action_name]
        self.add_tools(tools)
        self.tool_calling_prompt = self.ori_prompt + CUSTOM_TOOL_CALLER_PROMPT + "\n### Tools Available\n" + str(self.tools_schema)
        
    @property
    def tool_calling_action_name(self) -> str:  
        return self.get_action_name(action_cls=ToolCalling)
    
    def add_tool(self, tool: Tool) -> None:
        """Add a tool to the tool caller agent"""
        self.add_tools([tool])
    
    def add_tools(self, tools: list[Tool]):
        tools_schemas = [tool.get_tool_schemas() for tool in tools]
        tools_schemas = [j for i in tools_schemas for j in i]
        tools_callers = [tool.get_tools() for tool in tools]
        tools_callers = [j for i in tools_callers for j in i]
        tools_names = [i["function"]["name"] for i in tools_schemas]
        if not self.tools_schema:
            self.tools_schema = {}
            self.tools_caller = {}
        for tool_schema, tool_caller, tool_name in zip(tools_schemas, tools_callers, tools_names):
            self.tools_schema[tool_name] = tool_schema
            self.tools_caller[tool_name] = tool_caller
        
        self.tool_calling_action.add_tools(tools)
        self._initialized = True
    

