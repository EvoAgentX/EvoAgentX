from typing import Callable, Dict, Any, List, Optional, Union
from pydantic import Field

from .tool import Toolkit, Tool
from ..core.logging import logger





class ToolCollection(Toolkit):
    execution_order: List[Tool] = Field(default=None, description="Order to execute tools, a list of strings")
    default_order: Optional[List[Tool]] = Field(default=[], description="Default order to execute tools, a list of strings")
    default_argument_mapping_function: Optional[Dict[str, Dict[Any, Any]]] = Field(default={}, description="Default argument mapping dictionary that convert the unified input into correct format for each tool")
    default_output_mapping_function: Optional[Dict[str, Dict[Any, Any]]] = Field(default={}, description="Default output mapping dictionary that convert the unified output into correct format for each tool")

    def __init__(
        self, 
        kits: Optional[Union[Tool, Toolkit]], 
        execution_order: Optional[List[str]],
        argument_mapping_function: Optional[Dict[str, Callable]] = {},
        output_mapping_function: Optional[Dict[str, Callable]] = {},
        name:str="ToolCollection",
        ):
        tools = []
        for kit in kits:
            if isinstance(kit, Tool):
                tools.append(kit)
            elif isinstance(kit, Toolkit):
                tools += kit.get_tool()
            else:
                try:
                    cur_name = kit.name
                except:
                    cur_name = kit
                logger.warning(f"{cur_name} not a valid Tool/Toolkit, skip adding to the collection")
        
        super().__init__(name=name, tools=tools)
        self.argument_mapping_function = argument_mapping_function or self.default_argument_mapping_function
        self.output_mapping_function = output_mapping_function or self.default_output_mapping_function
        self._init_module()
    
    def _init_module(self, preferred_order: List[str]) -> None:
        ## Initialize execution order
        tool_mappings = {}
        if preferred_order: 
            order_list = []
        else:
            preferred_order = self.default_order
        
        for name in preferred_order:
            cur_tool = self.get_tool(name)
            if not cur_tool:
                logger.warning(f"Tool {name} not found in inherit tool lists...")
                continue
            if not self.argument_mapping_function.get(name, None):
                logger.warning(f"Argument mapping function for {name} cannot be found...")
            tool_mappings[name] = cur_tool
        
        self.execution_order = order_list
        self.tool_mappings = tool_mappings
    
    def _get_next_execute(self, execution_history: List[str]) -> str:
        pass
    
    def __call__(self, **kwargs) -> Any:
        execution_history = []
        cur_tool = self._get_next_execute(execution_history)
        while cur_tool:
            try:
                cur_input = self.argument_mapping_function[cur_tool](kwargs)
                result = self.tool_mappings[cur_tool](**cur_input)
            except Exception as e:
                result = {"tool_name": cur_tool, "success": False, "error": str(e), "result": None}
            if result:
                success_output = self.output_mapping_function[cur_tool](result)
                result = {"tool_name": cur_tool, "success": True, "error": None, "result": success_output}
                return result
            execution_history.append(result)
            cur_tool = self._get_next_execute(execution_history)
        
        return {"tool_name": None, "success": False, "error": "No tool executed successfully", "result": None}
    
    



