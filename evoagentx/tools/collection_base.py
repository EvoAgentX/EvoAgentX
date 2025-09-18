from typing import Callable, Dict, Any, List, Optional, Union
from pydantic import Field

from .tool import Tool, Toolkit
from ..core.logging import logger


class ToolCollection(Tool):
    """
    Base class for tool collections that orchestrate multiple tools.
    A ToolCollection is a Tool that manages and executes multiple tools in a specific order.
    """
    
    # Tool interface requirements
    name: str = "ToolCollection"
    description: str = "A collection that orchestrates multiple tools"
    inputs: Dict[str, Dict[str, str]] = {
        "query": {
            "type": "string", 
            "description": "The input query to process through the tool collection"
        }
    }
    required: Optional[List[str]] = ["query"]
    
    # Collection-specific attributes
    execution_order: List[str] = Field(default_factory=list, description="Order to execute tools")
    default_order: Optional[List[str]] = Field(default_factory=list, description="Default order to execute tools")
    argument_mapping_function: Optional[Dict[str, Callable]] = Field(default_factory=dict, description="Argument mapping functions for each tool")
    output_mapping_function: Optional[Dict[str, Callable]] = Field(default_factory=dict, description="Output mapping functions for each tool")

    def __init__(
        self, 
        name: str = "ToolCollection",
        description: str = "A collection that orchestrates multiple tools",
        kits: Optional[List[Union[Tool, Toolkit]]] = None, 
        execution_order: Optional[List[str]] = None,
        argument_mapping_function: Optional[Dict[str, Callable]] = None,
        output_mapping_function: Optional[Dict[str, Callable]] = None,
        **kwargs
        ):
        
        # Call parent constructor first
        super().__init__(**kwargs)
        
        # Initialize Tool attributes
        self.name = name
        self.description = description
        
        # Process tools from kits
        tools = []
        kits = kits or []
        for kit in kits:
            if isinstance(kit, Tool):
                tools.append(kit)
            elif isinstance(kit, Toolkit):
                tools += kit.get_tools()
            else:
                try:
                    cur_name = kit.name
                except:
                    cur_name = str(kit)
                logger.warning(f"{cur_name} not a valid Tool/Toolkit, skip adding to the collection")
        
        # Initialize collection attributes
        self.tools = tools
        self.argument_mapping_function = argument_mapping_function or {}
        self.output_mapping_function = output_mapping_function or {}
        self._init_module(execution_order)
    
    def _init_module(self, preferred_order: Optional[List[str]] = None) -> None:
        """Initialize the tool mappings and execution order."""
        tool_mappings = {}
        order_list = []
        
        if not preferred_order:
            preferred_order = self.default_order or []
        
        # Create tool name to tool mapping
        available_tools = {tool.name: tool for tool in self.tools}
        
        for name in preferred_order:
            cur_tool = available_tools.get(name)
            if not cur_tool:
                logger.warning(f"Tool {name} not found in available tools...")
                continue
            if not self.argument_mapping_function.get(name, None):
                logger.warning(f"Argument mapping function for {name} cannot be found...")
            tool_mappings[name] = cur_tool
            order_list.append(name)
        
        self.execution_order = order_list
        self.tool_mappings = tool_mappings
    
    def _get_next_execute(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> Optional[str]:
        """Get the next tool to execute based on current outputs."""
        for tool_name in self.execution_order:
            if tool_name not in outputs:
                return tool_name
        return None
    
    def __call__(self, **kwargs) -> Dict[str, Any]:
        """
        Execute the tool collection with the given arguments.
        
        Args:
            **kwargs: Arguments based on the collection's input schema
            
        Returns:
            Dict containing the results from all executed tools
        """
        # Validate required parameters based on the collection's inputs schema
        if self.required:
            for required_param in self.required:
                if required_param not in kwargs:
                    raise ValueError(f"Missing required parameter '{required_param}'")
            
        inputs = kwargs.copy()
        outputs = {}
        
        while True:
            next_tool_name = self._get_next_execute(inputs, outputs)
            if not next_tool_name:
                break
                
            tool = self.tool_mappings.get(next_tool_name)
            if not tool:
                logger.error(f"Tool {next_tool_name} not found in tool mappings")
                break
                
            try:
                mapped_args = self.argument_mapping_function[next_tool_name](inputs, outputs)
                result = tool(**mapped_args)
                outputs[next_tool_name] = self.output_mapping_function[next_tool_name](result)
            except Exception as e:
                logger.error(f"Error executing tool {next_tool_name}: {e}")
                outputs[next_tool_name] = {"error": str(e)}
                
        return outputs
