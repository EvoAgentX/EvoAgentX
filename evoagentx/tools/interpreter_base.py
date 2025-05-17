from typing import Dict, Any, Optional, List, Callable
from .tool import Tool

class BaseInterpreter(Tool):
    """
    Base class for interpreter tools that execute code securely.
    Implements the standard tool interface with get_tool_schemas and execute methods.
    """

    def __init__(
        self, 
        name: str = 'Base Interpreter',
        schemas: Optional[List[dict]] = None,
        descriptions: Optional[List[str]] = None,
        tools: Optional[List[Callable]] = None,
        **kwargs
    ):
        # Get default values for required attributes
        # name = kwargs.pop('name', 'Base Interpreter')
        schemas = schemas or self.get_tool_schemas()
        descriptions = descriptions or self.get_tool_descriptions()
        tools = tools or self.get_tools()
        
        # Pass these to the parent class initialization
        super().__init__(
            name=name,
            schemas=schemas,
            descriptions=descriptions,
            tools=tools,
            **kwargs
        )

    def get_tool_schemas(self) -> Dict[str, Any]:
        """
        Returns the OpenAI-compatible function schema for the interpreter.
        
        Returns:
            Dict[str, Any]: Function schema in OpenAI format
        """
        pass

    def get_tool_descriptions(self) -> str:
        """
        Returns a brief description of the interpreter tool.
        
        Returns:
            str: Tool description
        """
        pass

    def get_tools(self):
        """
        Returns a list of callable methods provided by this tool.
        
        Returns:
            list: List of callable methods
        """
        pass
