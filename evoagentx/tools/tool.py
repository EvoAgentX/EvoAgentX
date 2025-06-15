from ..core.module import BaseModule
from typing import Dict, Any, List, Callable

class Tool(BaseModule):
    """
    Base interface for all tools. All tools must implement:
    - `get_tool_schemas`: Returns the OpenAI-compatible function schema
    - `get_tools`: Returns a list of callable functions for all tools
    - `get_tool_descriptions`: Returns a list of descriptions for all tools
    """
    name: str

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """
        Returns the OpenAI-compatible function schema for this tool.
        The schema follows the format used by MCP servers and OpenAI function calling.
        
        Returns:
            Dict[str, Any]: The function schema in OpenAI format
        """
        raise NotImplementedError("All tools must implement get_tool_schemas")

    def get_tools(self) -> List[Callable]:
        """
        Returns a list of callable functions for all tools
        
        Returns:
            List[Callable]: A list of callable functions
        """
        raise NotImplementedError("All tools must implement get_tools")
    
    # Legacy method - to be deprecated
    def get_tool_descriptions(self) -> List[str]:
        raise NotImplementedError("All tools must implement get_tool_descriptions")

    def get_tool_prompt(self) -> str:
        return ""