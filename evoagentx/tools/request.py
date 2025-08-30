from typing import Dict, Any, Optional, List

from .tool import Tool, Toolkit
from .request_base import RequestBase


class HTTPRequestTool(Tool):
    """Universal HTTP request tool that handles all request methods and processing."""
    
    name: str = "http_request"
    description: str = "Make HTTP requests (GET, POST, PUT, DELETE, etc.) with automatic content processing and optional file saving"
    inputs: Dict[str, Dict[str, str]] = {
        "url": {
            "type": "string",
            "description": "The URL to make the request to"
        },
        "method": {
            "type": "string",
            "description": "HTTP method to use (GET, POST, PUT, DELETE, PATCH, etc.). Defaults to GET"
        },
        "headers": {
            "type": "object",
            "description": "Optional headers to include in the request"
        },
        "params": {
            "type": "object",
            "description": "Optional URL parameters to include in the request"
        },
        "data": {
            "type": "object",
            "description": "Optional form data to send in the request body"
        },
        "json_data": {
            "type": "object",
            "description": "Optional JSON data to send in the request body"
        },
        "return_raw": {
            "type": "boolean",
            "description": "If true, return raw response content. If false (default), return processed content (HTML converted to text, JSON parsed, etc.)"
        },
        "save_file_path": {
            "type": "string",
            "description": "Optional file path to save the response content"
        }
    }
    required: Optional[List[str]] = ["url"]
    
    def __init__(self, request_base: RequestBase = None):
        super().__init__()
        self.request_base = request_base
    
    def __call__(self, url: str, method: str = 'GET', headers: dict = None,
                 params: dict = None, data: dict = None,
                 json_data: dict = None, return_raw: bool = False,
                 save_file_path: str = None) -> Dict[str, Any]:
        """
        Make an HTTP request with comprehensive processing and error handling.
        
        Args:
            url: The URL to request
            method: HTTP method (GET, POST, PUT, DELETE, etc.)
            headers: Optional headers to include
            params: Optional URL parameters
            data: Optional form data to send
            json_data: Optional JSON data to send
            return_raw: If True, return raw content; if False, return processed content
            save_file_path: Optional path to save the response content
            
        Returns:
            Dictionary containing response data and metadata
        """
        return self.request_base.request_and_process(
            url=url,
            method=method,
            headers=headers,
            params=params,
            data=data,
            json_data=json_data,
            return_raw=return_raw,
            save_file_path=save_file_path
        )


class RequestToolkit(Toolkit):
    def __init__(self, name: str = "RequestToolkit"):
        # Create the shared request base instance
        request_base = RequestBase()
        
        # Create tools with the shared request base
        tools = [
            HTTPRequestTool(request_base=request_base)
        ]
        
        # Initialize parent with tools
        super().__init__(name=name, tools=tools)
        