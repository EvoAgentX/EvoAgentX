import json
import requests
from typing import Dict, List, Optional, Any, Union, Callable
from abc import ABC, abstractmethod


from .tool import Tool, Toolkit
from ..core.logging import logger
from ..core.module import BaseModule


class APITool(Tool):
    """
    API tool wrapper that encapsulates a single API endpoint as a Tool
    
    Attributes:
        name: Tool name
        description: Tool description
        inputs: Input parameter schema
        required: List of required parameters
        endpoint_config: API endpoint configuration
        auth_config: Authentication configuration
        function: Actual execution function
    """
    
    def __init__(
        self,
        name: str,
        description: str,
        inputs: Dict[str, Dict[str, Any]],
        required: Optional[List[str]] = None,
        endpoint_config: Dict[str, Any] = None,
        auth_config: Dict[str, Any] = None,
        function: Callable = None
    ):
        super().__init__(name=name, description=description, inputs=inputs, required=required)
        self.endpoint_config = endpoint_config or {}
        self.auth_config = auth_config or {}
        self.function = function
    
    @property
    def __name__(self):
        return self.name
    
    def __call__(self, **kwargs):
        """Execute the API call"""
        if not self.function:
            raise ValueError("Function not set for APITool")
        
        try:
            result = self.function(**kwargs)
            return self._process_result(result)
        except Exception as e:
            logger.error(f"Error calling API tool {self.name}: {str(e)}")
            raise
    
    def _process_result(self, result: Any) -> Any:
        """Process API response"""
        if isinstance(result, requests.Response):
            try:
                return result.json()
            except (ValueError, json.JSONDecodeError):
                return result.text
        return result
    
    @classmethod
    def validate_attributes(cls):
        """Validate attributes"""
        # APITool attributes are set during instantiation; skip class-level attribute validation
        # Only validate if class-level attributes are defined in subclasses
        if cls.__name__ == 'APITool':
            return
        
        # Inherit parent validation but relax the requirement for the __call__ method
        required_attributes = {
            "name": str,
            "description": str,
            "inputs": dict
        }
        
        for attr, attr_type in required_attributes.items():
            if not hasattr(cls, attr):
                raise ValueError(f"Attribute {attr} is required")
            if not isinstance(getattr(cls, attr), attr_type):
                raise ValueError(f"Attribute {attr} must be of type {attr_type}")

        if hasattr(cls, 'required') and cls.required:
            for required_input in cls.required:
                if required_input not in cls.inputs:
                    raise ValueError(f"Required input '{required_input}' is not found in inputs")


class APIToolkit(Toolkit):
    """
    API tool collection representing all endpoints of an API service
    
    Attributes:
        name: Service name
        tools: List of API tools
        base_url: Base URL
        auth_config: Authentication configuration
        common_headers: Common request headers
    """
    
    def __init__(
        self,
        name: str,
        tools: List[APITool],
        base_url: str = "",
        auth_config: Dict[str, Any] = None,
        common_headers: Dict[str, str] = None
    ):
        super().__init__(name=name, tools=tools)
        self.base_url = base_url
        self.auth_config = auth_config or {}
        self.common_headers = common_headers or {}
    
    def add_auth_to_headers(self, headers: Dict[str, str]) -> Dict[str, str]:
        """Add authentication information to request headers"""
        headers = headers.copy()
        headers.update(self.common_headers)
        
        # Handle different authentication types
        if "api_key" in self.auth_config:
            key_name = self.auth_config.get("key_name", "X-API-Key")
            headers[key_name] = self.auth_config["api_key"]
        
        if "bearer_token" in self.auth_config:
            headers["Authorization"] = f"Bearer {self.auth_config['bearer_token']}"
        
        return headers


class BaseAPIConverter(BaseModule, ABC):
    """
    Base API converter abstract class
    
    Responsible for converting an API specification into an APIToolkit
    """
    
    def __init__(
        self,
        input_schema: Union[str, Dict[str, Any]],
        description: str = "",
        auth_config: Dict[str, Any] = None
    ):
        """
        Initialize the API converter
        
        Args:
            input_schema: API specification, can be a file path or a dictionary
            description: Service description
            auth_config: Authentication configuration
        """
        super().__init__()
        self.input_schema = self._load_schema(input_schema)
        self.description = description
        self.auth_config = auth_config or {}
    
    def _load_schema(self, schema: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Load API specification"""
        if isinstance(schema, str):
            # If it's a file path
            try:
                with open(schema, 'r', encoding='utf-8') as f:
                    if schema.endswith('.json'):
                        return json.load(f)
                    elif schema.endswith(('.yaml', '.yml')):
                        import yaml
                        return yaml.safe_load(f)
                    else:
                        # Attempt JSON parsing
                        content = f.read()
                        return json.loads(content)
            except Exception as e:
                logger.error(f"Failed to load schema from {schema}: {e}")
                raise
        elif isinstance(schema, dict):
            return schema
        else:
            raise ValueError("input_schema must be a file path or dictionary")
    
    @abstractmethod
    def convert_to_toolkit(self) -> APIToolkit:
        """
        Convert API specification to APIToolkit
        
        Returns:
            APIToolkit: Converted toolkit
        """
        pass
    
    @abstractmethod
    def _create_api_function(self, endpoint_config: Dict[str, Any]) -> Callable:
        """
        Create an execution function for a single API endpoint
        
        Args:
            endpoint_config: Endpoint configuration
            
        Returns:
            Callable: API execution function
        """
        pass
    
    def _extract_parameters(self, endpoint_config: Dict[str, Any]) -> tuple:
        """
        Extract parameter information from endpoint configuration
        
        Args:
            endpoint_config: Endpoint configuration
            
        Returns:
            tuple: (inputs, required) parameter schema and list of required parameters
        """
        inputs = {}
        required = []
        
        # Implementation depends on the specific API specification format
        # Default implementation; subclasses can override
        parameters = endpoint_config.get("parameters", [])
        
        for param in parameters:
            param_name = param.get("name", "")
            param_type = param.get("type", "string")
            param_desc = param.get("description", "")
            is_required = param.get("required", False)
            
            inputs[param_name] = {
                "type": param_type,
                "description": param_desc
            }
            
            if is_required:
                required.append(param_name)
        
        return inputs, required


class OpenAPIConverter(BaseAPIConverter):
    """
    OpenAPI (Swagger) specification converter
    """
    
    def convert_to_toolkit(self) -> APIToolkit:
        """Convert OpenAPI specification to APIToolkit"""
        service_name = self.input_schema.get("info", {}).get("title", "API Service")
        base_url = self._get_base_url()
        
        tools = []
        paths = self.input_schema.get("paths", {})
        
        for path, methods in paths.items():
            for method, operation in methods.items():
                if method.upper() in ["GET", "POST", "PUT", "DELETE", "PATCH"]:
                    tool = self._create_tool_from_operation(path, method, operation, base_url)
                    if tool:
                        tools.append(tool)
        
        return APIToolkit(
            name=service_name,
            tools=tools,
            base_url=base_url,
            auth_config=self.auth_config,
            common_headers={"Content-Type": "application/json"}
        )
    
    def _get_base_url(self) -> str:
        """Get base URL from the OpenAPI specification"""
        servers = self.input_schema.get("servers", [])
        if servers:
            return servers[0].get("url", "")
        
        # Legacy fields: host and basePath
        host = self.input_schema.get("host", "")
        base_path = self.input_schema.get("basePath", "")
        schemes = self.input_schema.get("schemes", ["https"])
        
        if host:
            return f"{schemes[0]}://{host}{base_path}"
        
        return ""
    
    def _create_tool_from_operation(
        self, 
        path: str, 
        method: str, 
        operation: Dict[str, Any], 
        base_url: str
    ) -> Optional[APITool]:
        """Create a tool from an OpenAPI operation"""
        try:
            # Generate tool name
            operation_id = operation.get("operationId")
            if not operation_id:
                # If operationId is missing, generate based on path and method
                clean_path = path.replace("/", "_").replace("{", "").replace("}", "").strip("_")
                operation_id = f"{method.lower()}_{clean_path}"
            
            # Extract parameters
            inputs, required = self._extract_openapi_parameters(operation)
            
            # Create API execution function
            api_function = self._create_api_function({
                "url": base_url + path,
                "method": method.upper(),
                "operation": operation
            })
            
            return APITool(
                name=operation_id,
                description=operation.get("summary", operation.get("description", "")),
                inputs=inputs,
                required=required,
                endpoint_config={
                    "url": base_url + path,
                    "method": method.upper(),
                    "operation": operation
                },
                auth_config=self.auth_config,
                function=api_function
            )
        except Exception as e:
            logger.warning(f"Failed to create tool for {method.upper()} {path}: {e}")
            return None
    
    def _extract_openapi_parameters(self, operation: Dict[str, Any]) -> tuple:
        """Extract parameters from an OpenAPI operation"""
        inputs = {}
        required = []
        
        # Handle parameters
        parameters = operation.get("parameters", [])
        for param in parameters:
            param_name = param.get("name", "")
            param_schema = param.get("schema", {})
            param_type = param_schema.get("type", "string")
            
            inputs[param_name] = {
                "type": param_type,
                "description": param.get("description", "")
            }
            
            if param.get("required", False):
                required.append(param_name)
        
        # Handle requestBody
        request_body = operation.get("requestBody", {})
        if request_body:
            content = request_body.get("content", {})
            for media_type, media_schema in content.items():
                if "application/json" in media_type:
                    schema = media_schema.get("schema", {})
                    properties = schema.get("properties", {})
                    
                    for prop_name, prop_schema in properties.items():
                        inputs[prop_name] = {
                            "type": prop_schema.get("type", "string"),
                            "description": prop_schema.get("description", "")
                        }
                        
                        if prop_name in schema.get("required", []):
                            required.append(prop_name)
        
        return inputs, required
    
    def _create_api_function(self, endpoint_config: Dict[str, Any]) -> Callable:
        """Create OpenAPI execution function"""
        url = endpoint_config["url"]
        method = endpoint_config["method"]
        operation = endpoint_config["operation"]
        
        def api_call(**kwargs):
            # Separate path params, query params, and request body
            path_params = {}
            query_params = {}
            body_data = {}
            
            parameters = operation.get("parameters", [])
            param_locations = {param["name"]: param.get("in", "query") for param in parameters}
            
            for key, value in kwargs.items():
                if value is None:
                    continue
                    
                location = param_locations.get(key, "body")
                if location == "path":
                    path_params[key] = value
                elif location == "query":
                    query_params[key] = value
                else:
                    body_data[key] = value
            
            # Replace path parameters
            final_url = url
            for param_name, param_value in path_params.items():
                final_url = final_url.replace(f"{{{param_name}}}", str(param_value))
            
            # Prepare request
            headers = {"Content-Type": "application/json"}
            if hasattr(self, 'auth_config') and self.auth_config:
                if "api_key" in self.auth_config:
                    key_name = self.auth_config.get("key_name", "X-API-Key")
                    headers[key_name] = self.auth_config["api_key"]
            
            # Send request
            try:
                if method in ["GET", "DELETE"]:
                    response = requests.request(
                        method=method,
                        url=final_url,
                        params=query_params,
                        headers=headers,
                        timeout=30
                    )
                else:
                    response = requests.request(
                        method=method,
                        url=final_url,
                        params=query_params,
                        json=body_data if body_data else None,
                        headers=headers,
                        timeout=30
                    )
                
                response.raise_for_status()
                
                try:
                    return response.json()
                except (ValueError, json.JSONDecodeError):
                    return response.text
                    
            except requests.exceptions.RequestException as e:
                logger.error(f"API request failed: {e}")
                raise
        
        # Set function name for easier debugging
        api_call.__name__ = f"api_call_{method.lower()}"
        return api_call


class RapidAPIConverter(OpenAPIConverter):
    """
    RapidAPI-specific converter
    Inherits from OpenAPIConverter and adds RapidAPI-specific authentication and configuration
    """
    
    def __init__(
        self,
        input_schema: Union[str, Dict[str, Any]],
        description: str = "",
        rapidapi_key: str = "",
        rapidapi_host: str = "",
        **kwargs
    ):
        """
        Initialize the RapidAPI converter
        
        Args:
            input_schema: API specification
            description: Service description
            rapidapi_key: RapidAPI key
            rapidapi_host: RapidAPI host
        """
        # Set RapidAPI-specific authentication configuration
        if not rapidapi_key:
            from os import getenv
            from dotenv import load_dotenv
            load_dotenv()
            rapidapi_key = getenv("RAPIDAPI_KEY", "")
            if not rapidapi_key:
                raise ValueError("rapidapi_key not provided or RAPIDAPI_KEY environment variable not set")
        if not rapidapi_host:
            raise ValueError("rapidapi_host not provided or RAPIDAPI_HOST environment variable not set")
        
        auth_config = {
            "api_key": rapidapi_key,
            "key_name": "X-RapidAPI-Key",
            "rapidapi_host": rapidapi_host
        }
        
        super().__init__(
            input_schema=input_schema,
            description=description,
            auth_config=auth_config,
            **kwargs
        )
    
    def convert_to_toolkit(self) -> APIToolkit:
        """Convert to a RapidAPI toolkit"""
        toolkit = super().convert_to_toolkit()
        
        # Add RapidAPI-specific common headers
        rapidapi_headers = {
            "X-RapidAPI-Key": self.auth_config.get("api_key", ""),
            "X-RapidAPI-Host": self.auth_config.get("rapidapi_host", "")
        }
        
        toolkit.common_headers.update(rapidapi_headers)
        
        return toolkit
    
    def _create_api_function(self, endpoint_config: Dict[str, Any]) -> Callable:
        """Create RapidAPI execution function"""
        url = endpoint_config["url"]
        method = endpoint_config["method"]
        operation = endpoint_config["operation"]
        
        def rapidapi_call(**kwargs):
            # Separate parameters
            path_params = {}
            query_params = {}
            body_data = {}
            
            parameters = operation.get("parameters", [])
            param_locations = {param["name"]: param.get("in", "query") for param in parameters}
            
            for key, value in kwargs.items():
                if value is None:
                    continue
                    
                location = param_locations.get(key, "body")
                if location == "path":
                    path_params[key] = value
                elif location == "query":
                    query_params[key] = value
                else:
                    body_data[key] = value
            
            # Replace path parameters
            final_url = url
            for param_name, param_value in path_params.items():
                final_url = final_url.replace(f"{{{param_name}}}", str(param_value))
            
            # Prepare RapidAPI request headers
            headers = {
                "Content-Type": "application/json",
                "X-RapidAPI-Key": self.auth_config.get("api_key", ""),
                "X-RapidAPI-Host": self.auth_config.get("rapidapi_host", "")
            }
            
            # Send request
            try:
                if method in ["GET", "DELETE"]:
                    response = requests.request(
                        method=method,
                        url=final_url,
                        params=query_params,
                        headers=headers,
                        timeout=30
                    )
                else:
                    response = requests.request(
                        method=method,
                        url=final_url,
                        params=query_params,
                        json=body_data if body_data else None,
                        headers=headers,
                        timeout=30
                    )
                
                response.raise_for_status()
                
                try:
                    return response.json()
                except (ValueError, json.JSONDecodeError):
                    return response.text
                    
            except requests.exceptions.RequestException as e:
                logger.error(f"RapidAPI request failed: {e}")
                raise
        
        rapidapi_call.__name__ = f"rapidapi_call_{method.lower()}"
        return rapidapi_call


def create_openapi_toolkit(
    schema_path_or_dict: Union[str, Dict[str, Any]],
    service_name: str = None,
    auth_config: Dict[str, Any] = None
) -> APIToolkit:
    """
    Convenience function: create an APIToolkit from an OpenAPI specification
    
    Args:
        schema_path_or_dict: OpenAPI specification file path or dictionary
        service_name: Service name (optional, will be extracted from the spec)
        auth_config: Authentication configuration
    
    Returns:
        APIToolkit: Created toolkit
    """
    converter = OpenAPIConverter(
        input_schema=schema_path_or_dict,
        description=service_name or "",
        auth_config=auth_config
    )
    return converter.convert_to_toolkit()


def create_rapidapi_toolkit(
    schema_path_or_dict: Union[str, Dict[str, Any]],
    rapidapi_key: str,
    rapidapi_host: str,
    service_name: str = None
) -> APIToolkit:
    """
    Convenience function: create a RapidAPI toolkit
    
    Args:
        schema_path_or_dict: API specification file path or dictionary
        rapidapi_key: RapidAPI key
        rapidapi_host: RapidAPI host
        service_name: Service name (optional)
    
    Returns:
        APIToolkit: Created RapidAPI toolkit
    """
    converter = RapidAPIConverter(
        input_schema=schema_path_or_dict,
        description=service_name or "",
        rapidapi_key=rapidapi_key,
        rapidapi_host=rapidapi_host
    )
    return converter.convert_to_toolkit()
