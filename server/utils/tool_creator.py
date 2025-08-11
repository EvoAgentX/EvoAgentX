"""
Tool creation utilities for EvoAgentX server.
Database tools are currently disabled.
"""

from typing import Dict, Any, List


def create_tools_with_database(database_information: Dict[str, Any] = None) -> List[Any]:
    """
    Create tools with database support (currently disabled)
    
    Args:
        database_information: Database configuration information (ignored)
        
    Returns:
        Empty list (database tools disabled)
    """
    print("🔧 Database tools disabled in tool_creator")
    return []


def create_custom_tools(tool_configs: List[Dict[str, Any]]) -> List[Any]:
    """
    Create custom tools based on configuration
    
    Args:
        tool_configs: List of tool configurations
        
    Returns:
        List of created tools
    """
    tools = []
    
    for config in tool_configs:
        try:
            tool_type = config.get("type")
            tool_params = config.get("params", {})
            
            if tool_type == "mongodb":
                # Create MongoDB tool
                mongo_tool = create_mongodb_tool(tool_params)
                if mongo_tool:
                    tools.append(mongo_tool)
            elif tool_type == "api":
                # Create API tool
                api_tool = create_api_tool(tool_params)
                if api_tool:
                    tools.append(api_tool)
            elif tool_type == "file":
                # Create file tool
                file_tool = create_file_tool(tool_params)
                if file_tool:
                    tools.append(file_tool)
                    
        except Exception as e:
            print(f"Warning: Failed to create tool {config.get('name', 'unknown')}: {e}")
    
    return tools


def create_mongodb_tool(params: Dict[str, Any]) -> Any:
    """
    Create a MongoDB tool (currently disabled)
    
    Args:
        params: Tool parameters (ignored)
        
    Returns:
        None (database tools disabled)
    """
    print("🔧 MongoDB tool creation disabled")
    return None


def create_api_tool(params: Dict[str, Any]) -> Any:
    """
    Create an API tool
    
    Args:
        params: Tool parameters
        
    Returns:
        API tool instance or None
    """
    try:
        # This would be implemented based on your API tool requirements
        # For now, return None
        return None
        
    except Exception as e:
        print(f"Failed to create API tool: {e}")
        return None


def create_file_tool(params: Dict[str, Any]) -> Any:
    """
    Create a file tool
    
    Args:
        params: Tool parameters
        
    Returns:
        File tool instance or None
    """
    try:
        # This would be implemented based on your file tool requirements
        # For now, return None
        return None
        
    except Exception as e:
        print(f"Failed to create file tool: {e}")
        return None


def validate_tool_config(tool_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate tool configuration
    
    Args:
        tool_config: Tool configuration to validate
        
    Returns:
        Validation result
    """
    errors = []
    warnings = []
    
    # Check required fields
    required_fields = ["type", "name"]
    for field in required_fields:
        if field not in tool_config:
            errors.append(f"Missing required field: {field}")
    
    # Check tool type
    valid_types = ["mongodb", "api", "file", "custom"]
    if "type" in tool_config and tool_config["type"] not in valid_types:
        errors.append(f"Invalid tool type: {tool_config['type']}")
    
    # Check MongoDB specific requirements (disabled)
    if tool_config.get("type") == "mongodb":
        errors.append("MongoDB tools are currently disabled")
    
    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings
    }

