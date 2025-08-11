"""
Configuration management utilities for EvoAgentX server.
Handles loading and managing configuration settings.
"""

import os
import json
from typing import Dict, Any, Optional
from dotenv import load_dotenv


def load_env_config(config_path: str = None) -> Dict[str, Any]:
    """
    Load environment configuration from .env file
    
    Args:
        config_path: Path to .env file (defaults to app.env in server directory)
        
    Returns:
        Dictionary of configuration values
    """
    if config_path is None:
        # Default to app.env in the server directory
        server_dir = os.path.dirname(os.path.dirname(__file__))
        config_path = os.path.join(server_dir, "app.env")
    
    # Load environment variables
    load_dotenv(config_path)
    
    # Return configuration dictionary
    config = {
        "openai_api_key": os.getenv("OPENAI_API_KEY"),
        "openrouter_api_key": os.getenv("OPENROUTER_API_KEY"),
        "mongodb_uri": os.getenv("MONGODB_URI"),
        "mongodb_database": os.getenv("MONGODB_DATABASE"),
        "server_host": os.getenv("SERVER_HOST", "0.0.0.0"),
        "server_port": int(os.getenv("SERVER_PORT", "8000")),
        "debug": os.getenv("DEBUG", "false").lower() == "true",
        "log_level": os.getenv("LOG_LEVEL", "INFO"),
        "cors_origins": os.getenv("CORS_ORIGINS", "*").split(","),
        "max_workflow_execution_time": int(os.getenv("MAX_WORKFLOW_EXECUTION_TIME", "300")),
        "default_model": os.getenv("DEFAULT_MODEL", "gpt-4o-mini"),
        "default_temperature": float(os.getenv("DEFAULT_TEMPERATURE", "0.1"))
    }
    
    return config


def get_config(key: str = None, default: Any = None) -> Any:
    """
    Get configuration value by key
    
    Args:
        key: Configuration key to retrieve
        default: Default value if key not found
        
    Returns:
        Configuration value or default
    """
    config = load_env_config()
    
    if key is None:
        return config
    
    return config.get(key, default)


def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate configuration and return validation results
    
    Args:
        config: Configuration to validate
        
    Returns:
        Validation results with errors and warnings
    """
    errors = []
    warnings = []
    
    # Check required fields
    required_fields = ["mongodb_uri", "mongodb_database"]
    for field in required_fields:
        if not config.get(field):
            errors.append(f"Missing required configuration: {field}")
    
    # Check API keys
    if not config.get("openai_api_key") and not config.get("openrouter_api_key"):
        warnings.append("No API key configured - some features may not work")
    
    # Check server settings
    if config.get("server_port", 0) <= 0 or config.get("server_port", 0) > 65535:
        errors.append("Invalid server port")
    
    # Check execution time limits
    if config.get("max_workflow_execution_time", 0) <= 0:
        warnings.append("Invalid max workflow execution time")
    
    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings
    }


def load_mcp_config(config_path: str = None) -> Dict[str, Any]:
    """
    Load MCP configuration from file
    
    Args:
        config_path: Path to MCP config file
        
    Returns:
        MCP configuration dictionary
    """
    if config_path is None:
        # Default to mcp.config in the server directory
        server_dir = os.path.dirname(os.path.dirname(__file__))
        config_path = os.path.join(server_dir, "mcp.config")
    
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            return config
        else:
            print(f"Warning: MCP config file not found at {config_path}")
            return {}
    except Exception as e:
        print(f"Error loading MCP config: {e}")
        return {}


def save_config(config: Dict[str, Any], config_path: str) -> bool:
    """
    Save configuration to file
    
    Args:
        config: Configuration to save
        config_path: Path to save configuration
        
    Returns:
        True if saved successfully
    """
    try:
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving config: {e}")
        return False


def get_database_config() -> Dict[str, Any]:
    """
    Get database configuration
    
    Returns:
        Database configuration dictionary
    """
    config = load_env_config()
    
    return {
        "uri": config.get("mongodb_uri"),
        "database": config.get("mongodb_database"),
        "collections": {
            "workflows": "workflows",
            "requirements": "requirements",
            "executions": "executions"
        }
    }


def get_llm_config() -> Dict[str, Any]:
    """
    Get LLM configuration
    
    Returns:
        LLM configuration dictionary
    """
    config = load_env_config()
    
    return {
        "openai_key": config.get("openai_api_key"),
        "openrouter_key": config.get("openrouter_api_key"),
        "model": config.get("default_model"),
        "temperature": config.get("default_temperature")
    }

