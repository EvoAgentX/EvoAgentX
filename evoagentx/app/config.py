"""
Configuration settings for the EvoAgentX application.
"""
# import os
from pydantic import BaseModel, Field, validator
from pydantic_settings import BaseSettings
from typing import Optional, Dict, Any, List

class Settings(BaseSettings):
    # Application settings
    APP_NAME: str
    DEBUG: bool
    API_PREFIX: str
    HOST: str
    PORT: int
    
    # MongoDB settings
    MONGODB_URL: str
    MONGODB_DB_NAME: str
    
    # JWT Authentication
    SECRET_KEY: str
    ACCESS_TOKEN_EXPIRE_MINUTES: int
    ALGORITHM: str
    
    # Logging configuration
    LOG_LEVEL: str
    
    # Add CORS settings
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:8000"]
    CORS_ALLOW_CREDENTIALS: bool = True
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        env_delimiter = ","



# Global settings instance
settings = Settings()

# Agent and Workflow configuration
class AgentConfig(BaseModel):
    """Base configuration for an LLM agent."""
    model_name: str
    temperature: float = 0.7
    max_tokens: int = 2048
    api_key_env_var: Optional[str] = None
    system_prompt: Optional[str] = None
    extra_params: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('temperature')
    def validate_temperature(cls, v):
        if v < 0 or v > 1:
            raise ValueError('Temperature must be between 0 and 1')
        return v

class WorkflowStepConfig(BaseModel):
    """Configuration for a single step in a workflow."""
    step_id: str
    agent_id: str
    action: str
    input_mapping: Dict[str, str] = Field(default_factory=dict)
    output_mapping: Dict[str, str] = Field(default_factory=dict)
    timeout_seconds: int = 300
    retry_count: int = 3
    
class WorkflowConfig(BaseModel):
    """Configuration for a workflow composed of agent steps."""
    name: str
    description: Optional[str] = None
    steps: List[WorkflowStepConfig]
    parallel_execution: bool = False
    timeout_seconds: int = 3600  # Default to 1 hour total timeout

class ExecutionConfig(BaseModel):
    """Configuration for a workflow execution."""
    workflow_id: str
    input_params: Dict[str, Any] = Field(default_factory=dict)
    user_id: Optional[str] = None
    priority: int = 1  # Higher number means higher priority
    callback_url: Optional[str] = None