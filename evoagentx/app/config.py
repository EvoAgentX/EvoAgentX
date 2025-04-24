"""
Configuration settings for the EvoAgentX application.
"""
# import os
from pydantic import BaseModel, Field, validator
from pydantic_settings import BaseSettings
from typing import Optional, Dict, Any, List

class Settings(BaseSettings):
    """Application configuration settings loaded from environment variables.
    
    This class defines all configuration parameters for the EvoAgentX application,
    loading values from environment variables or a .env file. It includes settings
    for the web server, database connection, authentication, and CORS.
    
    Attributes:
        APP_NAME: Name of the application
        DEBUG: Boolean flag for debug mode
        API_PREFIX: Prefix for all API routes
        HOST: Hostname for the server
        PORT: Port number for the server
        MONGODB_URL: Connection string for MongoDB
        MONGODB_DB_NAME: Name of the MongoDB database
        SECRET_KEY: Secret key for JWT token encryption
        ACCESS_TOKEN_EXPIRE_MINUTES: JWT token expiration time in minutes
        ALGORITHM: Algorithm used for JWT token encryption
        LOG_LEVEL: Logging level (e.g., DEBUG, INFO, WARNING)
        CORS_ORIGINS: List of allowed origins for CORS
        CORS_ALLOW_CREDENTIALS: Whether to allow credentials in CORS requests
    """
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
        """Configuration for environment variable loading.
        
        Specifies how environment variables are loaded and processed.
        """
        env_file = ".env"
        case_sensitive = True
        env_delimiter = ","



# Global settings instance
settings = Settings()

# Agent and Workflow configuration
class AgentConfig(BaseModel):
    """Base configuration for an LLM agent.
    
    Defines the configuration parameters for a language model agent,
    including model selection, generation parameters, and prompting.
    
    Attributes:
        model_name: Name of the language model to use
        temperature: Sampling temperature between 0 and 1 (higher = more creative)
        max_tokens: Maximum number of tokens to generate
        api_key_env_var: Environment variable name containing the API key
        system_prompt: System prompt to use for the agent
        extra_params: Additional model-specific parameters
    """
    model_name: str
    temperature: float = 0.7
    max_tokens: int = 2048
    api_key_env_var: Optional[str] = None
    system_prompt: Optional[str] = None
    extra_params: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('temperature')
    def validate_temperature(cls, v):
        """Validate that temperature is between 0 and 1.
        
        Args:
            v: The temperature value to validate
            
        Returns:
            The validated temperature value
            
        Raises:
            ValueError: If temperature is not between 0 and 1
        """
        if v < 0 or v > 1:
            raise ValueError('Temperature must be between 0 and 1')
        return v

class WorkflowStepConfig(BaseModel):
    """Configuration for a single step in a workflow.
    
    Defines how a single step in a workflow should be executed,
    including which agent to use, how to map inputs and outputs,
    and error handling configuration.
    
    Attributes:
        step_id: Unique identifier for this step
        agent_id: ID of the agent to use for this step
        action: The action to perform (e.g., "generate", "analyze")
        input_mapping: Mapping from workflow context to step inputs
        output_mapping: Mapping from step outputs to workflow context
        timeout_seconds: Maximum time in seconds to wait for step completion
        retry_count: Number of retry attempts if the step fails
    """
    step_id: str
    agent_id: str
    action: str
    input_mapping: Dict[str, str] = Field(default_factory=dict)
    output_mapping: Dict[str, str] = Field(default_factory=dict)
    timeout_seconds: int = 300
    retry_count: int = 3
    
class WorkflowConfig(BaseModel):
    """Configuration for a workflow composed of agent steps.
    
    Defines a complete workflow with multiple steps, execution options,
    and overall constraints.
    
    Attributes:
        name: Name of the workflow
        description: Optional description of the workflow's purpose
        steps: List of step configurations that make up the workflow
        parallel_execution: Whether steps can be executed in parallel
        timeout_seconds: Maximum time in seconds for the entire workflow
    """
    name: str
    description: Optional[str] = None
    steps: List[WorkflowStepConfig]
    parallel_execution: bool = False
    timeout_seconds: int = 3600  # Default to 1 hour total timeout

class ExecutionConfig(BaseModel):
    """Configuration for a workflow execution.
    
    Defines the parameters for a specific execution of a workflow,
    including inputs and execution options.
    
    Attributes:
        workflow_id: ID of the workflow to execute
        input_params: Input parameters for the workflow execution
        user_id: Optional ID of the user initiating the execution
        priority: Execution priority (higher = more priority)
        callback_url: Optional URL to call with results when execution completes
    """
    workflow_id: str
    input_params: Dict[str, Any] = Field(default_factory=dict)
    user_id: Optional[str] = None
    priority: int = 1  # Higher number means higher priority
    callback_url: Optional[str] = None