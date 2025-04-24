"""
Pydantic models for request/response validation in the EvoAgentX API.
"""
from datetime import datetime
from typing import Optional, List, Dict, Any # , Union
from pydantic import BaseModel, Field # , validator
from bson import ObjectId
from evoagentx.app.db import AgentStatus, WorkflowStatus, ExecutionStatus

# Helper for ObjectId validation
class PyObjectId(ObjectId):
    """MongoDB ObjectId field for Pydantic models.
    
    Extends MongoDB's ObjectId class to make it compatible with Pydantic models,
    providing validation and schema modification for proper API documentation.
    """
    @classmethod
    def __get_validators__(cls):
        """Return a list of validator methods.
        
        Returns:
            Generator yielding validate method
        """
        yield cls.validate
    
    @classmethod
    def validate(cls, v):
        """Validate that a value is a valid ObjectId.
        
        Args:
            v: Value to validate
            
        Returns:
            ObjectId: The validated ObjectId
            
        Raises:
            ValueError: If value is not a valid ObjectId
        """
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid ObjectId")
        return ObjectId(v)
    
    @classmethod
    def __modify_schema__(cls, field_schema):
        """Modify JSON schema to represent ObjectId as a string.
        
        Args:
            field_schema: The schema to modify
        """
        field_schema.update(type="string")

# Base Schema Models
class BaseSchema(BaseModel):
    """Base model for all API schemas.
    
    Provides common configuration for Pydantic models used in the API,
    including proper handling of MongoDB ObjectIds and datetime serialization.
    """
    class Config:
        """Configuration for Pydantic models.
        
        Defines common settings for all schema models.
        """
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {
            ObjectId: str,
            datetime: lambda dt: dt.isoformat()
        }

# Agent Schemas
class AgentCreate(BaseSchema):
    """Schema for creating a new agent.
    
    Defines the required and optional fields when creating a new agent.
    
    Attributes:
        name: Unique name for the agent
        description: Optional description of the agent's purpose
        config: Configuration dictionary for the agent
        runtime_params: Optional runtime parameters for the agent
        tags: Optional list of tags for categorization
    """
    name: str
    description: Optional[str] = None
    config: Dict[str, Any]
    runtime_params: Dict[str, Any] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)

class AgentUpdate(BaseSchema):
    """Schema for updating an existing agent.
    
    Defines the fields that can be updated for an existing agent.
    All fields are optional since updates may be partial.
    
    Attributes:
        name: Optional new name for the agent
        description: Optional new description
        config: Optional new configuration dictionary
        runtime_params: Optional new runtime parameters
        status: Optional new status
        tags: Optional new list of tags
    """
    name: Optional[str] = None
    description: Optional[str] = None
    config: Optional[Dict[str, Any]] = None
    runtime_params: Optional[Dict[str, Any]] = None
    status: Optional[AgentStatus] = None
    tags: Optional[List[str]] = None

class AgentResponse(BaseSchema):
    """Schema for agent response data.
    
    Defines the structure of agent data returned in API responses.
    
    Attributes:
        id: Unique identifier for the agent
        name: Name of the agent
        description: Optional description of the agent
        config: Configuration dictionary for the agent
        status: Current status of the agent
        runtime_params: Runtime parameters for the agent
        created_at: Timestamp when the agent was created
        updated_at: Timestamp when the agent was last updated
        created_by: ID of the user who created the agent
        tags: List of tags for categorization
    """
    id: str = Field(..., alias="_id")
    name: str
    description: Optional[str] = None
    config: Dict[str, Any]
    status: AgentStatus
    runtime_params: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    created_by: Optional[str] = None
    tags: List[str]

# Workflow Schemas
class WorkflowStepDefinition(BaseSchema):
    """Schema for a single step in a workflow.
    
    Defines the structure of a workflow step, including agent,
    action, and data mapping configuration.
    
    Attributes:
        step_id: Unique identifier for the step
        agent_id: ID of the agent to execute this step
        action: Action to perform in this step
        input_mapping: Mapping of workflow data to step inputs
        output_mapping: Mapping of step outputs to workflow data
        timeout_seconds: Maximum time in seconds for step execution
        retry_count: Number of times to retry on failure
        depends_on: List of step IDs that must complete before this step
    """
    step_id: str
    agent_id: str
    action: str
    input_mapping: Dict[str, str] = Field(default_factory=dict)
    output_mapping: Dict[str, str] = Field(default_factory=dict)
    timeout_seconds: int = 300
    retry_count: int = 3
    depends_on: List[str] = Field(default_factory=list)

class WorkflowCreate(BaseSchema):
    """Schema for creating a new workflow.
    
    Defines the required and optional fields when creating a new workflow.
    
    Attributes:
        name: Name of the workflow
        description: Optional description of the workflow
        definition: Complete workflow definition including steps
        tags: Optional list of tags for categorization
    """
    name: str
    description: Optional[str] = None
    definition: Dict[str, Any]
    tags: List[str] = Field(default_factory=list)

class WorkflowUpdate(BaseSchema):
    """Schema for updating an existing workflow.
    
    Defines the fields that can be updated for an existing workflow.
    All fields are optional since updates may be partial.
    
    Attributes:
        name: Optional new name for the workflow
        description: Optional new description
        definition: Optional new workflow definition
        status: Optional new status
        tags: Optional new list of tags
    """
    name: Optional[str] = None
    description: Optional[str] = None
    definition: Optional[Dict[str, Any]] = None
    status: Optional[WorkflowStatus] = None
    tags: Optional[List[str]] = None

class WorkflowResponse(BaseSchema):
    """Schema for workflow response data.
    
    Defines the structure of workflow data returned in API responses.
    
    Attributes:
        id: Unique identifier for the workflow
        name: Name of the workflow
        description: Optional description of the workflow
        definition: Complete workflow definition including steps
        agent_ids: List of agent IDs used in this workflow
        status: Current status of the workflow
        created_at: Timestamp when the workflow was created
        updated_at: Timestamp when the workflow was last updated
        created_by: ID of the user who created the workflow
        tags: List of tags for categorization
        version: Version number of the workflow
    """
    id: str = Field(..., alias="_id")
    name: str
    description: Optional[str] = None
    definition: Dict[str, Any]
    agent_ids: List[str]
    status: WorkflowStatus
    created_at: datetime
    updated_at: datetime
    created_by: Optional[str] = None
    tags: List[str]
    version: int

# Execution Schemas
class ExecutionCreate(BaseSchema):
    """Schema for creating a new workflow execution.
    
    Defines the required and optional fields when initiating a workflow execution.
    
    Attributes:
        workflow_id: ID of the workflow to execute
        input_params: Input parameters for the workflow execution
        callback_url: Optional URL to call when execution completes
    """
    workflow_id: str
    input_params: Dict[str, Any] = Field(default_factory=dict)
    callback_url: Optional[str] = None

class ExecutionResponse(BaseSchema):
    """Schema for workflow execution response data.
    
    Defines the structure of execution data returned in API responses.
    
    Attributes:
        id: Unique identifier for the execution
        workflow_id: ID of the executed workflow
        status: Current status of the execution
        start_time: Timestamp when execution started
        end_time: Timestamp when execution completed
        input_params: Input parameters for the execution
        results: Results produced by the execution
        created_by: ID of the user who initiated the execution
        step_results: Results from individual workflow steps
        current_step: ID of the step currently being executed
        error_message: Error message if execution failed
        created_at: Timestamp when the execution was created
    """
    id: str = Field(..., alias="_id")
    workflow_id: str
    status: ExecutionStatus
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    input_params: Dict[str, Any]
    results: Dict[str, Any]
    created_by: Optional[str] = None
    step_results: Dict[str, Dict[str, Any]]
    current_step: Optional[str] = None
    error_message: Optional[str] = None
    created_at: datetime

class ExecutionLogResponse(BaseSchema):
    """Schema for execution log response data.
    
    Defines the structure of execution log entries returned in API responses.
    
    Attributes:
        id: Unique identifier for the log entry
        workflow_id: ID of the workflow
        execution_id: ID of the execution
        step_id: Optional ID of the workflow step
        agent_id: Optional ID of the agent
        timestamp: Timestamp when the log entry was created
        level: Log level (INFO, WARNING, ERROR, etc.)
        message: Log message text
        details: Additional structured details about the event
    """
    id: str = Field(..., alias="_id")
    workflow_id: str
    execution_id: str
    step_id: Optional[str] = None
    agent_id: Optional[str] = None
    timestamp: datetime
    level: str
    message: str
    details: Dict[str, Any]

# User auth schemas
class Token(BaseSchema):
    """Schema for authentication token response.
    
    Defines the structure of the authentication token returned after successful login.
    
    Attributes:
        access_token: JWT access token for authentication
        token_type: Type of token (typically "bearer")
    """
    access_token: str
    token_type: str

class TokenPayload(BaseSchema):
    """Schema for JWT token payload.
    
    Defines the structure of the data encoded in JWT tokens.
    
    Attributes:
        sub: Subject of the token (typically user ID or email)
        exp: Expiration timestamp
    """
    sub: Optional[str] = None
    exp: Optional[int] = None

class UserCreate(BaseSchema):
    """Schema for creating a new user.
    
    Defines the required and optional fields when creating a new user.
    
    Attributes:
        email: Email address of the user (used as username)
        password: Password for the user account
        full_name: Optional full name of the user
    """
    email: str
    password: str
    full_name: Optional[str] = None

class UserLogin(BaseSchema):
    """Schema for user login.
    
    Defines the fields required for user authentication.
    
    Attributes:
        email: Email address of the user
        password: Password for the user account
    """
    email: str
    password: str

class UserResponse(BaseSchema):
    """Schema for user response data.
    
    Defines the structure of user data returned in API responses.
    
    Attributes:
        id: Unique identifier for the user
        email: Email address of the user
        full_name: Optional full name of the user
        is_active: Whether the user account is active
        is_admin: Whether the user has admin privileges
        created_at: Timestamp when the user was created
    """
    id: str = Field(..., alias="_id")
    email: str
    full_name: Optional[str] = None
    is_active: bool
    is_admin: bool
    created_at: datetime

# Query parameters
class PaginationParams(BaseSchema):
    """Schema for pagination parameters.
    
    Defines common pagination parameters used across list endpoints.
    
    Attributes:
        skip: Number of items to skip (for pagination)
        limit: Maximum number of items to return
    """
    skip: int = 0
    limit: int = 100
    
class SearchParams(BaseSchema):
    """Schema for search and filtering parameters.
    
    Defines common search and filter parameters used across list endpoints.
    
    Attributes:
        query: Optional search query string
        tags: Optional list of tags to filter by
        status: Optional status to filter by
        start_date: Optional minimum creation date
        end_date: Optional maximum creation date
    """
    query: Optional[str] = None
    tags: Optional[List[str]] = None
    status: Optional[str] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None