"""
Pydantic models for request/response validation in the EvoAgentX API.
"""
from datetime import datetime
from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field, validator
from bson import ObjectId
from evoagentx.app.db import AgentStatus, WorkflowStatus, ExecutionStatus

# Helper for ObjectId validation
class PyObjectId(ObjectId):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate
    
    @classmethod
    def validate(cls, v):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid ObjectId")
        return ObjectId(v)
    
    @classmethod
    def __modify_schema__(cls, field_schema):
        field_schema.update(type="string")

# Base Schema Models
class BaseSchema(BaseModel):
    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {
            ObjectId: str,
            datetime: lambda dt: dt.isoformat()
        }

# Agent Schemas
class AgentCreate(BaseSchema):
    name: str
    description: Optional[str] = None
    config: Dict[str, Any]
    runtime_params: Dict[str, Any] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)

class AgentUpdate(BaseSchema):
    name: Optional[str] = None
    description: Optional[str] = None
    config: Optional[Dict[str, Any]] = None
    runtime_params: Optional[Dict[str, Any]] = None
    status: Optional[AgentStatus] = None
    tags: Optional[List[str]] = None

class AgentResponse(BaseSchema):
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
    step_id: str
    agent_id: str
    action: str
    input_mapping: Dict[str, str] = Field(default_factory=dict)
    output_mapping: Dict[str, str] = Field(default_factory=dict)
    timeout_seconds: int = 300
    retry_count: int = 3
    depends_on: List[str] = Field(default_factory=list)

class WorkflowCreate(BaseSchema):
    name: str
    description: Optional[str] = None
    definition: Dict[str, Any]
    tags: List[str] = Field(default_factory=list)

class WorkflowUpdate(BaseSchema):
    name: Optional[str] = None
    description: Optional[str] = None
    definition: Optional[Dict[str, Any]] = None
    status: Optional[WorkflowStatus] = None
    tags: Optional[List[str]] = None

class WorkflowResponse(BaseSchema):
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
    workflow_id: str
    input_params: Dict[str, Any] = Field(default_factory=dict)
    callback_url: Optional[str] = None

class ExecutionResponse(BaseSchema):
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
    access_token: str
    token_type: str

class TokenPayload(BaseSchema):
    sub: Optional[str] = None
    exp: Optional[int] = None

class UserCreate(BaseSchema):
    email: str
    password: str
    full_name: Optional[str] = None

class UserLogin(BaseSchema):
    email: str
    password: str

class UserResponse(BaseSchema):
    id: str = Field(..., alias="_id")
    email: str
    full_name: Optional[str] = None
    is_active: bool
    is_admin: bool
    created_at: datetime

# Query parameters
class PaginationParams(BaseSchema):
    skip: int = 0
    limit: int = 100
    
class SearchParams(BaseSchema):
    query: Optional[str] = None
    tags: Optional[List[str]] = None
    status: Optional[str] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None