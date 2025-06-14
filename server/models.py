from pydantic import BaseModel
from typing import Dict, Any, Optional

class Config(BaseModel):
    """Configuration model for processing requests"""
    parameters: Dict[str, Any]
    timeout: Optional[int] = 30  # default timeout in seconds

class WorkflowGenerationConfig(BaseModel):
    """Configuration model for workflow generation requests"""
    goal: str
    llm_config: Dict[str, Any]
    mcp_config: Optional[Dict[str, Any]] = None  # MCP config is optional
    timeout: Optional[int] = 60  # longer timeout for workflow generation

class ProcessResponse(BaseModel):
    """Response model for processing requests"""
    task_id: str
    status: str
    result: Optional[Dict[str, Any]] = None

# New models for client session functionality
class ClientConnectResponse(BaseModel):
    """Response model for client connection"""
    client_id: str
    stream_url: str

class ClientTaskResponse(BaseModel):
    """Response model for client task start"""
    task_id: str
    status: str
    client_id: str 