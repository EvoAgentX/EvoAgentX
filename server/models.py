from pydantic import BaseModel
from typing import Dict, Any, Optional, List



# Updated workflow-based models (replacing project-based approach)
class ProjectSetupRequest(BaseModel):
    """Request model for workflow setup - Phase 1"""
    project_short_id: str  # Project identifier

class ProjectSetupResponse(BaseModel):
    """Response model for workflow setup - Phase 1"""
    workflow_graphs: List[Dict[str, Any]]  # List of workflow graphs
    message: str

class ProjectWorkflowGenerationRequest(BaseModel):
    """Request model for workflow generation - Phase 2"""
    # No body required for generation endpoint - workflow_id comes from URL path
    pass

class ProjectWorkflowGenerationResponse(BaseModel):
    """Response model for workflow generation - Phase 2"""
    workflow_graph: Dict[str, Any]
    status: str  # success, failed

class ProjectWorkflowExecutionRequest(BaseModel):
    """Request model for workflow execution - Phase 3"""
    inputs: Dict[str, Any]

class ProjectWorkflowExecutionResponse(BaseModel):
    """Response model for workflow execution - Phase 3"""
    execution_result: Optional[Dict[str, Any]] = None  # running result / status

class WorkflowGraphResponse(BaseModel):
    """Response model for workflow graph retrieval"""
    workflow_graph: Optional[Dict[str, Any]] = None

# Legacy aliases for backward compatibility
setup_input = ProjectSetupRequest
setup_output = ProjectSetupResponse
workflow_generation_input = ProjectWorkflowGenerationRequest
workflow_generation_output = ProjectWorkflowGenerationResponse
workflow_execution_input = ProjectWorkflowExecutionRequest
workflow_execution_output = ProjectWorkflowExecutionResponse

## Database Schema for Workflow Storage:
# workflow_id: str (Primary Key)
# project_short_id: str  # Project identifier
# task_info: Optional[Dict[str, Any]] = None       # saved after setup phase
# workflow_graph: Optional[Dict[str, Any]] = None  # saved after generation phase
# execution_result: Optional[Dict[str, Any]] = None # saved after execution phase
# created_at: datetime
# updated_at: datetime


### _____________________________________________
### User Query Router Models
### _____________________________________________

class UserQueryRequest(BaseModel):
    """Request model for user query analysis"""
    query: str  # The user's query string to analyze

class UserQueryResponse(BaseModel):
    """Response model for user query analysis"""
    result: Dict[str, Any]  # Contains all analysis results in a single dict


