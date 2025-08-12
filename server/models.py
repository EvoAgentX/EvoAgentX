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


class WorkflowGraphResponse(BaseModel):
    """Response model for workflow graph retrieval"""
    workflow_graph: Optional[Dict[str, Any]] = None

# Legacy aliases for backward compatibility
setup_input = ProjectSetupRequest
setup_output = ProjectSetupResponse
workflow_generation_input = ProjectWorkflowGenerationRequest
workflow_generation_output = ProjectWorkflowGenerationResponse
workflow_execution_input = ProjectWorkflowExecutionRequest

### _____________________________________________
### Parallel Workflow Generation Models
### _____________________________________________

class WorkflowGenerationStatus(BaseModel):
    """Status model for individual workflow generation"""
    workflow_id: str
    workflow_name: str
    status: str  # pending, generating, completed, failed
    progress: Optional[float] = None  # 0.0 to 1.0
    error_message: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None

class ParallelWorkflowGenerationResponse(BaseModel):
    """Response model for parallel workflow generation status"""
    project_short_id: str
    total_workflows: int
    completed_workflows: int
    failed_workflows: int
    workflows: List[WorkflowGenerationStatus]
    overall_status: str  # running, completed, failed
    estimated_completion_time: Optional[str] = None

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


