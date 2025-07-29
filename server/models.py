from pydantic import BaseModel
from typing import Dict, Any, Optional, List



# Updated workflow-based models (replacing project-based approach)
class ProjectSetupRequest(BaseModel):
    """Request model for workflow setup - Phase 1"""
    detailed_requirements: str  # Only input needed - system generates workflow_id and user_id

class ProjectSetupResponse(BaseModel):
    """Response model for workflow setup - Phase 1"""
    workflow_id: str
    user_id: str
    workflows: List[Dict[str, Any]]
    database_information: Dict[str, Any]
    total_workflows: int
    message: str

class ProjectWorkflowGenerationRequest(BaseModel):
    """Request model for workflow generation - Phase 2"""
    workflow_id: str

class ProjectWorkflowGenerationResponse(BaseModel):
    """Response model for workflow generation - Phase 2"""
    workflow_graph: Dict[str, Any]
    status: str  # success, failed

class ProjectWorkflowExecutionRequest(BaseModel):
    """Request model for workflow execution - Phase 3"""
    workflow_id: str
    inputs: Dict[str, Any]

class ProjectWorkflowExecutionResponse(BaseModel):
    """Response model for workflow execution - Phase 3"""
    execution_result: Optional[Dict[str, Any]] = None  # running result / status

# Legacy aliases for backward compatibility
setup_input = ProjectSetupRequest
setup_output = ProjectSetupResponse
workflow_generation_input = ProjectWorkflowGenerationRequest
workflow_generation_output = ProjectWorkflowGenerationResponse
workflow_execution_input = ProjectWorkflowExecutionRequest
workflow_execution_output = ProjectWorkflowExecutionResponse

## Database Schema for Workflow Storage:
# workflow_id: str (Primary Key)
# user_id: str
# requirement_id: str
# task_info: Optional[Dict[str, Any]] = None       # saved after setup phase
# workflow_graph: Optional[Dict[str, Any]] = None  # saved after generation phase
# execution_result: Optional[Dict[str, Any]] = None # saved after execution phase
# created_at: datetime
# updated_at: datetime


