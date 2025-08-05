from fastapi import FastAPI, HTTPException, Depends, Header, WebSocket, WebSocketDisconnect
from typing import Dict, Any, Optional, List
import json
import asyncio
import uuid
import os
from dotenv import load_dotenv
from datetime import datetime
from contextlib import asynccontextmanager

from .models import (
    ProjectSetupRequest, ProjectSetupResponse, ProjectWorkflowGenerationRequest, ProjectWorkflowGenerationResponse,
    ProjectWorkflowExecutionRequest, ProjectWorkflowExecutionResponse, UserQueryRequest, UserQueryResponse,
    WorkflowGraphResponse
)
from .service import (
    setup_project, get_workflow, list_workflows, 
    generate_workflow, execute_workflow, execute_workflow_with_websocket
)

load_dotenv('server/app.env', override = True)

# Access control dependency
async def verify_access_token(eax_access_token: Optional[str] = Header(None, alias="eax-access-token")):
    """
    Verify the access token from the header.
    Returns the token if valid, raises HTTPException if invalid.
    """
    expected_token = os.getenv("EAX_ACCESS_TOKEN", "default_secret_token_change_me")
    
    if not eax_access_token:
        raise HTTPException(
            status_code=401, 
            detail="Access token required. Please provide 'eax-access-token' in the header."
        )
    
    if eax_access_token != expected_token:
        raise HTTPException(
            status_code=403, 
            detail="Invalid access token. Access denied."
        )
    
    return eax_access_token

app = FastAPI(title="Processing Server")


### _____________________________________________
### Workflow Management 
### _____________________________________________

@app.get("/health")
async def health_check():
    """Basic health check endpoint"""
    return {"status": "healthy"}

# Workflow management endpoints
@app.get("/workflow/{workflow_id}/status")
async def get_workflow_status(workflow_id: str, token: str = Depends(verify_access_token)):
    """
    Get the current status and details of a workflow.
    Shows which phase the workflow is in and all stored data.
    """
    try:
        workflow = await get_workflow(workflow_id)
        if not workflow:
            raise HTTPException(status_code=404, detail="Workflow not found")
        
        return {
            "workflow_id": workflow_id,
            "status": workflow.get("status", "unknown"),
            "created_at": workflow.get("created_at"),
            "updated_at": workflow.get("updated_at"),
            "phases": {
                "setup_complete": workflow.get("workflow_graph") is not None,
                "execution_complete": workflow.get("execution_result") is not None
            },
            "workflows": workflow.get("workflows", []),
            "database_information": workflow.get("database_information"),
            "workflow_graph": workflow.get("workflow_graph"),
            "execution_result": workflow.get("execution_result")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting workflow status: {str(e)}")

@app.get("/workflow/{workflow_id}/get_graph", response_model=WorkflowGraphResponse)
async def get_workflow_graph_endpoint(workflow_id: str, token: str = Depends(verify_access_token)) -> WorkflowGraphResponse:
    """
    Get the workflow graph for a specific workflow ID.
    
    Args:
        workflow_id: The unique identifier of the workflow
        token: Access token for authentication
        
    Returns:
        WorkflowGraphResponse containing the workflow graph
        
    Raises:
        HTTPException: 404 if workflow not found, 500 for server errors
    """
    try:
        workflow = await get_workflow(workflow_id)
        if not workflow:
            raise HTTPException(status_code=404, detail=f"Workflow with ID '{workflow_id}' not found")
        
        return WorkflowGraphResponse(workflow_graph=workflow.get("workflow_graph"))
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving workflow graph: {str(e)}")

### _____________________________________________
### Workflow CRUD 
### _____________________________________________


# Updated workflow-based endpoints (using original project endpoints)
@app.post("/project/setup", response_model=ProjectSetupResponse)
async def setup_new_project(request: ProjectSetupRequest, token: str = Depends(verify_access_token)) -> ProjectSetupResponse:
    """
    Phase 1: Setup workflow with extraction AND generation.
    This is the first phase of the workflow process.
    """
    try:
        workflow_graphs = await setup_project(request.project_short_id)
        return ProjectSetupResponse(
            workflow_graphs=workflow_graphs,
            message="Project setup completed successfully with workflow generation"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error setting up workflow: {str(e)}")

@app.post("/workflow/{workflow_id}/generate", response_model=ProjectWorkflowGenerationResponse)
async def generate_workflow_with_workflow_id(workflow_id: str, token: str = Depends(verify_access_token)) -> ProjectWorkflowGenerationResponse:
    """
    Phase 2: Generate workflow graph based on task_info.
    This is the second phase of the workflow process.
    """
    try:
        result = await generate_workflow(workflow_id)
        return ProjectWorkflowGenerationResponse(
            workflow_graph=result["workflow_graph"],
            status=result["status"]
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=f"Workflow generation failed: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error during workflow generation: {str(e)}")

@app.post("/workflow/{workflow_id}/execute", response_model=ProjectWorkflowExecutionResponse)
async def execute_workflow_with_workflow_id(workflow_id: str, request: ProjectWorkflowExecutionRequest, token: str = Depends(verify_access_token)) -> ProjectWorkflowExecutionResponse:
    """
    Phase 3: Execute workflow with provided inputs.
    This is the third phase of the workflow process.
    """
    try:
        result = await execute_workflow(workflow_id, request.inputs)
        return ProjectWorkflowExecutionResponse(execution_result=result)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=f"Workflow execution failed: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error during workflow execution: {str(e)}")

### _____________________________________________
### WebSocket-based Workflow Execution
### _____________________________________________

@app.websocket("/workflow/{workflow_id}/execute_ws")
async def execute_workflow_websocket(
    websocket: WebSocket,
    workflow_id: str,
    token: str = Depends(verify_access_token)
):
    """
    WebSocket endpoint for executing workflows with real-time progress updates.
    
    Expected message format:
    {
        "inputs": {
            "key1": "value1",
            "key2": "value2"
        }
    }
    
    Returns real-time progress updates and log messages via WebSocket.
    """
    try:
        # Accept WebSocket connection
        await websocket.accept()
        
        # Send connection confirmation
        await websocket.send_text(json.dumps({
            "type": "connection",
            "timestamp": datetime.now().isoformat(),
            "message": "WebSocket connection established",
            "workflow_id": workflow_id
        }))
        
        # Wait for execution inputs
        try:
            data = await websocket.receive_text()
            request_data = json.loads(data)
            
            if "inputs" not in request_data:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "timestamp": datetime.now().isoformat(),
                    "error": "Missing 'inputs' field in request",
                    "workflow_id": workflow_id
                }))
                return
            
            inputs = request_data["inputs"]
            
            # Send execution start confirmation
            await websocket.send_text(json.dumps({
                "type": "start",
                "timestamp": datetime.now().isoformat(),
                "message": "Workflow execution started",
                "workflow_id": workflow_id,
                "inputs": inputs
            }))
            
            # Execute workflow with WebSocket progress updates
            async def send_websocket_message(message: str):
                """Helper function to send messages via WebSocket."""
                try:
                    await websocket.send_text(message)
                except Exception as e:
                    print(f"Error sending WebSocket message: {e}")
            
            # Execute the workflow
            result = await execute_workflow_with_websocket(
                workflow_id=workflow_id,
                inputs=inputs,
                websocket_send_func=send_websocket_message
            )
            
            # Send final result
            await websocket.send_text(json.dumps({
                "type": "complete",
                "timestamp": datetime.now().isoformat(),
                "result": result,
                "workflow_id": workflow_id
            }))
            
        except json.JSONDecodeError:
            await websocket.send_text(json.dumps({
                "type": "error",
                "timestamp": datetime.now().isoformat(),
                "error": "Invalid JSON format in request",
                "workflow_id": workflow_id
            }))
        except Exception as e:
            await websocket.send_text(json.dumps({
                "type": "error",
                "timestamp": datetime.now().isoformat(),
                "error": f"Execution error: {str(e)}",
                "workflow_id": workflow_id
            }))
    
    except WebSocketDisconnect:
        print(f"WebSocket disconnected for workflow {workflow_id}")
    except Exception as e:
        print(f"WebSocket error for workflow {workflow_id}: {e}")
        try:
            await websocket.send_text(json.dumps({
                "type": "error",
                "timestamp": datetime.now().isoformat(),
                "error": f"WebSocket error: {str(e)}",
                "workflow_id": workflow_id
            }))
        except:
            pass  # Connection might be closed 

### _____________________________________________
### User Query Router API
### _____________________________________________

@app.post("/project/{project_short_id}/user_query", response_model=UserQueryResponse)
async def analyze_user_query_endpoint(
    project_short_id: str,
    request: UserQueryRequest,
    token: str = Depends(verify_access_token)
) -> UserQueryResponse:
    """
    Analyze user query using UserQueryRouter.
    
    This endpoint:
    1. Takes a project_short_id and user query
    2. Collects all workflows for the project
    3. Uses UserQueryRouter to analyze the query
    4. Returns structured analysis result
    
    Args:
        project_short_id: The project identifier (from URL path)
        request: UserQueryRequest containing the query string
        token: Access token for authentication
        
    Returns:
        UserQueryResponse containing the analysis result
        
    Raises:
        HTTPException: If project not found or analysis fails
    """
    try:
        from .service import analyze_user_query
        
        # Call the service function
        result = await analyze_user_query(project_short_id, request.query)
        
        # Return result as a single dict
        response = UserQueryResponse(result=result)
        
        return response
        
    except ValueError as e:
        # Handle service-level errors (project not found, etc.)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Handle unexpected errors
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}") 














