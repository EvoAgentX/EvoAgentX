from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Depends, Header
from sse_starlette.sse import EventSourceResponse
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
    ProjectWorkflowExecutionRequest, ProjectWorkflowExecutionResponse
)
from .service import (
    setup_project, get_project, list_projects, 
    generate_workflow_for_project, execute_workflow_for_project, start_streaming_workflow_execution
)
from .db import initialize_database, close_database, seed_database
from .task_manager import (
    get_stream_task, get_stream_task_updates, is_stream_task_completed
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



# Updated workflow-based endpoints (using original project endpoints)
@app.post("/project/setup", response_model=ProjectSetupResponse)
async def setup_new_project(request: ProjectSetupRequest, token: str = Depends(verify_access_token)) -> ProjectSetupResponse:
    """
    Phase 1: Setup workflow with extraction AND generation.
    This is the first phase of the workflow process.
    """
    try:
        result = await setup_project(request.detailed_requirements)
        return ProjectSetupResponse(
            workflow_id=result["workflow_id"],
            user_id=result["user_id"],
            workflows=result["workflows"],
            database_information=result["database_information"],
            total_workflows=result["total_workflows"],
            message="Project setup completed successfully with workflow generation"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error setting up workflow: {str(e)}")

@app.post("/workflow/generate", response_model=ProjectWorkflowGenerationResponse)
async def generate_workflow_for_project_api(request: ProjectWorkflowGenerationRequest, token: str = Depends(verify_access_token)) -> ProjectWorkflowGenerationResponse:
    """
    Phase 2: Generate workflow graph based on task_info.
    This is the second phase of the workflow process.
    """
    try:
        result = await generate_workflow_for_project(request.workflow_id)
        return ProjectWorkflowGenerationResponse(
            workflow_graph=result["workflow_graph"],
            status=result["status"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating workflow: {str(e)}")

@app.post("/workflow/execute", response_model=ProjectWorkflowExecutionResponse)
async def execute_workflow_for_project_api(request: ProjectWorkflowExecutionRequest, token: str = Depends(verify_access_token)) -> ProjectWorkflowExecutionResponse:
    """
    Phase 3: Execute workflow with provided inputs.
    This is the third phase of the workflow process.
    """
    try:
        result = await execute_workflow_for_project(request.workflow_id, request.inputs)
        return ProjectWorkflowExecutionResponse(execution_result=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error executing workflow: {str(e)}")

# Workflow management endpoints
@app.get("/workflow/{workflow_id}/status")
async def get_workflow_status(workflow_id: str, token: str = Depends(verify_access_token)):
    """
    Get the current status and details of a workflow.
    Shows which phase the workflow is in and all stored data.
    """
    try:
        workflow = await get_project(workflow_id)
        if not workflow:
            raise HTTPException(status_code=404, detail="Workflow not found")
        
        return {
            "workflow_id": workflow_id,
            "status": workflow.get("status", "unknown"),
            "user_id": workflow.get("user_id"),
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



async def event_generator(task_id: str, timeout: int = 30):
    """Generate SSE events for a given task"""
    start_time = datetime.now()
    last_index = 0
    
    while True:
        # Check timeout
        if (datetime.now() - start_time).seconds > timeout:
            yield {
                "event": "error",
                "data": json.dumps({"error": "Stream timeout"})
            }
            break
            
        # Check if task exists
        if not get_stream_task(task_id):
            yield {
                "event": "error",
                "data": json.dumps({"error": "Task not found"})
            }
            break
            
        # Get new updates
        updates = get_stream_task_updates(task_id, last_index)
        for update in updates:
            yield {
                "event": "update" if not is_stream_task_completed(task_id) else "complete",
                "data": json.dumps(update)
            }
            last_index += 1
            
        # If task is completed, end stream
        if is_stream_task_completed(task_id):
            break
            
        # Wait before checking for new updates
        await asyncio.sleep(0.5)

async def client_event_generator(client_id: str, timeout: int = 3600):
    """Generate SSE events for a client session"""
    start_time = datetime.now()
    last_index = 0
    
    while True:
        # Check if client session is still active
        if not is_client_session_active(client_id):
            yield {
                "event": "session_closed",
                "data": json.dumps({"message": "Client session closed"})
            }
            break
            
        # Check timeout
        if (datetime.now() - start_time).seconds > timeout:
            yield {
                "event": "session_timeout",
                "data": json.dumps({"message": "Session timed out"})
            }
            break
        
        # Get new updates for this client
        updates = get_client_updates(client_id, last_index)
        for update in updates:
            event_type = update.get("event_type", "update")
            yield {
                "event": event_type,
                "data": json.dumps(update)
            }
            last_index += 1
        
        # Wait before checking for new updates
        await asyncio.sleep(0.5)

@app.get("/stream/{task_id}")
async def stream_results(task_id: str, token: str = Depends(verify_access_token)):
    """
    Stream results for a given task ID using Server-Sent Events.
    """
    if not get_stream_task(task_id):
        raise HTTPException(status_code=404, detail="Task not found")
        
    task_config = get_stream_task(task_id)["config"]
    return EventSourceResponse(
        event_generator(task_id, timeout=task_config["timeout"])
    )

@app.post("/workflow/execute_stream")
async def execute_workflow_for_project_stream_api(request: ProjectWorkflowExecutionRequest, token: str = Depends(verify_access_token)):
    """
    Phase 3: Execute workflow with provided inputs (Streaming version).
    Returns stream information to connect to the execution stream.
    """
    try:
        result = await start_streaming_workflow_execution(request.workflow_id, request.inputs)
        return {
            "task_id": result["task_id"],
            "status": result["status"],
            "stream_url": result["stream_url"],
            "workflow_id": result["workflow_id"],
            "message": f"Workflow execution started. Connect to {result['stream_url']} to receive real-time updates."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error starting workflow execution stream: {str(e)}")

@app.websocket("/ws/workflow/{task_id}")
async def websocket_endpoint(websocket: WebSocket, task_id: str):
    """
    WebSocket endpoint for streaming workflow execution updates.
    Note: WebSocket connections don't support headers in the same way, 
    so we'll need to handle authentication differently for WebSocket.
    """
    await websocket.accept()
    
    try:
        # For WebSocket, we'll accept the connection but could add token validation
        # in the message protocol if needed
        start_time = datetime.now()
        last_index = 0
        
        while True:
            # Check if task exists
            if not get_stream_task(task_id):
                await websocket.send_json({
                    "event": "error",
                    "data": {"error": "Task not found"}
                })
                break
                
            # Get new updates
            updates = get_stream_task_updates(task_id, last_index)
            for update in updates:
                event_type = "update" if not is_stream_task_completed(task_id) else "complete"
                await websocket.send_json({
                    "event": event_type,
                    "data": update
                })
                last_index += 1
                
            # If task is completed, end stream
            if is_stream_task_completed(task_id):
                break
                
            # Wait before checking for new updates
            await asyncio.sleep(0.5)
            
    except WebSocketDisconnect:
        print(f"WebSocket client disconnected: {task_id}")
    except Exception as e:
        try:
            await websocket.send_json({
                "event": "error",
                "data": {"error": str(e)}
            })
        except:
            pass
    finally:
        try:
            await websocket.close()
        except:
            pass

@app.post("/workflow/execute_ws")
async def execute_workflow_for_project_ws_api(request: ProjectWorkflowExecutionRequest, token: str = Depends(verify_access_token)):
    """
    Phase 3: Execute workflow with provided inputs (WebSocket version).
    Returns WebSocket connection information.
    """
    try:
        result = await start_streaming_workflow_execution(request.workflow_id, request.inputs)
        return {
            "task_id": result["task_id"],
            "status": result["status"],
            "ws_url": f"/ws/workflow/{result['task_id']}",  # WebSocket URL instead of SSE
            "workflow_id": result["workflow_id"],
            "message": f"Workflow execution started. Connect to WebSocket at /ws/workflow/{result['task_id']} for real-time updates."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error starting workflow execution: {str(e)}")

@app.get("/health")
async def health_check():
    """Basic health check endpoint"""
    return {"status": "healthy"}



@app.get("/projects")
async def get_all_projects(token: str = Depends(verify_access_token)):
    """
    List all projects in the system.
    """
    return await list_projects() 