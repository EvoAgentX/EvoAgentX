from fastapi import FastAPI, HTTPException, Depends, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
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
    ProjectWorkflowExecutionRequest, UserQueryRequest, UserQueryResponse,
    WorkflowGraphResponse, ParallelWorkflowGenerationResponse
)
from .core import (
    setup_project, generate_workflow, execute_workflow, execute_workflow_with_websocket,
    get_workflow, list_workflows
)
from .config.cors_config import get_cors_config
from evoagentx.core.logging import logger

load_dotenv('config/app.env', override = True)



app = FastAPI(title="Processing Server")

# Add CORS middleware using structured configuration
cors_config = get_cors_config()
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_config["allow_origins"],
    allow_credentials=cors_config["allow_credentials"],
    allow_methods=cors_config["allow_methods"],
    allow_headers=cors_config["allow_headers"],
)


### _____________________________________________
### Workflow Management 
### _____________________________________________

@app.get("/health")
async def health_check():
    """Basic health check endpoint"""
    return {"status": "healthy"}

# Workflow management endpoints
@app.get("/workflow/{workflow_id}/status")
async def get_workflow_status(workflow_id: str):
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
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error(f"Error getting workflow status for workflow_id {workflow_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting workflow status: {str(e)}")

@app.get("/workflow/{workflow_id}/get_graph", response_model=WorkflowGraphResponse)
async def get_workflow_graph_endpoint(workflow_id: str) -> WorkflowGraphResponse:
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
        logger.error(f"Error retrieving workflow graph for workflow_id {workflow_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving workflow graph: {str(e)}")

### _____________________________________________
### Workflow CRUD 
### _____________________________________________


# Updated workflow-based endpoints (using original project endpoints)
@app.post("/project/setup", response_model=ProjectSetupResponse)
async def setup_new_project(request: ProjectSetupRequest) -> ProjectSetupResponse:
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
        logger.error(f"Error setting up project for project_short_id {request.project_short_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error setting up workflow: {str(e)}")

@app.post("/project/setup-parallel", response_model=ProjectSetupResponse)
async def setup_new_project_parallel(request: ProjectSetupRequest) -> ProjectSetupResponse:
    """
    Phase 1: Setup workflow with extraction AND parallel generation with retry logic.
    This is an enhanced version with parallel execution and automatic retries.
    Takes the same input as the regular setup: {"project_short_id": "string"}
    """
    try:
        from .core.workflow_setup import setup_project_parallel
        
        workflow_graphs = await setup_project_parallel(request.project_short_id)
        return ProjectSetupResponse(
            workflow_graphs=workflow_graphs,
            message="Project setup completed successfully with parallel workflow generation and retry logic"
        )
    except Exception as e:
        logger.error(f"Error setting up project in parallel for project_short_id {request.project_short_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error setting up parallel workflow: {str(e)}")

@app.post("/workflow/{workflow_id}/generate", response_model=ProjectWorkflowGenerationResponse)
async def generate_workflow_with_workflow_id(workflow_id: str) -> ProjectWorkflowGenerationResponse:
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
        logger.error(f"Workflow generation failed for workflow_id {workflow_id}: {str(e)}")
        raise HTTPException(status_code=422, detail=f"Workflow generation failed: {str(e)}")
    except Exception as e:
        logger.error(f"Internal server error during workflow generation for workflow_id {workflow_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error during workflow generation: {str(e)}")

@app.get("/project/{project_short_id}/parallel-generation-status", response_model=ParallelWorkflowGenerationResponse)
async def get_parallel_workflow_generation_status(project_short_id: str) -> ParallelWorkflowGenerationResponse:
    """
    Get the status of parallel workflow generation for a project.
    Shows progress of all workflows being generated concurrently.
    """
    try:
        from .core.workflow_setup import get_project_workflow_status
        
        # Get the parallel generation status
        status = await get_project_workflow_status(project_short_id)
        
        return ParallelWorkflowGenerationResponse(**status)
        
    except ValueError as e:
        logger.error(f"Failed to get parallel generation status for project {project_short_id}: {str(e)}")
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Internal server error getting parallel generation status for project {project_short_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/workflow/{workflow_id}/execute")
async def execute_workflow_with_workflow_id(workflow_id: str, request: ProjectWorkflowExecutionRequest) -> Dict[str, Any]:
    """
    Phase 3: Execute workflow with provided inputs.
    This is the third phase of the workflow process.
    """
    try:
        result = await execute_workflow(workflow_id, request.inputs)
        return result
    except ValueError as e:
        logger.error(f"Workflow execution failed for workflow_id {workflow_id}: {str(e)}")
        raise HTTPException(status_code=422, detail=f"Workflow execution failed: {str(e)}")
    except Exception as e:
        logger.error(f"Internal server error during workflow execution for workflow_id {workflow_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error during workflow execution: {str(e)}")

### _____________________________________________
### WebSocket-based Workflow Execution
### _____________________________________________

@app.websocket("/workflow/{workflow_id}/execute_ws")
async def execute_workflow_websocket(
    websocket: WebSocket,
    workflow_id: str
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
            "content": "WebSocket connection established",
            "result": None
        }))
        
        # Wait for execution inputs
        try:
            data = await websocket.receive_text()
            request_data = json.loads(data)
            
            if "inputs" not in request_data:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "content": "Missing 'inputs' field in request",
                    "result": None
                }))
                return
            
            inputs = request_data["inputs"]
            
            # Send execution start confirmation
            await websocket.send_text(json.dumps({
                "type": "start",
                "content": "Workflow execution started",
                "result": None
            }))
            
            # Execute workflow with WebSocket progress updates
            async def send_websocket_message(message: str):
                """Helper function to send messages via WebSocket."""
                try:
                    await websocket.send_text(message)
                except Exception as e:
                    print(f"Error sending WebSocket message: {e}")
            
            # Execute the workflow
            try:
                result = await execute_workflow_with_websocket(
                    workflow_id=workflow_id,
                    inputs=inputs,
                    websocket_send_func=send_websocket_message
                )
                
                # Send final result
                await websocket.send_text(json.dumps({
                    "type": "complete",
                    "content": "Workflow execution completed successfully",
                    "result": result
                }))
            except Exception as e:
                logger.error(f"Error during WebSocket workflow execution for workflow_id {workflow_id}: {str(e)}")
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "content": f"Workflow execution error: {str(e)}",
                    "result": None
                }))
            
        except json.JSONDecodeError:
            await websocket.send_text(json.dumps({
                "type": "error",
                "content": "Invalid JSON format in request",
                "result": None
            }))
        except Exception as e:
            await websocket.send_text(json.dumps({
                "type": "error",
                "content": f"Execution error: {str(e)}",
                "result": None
            }))
    
    except WebSocketDisconnect:
        print(f"WebSocket disconnected for workflow {workflow_id}")
    except Exception as e:
        print(f"WebSocket error for workflow {workflow_id}: {e}")
        try:
            await websocket.send_text(json.dumps({
                "type": "error",
                "content": f"WebSocket error: {str(e)}",
                "result": None
            }))
        except:
            print("Connection might be closed")

@app.websocket("/project/{project_short_id}/parallel-generation-progress")
async def parallel_workflow_generation_progress(
    websocket: WebSocket,
    project_short_id: str
):
    """
    WebSocket endpoint for real-time parallel workflow generation progress.
    Provides live updates on the status of all workflows being generated concurrently.
    """
    try:
        # Accept WebSocket connection
        await websocket.accept()
        
        # Send connection confirmation
        await websocket.send_text(json.dumps({
            "type": "connection",
            "content": "Parallel workflow generation progress WebSocket connected",
            "project_short_id": project_short_id
        }))
        
        # Send initial status
        try:
            from .core.workflow_setup import get_project_workflow_status
            initial_status = await get_project_workflow_status(project_short_id)
            
            await websocket.send_text(json.dumps({
                "type": "status_update",
                "content": "Initial parallel generation status",
                "status": initial_status
            }))
            
            # Check if workflows need to be generated
            workflows = initial_status.get("workflows", [])
            uninitialized_workflows = [wf for wf in workflows if wf.get("status") == "uninitialized"]
            
            if uninitialized_workflows:
                # Send status update: uninitialized for each workflow
                for workflow in uninitialized_workflows:
                    await websocket.send_text(json.dumps({
                        "type": "workflow_status",
                        "status": "uninitialized",
                        "workflow_id": workflow.get("workflow_id")
                    }))
                
                # Start parallel generation with status messages
                from .core.workflow_setup import setup_project_parallel_with_status_messages
                
                async def send_websocket_message(message: str):
                    await websocket.send_text(message)
                
                # Run the parallel setup with WebSocket status messages
                try:
                    await setup_project_parallel_with_status_messages(project_short_id, send_websocket_message)
                    
                    # Send completion message
                    await websocket.send_text(json.dumps({
                        "type": "completion",
                        "content": "Parallel workflow generation completed",
                        "status": "completed"
                    }))
                    
                except Exception as e:
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "content": f"Error during parallel generation: {str(e)}"
                    }))
            else:
                # All workflows are already processed, just monitor for changes
                await websocket.send_text(json.dumps({
                    "type": "info",
                    "content": "All workflows already processed, monitoring for status changes"
                }))
                
                # Monitor progress and send updates
                last_status = initial_status
                while True:
                    # Wait a bit before checking for updates
                    await asyncio.sleep(2)
                    
                    try:
                        current_status = await get_project_workflow_status(project_short_id)
                        
                        # Only send update if status changed
                        if current_status != last_status:
                            await websocket.send_text(json.dumps({
                                "type": "status_update",
                                "content": "Parallel generation status updated",
                                "status": current_status
                            }))
                            last_status = current_status
                            
                            # If all workflows are completed or failed, send final update and close
                            if current_status["overall_status"] in ["completed", "failed"]:
                                await websocket.send_text(json.dumps({
                                    "type": "completion",
                                    "content": f"Parallel workflow generation {current_status['overall_status']}",
                                    "status": current_status
                                }))
                                break
                                
                    except Exception as e:
                        await websocket.send_text(json.dumps({
                            "type": "error",
                            "content": f"Error getting status update: {str(e)}"
                        }))
                        break
                        
        except Exception as e:
            await websocket.send_text(json.dumps({
                "type": "error",
                "content": f"Error in parallel generation progress: {str(e)}"
            }))
    
    except WebSocketDisconnect:
        print(f"Parallel generation progress WebSocket disconnected for project {project_short_id}")
    except Exception as e:
        print(f"Parallel generation progress WebSocket error for project {project_short_id}: {e}")
        try:
            await websocket.send_text(json.dumps({
                "type": "error",
                "content": f"WebSocket error: {str(e)}"
            }))
        except:
            pass  # Connection might be closed

### _____________________________________________
### User Query Router API
### _____________________________________________

@app.post("/project/{project_short_id}/user_query", response_model=UserQueryResponse)
async def analyze_user_query_endpoint(
    project_short_id: str,
    request: UserQueryRequest
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
        logger.error(f"User query analysis failed for project_short_id {project_short_id}: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Handle unexpected errors
        logger.error(f"Internal server error during user query analysis for project_short_id {project_short_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}") 














