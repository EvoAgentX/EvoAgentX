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

from .models.models import (
    ProjectSetupRequest, ProjectSetupResponse, ProjectWorkflowGenerationRequest, ProjectWorkflowGenerationResponse,
    ProjectWorkflowExecutionRequest, ProjectWorkflowExecutionResponse, UserQueryRequest, UserQueryResponse,
    WorkflowGraphResponse
)
from .core import (
    setup_project, execute_workflow, execute_workflow_with_websocket,
    get_workflow, list_workflows
)
from .config.cors_config import get_cors_config
from evoagentx.core.logging import logger

# Import socket management service
from .socket_management import socket_service
from .socket_management.message_store import message_store

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
        # Create a websocket send function if there's an active connection for this project
        websocket_send_func = None
        if socket_service.is_project_connected(request.project_short_id):
            async def send_func(message):
                return await socket_service.send_to_project(request.project_short_id, message)
            websocket_send_func = send_func
        
        workflow_graphs = await setup_project(request.project_short_id, websocket_send_func)
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

@app.post("/workflow/{workflow_id}/execute", response_model=ProjectWorkflowExecutionResponse)
async def execute_workflow_with_workflow_id(workflow_id: str, request: ProjectWorkflowExecutionRequest) -> ProjectWorkflowExecutionResponse:
    """
    Phase 3: Execute workflow with provided inputs.
    This is the third phase of the workflow process.
    """
    try:
        result = await execute_workflow(workflow_id, request.inputs)
        return ProjectWorkflowExecutionResponse(parsed_json=result)
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

@app.websocket("/project/{project_short_id}/parallel-setup")
async def parallel_workflow_setup(
    websocket: WebSocket,
    project_short_id: str
):
    """
    WebSocket endpoint for parallel workflow setup with persistent connection.
    Generates all workflows for a project in parallel and keeps socket open for execution.
    """
    try:
        # Connect via socket service for persistent management
        await socket_service.connect_project(project_short_id, websocket)
        logger.info(f"Parallel generation progress WebSocket connected for project {project_short_id} via socket service")
        
        # Send setup start message using README format
        from .socket_management.protocols import create_message, MessageType
        start_message = create_message(
            MessageType.SETUP_LOG,
            status=None,
            workflow_id=None,
            content="Setup start...",
            result=None
        )
        await socket_service.send_to_project(project_short_id, start_message)
        
        # Start workflow extraction and generation directly
        try:
            logger.info(f"Starting parallel workflow generation for project {project_short_id}")
            from .core.workflow_setup import setup_project_parallel_with_status_messages
            
            async def send_websocket_message(message: str):
                try:
                    # Parse message to get proper format
                    message_data = json.loads(message)
                    await socket_service.send_to_project(project_short_id, message_data)
                except Exception as e:
                    logger.error(f"Failed to send WebSocket message via socket service: {str(e)}")
                    # If we can't send messages, the connection might be broken
                    raise e
            
            # Run the parallel setup with WebSocket status messages and timeout protection
            try:
                workflow_results = await asyncio.wait_for(
                    setup_project_parallel_with_status_messages(project_short_id, send_websocket_message),
                    timeout=300.0  # 5 minute timeout
                )
            except asyncio.TimeoutError:
                logger.error(f"Parallel generation timed out for project {project_short_id} after 5 minutes")
                timeout_message = create_message(
                    MessageType.SETUP_LOG,
                    status=None,
                    workflow_id=None,
                    content="Parallel generation timed out after 5 minutes",
                    result=None
                )
                await socket_service.send_to_project(project_short_id, timeout_message)
                # Keep connection open even after timeout for potential retry
                return
            
            # Extract just the workflow_graphs from the results
            logger.info(f"Processing {len(workflow_results)} workflow results for project {project_short_id}")
            workflow_graphs = []
            for i, workflow in enumerate(workflow_results):
                if "workflow_graph" in workflow:
                    workflow_graphs.append(workflow["workflow_graph"])
                    logger.debug(f"Added workflow {i+1} graph to results")
                else:
                    logger.warning(f"Workflow {i+1} missing workflow_graph field: {workflow.keys()}")
            
            logger.info(f"Extracted {len(workflow_graphs)} workflow graphs from {len(workflow_results)} results")
            
            # Send setup_complete message with workflow graph dicts using new format
            try:
                setup_complete_msg = create_message(
                    MessageType.SETUP_COMPLETE,
                    status=None,
                    workflow_id=None,
                    content="Workflow setup completed successfully",
                    result=workflow_graphs
                )
                await socket_service.send_to_project(project_short_id, setup_complete_msg)
                
                logger.info(f"Successfully sent setup_complete message for project {project_short_id} with {len(workflow_graphs)} workflow graphs")
                
                # Keep the WebSocket connection open for future execution requests
                logger.info(f"WebSocket connection remains open for project {project_short_id} for future execution requests")
                
            except Exception as e:
                logger.error(f"Failed to send setup_complete message for project {project_short_id}: {str(e)}")
                # Don't close the connection on message send failure, just log the error
            
        except Exception as e:
            logger.error(f"Error during parallel generation for project {project_short_id}: {str(e)}")
            logger.error(f"Exception type: {type(e).__name__}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            try:
                error_message = create_message(
                    MessageType.SETUP_LOG,
                    status=None,
                    workflow_id=None,
                    content=f"Error during parallel generation: {str(e)}",
                    result=None
                )
                await socket_service.send_to_project(project_short_id, error_message)
                
                # Keep connection open even after error, just log it
                logger.error(f"Error occurred but keeping connection open for recovery: {str(e)}")
            except Exception as send_error:
                logger.error(f"Failed to send error message: {str(send_error)}")
                # Keep connection open for potential recovery
    
    except WebSocketDisconnect:
        logger.info(f"Parallel generation progress WebSocket disconnected for project {project_short_id}")
    except Exception as e:
        logger.error(f"Parallel generation progress WebSocket error for project {project_short_id}: {e}")
        logger.error(f"Exception type: {type(e).__name__}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        try:
            await websocket.send_text(json.dumps({
                "type": "error",
                "data": {
                    "status": "error",
                    "workflow_id": None,
                    "content": f"WebSocket error: {str(e)}",
                    "result": None
                }
            }))
        except Exception as send_error:
            logger.error(f"Failed to send error message: {str(send_error)}")
            # Connection might be closed, can't send error message

### _____________________________________________
### Socket Management API
### _____________________________________________

@app.post("/workflow/{workflow_id}/execute")
async def execute_workflow_with_socket_integration(
    workflow_id: str,
    request: ProjectWorkflowExecutionRequest
):
    """
    Execute workflow with socket integration for real-time updates.
    If socket exists for the project, execution updates are sent via WebSocket.
    Otherwise, falls back to regular execution without real-time updates.
    """
    try:
        # Get workflow to find project_short_id
        from .core.workflow_execution import get_workflow
        workflow = await get_workflow(workflow_id)
        
        if not workflow:
            raise HTTPException(status_code=404, detail=f"Workflow {workflow_id} not found")
        
        project_short_id = workflow.get("project_short_id")
        
        if not project_short_id:
            raise HTTPException(status_code=400, detail="Workflow missing project_short_id")
        
        # Check if socket exists for this project
        if socket_service.is_project_connected(project_short_id):
            logger.info(f"Executing workflow {workflow_id} with socket integration for project {project_short_id}")
            
            # Create WebSocket send function
            async def websocket_send_func(message_str: str):
                message_data = json.loads(message_str) if isinstance(message_str, str) else message_str
                await socket_service.send_to_project(project_short_id, message_data)
            
            # Execute with socket integration
            from .core.workflow_execution import execute_workflow_with_websocket
            result = await execute_workflow_with_websocket(workflow_id, request.inputs, websocket_send_func)
            
            return ProjectWorkflowExecutionResponse(
                workflow_id=workflow_id,
                status="completed",
                result=result,
                message="Workflow executed successfully with real-time updates"
            )
        else:
            logger.info(f"Executing workflow {workflow_id} without socket integration (no active connection)")
            
            # Execute without socket integration
            from .core.workflow_execution import execute_workflow
            result = await execute_workflow(workflow_id, request.inputs)
            
            return ProjectWorkflowExecutionResponse(
                workflow_id=workflow_id,
                status="completed", 
                result=result,
                message="Workflow executed successfully"
            )
            
    except Exception as e:
        logger.error(f"Error executing workflow {workflow_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/socket/{project_short_id}/timeout")
async def update_socket_timeout(
    project_short_id: str,
    timeout: int
):
    """
    Update socket timeout for a specific project.
    
    Args:
        project_short_id: The project identifier
        timeout: Timeout in seconds (use -1 for no expiry)
    
    Returns:
        Success status and current timeout settings
    """
    try:
        if timeout < -1:
            raise HTTPException(status_code=400, detail="Timeout must be -1 (no expiry) or positive integer")
        
        success = socket_service.update_socket_timeout(project_short_id, timeout)
        
        if not success:
            raise HTTPException(status_code=404, detail=f"No active socket found for project {project_short_id}")
        
        current_timeout = socket_service.get_socket_timeout(project_short_id)
        
        return {
            "success": True,
            "project_short_id": project_short_id,
            "timeout": current_timeout,
            "message": f"Socket timeout updated to {current_timeout}s {'(no expiry)' if current_timeout == -1 else ''}"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating socket timeout for {project_short_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/socket/{project_short_id}/status")
async def get_socket_status(project_short_id: str):
    """
    Get socket status and configuration for a specific project.
    
    Args:
        project_short_id: The project identifier
        
    Returns:
        Socket status, timeout, and connection details
    """
    try:
        is_connected = socket_service.is_project_connected(project_short_id)
        
        if not is_connected:
            return {
                "connected": False,
                "project_short_id": project_short_id,
                "message": "No active socket connection"
            }
        
        timeout = socket_service.get_socket_timeout(project_short_id)
        connection_info = socket_service.active_connections.get(project_short_id, {})
        
        return {
            "connected": True,
            "project_short_id": project_short_id,
            "timeout": timeout,
            "timeout_description": "No expiry" if timeout == -1 else f"{timeout} seconds",
            "connected_at": connection_info.get("connected_at").isoformat() if connection_info.get("connected_at") else None,
            "last_ping": connection_info.get("last_ping").isoformat() if connection_info.get("last_ping") else None
        }
        
    except Exception as e:
        logger.error(f"Error getting socket status for {project_short_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/socket/stats")
async def get_socket_service_stats():
    """
    Get overall socket service statistics.
    
    Returns:
        Statistics about all active socket connections
    """
    try:
        stats = socket_service.get_connection_stats()
        return stats
    except Exception as e:
        logger.error(f"Error getting socket service stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/socket/{project_short_id}")
async def project_socket_endpoint(websocket: WebSocket, project_short_id: str):
    """
    WebSocket endpoint for project-specific socket connections.
    Provides real-time communication for workflow monitoring and execution.
    
    This endpoint supports:
    - Project setup with real-time progress
    - Workflow execution with live monitoring
    - Query analysis
    - System health checks
    
    Message format matches existing EvoAgentX WebSocket format:
    {
        "type": "message_type",
        "data": {
            "status": "status_value",
            "workflow_id": "workflow_id_or_null",
            "content": "human_readable_message", 
            "result": "actual_data_or_null"
        }
    }
    """
    try:
        # Connect the project socket
        await socket_service.connect_project(project_short_id, websocket)
        
        # Handle incoming messages
        while True:
            try:
                message = await websocket.receive_text()
                message_data = json.loads(message)
                message_data["project_short_id"] = project_short_id
                
                # Add message_id if not present
                if "message_id" not in message_data:
                    message_data["message_id"] = uuid.uuid4().hex
                
                # Handle the message
                await socket_service.handle_message(project_short_id, message_data)
                
            except json.JSONDecodeError:
                await socket_service.send_to_project(project_short_id, {
                    "type": "error",
                    "data": {
                        "status": "error",
                        "workflow_id": None,
                        "content": "Invalid JSON format",
                        "result": None
                    }
                })
            except WebSocketDisconnect:
                logger.info(f"WebSocket disconnected for project: {project_short_id}")
                break
            except Exception as e:
                logger.error(f"Error processing socket message: {e}")
                await socket_service.send_to_project(project_short_id, {
                    "type": "error",
                    "data": {
                        "status": "error",
                        "workflow_id": None,
                        "content": f"Message processing error: {str(e)}",
                        "result": None
                    }
                })
                break
    
    except WebSocketDisconnect:
        logger.info(f"Socket disconnected for project: {project_short_id}")
    except Exception as e:
        logger.error(f"Socket error for project {project_short_id}: {e}")

@app.get("/socket/status")
async def socket_status():
    """Get status of all active socket connections."""
    return socket_service.get_connection_stats()

@app.get("/socket/{project_short_id}/messages")
async def get_project_messages(
    project_short_id: str,
    limit: int = 100,
    message_type: str = None,
    since_hours: int = None
):
    """
    Get stored messages for a specific project.
    
    Args:
        project_short_id: Project identifier
        limit: Maximum number of messages to return (default: 100)
        message_type: Filter by message type (optional)
        since_hours: Only return messages from last N hours (optional)
    """
    from datetime import datetime, timedelta
    
    since = None
    if since_hours:
        since = datetime.now() - timedelta(hours=since_hours)
    
    messages = message_store.get_project_messages(
        project_short_id=project_short_id,
        limit=limit,
        message_type=message_type,
        since=since
    )
    
    return {
        "project_short_id": project_short_id,
        "messages": messages,
        "total_returned": len(messages),
        "filters": {
            "limit": limit,
            "message_type": message_type,
            "since_hours": since_hours
        }
    }

@app.get("/socket/{project_short_id}/stats")
async def get_project_message_stats(project_short_id: str):
    """Get message statistics for a specific project."""
    return message_store.get_project_stats(project_short_id)

@app.get("/socket/messages/stats")
async def get_all_message_stats():
    """Get message statistics for all projects."""
    return message_store.get_all_projects_stats()

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
        from .services.user_query_service import analyze_user_query
        
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


