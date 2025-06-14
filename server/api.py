from fastapi import FastAPI, HTTPException
from sse_starlette.sse import EventSourceResponse
import json
import asyncio
import uuid
from datetime import datetime

from .models import Config, ProcessResponse, WorkflowGenerationConfig, ClientConnectResponse, ClientTaskResponse
from .service import handle_process_request, start_streaming_task, process_workflow_generation_for_client, generate_workflow_from_goal
from .task_manager import (
    get_stream_task, get_stream_task_updates, is_stream_task_completed,
    create_client_session, get_client_session, get_client_updates, 
    is_client_session_active, add_task_to_client, send_to_client
)

app = FastAPI(title="Processing Server")

@app.post("/process", response_model=ProcessResponse)
async def process_request(config: Config) -> ProcessResponse:
    """
    Process the incoming request with the given configuration.
    Returns a task ID that can be used to retrieve results.
    """
    try:
        return await handle_process_request(config.parameters)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Simple synchronous workflow generation endpoint
@app.post("/workflow/generate")
async def generate_workflow_sync(config: WorkflowGenerationConfig):
    """
    Generate a workflow synchronously and return it directly.
    This is a simple blocking API that returns the workflow in the response.
    """
    try:
        goal = config.goal
        llm_config_dict = config.llm_config
        mcp_config = config.mcp_config or {}
        
        if not goal or not llm_config_dict:
            raise HTTPException(
                status_code=400, 
                detail="Missing required parameters: goal and llm_config are required"
            )
        
        # Generate the workflow directly (this will block until complete)
        workflow_graph = await generate_workflow_from_goal(goal, llm_config_dict, mcp_config)
        
        if workflow_graph is None:
            raise HTTPException(status_code=500, detail="Failed to generate workflow")
        
        # Convert workflow_graph to serializable format
        try:
            if hasattr(workflow_graph, 'get_config'):
                workflow_dict = workflow_graph.get_config()
            elif hasattr(workflow_graph, 'get_workflow_description'):
                workflow_dict = {
                    "goal": workflow_graph.goal,
                    "description": workflow_graph.get_workflow_description()
                }
            else:
                workflow_dict = str(workflow_graph)
        except Exception as e:
            workflow_dict = f"Workflow generated successfully (serialization error: {str(e)})"
        
        # Return the workflow directly
        return {
            "success": True,
            "goal": goal,
            "workflow_graph": workflow_dict,
            "message": "Workflow generated successfully",
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating workflow: {str(e)}")

@app.post("/stream/process")
async def start_stream_process(config: Config):
    """
    Start a streaming process and return a task ID.
    """
    return await start_streaming_task(config.dict())

@app.post("/stream/workflow/generate")
async def start_workflow_generation(config: WorkflowGenerationConfig):
    """
    Start a streaming workflow generation process and return a task ID.
    """
    workflow_config = {
        "task_type": "workflow_generation",
        "parameters": {
            "goal": config.goal,
            "llm_config": config.llm_config,
            "mcp_config": config.mcp_config
        },
        "timeout": config.timeout
    }
    return await start_streaming_task(workflow_config)

# New client-session endpoints
@app.post("/connect", response_model=ClientConnectResponse)
async def connect_client() -> ClientConnectResponse:
    """
    Create a new client session and return client ID with stream URL.
    """
    client_id = str(uuid.uuid4())
    create_client_session(client_id)
    
    return ClientConnectResponse(
        client_id=client_id,
        stream_url=f"/stream/client/{client_id}"
    )

@app.post("/client/{client_id}/workflow/generate", response_model=ClientTaskResponse)
async def start_client_workflow_generation(client_id: str, config: WorkflowGenerationConfig) -> ClientTaskResponse:
    """
    Start workflow generation for a specific client session.
    """
    if not get_client_session(client_id):
        raise HTTPException(status_code=404, detail="Client session not found")
    
    task_id = str(uuid.uuid4())
    add_task_to_client(client_id, task_id)
    
    # Send initial status to client
    send_to_client(client_id, {
        "event_type": "task_started",
        "task_id": task_id,
        "task_type": "workflow_generation",
        "goal": config.goal
    })
    
    # Start processing in background
    asyncio.create_task(
        process_workflow_generation_for_client(client_id, task_id, config.dict())
    )
    
    return ClientTaskResponse(
        task_id=task_id,
        status="started",
        client_id=client_id
    )

@app.get("/stream/client/{client_id}")
async def stream_client_updates(client_id: str):
    """
    Persistent SSE stream for a client session.
    """
    if not get_client_session(client_id):
        raise HTTPException(status_code=404, detail="Client session not found")
        
    return EventSourceResponse(
        client_event_generator(client_id, timeout=3600)  # 1 hour timeout
    )

@app.delete("/client/{client_id}")
async def disconnect_client(client_id: str):
    """
    Disconnect a client session.
    """
    if not get_client_session(client_id):
        raise HTTPException(status_code=404, detail="Client session not found")
    
    from .task_manager import close_client_session
    close_client_session(client_id)
    
    return {"status": "disconnected", "client_id": client_id}

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
async def stream_results(task_id: str):
    """
    Stream results for a given task ID using Server-Sent Events.
    """
    if not get_stream_task(task_id):
        raise HTTPException(status_code=404, detail="Task not found")
        
    task_config = get_stream_task(task_id)["config"]
    return EventSourceResponse(
        event_generator(task_id, timeout=task_config["timeout"])
    )

@app.get("/health")
async def health_check():
    """Basic health check endpoint"""
    return {"status": "healthy"}

@app.get("/clients")
async def list_clients():
    """List all active client sessions (for debugging)"""
    from .task_manager import client_sessions
    active_clients = []
    
    for client_id, session in client_sessions.items():
        if session["is_active"]:
            active_clients.append({
                "client_id": client_id,
                "created_at": session["created_at"].isoformat(),
                "last_activity": session["last_activity"].isoformat(),
                "active_tasks": session["active_tasks"]
            })
    
    return {"active_clients": active_clients, "total": len(active_clients)} 