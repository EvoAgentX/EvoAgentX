"""
API routes for EvoAgentX application.
"""
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from fastapi.security import OAuth2PasswordRequestForm
from typing import List, Dict, Any # , Optional
from fastapi import Response

from datetime import timedelta

from evoagentx.app.config import settings
from evoagentx.app.schemas import (
    AgentCreate, AgentUpdate, AgentResponse,
    WorkflowCreate, WorkflowUpdate, WorkflowResponse,
    ExecutionCreate, ExecutionResponse,
    PaginationParams, SearchParams,
    Token, UserCreate, UserResponse, # UserLogin, 
)
from evoagentx.app.services import AgentService, WorkflowService, WorkflowExecutionService
from evoagentx.app.security import (
    create_access_token, 
    authenticate_user, 
    create_user, 
    get_current_active_user,
    get_current_admin_user
)
from evoagentx.app.db import Database, ExecutionStatus 

# Create routers for different route groups
auth_router = APIRouter(prefix=settings.API_PREFIX)
agents_router = APIRouter(prefix=settings.API_PREFIX)
workflows_router = APIRouter(prefix=settings.API_PREFIX)
executions_router = APIRouter(prefix=settings.API_PREFIX)
system_router = APIRouter(prefix=settings.API_PREFIX)

# Authentication Routes
@auth_router.post("/auth/register", response_model=UserResponse, tags=["Authentication"])
async def register_user(user: UserCreate):
    """Register a new user."""
    return await create_user(user)

@auth_router.post("/auth/login", response_model=Token, tags=["Authentication"])
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """Login and return access token."""
    user = await authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        subject=user['email'], 
        expires_delta=access_token_expires
    )
    
    return {
        "access_token": access_token, 
        "token_type": "bearer"
    }

# Agent Routes
@agents_router.post("/agents", response_model=AgentResponse, tags=["Agents"])
async def create_agent(
    agent: AgentCreate, 
    current_user: Dict[str, Any] = Depends(get_current_active_user)
):
    """Create a new agent."""
    try:
        created_agent = await AgentService.create_agent(
            agent, 
            user_id=str(current_user['_id'])
        )
        # Convert the ObjectId to string before creating the response model
        created_agent["_id"] = str(created_agent["_id"])
        return AgentResponse(**created_agent)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@agents_router.get("/agents/{agent_id}", response_model=AgentResponse, tags=["Agents"])
async def get_agent(
    agent_id: str, 
    current_user: Dict[str, Any] = Depends(get_current_active_user)
):
    """Retrieve a specific agent by ID."""
    agent = await AgentService.get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    agent["_id"] = str(agent["_id"])
    return AgentResponse(**agent)

@agents_router.put("/agents/{agent_id}", response_model=AgentResponse, tags=["Agents"])
async def update_agent(
    agent_id: str, 
    agent_update: AgentUpdate, 
    current_user: Dict[str, Any] = Depends(get_current_active_user)
):
    """Update an existing agent."""
    try:
        updated_agent = await AgentService.update_agent(agent_id, agent_update)
        if not updated_agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        updated_agent["_id"] = str(updated_agent["_id"])
        return AgentResponse(**updated_agent)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@agents_router.get("/agents", response_model=List[AgentResponse], tags=["Agents"])
async def list_agents(
    pagination: PaginationParams = Depends(),
    search: SearchParams = Depends(),
    current_user: Dict[str, Any] = Depends(get_current_active_user)
):
    """List agents with optional pagination and search."""
    agents, total = await AgentService.list_agents(pagination, search)
    # Convert _id to string for each agent in the list
    for agent in agents:
        agent["_id"] = str(agent["_id"])
    return [AgentResponse(**agent) for agent in agents]

@agents_router.delete("/agents/{agent_id}", status_code=204, tags=["Agents"])
async def delete_agent(
    agent_id: str, 
    current_user: Dict[str, Any] = Depends(get_current_admin_user)
):
    """Delete an agent (admin-only)."""
    try:
        success = await AgentService.delete_agent(agent_id)
        if not success:
            raise HTTPException(status_code=404, detail="Agent not found")
        return  # With 204, no content is returned
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))



# Workflow Routes
@workflows_router.post("/workflows", response_model=WorkflowResponse,status_code=201, tags=["Workflows"])
async def create_workflow(
    workflow: WorkflowCreate, 
    current_user: Dict[str, Any] = Depends(get_current_active_user)
):
    """Create a new workflow."""
    try:
        created_workflow = await WorkflowService.create_workflow(
            workflow, 
            user_id=str(current_user['_id'])
        )
        # Convert the ObjectId to string for consistency
        created_workflow["_id"] = str(created_workflow["_id"])
        return WorkflowResponse(**created_workflow)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))



@workflows_router.get("/workflows/{workflow_id}", response_model=WorkflowResponse, tags=["Workflows"])
async def get_workflow(
    workflow_id: str, 
    current_user: Dict[str, Any] = Depends(get_current_active_user)
):
    """Retrieve a specific workflow by ID."""
    workflow = await WorkflowService.get_workflow(workflow_id)
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    # Convert ObjectId to string
    workflow["_id"] = str(workflow["_id"])
    return WorkflowResponse(**workflow)

@workflows_router.put("/workflows/{workflow_id}", response_model=WorkflowResponse, tags=["Workflows"])
async def update_workflow(
    workflow_id: str, 
    workflow_update: WorkflowUpdate, 
    current_user: Dict[str, Any] = Depends(get_current_active_user)
):
    """Update an existing workflow."""
    try:
        updated_workflow = await WorkflowService.update_workflow(workflow_id, workflow_update)
        if not updated_workflow:
            raise HTTPException(status_code=404, detail="Workflow not found")
        
        updated_workflow["_id"] = str(updated_workflow["_id"])
        return WorkflowResponse(**updated_workflow)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@workflows_router.delete("/workflows/{workflow_id}", status_code=204, tags=["Workflows"])
async def delete_workflow(
    workflow_id: str, 
    current_user: Dict[str, Any] = Depends(get_current_admin_user)
):
    """Delete a workflow (admin-only)."""
    try:
        success = await WorkflowService.delete_workflow(workflow_id)
        if not success:
            raise HTTPException(status_code=404, detail="Workflow not found")
        return Response(status_code=204)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@workflows_router.get("/workflows", response_model=List[WorkflowResponse], tags=["Workflows"])
async def list_workflows(
    pagination: PaginationParams = Depends(),
    search: SearchParams = Depends(),
    current_user: Dict[str, Any] = Depends(get_current_active_user)
):
    """List workflows with optional pagination and search."""
    workflows, total = await WorkflowService.list_workflows(pagination, search)
    
    # Convert ObjectId to string for each workflow
    converted_workflows = [
        {**workflow, "_id": str(workflow["_id"])}
        for workflow in workflows
    ]
    
    return [WorkflowResponse(**workflow) for workflow in converted_workflows]


# Workflow Execution Routes
@executions_router.post("/executions", response_model=ExecutionResponse, status_code=202)
async def create_execution(
    execution: ExecutionCreate,
    background_tasks: BackgroundTasks,
    current_user: Dict[str, Any] = Depends(get_current_active_user)
):
    """Create and start a workflow execution."""
    try:
        execution_result = await WorkflowExecutionService.create_execution(
            execution_data=execution,
            user_id=str(current_user['_id'])
        )
        # Convert _id to string for consistency
        execution_result["_id"] = str(execution_result["_id"])
        return ExecutionResponse(**execution_result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@executions_router.get("/executions/{execution_id}", response_model=ExecutionResponse)
async def get_execution(
    execution_id: str,
    current_user: Dict[str, Any] = Depends(get_current_active_user)
):
    """Retrieve a specific workflow execution by ID."""
    try:
        execution = await WorkflowExecutionService.get_execution(execution_id)
        if not execution:
            raise HTTPException(status_code=404, detail="Execution not found")
        execution["_id"] = str(execution["_id"])
        return ExecutionResponse(**execution)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@executions_router.post("/executions/{execution_id}/stop", response_model=ExecutionResponse)
async def stop_execution(
    execution_id: str,
    current_user: Dict[str, Any] = Depends(get_current_active_user)
):
    """Stop (cancel) a workflow execution."""
    try:
        updated_execution = await WorkflowExecutionService.update_execution_status(
            execution_id=execution_id,
            status=ExecutionStatus.CANCELLED
        )
        if not updated_execution:
            raise HTTPException(status_code=404, detail="Execution not found")
        # Convert ObjectId to string for consistency
        updated_execution["_id"] = str(updated_execution["_id"])
        return ExecutionResponse(**updated_execution)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@executions_router.get("/executions", response_model=List[ExecutionResponse])
async def list_executions(
    pagination: PaginationParams = Depends(),
    search: SearchParams = Depends(), 
    current_user: Dict[str, Any] = Depends(get_current_active_user)
):
    """List workflow executions with optional pagination and search."""
    executions, total = await WorkflowExecutionService.list_executions(
        params=pagination, 
        search=search
    )
    # Convert _id to string for each execution
    for exec_item in executions:
        exec_item["_id"] = str(exec_item["_id"])
    return [ExecutionResponse(**exec_item) for exec_item in executions]


@executions_router.get("/executions/{execution_id}/logs", response_model=List[Dict[str, Any]])
async def get_execution_logs(
    execution_id: str,
    pagination: PaginationParams = Depends(),
    current_user: Dict[str, Any] = Depends(get_current_active_user)
):
    """Retrieve logs for a specific execution."""
    logs, total = await WorkflowExecutionService.get_execution_logs(execution_id, params=pagination)
    # Convert _id in each log entry to string
    for log in logs:
        log["_id"] = str(log["_id"])
    return logs

# Health Check Route
@system_router.get("/health", tags=["System"])
async def health_check():
    """Simple health check endpoint."""
    try:
        # You can add more comprehensive health checks here
        await Database.db.command('ping')
        return {
            "status": "healthy", 
            "version": "1.0.0"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database connection error: {str(e)}")

# Export the routers
__all__ = [
    'auth_router',
    'agents_router',
    'workflows_router',
    'executions_router',
    'system_router'
]