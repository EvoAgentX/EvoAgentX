"""
API Integration Tests for EvoAgentX.

These tests cover:
1. Authentication flow (successful and failed login)
2. Agent CRUD operations (create, get, update, list, delete)
3. Workflow CRUD operations (create, get, update, list, execute)
4. Unauthorized access to protected endpoints
5. Agent search/filtering capabilities
6. Workflow execution with multiple agents and steps
"""
import pytest
import pytest_asyncio
import uuid
from urllib.parse import urljoin
import httpx

# Base API URL configuration
BASE_URL = "http://localhost:8000/api/v1/"

# --- Helper functions --- #
async def create_agent(client: httpx.AsyncClient, headers: dict, payload: dict) -> str:
    """Helper to create an agent and return its ID."""
    response = await client.post(urljoin(BASE_URL, "agents"), headers=headers, json=payload)
    assert response.status_code in (200, 201), f"Unexpected status: {response.status_code}"
    agent_data = response.json()
    return agent_data["_id"]

async def delete_agent(client: httpx.AsyncClient, headers: dict, agent_id: str):
    """Helper to delete an agent."""
    response = await client.delete(urljoin(BASE_URL, f"agents/{agent_id}"), headers=headers)
    assert response.status_code == 204

async def create_workflow(client: httpx.AsyncClient, headers: dict, payload: dict) -> str:
    """Helper to create a workflow and return its ID."""
    response = await client.post(urljoin(BASE_URL, "workflows"), headers=headers, json=payload)
    assert response.status_code == 201, f"Unexpected status: {response.status_code}"
    workflow_data = response.json()
    return workflow_data["_id"]

async def delete_workflow(client: httpx.AsyncClient, headers: dict, workflow_id: str):
    """Helper to delete a workflow."""
    response = await client.delete(urljoin(BASE_URL, f"workflows/{workflow_id}"), headers=headers)
    assert response.status_code == 204

# --- Fixtures --- #
@pytest_asyncio.fixture
async def client():
    async with httpx.AsyncClient() as client:
        yield client

@pytest_asyncio.fixture
async def access_token(client: httpx.AsyncClient):
    response = await client.post(
        urljoin(BASE_URL, "auth/login"),
        data={"username": "admin@clayx.ai", "password": "adminpassword"}
    )
    assert response.status_code == 200
    return response.json()["access_token"]



@pytest.mark.asyncio
async def test_create_execution(client: httpx.AsyncClient, access_token: str):
    headers = {"Authorization": f"Bearer {access_token}"}
    # Create an agent for the workflow
    agent_payload = {
        "name": f"ExecAgent_{uuid.uuid4().hex[:8]}",
        "description": "Agent for execution test",
        "config": {"model": "dummy-model"},
        "runtime_params": {},
        "tags": []
    }
    agent_id = await create_agent(client, headers, agent_payload)
    # Create a workflow that uses the agent
    workflow_definition = {
        "steps": [
            {
                "step_id": "step1",
                "agent_id": agent_id,
                "action": "test_action",
                "input_mapping": {},
                "output_mapping": {}
            }
        ]
    }
    workflow_payload = {
        "name": f"ExecWorkflow_{uuid.uuid4().hex[:8]}",
        "description": "Workflow for execution test",
        "definition": workflow_definition,
        "tags": []
    }
    workflow_id = await create_workflow(client, headers, workflow_payload)
    try:
        # Create an execution via POST /executions
        execution_payload = {
            "workflow_id": workflow_id,
            "input_params": {"test_key": "test_value"}
        }
        exec_resp = await client.post(
            urljoin(BASE_URL, "executions"),
            headers=headers,
            json=execution_payload
        )
        assert exec_resp.status_code == 202, f"Expected 202, got {exec_resp.status_code}"
        exec_data = exec_resp.json()
        assert "id" in exec_data or "_id" in exec_data
    finally:
        await delete_workflow(client, headers, workflow_id)
        await delete_agent(client, headers, agent_id)

@pytest.mark.asyncio
async def test_stop_execution(client: httpx.AsyncClient, access_token: str):
    headers = {"Authorization": f"Bearer {access_token}"}
    
    # Create an agent for the workflow
    agent_payload = {
        "name": f"ExecAgentForStop_{uuid.uuid4().hex[:8]}",
        "description": "Agent for stop execution test",
        "config": {"model": "dummy-model"},
        "runtime_params": {},
        "tags": []
    }
    agent_id = await create_agent(client, headers, agent_payload)
    
    # Create a workflow that uses the agent
    workflow_definition = {
        "steps": [
            {
                "step_id": "step1",
                "agent_id": agent_id,
                "action": "test_action",
                "input_mapping": {},
                "output_mapping": {}
            }
        ]
    }
    workflow_payload = {
        "name": f"ExecWorkflowForStop_{uuid.uuid4().hex[:8]}",
        "description": "Workflow for stop execution test",
        "definition": workflow_definition,
        "tags": []
    }
    workflow_id = await create_workflow(client, headers, workflow_payload)
    
    try:
        # Create an execution via POST /executions
        execution_payload = {
            "workflow_id": workflow_id,
            "input_params": {"test_key": "test_value"}
        }
        exec_resp = await client.post(
            urljoin(BASE_URL, "executions"),
            headers=headers,
            json=execution_payload
        )
        assert exec_resp.status_code == 202, f"Expected 202, got {exec_resp.status_code}"
        execution_data = exec_resp.json()
        execution_id = execution_data.get("id") or execution_data.get("_id")
        assert execution_id, "Execution ID should be present"
        
        # Stop the execution by calling POST /executions/{execution_id}/stop
        stop_resp = await client.post(
            urljoin(BASE_URL, f"executions/{execution_id}/stop"),
            headers=headers
        )
        assert stop_resp.status_code == 200, f"Expected 200, got {stop_resp.status_code}"
        stop_data = stop_resp.json()
        # Verify the execution status is now "cancelled"
        assert stop_data["status"].lower() == "cancelled", f"Expected status 'cancelled', got {stop_data['status']}"
        
    finally:
        await delete_workflow(client, headers, workflow_id)
        await delete_agent(client, headers, agent_id)


@pytest.mark.asyncio
async def test_get_execution(client: httpx.AsyncClient, access_token: str):
    headers = {"Authorization": f"Bearer {access_token}"}
    # Create an agent and workflow for execution
    agent_payload = {
        "name": f"ExecAgent2_{uuid.uuid4().hex[:8]}",
        "description": "Agent for get execution test",
        "config": {"model": "dummy-model"},
        "runtime_params": {},
        "tags": []
    }
    agent_id = await create_agent(client, headers, agent_payload)
    workflow_definition = {
        "steps": [
            {
                "step_id": "step1",
                "agent_id": agent_id,
                "action": "test_action",
                "input_mapping": {},
                "output_mapping": {}
            }
        ]
    }
    workflow_payload = {
        "name": f"ExecWorkflow2_{uuid.uuid4().hex[:8]}",
        "description": "Workflow for get execution test",
        "definition": workflow_definition,
        "tags": []
    }
    workflow_id = await create_workflow(client, headers, workflow_payload)
    try:
        execution_payload = {
            "workflow_id": workflow_id,
            "input_params": {"param": "value"}
        }
        create_resp = await client.post(
            urljoin(BASE_URL, "executions"),
            headers=headers,
            json=execution_payload
        )
        assert create_resp.status_code == 202
        execution_data = create_resp.json()
        execution_id = execution_data.get("id") or execution_data.get("_id")
        assert execution_id, "Execution ID should be present"
        
        get_resp = await client.get(urljoin(BASE_URL, f"executions/{execution_id}"), headers=headers)
        assert get_resp.status_code == 200
        get_data = get_resp.json()
        assert get_data["workflow_id"] == workflow_id
    finally:
        await delete_workflow(client, headers, workflow_id)
        await delete_agent(client, headers, agent_id)


@pytest.mark.asyncio
async def test_list_executions(client: httpx.AsyncClient, access_token: str):
    headers = {"Authorization": f"Bearer {access_token}"}
    # Create an agent and workflow for executions
    agent_payload = {
        "name": f"ExecAgent3_{uuid.uuid4().hex[:8]}",
        "description": "Agent for list executions test",
        "config": {"model": "dummy-model"},
        "runtime_params": {},
        "tags": []
    }
    agent_id = await create_agent(client, headers, agent_payload)
    workflow_definition = {
        "steps": [
            {
                "step_id": "step1",
                "agent_id": agent_id,
                "action": "test_action",
                "input_mapping": {},
                "output_mapping": {}
            }
        ]
    }
    workflow_payload = {
        "name": f"ExecWorkflow3_{uuid.uuid4().hex[:8]}",
        "description": "Workflow for list executions test",
        "definition": workflow_definition,
        "tags": []
    }
    workflow_id = await create_workflow(client, headers, workflow_payload)
    try:
        execution_payload = {
            "workflow_id": workflow_id,
            "input_params": {"param": "value1"}
        }
        await client.post(urljoin(BASE_URL, "executions"), headers=headers, json=execution_payload)
        
        execution_payload2 = {
            "workflow_id": workflow_id,
            "input_params": {"param": "value2"}
        }
        await client.post(urljoin(BASE_URL, "executions"), headers=headers, json=execution_payload2)
        
        list_resp = await client.get(
            urljoin(BASE_URL, "executions"),
            headers=headers,
            params={"skip": 0, "limit": 10}
        )
        assert list_resp.status_code == 200
        executions = list_resp.json()
        assert isinstance(executions, list)
        assert len(executions) >= 2
    finally:
        await delete_workflow(client, headers, workflow_id)
        await delete_agent(client, headers, agent_id)


@pytest.mark.asyncio
async def test_get_execution_logs(client: httpx.AsyncClient, access_token: str):
    headers = {"Authorization": f"Bearer {access_token}"}
    # Create an agent, workflow, and execution
    agent_payload = {
        "name": f"ExecAgentLog_{uuid.uuid4().hex[:8]}",
        "description": "Agent for logs test",
        "config": {"model": "dummy-model"},
        "runtime_params": {},
        "tags": []
    }
    agent_id = await create_agent(client, headers, agent_payload)
    workflow_definition = {
        "steps": [
            {
                "step_id": "step1",
                "agent_id": agent_id,
                "action": "test_action",
                "input_mapping": {},
                "output_mapping": {}
            }
        ]
    }
    workflow_payload = {
        "name": f"ExecWorkflowLog_{uuid.uuid4().hex[:8]}",
        "description": "Workflow for logs test",
        "definition": workflow_definition,
        "tags": []
    }
    workflow_id = await create_workflow(client, headers, workflow_payload)
    try:
        execution_payload = {
            "workflow_id": workflow_id,
            "input_params": {"param": "value"}
        }
        create_resp = await client.post(
            urljoin(BASE_URL, "executions"),
            headers=headers,
            json=execution_payload
        )
        assert create_resp.status_code == 202
        execution_data = create_resp.json()
        execution_id = execution_data.get("id") or execution_data.get("_id")
        assert execution_id, "Execution ID should be present"
        
        log_resp = await client.get(
            urljoin(BASE_URL, f"executions/{execution_id}/logs"),
            headers=headers,
            params={"skip": 0, "limit": 10}
        )
        assert log_resp.status_code == 200
        logs = log_resp.json()
        assert isinstance(logs, list)
    finally:
        await delete_workflow(client, headers, workflow_id)
        await delete_agent(client, headers, agent_id)