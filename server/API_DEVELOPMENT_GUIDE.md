# EvoAgentX Server API Development Guide

## Table of Contents
- [Overview](#overview)
- [Server Configuration](#server-configuration)
- [Authentication](#authentication)
- [HTTP Endpoints](#http-endpoints)
- [WebSocket Endpoints](#websocket-endpoints)
- [Data Models & Schemas](#data-models--schemas)
- [Error Handling](#error-handling)
- [Testing & Examples](#testing--examples)
- [Deployment](#deployment)

---

## Overview

The EvoAgentX Server provides a comprehensive API for managing AI workflow execution with both synchronous HTTP endpoints and asynchronous WebSocket streaming for real-time updates. The server manages the complete workflow lifecycle: setup → generation → execution → monitoring.

**Key Features:**
- Real-time streaming via WebSocket for workflow execution
- Parallel workflow generation with progress tracking
- Comprehensive workflow lifecycle management
- User query analysis for workflow modifications
- Health monitoring and status tracking
- Flexible input/output handling for different workflow types

---

## Server Configuration

### Base URLs
- **HTTP Base URL**: `http://localhost:8001` (configurable)
- **WebSocket Base URL**: `ws://localhost:8001` (derived from HTTP base)
- **Internal Port**: 8001 (as configured in `fly.toml`)

### Environment Variables
```bash
# Required
EAX_ACCESS_TOKEN=your_access_token_here

# Optional
LOG_LEVEL=warning
SUPPRESS_WARNINGS=true
VERBOSE_STARTUP=false
```

### Infrastructure (fly.toml)
```toml
[env]
  LOG_LEVEL = 'warning'
  SUPPRESS_WARNINGS = 'true'
  VERBOSE_STARTUP = 'false'

[http_service]
  internal_port = 8001
  force_https = true
  auto_stop_machines = 'stop'
  auto_start_machines = true
  min_machines_running = 1

[[vm]]
  memory = '2gb'
  cpu_kind = 'shared'
  cpus = 2
```

---

## Authentication

All endpoints (except `/health`) require authentication via the `eax-access-token` header.

```bash
# Example header
eax-access-token: your_access_token_here
```

**Default Token**: `"default_secret_token_change_me"` (change in production!)

---

## HTTP Endpoints

### 1. Health Check

**Endpoint**: `GET /health`

**Description**: Basic health check endpoint for monitoring and load balancers.

**Request**: No authentication required

**Response**:
```json
{
  "status": "healthy"
}
```

**Usage Example**:
```bash
curl -X GET "http://localhost:8001/health"
```

---

### 2. Project Setup (Phase 1)

**Endpoint**: `POST /project/setup`

**Description**: Initializes workflow setup with extraction and generation. This is the first phase of the workflow process.

**Authentication**: Required

**Request Schema**:
```json
{
  "project_short_id": "string"
}
```

**Response Schema**:
```json
{
  "workflow_graphs": [
    {
      "workflow_id": "string",
      "workflow_name": "string",
      "workflow_inputs": [
        {
          "name": "string",
          "type": "string",
          "description": "string"
        }
      ],
      "workflow_outputs": [
        {
          "name": "string",
          "type": "string"
        }
      ],
      "workflow_graph": "object|string|null",
      "status": "string"
    }
  ],
  "message": "string"
}
```

**Usage Example**:
```bash
curl -X POST "http://localhost:8001/project/setup" \
  -H "eax-access-token: your_token" \
  -H "Content-Type: application/json" \
  -d '{"project_short_id": "xf35dy4"}'
```

---

### 3. Parallel Project Setup

**Endpoint**: `POST /project/setup-parallel`

**Description**: Enhanced version with parallel execution and automatic retries. Takes the same input as regular setup.

**Authentication**: Required

**Request Schema**: Same as regular setup

**Response Schema**: Same as regular setup

**Features**: 
- Parallel workflow generation with configurable concurrency (default: 5 concurrent workflows)
- Automatic retry logic with exponential backoff (default: 2 retries)
- Enhanced error handling and fallback workflow creation
- Real-time status updates for each workflow phase
- Automatic WebSocket connection management

**Usage Example**:
```bash
curl -X POST "http://localhost:8001/project/setup-parallel" \
  -H "eax-access-token: your_token" \
  -H "Content-Type: application/json" \
  -d '{"project_short_id": "xf35dy4"}'
```

---

### 4. Workflow Generation (Phase 2)

**Endpoint**: `POST /workflow/{workflow_id}/generate`

**Description**: Generates workflow graph based on task_info. This is the second phase of the workflow process.

**Authentication**: Required

**Path Parameters**:
- `workflow_id`: string (UUID format)

**Request**: No body required

**Response Schema**:
```json
{
  "workflow_graph": "object|string",
  "status": "success|failed"
}
```

**Usage Example**:
```bash
curl -X POST "http://localhost:8001/workflow/550e8400-e29b-41d4-a716-446655440001/generate" \
  -H "eax-access-token: your_token"
```

---

### 5. Workflow Execution (Phase 3) - HTTP

**Endpoint**: `POST /workflow/{workflow_id}/execute`

**Description**: Executes workflow with provided inputs. This is the third phase of the workflow process.

**Authentication**: Required

**Path Parameters**:
- `workflow_id`: string

**Request Schema**:
```json
{
  "inputs": {
    "prompt": "string",
    "negative_prompt": "string",
    "style": "string",
    "dimensions": {
      "width": "integer",
      "height": "integer"
    },
    "quality": "string"
  }
}
```

**Response Schema**:
```json
{
  "parsed_json": "object|null"
}
```

**Usage Example**:
```bash
curl -X POST "http://localhost:8001/workflow/550e8400-e29b-41d4-a716-446655440001/execute" \
  -H "eax-access-token: your_token" \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": {
      "prompt": "A beautiful sunset over a calm ocean",
      "negative_prompt": "No text, no watermark, no logo",
      "style": "realistic",
      "dimensions": {"width": 1024, "height": 1024},
      "quality": "high"
    }
  }'
```

---

### 6. Workflow Status

**Endpoint**: `GET /workflow/{workflow_id}/status`

**Description**: Retrieves the current status and details of a workflow, showing which phase the workflow is in and all stored data.

**Authentication**: Required

**Path Parameters**:
- `workflow_id`: string

**Response Schema**:
```json
{
  "workflow_id": "string",
  "status": "string",
  "created_at": "datetime",
  "updated_at": "datetime",
  "phases": {
    "setup_complete": "boolean",
    "execution_complete": "boolean"
  },
  "workflows": "array",
  "database_information": "object",
  "workflow_graph": "object|null",
  "execution_result": "object|null"
}
```

**Usage Example**:
```bash
curl -X GET "http://localhost:8001/workflow/550e8400-e29b-41d4-a716-446655440001/status" \
  -H "eax-access-token: your_token"
```

---

### 7. Workflow Graph Retrieval

**Endpoint**: `GET /workflow/{workflow_id}/get_graph`

**Description**: Retrieves the workflow graph for a specific workflow ID.

**Authentication**: Required

**Path Parameters**:
- `workflow_id`: string

**Response Schema**:
```json
{
  "workflow_graph": "object|null"
}
```

**Usage Example**:
```bash
curl -X GET "http://localhost:8001/workflow/550e8400-e29b-41d4-a716-446655440001/get_graph" \
  -H "eax-access-token: your_token"
```

---

### 8. User Query Analysis

**Endpoint**: `POST /project/{project_short_id}/user_query`

**Description**: Analyzes user query using UserQueryRouter to understand workflow modification requests.

**Authentication**: Required

**Path Parameters**:
- `project_short_id`: string

**Request Schema**:
```json
{
  "query": "string"
}
```

**Response Schema**:
```json
{
  "result": {
    "original_query": "string",
    "total_operations": "integer",
    "is_composite": "boolean",
    "has_frontend": "boolean",
    "has_backend": "boolean",
    "classified_operations": [
      {
        "category": "string",
        "atomic_query": "string",
        "not_clear": "boolean",
        "follow_up_questions": ["string"],
        "clarity_reasoning": "string"
      }
    ]
  }
}
```

**Usage Example**:
```bash
curl -X POST "http://localhost:8001/project/xf35dy4/user_query" \
  -H "eax-access-token: your_token" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "I want to add a new node at the end of the workflow to print all middle variables"
  }'
```

---

## WebSocket Endpoints

### 1. Workflow Execution WebSocket

**Endpoint**: `ws://localhost:8001/workflow/{workflow_id}/execute_ws`

**Description**: Real-time streaming for workflow execution with live progress updates, log messages, and execution results.

**Authentication**: Required via `eax-access-token` header

**Connection Headers**:
```
eax-access-token: your_access_token_here
```

**Input Message Schema**:
```json
{
  "inputs": {
    "prompt": "string",
    "negative_prompt": "string",
    "style": "string",
    "dimensions": {
      "width": "integer",
      "height": "integer"
    },
    "quality": "string"
  }
}
```

**Output Message Types & Schemas**:

#### Connection Confirmation
```json
{
  "type": "connection",
  "content": "WebSocket connection established",
  "result": null
}
```

#### Execution Start
```json
{
  "type": "start",
  "content": "Workflow execution started",
  "result": null
}
```

#### Progress Updates
```json
{
  "type": "progress",
  "content": "Executing workflow: story_generation (75% complete)",
  "result": null
}
```

#### Log Messages
```json
{
  "type": "log",
  "content": "INFO: Starting workflow execution",
  "result": null
}
```

#### Output Messages
```json
{
  "type": "output",
  "content": "Processing user data...",
  "result": null
}
```

#### Input Messages
```json
{
  "type": "input",
  "content": "Waiting for user input...",
  "result": null
}
```

#### Workflow Status Updates
```json
{
  "type": "workflow_status",
  "status": "running|completed|failed",
  "workflow_id": "string"
}
```

#### Completion
```json
{
  "type": "complete",
  "content": "Workflow execution completed successfully",
  "result": {
    "parsed_json": "object",
    "execution_time": "string",
    "status": "string"
  }
}
```

#### Error Messages
```json
{
  "type": "error",
  "content": "Failed to connect to database",
  "result": null
}
```

**Usage Example** (JavaScript):
```javascript
const ws = new WebSocket('ws://localhost:8001/workflow/550e8400-e29b-41d4-a716-446655440001/execute_ws');

ws.onopen = function() {
  // Send execution inputs
  ws.send(JSON.stringify({
    inputs: {
      prompt: "A beautiful sunset over a calm ocean",
      negative_prompt: "No text, no watermark, no logo",
      style: "realistic",
      dimensions: {width: 1024, height: 1024},
      quality: "high"
    }
  }));
};

ws.onmessage = function(event) {
  const data = JSON.parse(event.data);
  console.log(`Message type: ${data.type}`);
  console.log(`Content: ${data.content}`);
  console.log(`Result:`, data.result);
};
```

---

### 2. Parallel Generation WebSocket

**Endpoint**: `ws://localhost:8001/project/{project_short_id}/parallel-setup`

**Description**: Real-time progress updates for parallel workflow generation with live status tracking. This WebSocket automatically starts the parallel workflow generation process upon connection and provides detailed status updates for each workflow being generated.

**Authentication**: Required via `eax-access-token` header

**Input**: None (auto-starts on connection)

**Output Message Types & Schemas**:

#### Connection Confirmation
```json
{
  "type": "connection",
  "data": {
    "status": "connected",
    "workflow_id": null,
    "content": "Parallel workflow generation progress WebSocket connected",
    "result": {
      "project_short_id": "string"
    }
  }
}
```

#### Workflow Status Updates
```json
{
  "type": "workflow_status",
  "data": {
    "status": "uninitialized",
    "workflow_id": "string",
    "content": "workflow extracted",
    "result": null
  }
}
```

**Status Values**:
- `"uninitialized"`: Workflow has been extracted and added to database
- `"pending"`: Workflow generation completed successfully, ready for execution
- `"failed"`: Workflow generation failed after retries

#### Setup Complete
```json
{
  "type": "setup_complete",
  "data": {
    "status": "complete",
    "workflow_id": null,
    "content": "setup successful",
    "result": [
      "workflow_graph_1",
      "workflow_graph_2",
      "workflow_graph_3"
    ]
  }
}
```

**Note**: The `result` field contains an array of workflow graphs only. Each workflow graph can be:
- An object (if generation succeeded)
- A string (if generation failed, contains error message)
- The full workflow metadata (workflow_id, workflow_name, inputs, outputs) is available in the individual `workflow_status` messages sent during the generation process

#### Error Messages
```json
{
  "type": "error",
  "data": {
    "status": "error",
    "workflow_id": "string|null",
    "content": "Error during parallel generation: specific error message",
    "result": null
  }
}
```

**Message Flow**:
1. **Connection**: WebSocket connects and sends connection confirmation
2. **Workflow Extraction**: For each extracted workflow, sends `workflow_status` with `"uninitialized"` status
3. **Parallel Generation**: Workflows are generated concurrently with retry logic
4. **Status Updates**: After successful generation, sends `workflow_status` with `"pending"` status and full workflow metadata
5. **Completion**: Sends `setup_complete` with array of workflow graphs only (design decision for consistency with HTTP endpoints)
6. **Connection Close**: WebSocket automatically closes after completion

**Design Note**: The `setup_complete` message returns only workflow graphs (not full metadata) to maintain consistency with the HTTP `/project/setup` and `/project/setup-parallel` endpoints. Full workflow information including workflow_id, workflow_name, inputs, and outputs is available in the individual `workflow_status` messages sent during the generation process.

**Usage Example** (JavaScript):
```javascript
const ws = new WebSocket('ws://localhost:8001/project/xf35dy4/parallel-setup');

ws.onopen = function() {
  console.log('Connected to parallel generation WebSocket');
  // Note: Generation starts automatically upon connection
};

ws.onmessage = function(event) {
  const data = JSON.parse(event.data);
  console.log(`Message type: ${data.type}`);
  
  if (data.type === 'workflow_status') {
    const status = data.data.status;
    const workflowId = data.data.workflow_id;
    const content = data.data.content;
    
    if (status === 'uninitialized') {
      console.log(`Workflow ${workflowId} extracted: ${content}`);
    } else if (status === 'pending') {
      console.log(`Workflow ${workflowId} generated successfully: ${content}`);
      // Note: Full workflow metadata (name, inputs, outputs) is available in data.data.result
      if (data.data.result) {
        console.log(`Workflow details:`, data.data.result);
      }
    }
  } else if (data.type === 'setup_complete') {
    console.log('Setup completed with workflow graphs:', data.data.result);
    console.log(`Generated ${data.data.result.length} workflow graphs`);
    // Note: WebSocket will automatically close after this message
    // The result contains only the workflow graphs, not full metadata
  } else if (data.type === 'error') {
    console.error('Error during generation:', data.data.content);
  }
};

ws.onclose = function(event) {
  if (event.code === 1000) {
    console.log('WebSocket closed normally - setup completed successfully');
  } else if (event.code === 1011) {
    console.error('WebSocket closed due to error');
  }
};
```

---

## Data Models & Schemas

### Pydantic Models

#### ProjectSetupRequest
```python
class ProjectSetupRequest(BaseModel):
    """Request model for workflow setup - Phase 1"""
    project_short_id: str  # Project identifier
```

#### ProjectSetupResponse
```python
class ProjectSetupResponse(BaseModel):
    """Response model for workflow setup - Phase 1"""
    workflow_graphs: List[Dict[str, Any]]  # List of workflow graphs
    message: str
```

#### ProjectWorkflowExecutionRequest
```python
class ProjectWorkflowExecutionRequest(BaseModel):
    """Request model for workflow execution - Phase 3"""
    inputs: Dict[str, Any]
```

#### UserQueryRequest
```python
class UserQueryRequest(BaseModel):
    """Request model for user query analysis"""
    query: str  # The user's query string to analyze
```

### Database Schema

#### Workflow Document Structure
```json
{
  "id": "string (Primary Key)",
  "project_short_id": "string",
  "task_info": "object (saved after setup phase)",
  "workflow_graph": "object|null (saved after generation phase)",
  "execution_result": "object|null (saved after execution phase)",
  "status": "string (created|generating|completed|failed)",
  "created_at": "datetime",
  "updated_at": "datetime"
}
```

---

## Error Handling

### HTTP Error Responses

**400 Bad Request**:
```json
{
  "detail": "Validation error or business logic error"
}
```

**404 Not Found**:
```json
{
  "detail": "Workflow not found"
}
```

**422 Unprocessable Entity**:
```json
{
  "detail": "Workflow generation failed: specific error message"
}
```

**500 Internal Server Error**:
```json
{
  "detail": "Internal server error during workflow execution: error details"
}
```

### WebSocket Error Responses
```json
{
  "type": "error",
  "content": "Detailed error message",
  "result": null
}
```

### Common Error Scenarios

1. **Authentication Failed**: Missing or invalid `eax-access-token`
2. **Workflow Not Found**: Invalid `workflow_id` in URL path
3. **Generation Failed**: Workflow generation process failed
4. **Execution Failed**: Workflow execution process failed
5. **WebSocket Connection Issues**: Network or authentication problems

---

## Testing & Examples

### Test Input Examples

#### Image Generation Workflow
```json
{
  "prompt": "A beautiful sunset over a calm ocean, with a small boat in the foreground. The sky is painted with soft, warm colors, and the water reflects the sunset's glow.",
  "negative_prompt": "No text, no watermark, no logo",
  "style": "realistic",
  "dimensions": {
    "width": 1024,
    "height": 1024
  },
  "quality": "high"
}
```

#### Story Generation Workflow
```json
{
  "character_name": "Alice",
  "character_type": "human",
  "setting": "a magical forest",
  "genre": "fantasy",
  "target_age": "8-12",
  "moral_lesson": "friendship and courage",
  "story_length": "medium",
  "language": "English"
}
```

### Complete Workflow Example

1. **Setup Project**:
```bash
curl -X POST "http://localhost:8001/project/setup" \
  -H "eax-access-token: your_token" \
  -H "Content-Type: application/json" \
  -d '{"project_short_id": "xf35dy4"}'
```

2. **Generate Workflow**:
```bash
curl -X POST "http://localhost:8001/workflow/550e8400-e29b-41d4-a716-446655440001/generate" \
  -H "eax-access-token: your_token"
```

3. **Execute Workflow**:
```bash
curl -X POST "http://localhost:8001/workflow/550e8400-e29b-41d4-a716-446655440001/execute" \
  -H "eax-access-token: your_token" \
  -H "Content-Type: application/json" \
  -d '{"inputs": {"prompt": "A beautiful sunset"}}'
```

4. **Check Status**:
```bash
curl -X GET "http://localhost:8001/workflow/550e8400-e29b-41d4-a716-446655440001/status" \
  -H "eax-access-token: your_token"
```

### WebSocket Testing

Use the test server (`test_server.py`) to test WebSocket functionality:

```bash
cd server
python test_server.py
```

---

## Deployment

### Local Development

1. **Install Dependencies**:
```bash
pip install -r requirements.txt
```

2. **Set Environment Variables**:
```bash
export EAX_ACCESS_TOKEN="your_token_here"
```

3. **Run Server**:
```bash
python -m uvicorn server.api:app --host 0.0.0.0 --port 8001 --reload
```

### Production Deployment (Fly.io)

1. **Configure fly.toml**:
```toml
app = 'evoagentx-server'
primary_region = 'sea'

[env]
  EAX_ACCESS_TOKEN = 'your_production_token'
  LOG_LEVEL = 'warning'

[http_service]
  internal_port = 8001
  force_https = true
```

2. **Deploy**:
```bash
fly deploy
```

### Health Checks

The server includes built-in health checks:
- **Endpoint**: `/health`
- **Interval**: Every 10 minutes
- **Timeout**: 10 minutes 40 seconds
- **Grace Period**: 600 seconds

---

## CORS Configuration

```python
{
  "allow_origins": ["*"],
  "allow_credentials": true,
  "allow_methods": ["*"],
  "allow_headers": ["*"]
}
```

---

## Performance & Scaling

- **Memory**: 2GB per VM instance
- **CPU**: 2 shared CPUs
- **Auto-scaling**: Enabled with min 1 machine running
- **Health Checks**: Every 10 minutes with 10-minute timeout
- **Grace Period**: 600 seconds for health check failures

---

## Troubleshooting

### Common Issues

1. **Authentication Errors**: Ensure `eax-access-token` header is set correctly
2. **WebSocket Connection Failures**: Check network connectivity and authentication
3. **Workflow Generation Failures**: Verify project setup completed successfully
4. **Execution Timeouts**: Check workflow complexity and resource availability

### Debug Mode

Enable verbose logging by setting:
```bash
export LOG_LEVEL=debug
export VERBOSE_STARTUP=true
```

### Log Analysis

Check server logs for detailed error information:
```bash
fly logs -a evoagentx-server
```

---

## Support & Contributing

For issues, questions, or contributions:
1. Check the logs for error details
2. Review the test examples in `test_server.py`
3. Verify environment variables are set correctly
4. Test with the provided curl examples

---

*Last Updated: 2025-01-27*
*Version: 1.0.0*
