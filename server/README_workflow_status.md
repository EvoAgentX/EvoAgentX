# Workflow Status Messaging System

This document describes the new workflow status messaging system that provides real-time updates on workflow generation progress.

## Overview

The workflow status messaging system sends real-time WebSocket messages to track the status of workflows during parallel generation. This allows clients to monitor progress and react to status changes immediately.

## Message Format

All workflow status messages follow this format:

```json
{
  "type": "workflow_status",
  "status": "uninitialized",
  "workflow_id": "SAMPLE_UUID"
}
```

### Message Fields

- **type**: Always `"workflow_status"` for status messages
- **status**: Current workflow status (see Status Values below)
- **workflow_id**: Unique identifier for the workflow (UUID string)

### Status Values

1. **`"uninitialized"`**: Workflow requirements extracted, but generation not yet started
2. **`"pending"`**: Workflow generation completed successfully, ready for execution
3. **`"error"`**: Workflow generation failed after retries
4. **`"complete"`**: Workflow execution completed successfully

## WebSocket Endpoint

Connect to the parallel generation progress WebSocket to receive status updates:

```
ws://localhost:8001/project/{project_short_id}/parallel-generation-progress
```

## Message Flow

### 1. Initial Connection
```
{"type": "connection", "content": "Parallel workflow generation progress WebSocket connected", "project_short_id": "..."}
```

### 2. Status Updates
For each workflow, you'll receive status messages in sequence:

```
{"type": "workflow_status", "status": "uninitialized", "workflow_id": "uuid-1"}
{"type": "workflow_status", "status": "uninitialized", "workflow_id": "uuid-2"}
...
```

### 3. Generation Progress
As workflows are generated, status changes to "pending":

```
{"type": "workflow_status", "status": "pending", "workflow_id": "uuid-1"}
{"type": "workflow_status", "status": "pending", "workflow_id": "uuid-2"}
...
```

### 4. Completion
When all workflows are processed:

```
{"type": "completion", "content": "Parallel workflow generation completed", "status": "completed"}
```

## Implementation Details

### Core Functions

- **`setup_project_parallel_with_status_messages()`**: Main function that sends status messages
- **`send_workflow_status_message()`**: Utility function for sending status messages
- **`WebSocketProgressTracker.send_workflow_status()`**: Class method for sending status messages

### Database Integration

The system automatically:
1. Creates workflow records with "uninitialized" status
2. Updates status to "pending" after successful generation
3. Updates status to "failed" if generation fails
4. Sends WebSocket messages at each status change

### Error Handling

- Failed workflow generations are marked as "error" status
- WebSocket errors are sent as error messages
- Timeout handling prevents infinite waiting

## Testing

Use the provided test script to verify the system:

```bash
python server/test_workflow_status.py
```

The test script will:
1. Connect to the WebSocket endpoint
2. Listen for workflow status messages
3. Validate message format and content
4. Analyze status transitions
5. Report success/failure

## Client Integration

### JavaScript Example

```javascript
const ws = new WebSocket(`ws://localhost:8001/project/${projectId}/parallel-generation-progress`);

ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    
    if (data.type === 'workflow_status') {
        console.log(`Workflow ${data.workflow_id}: ${data.status}`);
        
        // Update UI based on status
        updateWorkflowStatus(data.workflow_id, data.status);
    }
};
```

### Python Example

```python
import websockets
import json

async def listen_for_status(project_id):
    uri = f"ws://localhost:8001/project/{project_id}/parallel-generation-progress"
    
    async with websockets.connect(uri) as websocket:
        while True:
            message = await websocket.recv()
            data = json.loads(message)
            
            if data["type"] == "workflow_status":
                print(f"Workflow {data['workflow_id']}: {data['status']}")
```

## Benefits

1. **Real-time Updates**: Immediate notification of status changes
2. **Progress Tracking**: Monitor workflow generation progress
3. **Error Detection**: Quickly identify failed workflows
4. **UI Integration**: Build responsive user interfaces
5. **Debugging**: Track workflow lifecycle for troubleshooting

## Configuration

The system respects these environment variables:

- `PARALLEL_WORKFLOW_CONCURRENCY`: Maximum concurrent workflow generations (default: 5)

## Troubleshooting

### Common Issues

1. **No status messages received**: Check if workflows exist in "uninitialized" status
2. **WebSocket connection fails**: Verify server is running and endpoint is correct
3. **Status not updating**: Check database connection and workflow generation process

### Debug Mode

Enable debug logging by setting log level to DEBUG in your environment.

## Future Enhancements

- Add more granular status values (e.g., "generating", "validating")
- Include progress percentages in status messages
- Add estimated completion time estimates
- Support for workflow execution status updates
