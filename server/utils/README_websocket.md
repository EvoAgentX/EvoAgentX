# WebSocket Utilities for Real-Time Progress Tracking

This module provides generalized WebSocket utilities that can be reused across different parts of the EvoAgentX project for real-time progress tracking and messaging.

## Overview

The websocket utilities provide a standardized way to send real-time updates via WebSocket connections, following the expected message format:

```json
{
    "type": "message_type",
    "content": "Message content",
    "result": null
}
```

## Message Types

### 1. Connection Messages
- **`connection`**: Initial connection confirmation
- **`start`**: Operation start notification

### 2. Progress Messages
- **`progress`**: Real-time progress updates with percentage completion
- **`log`**: Log messages from operations
- **`output`**: Execution output messages

### 3. Completion Messages
- **`complete`**: Final operation result
- **`error`**: Error messages

## Core Classes

### WebSocketProgressTracker

The main class for tracking progress and sending standardized messages.

```python
from ..utils.websocket_utils import WebSocketProgressTracker

# Initialize tracker
progress_tracker = WebSocketProgressTracker(
    websocket_send_func=websocket_send_func,
    context_id="project_123",
    context_type="project_setup"
)

# Send connection confirmation
await progress_tracker.send_connection_confirmation()

# Send start notification
await progress_tracker.send_start_notification("Project setup")

# Send progress updates
await progress_tracker.send_progress_update("initializing", 0.0, "Starting...")
await progress_tracker.send_progress_update("processing", 0.5, "Halfway done...")
await progress_tracker.send_progress_update("completed", 1.0, "Finished!")

# Send log messages
await progress_tracker.send_log_message("INFO", "Database connected")
await progress_tracker.send_log_message("WARNING", "Slow response detected")

# Send output messages
await progress_tracker.send_output_message("Processing user data...")

# Send completion
await progress_tracker.send_completion(result_data, "Operation completed successfully")

# Send errors
await progress_tracker.send_error("Failed to connect to database")
```

### WebSocketEnhancedSink

Captures stdout/stderr and loguru messages for real-time streaming.

```python
from ..utils.websocket_utils import WebSocketEnhancedSink

# Initialize sink
websocket_sink = WebSocketEnhancedSink(
    websocket_send_func=websocket_send_func,
    context_id="workflow_123",
    context_type="workflow"
)

# Add to loguru (captures all log messages)
sink_id = logger.add(websocket_sink.write, level="INFO")

# Capture stdin input
websocket_sink.capture_stdin_input("user input")

# Cleanup when done
websocket_sink.stop()
logger.remove(sink_id)
```

## Convenience Functions

For simple operations, use the convenience functions:

```python
from ..utils.websocket_utils import (
    send_simple_message,
    send_progress_message,
    send_log_message,
    send_error_message
)

# Send simple messages
await send_simple_message(websocket_send_func, "log", "Simple log message")

# Send progress updates
await send_progress_message(websocket_send_func, "processing", 0.75, "Almost done")

# Send log messages
await send_log_message(websocket_send_func, "INFO", "Operation successful")

# Send error messages
await send_error_message(websocket_send_func, "Connection failed")
```

## Usage Examples

### 1. Workflow Execution

```python
async def execute_workflow_with_websocket(workflow_id: str, inputs: dict, websocket_send_func):
    progress_tracker = WebSocketProgressTracker(websocket_send_func, workflow_id, "workflow")
    
    try:
        await progress_tracker.send_connection_confirmation()
        await progress_tracker.send_start_notification("Workflow execution")
        
        # Execute workflow with progress updates
        await progress_tracker.send_progress_update("initializing", 0.0, "Starting...")
        # ... workflow execution logic ...
        await progress_tracker.send_progress_update("completed", 1.0, "Finished!")
        
        await progress_tracker.send_completion(result, "Workflow completed successfully")
        return result
        
    except Exception as e:
        await progress_tracker.send_error(f"Execution failed: {str(e)}")
        raise e
```

### 2. Parallel Project Setup

```python
async def setup_project_parallel_with_websocket(project_id: str, websocket_send_func):
    progress_tracker = WebSocketProgressTracker(websocket_send_func, project_id, "project_setup")
    
    try:
        await progress_tracker.send_connection_confirmation()
        await progress_tracker.send_start_notification("Parallel project setup")
        
        # Setup steps with progress
        await progress_tracker.send_progress_update("retrieving", 0.1, "Getting requirements...")
        # ... retrieve requirements ...
        
        await progress_tracker.send_progress_update("generating", 0.5, "Generating workflows...")
        # ... generate workflows in parallel ...
        
        await progress_tracker.send_progress_update("completed", 1.0, "Setup completed!")
        await progress_tracker.send_completion(result, "Project setup completed successfully")
        
        return result
        
    except Exception as e:
        await progress_tracker.send_error(f"Setup failed: {str(e)}")
        raise e
```

### 3. Database Operations

```python
async def migrate_database_with_websocket(websocket_send_func):
    progress_tracker = WebSocketProgressTracker(websocket_send_func, "db_migration", "database")
    
    try:
        await progress_tracker.send_connection_confirmation()
        await progress_tracker.send_start_notification("Database migration")
        
        # Migration steps
        await progress_tracker.send_progress_update("backup", 0.2, "Creating backup...")
        # ... backup logic ...
        
        await progress_tracker.send_progress_update("migrating", 0.6, "Running migrations...")
        # ... migration logic ...
        
        await progress_tracker.send_progress_update("verifying", 0.9, "Verifying data...")
        # ... verification logic ...
        
        await progress_tracker.send_progress_update("completed", 1.0, "Migration completed!")
        await progress_tracker.send_completion({"status": "success"}, "Database migration completed")
        
    except Exception as e:
        await progress_tracker.send_error(f"Migration failed: {str(e)}")
        raise e
```

## Integration with Existing Code

### 1. Import the utilities
```python
from ..utils.websocket_utils import WebSocketProgressTracker, WebSocketEnhancedSink
```

### 2. Replace existing websocket code
```python
# Old way (complex message format)
await websocket_send_func(json.dumps({
    "type": "status",
    "data": {"status": "running", "progress": 0.5},
    "timestamp": "2024-01-01T00:00:00"
}))

# New way (standardized format)
await progress_tracker.send_progress_update("processing", 0.5, "Halfway done")
```

### 3. Use in new operations
```python
async def new_operation_with_websocket(websocket_send_func):
    progress_tracker = WebSocketProgressTracker(websocket_send_func, "op_id", "operation")
    
    await progress_tracker.send_connection_confirmation()
    await progress_tracker.send_start_notification("New operation")
    
    # ... operation logic with progress updates ...
    
    await progress_tracker.send_completion(result, "Operation completed")
    return result
```

## Benefits

1. **Standardized Format**: All websocket messages follow the same structure
2. **Reusable**: Can be used across different project components
3. **Consistent**: Same progress tracking interface everywhere
4. **Maintainable**: Centralized websocket logic
5. **Extensible**: Easy to add new message types or features

## Error Handling

The utilities include built-in error handling:

```python
try:
    await progress_tracker.send_progress_update("processing", 0.5, "Working...")
except Exception as e:
    print(f"Failed to send progress update: {e}")
    # Continue execution - websocket failure shouldn't stop the main operation
```

## Best Practices

1. **Always initialize** the progress tracker at the start of operations
2. **Send connection confirmation** immediately after websocket connection
3. **Use descriptive progress messages** that help users understand what's happening
4. **Handle websocket errors gracefully** - don't let them stop main operations
5. **Clean up resources** when operations complete or fail
6. **Use appropriate message types** for different kinds of updates
7. **Keep progress percentages meaningful** and roughly accurate
