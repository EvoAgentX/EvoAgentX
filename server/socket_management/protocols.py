"""
Socket communication protocols and message formats.
Uses the existing EvoAgentX message format for compatibility.
"""

from enum import Enum
from typing import Dict, Any, Optional
from datetime import datetime

class MessageType(str, Enum):
    """Types of messages that can be sent via socket - matches new specification."""
    # Setup phase messages
    SETUP_LOG = "setup-log"             # Setup phase log messages
    SETUP_COMPLETE = "setup-complete"   # Setup phase completion
    
    # Execution phase messages  
    RUNTIME_LOG = "runtime-log"         # Execution phase log messages
    
    # Legacy compatibility (keeping for backward compatibility)
    CONNECTION = "connection"           # Initial connection confirmation
    WORKFLOW_STATUS = "workflow_status" # Workflow status updates
    EXECUTION_COMPLETE = "execution_complete"  # Workflow execution completion
    ERROR = "error"                     # Error messages
    PROGRESS = "progress"               # Progress updates (from workflow execution)
    LOG = "log"                         # Log messages (from workflow execution)
    OUTPUT = "output"                   # Output messages (from workflow execution)
    INPUT = "input"                     # Input messages (from workflow execution)
    COMPLETE = "complete"               # Final completion (from workflow execution)
    START = "start"                     # Execution start (from workflow execution)

class CommandType(str, Enum):
    """Available command types for client requests."""
    # Project management
    PROJECT_SETUP = "project.setup"
    PROJECT_STATUS = "project.status"
    
    # Workflow management
    WORKFLOW_EXECUTE = "workflow.execute"
    WORKFLOW_STOP = "workflow.stop"
    WORKFLOW_LIST = "workflow.list"
    WORKFLOW_STATUS = "workflow.status"
    
    # Query processing
    QUERY_ANALYZE = "query.analyze"
    
    # System commands
    SYSTEM_HEALTH = "system.health"
    HEARTBEAT = "heartbeat"

class WorkflowStatus(str, Enum):
    """Workflow status values - matches existing database statuses."""
    UNINITIALIZED = "uninitialized"
    PENDING = "pending"
    RUNNING = "running"
    COMPLETE = "complete"
    COMPLETED = "completed"
    ERROR = "error"
    FAILED = "failed"
    CONNECTED = "connected"

def create_message(message_type: MessageType, status: str = None, workflow_id: str = None, 
                  content: str = "", result: Any = None) -> Dict[str, Any]:
    """
    Create a message in the existing EvoAgentX format.
    
    Format:
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
    return {
        "type": message_type,
        "data": {
            "status": status,
            "workflow_id": workflow_id,
            "content": content,
            "result": result
        }
    }

def create_connection_message(project_short_id: str) -> Dict[str, Any]:
    """Create connection confirmation message."""
    return create_message(
        MessageType.CONNECTION,
        status="connected",
        workflow_id=None,
        content="Socket connection established for project",
        result={"project_short_id": project_short_id}
    )

def create_workflow_status_message(status: str, workflow_id: str, content: str, result: Any = None) -> Dict[str, Any]:
    """Create workflow status update message."""
    return create_message(
        MessageType.WORKFLOW_STATUS,
        status=status,
        workflow_id=workflow_id,
        content=content,
        result=result
    )

def create_setup_complete_message(workflow_graphs: list) -> Dict[str, Any]:
    """Create setup completion message."""
    return create_message(
        MessageType.SETUP_COMPLETE,
        status="complete",
        workflow_id=None,
        content="setup successful",
        result=workflow_graphs
    )

def create_execution_complete_message(execution_result: Dict[str, Any]) -> Dict[str, Any]:
    """Create execution completion message."""
    return create_message(
        MessageType.EXECUTION_COMPLETE,
        status="complete",
        workflow_id=None,
        content="execution successful",
        result=execution_result
    )

def create_error_message(error_message: str, workflow_id: str = None) -> Dict[str, Any]:
    """Create error message."""
    return create_message(
        MessageType.ERROR,
        status="error",
        workflow_id=workflow_id,
        content=error_message,
        result=None
    )

def create_setup_log_message(workflow_id: str, content: str, result: Any = None) -> Dict[str, Any]:
    """Create setup log message according to new specification."""
    return {
        "type": "setup-log",
        "data": {
            "workflow_id": workflow_id,
            "content": content,
            "result": result
        }
    }

def create_setup_complete_message_new(workflow_id: str, content: str, result: Any) -> Dict[str, Any]:
    """Create setup complete message according to new specification."""
    return {
        "type": "setup-complete",
        "data": {
            "workflow_id": workflow_id,
            "content": content,
            "result": result
        }
    }

def create_runtime_log_message(workflow_id: str, content: str, result: Any = None) -> Dict[str, Any]:
    """Create runtime log message according to new specification."""
    return {
        "type": "runtime-log",
        "data": {
            "workflow_id": workflow_id,
            "content": content,
            "result": result
        }
    }

def create_connection_confirmation_message() -> Dict[str, Any]:
    """Create connection confirmation message according to new specification."""
    return {
        "type": "setup-log",
        "data": {
            "workflow_id": None,
            "content": "WebSocket connection established",
            "result": None
        }
    }

def create_setup_start_message() -> Dict[str, Any]:
    """Create setup start message according to new specification."""
    return {
        "type": "setup-log",
        "data": {
            "workflow_id": None,
            "content": "Setup start...",
            "result": None
        }
    }

# Example messages for documentation
EXAMPLE_MESSAGES = {
    "connection": {
        "type": "connection",
        "data": {
            "status": "connected",
            "workflow_id": None,
            "content": "Socket connection established for project",
            "result": {"project_short_id": "my_project"}
        }
    },
    
    "workflow_status_uninitialized": {
        "type": "workflow_status",
        "data": {
            "status": "uninitialized",
            "workflow_id": "workflow-uuid-here",
            "content": "workflow extracted",
            "result": None
        }
    },
    
    "workflow_status_pending": {
        "type": "workflow_status",
        "data": {
            "status": "pending", 
            "workflow_id": "workflow-uuid-here",
            "content": "workflow generated",
            "result": {"workflow_data": "..."}
        }
    },
    
    "setup_complete": {
        "type": "setup_complete",
        "data": {
            "status": "complete",
            "workflow_id": None,
            "content": "setup successful",
            "result": [{"workflow_graph": "..."}]
        }
    },
    
    "execution_complete": {
        "type": "execution_complete",
        "data": {
            "status": "complete",
            "workflow_id": None,
            "content": "execution successful", 
            "result": {"parsed_json": "...", "execution_time": "..."}
        }
    },
    
    "error": {
        "type": "error",
        "data": {
            "status": "error",
            "workflow_id": None,
            "content": "Error message here",
            "result": None
        }
    }
}
