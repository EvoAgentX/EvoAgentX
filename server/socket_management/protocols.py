"""
Message protocols for socket communication.
Only includes message types specified in README.
"""

from enum import Enum
from typing import Dict, Any, List
from datetime import datetime

class MessageType(str, Enum):
    """Message types as specified in README."""
    # Setup phase messages (README compliant)
    SETUP_LOG = "setup-log"             # Setup phase log messages
    SETUP_COMPLETE = "setup-complete"   # Setup phase completion
    
    # Execution phase messages (README compliant)  
    RUNTIME_LOG = "runtime-log"         # Execution phase log messages
    
    # Connection messages
    ERROR = "error"                     # Error messages

def create_message(message_type: MessageType, status: str = None, workflow_id: str = None, 
                  content: str = "", result: Any = None) -> Dict[str, Any]:
    """
    Create a message in the README-specified format.
    
    Format:
    {
        "type": "message_type",
        "data": {
            "workflow_id": "workflow_id_or_null", 
            "content": "human_readable_message",
            "result": "actual_data_or_null"
        }
    }
    """
    return {
        "type": message_type,
        "data": {
            "workflow_id": workflow_id,
            "content": content,
            "result": result
        }
    }



def create_error_message(error_content: str) -> Dict[str, Any]:
    """Create error message."""
    return create_message(
        MessageType.ERROR,
        status="error",
        workflow_id=None,
        content=error_content,
        result=None
    )