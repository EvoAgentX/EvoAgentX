"""
Simplified process monitor for WebSocket logging support.
Only provides websocket send function creation for setup logging.
"""

import json
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .socket_service import SocketService

# Configure logging for this module
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Set to DEBUG level for troubleshooting

# Add console handler if none exists
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

class ProcessMonitor:
    """
    Simplified process monitor that only provides websocket send function creation.
    Used to create websocket send functions for setup logging.
    """
    
    def __init__(self, socket_service: 'SocketService'):
        self.socket_service = socket_service
    
    def create_websocket_send_function(self, project_short_id: str):
        """
        Create a websocket send function compatible with existing workflow execution.
        This allows existing workflow code to send messages through our socket service.
        """
        async def websocket_send_func(message_input):
            """
            Function that mimics the existing websocket send function.
            Accepts both JSON strings and already-formatted message objects.
            """
            try:
                # Handle different input formats
                if isinstance(message_input, dict):
                    # Already a formatted message object - send directly
                    await self.socket_service.send_to_project(project_short_id, message_input)
                elif isinstance(message_input, str):
                    # Try to parse as JSON first
                    try:
                        message_data = json.loads(message_input)
                        
                        # If it's already a properly formatted message, send it directly
                        if isinstance(message_data, dict) and "type" in message_data:
                            await self.socket_service.send_to_project(project_short_id, message_data)
                        else:
                            # If it's not properly formatted, wrap it in setup-log format
                            log_message = {
                                "type": "setup-log",
                                "data": {
                                    "workflow_id": None,
                                    "content": str(message_data),
                                    "result": None
                                }
                            }
                            await self.socket_service.send_to_project(project_short_id, log_message)
                            
                    except json.JSONDecodeError:
                        # If not JSON, treat as plain text log message
                        log_message = {
                            "type": "setup-log",
                            "data": {
                                "workflow_id": None,
                                "content": message_input,
                                "result": None
                            }
                        }
                        await self.socket_service.send_to_project(project_short_id, log_message)
                else:
                    # Convert other types to string and wrap
                    log_message = {
                        "type": "setup-log",
                        "data": {
                            "workflow_id": None,
                            "content": str(message_input),
                            "result": None
                        }
                    }
                    await self.socket_service.send_to_project(project_short_id, log_message)
                    
            except Exception as e:
                logger.error(f"Error sending websocket message for {project_short_id}: {e}")
                # Try to send error message to client
                try:
                    error_message = {
                        "type": "error",
                        "data": {
                            "workflow_id": None,
                            "content": f"Failed to send message: {str(e)}",
                            "result": None
                        }
                    }
                    await self.socket_service.send_to_project(project_short_id, error_message)
                except Exception as send_error:
                    logger.error(f"Failed to send error message to {project_short_id}: {send_error}")
        
        return websocket_send_func