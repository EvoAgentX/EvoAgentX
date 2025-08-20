"""
Simplified process monitor for WebSocket logging support.
Only provides websocket send function creation for setup logging.
"""

import json
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .socket_service import SocketService

logger = logging.getLogger(__name__)

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
        async def websocket_send_func(message_str: str):
            """
            Function that mimics the existing websocket send function.
            Parses the message string and sends it via our socket service.
            """
            try:
                # Try to parse as JSON first
                try:
                    message_data = json.loads(message_str)
                    await self.socket_service.send_to_project(project_short_id, message_data)
                except json.JSONDecodeError:
                    # If not JSON, treat as plain text log message
                    log_message = {
                        "type": "setup-log",
                        "data": {
                            "workflow_id": None,
                            "content": message_str,
                            "result": None
                        }
                    }
                    await self.socket_service.send_to_project(project_short_id, log_message)
                    
            except Exception as e:
                logger.error(f"Error sending websocket message for {project_short_id}: {e}")
        
        return websocket_send_func