"""
Message Handler for processing socket commands.
Implements only the message types specified in the README.
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime
from typing import Dict, Any, TYPE_CHECKING

from .protocols import (
    MessageType, create_error_message, create_message
)
from .process_monitor import ProcessMonitor

if TYPE_CHECKING:
    from .socket_service import SocketService

logger = logging.getLogger(__name__)

class MessageHandler:
    """
    Handles socket messages according to README specification.
    Only supports: setup messages, heartbeat, and error handling.
    """
    
    def __init__(self, socket_service: 'SocketService'):
        self.socket_service = socket_service
        self.process_monitor = ProcessMonitor(socket_service)
        
        # Import setup function
        self._import_setup_function()
    
    def _import_setup_function(self):
        """Import the setup function."""
        try:
            from ..core.workflow_setup import setup_project_parallel_with_status_messages
            self.setup_project = setup_project_parallel_with_status_messages
        except ImportError as e:
            logger.error(f"Failed to import setup function: {e}")
            self.setup_project = None

    async def handle_message(self, project_short_id: str, message_data: Dict[str, Any]):
        """
        Main message handling entry point.
        Only handles messages specified in README:
        - {"type": "setup", "data": {"project_short_id": "..."}}
        - {"type": "heartbeat"}
        """
        try:
            message_type = message_data.get("type")
            
            if message_type == "setup":
                # Setup message as per README
                await self._handle_setup_message(project_short_id, message_data)
            elif message_type == "heartbeat":
                # Heartbeat for connection keep-alive
                await self._handle_heartbeat(project_short_id, message_data)
            else:
                # Unknown message type
                await self._send_error(project_short_id, f"Unsupported message type: {message_type}. Only 'setup' and 'heartbeat' are supported.")
                
        except Exception as e:
            logger.error(f"Error handling message from {project_short_id}: {e}")
            await self._send_error(project_short_id, f"Message processing error: {str(e)}")

    async def _handle_setup_message(self, project_short_id: str, message_data: Dict[str, Any]):
        """
        Handle setup message as per README specification.
        Expected format: {"type": "setup", "data": {"project_short_id": "..."}}
        """
        try:
            data = message_data.get("data", {})
            msg_project_id = data.get("project_short_id")
            
            # Validate project_short_id matches
            if msg_project_id != project_short_id:
                await self._send_error(project_short_id, f"Project ID mismatch: expected {project_short_id}, got {msg_project_id}")
                return
            
            if not self.setup_project:
                await self._send_error(project_short_id, "Project setup function not available")
                return
            
            # Send setup start message (as per README)
            start_message = create_message(
                MessageType.SETUP_LOG,
                status=None,
                workflow_id=None,
                content="Setup start...",
                result=None
            )
            await self.socket_service.send_to_project(project_short_id, start_message)
            
            # Create WebSocket send function for setup process
            websocket_send_func = self.process_monitor.create_websocket_send_function(project_short_id)
            
            # Run setup in background task
            async def run_setup():
                try:
                    logger.info(f"Starting setup for project {project_short_id}")
                    
                    # Use the setup function with websocket logging
                    workflow_results = await self.setup_project(project_short_id, websocket_send_func)
                    
                    logger.info(f"Setup completed for project {project_short_id}, processing {len(workflow_results)} results")
                    
                    # Send individual completion messages for each workflow (as per README)
                    completion_count = 0
                    for workflow in workflow_results:
                        if "workflow_graph" in workflow:
                            workflow_id = workflow.get("workflow_id", "unknown")
                            workflow_graph = workflow["workflow_graph"]
                            
                            # Send completion message for this specific workflow
                            completion_message = create_message(
                                MessageType.SETUP_COMPLETE,
                                status=None,
                                workflow_id=workflow_id,
                                content=f"Workflow {workflow_id} generated successfully",
                                result=workflow_graph
                            )
                            
                            success = await self.socket_service.send_to_project(project_short_id, completion_message)
                            if success:
                                completion_count += 1
                                logger.info(f"Successfully sent completion for workflow {workflow_id}")
                            else:
                                logger.error(f"Failed to send completion for workflow {workflow_id}")
                    
                    logger.info(f"Sent {completion_count} workflow completion messages for project {project_short_id}")
                    
                except Exception as e:
                    logger.error(f"Setup failed for project {project_short_id}: {e}")
                    # Send error message (as per README - uses setup-complete type for errors)
                    error_message = create_message(
                        MessageType.SETUP_COMPLETE,
                        status="error",
                        workflow_id=None,
                        content=f"Setup failed: {str(e)}",
                        result=None
                    )
                    await self.socket_service.send_to_project(project_short_id, error_message)
            
            # Start setup as background task
            asyncio.create_task(run_setup())
            
        except Exception as e:
            logger.error(f"Error handling setup message: {e}")
            await self._send_error(project_short_id, f"Setup message processing error: {str(e)}")

    async def _handle_heartbeat(self, project_short_id: str, message_data: Dict[str, Any]):
        """Handle heartbeat messages for connection keep-alive."""
        try:
            # Update last ping time
            if project_short_id in self.socket_service.active_connections:
                self.socket_service.active_connections[project_short_id]["last_ping"] = datetime.now()
            
            # Send heartbeat response
            heartbeat_response = {
                "type": "heartbeat",
                "data": {
                    "status": "alive",
                    "timestamp": datetime.now().isoformat()
                }
            }
            
            await self.socket_service.send_to_project(project_short_id, heartbeat_response)
            
        except Exception as e:
            logger.error(f"Error handling heartbeat: {e}")

    async def _send_error(self, project_short_id: str, error_message: str):
        """Send error message to client."""
        try:
            error_msg = create_error_message(error_message)
            await self.socket_service.send_to_project(project_short_id, error_msg)
        except Exception as e:
            logger.error(f"Failed to send error message: {e}")