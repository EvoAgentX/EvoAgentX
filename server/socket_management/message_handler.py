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

class MessageHandler:
    """
    Handles socket messages according to README specification.
    Only supports: setup messages and error handling.
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
            logger.info("Successfully imported setup_project_parallel_with_status_messages function")
        except ImportError as e:
            logger.error(f"Failed to import setup function: {e}")
            self.setup_project = None
        except Exception as e:
            logger.error(f"Unexpected error importing setup function: {e}")
            self.setup_project = None

    async def handle_message(self, project_short_id: str, message_data: Dict[str, Any]):
        """
        Main message handling entry point.
        Handles all message types and routes them appropriately.
        """
        try:
            message_type = message_data.get("type")
            logger.info(f"Handling message from project {project_short_id}: {message_type}")
            
            if message_type == "setup":
                # Handle setup message non-blocking - start in background, don't await
                self._handle_setup_message_non_blocking(project_short_id, message_data)
            else:
                # Unknown message type
                await self._send_error(project_short_id, f"Unsupported message type: {message_type}. Only 'setup' is supported.")
                
        except Exception as e:
            logger.error(f"Error handling message from {project_short_id}: {e}")
            await self._send_error(project_short_id, f"Message processing error: {str(e)}")

    def _handle_setup_message_non_blocking(self, project_short_id: str, message_data: Dict[str, Any]):
        """
        Handle setup message non-blocking - creates new thread for workflow generation.
        Expected format: {"type": "setup", "data": {"project_short_id": "..."}}
        """
        try:
            logger.info(f"Processing setup message for project {project_short_id}")
            data = message_data.get("data", {})
            msg_project_id = data.get("project_short_id")
            
            logger.info(f"Setup message data: {data}")
            logger.info(f"Message project ID: {msg_project_id}, expected: {project_short_id}")
            
            # Validate project_short_id matches
            if msg_project_id != project_short_id:
                logger.error(f"Project ID mismatch: expected {project_short_id}, got {msg_project_id}")
                # Schedule error message to be sent asynchronously
                asyncio.create_task(self._send_error(project_short_id, f"Project ID mismatch: expected {project_short_id}, got {msg_project_id}"))
                return
            
            if not self.setup_project:
                logger.error("Project setup function not available")
                # Schedule error message to be sent asynchronously
                asyncio.create_task(self._send_error(project_short_id, "Project setup function not available"))
                return
            
            logger.info(f"Setup function available, starting setup process for project {project_short_id}")
            
            # Start setup in background task (non-blocking)
            logger.info(f"Starting setup background task for project {project_short_id}")
            
            status_response = create_message(
                MessageType.SETUP_LOG,
                status="processing",
                workflow_id=None,
                content="Start creating setup worker",
                result={"timestamp": datetime.now().isoformat()}
            )
            asyncio.create_task(self.socket_service.send_to_project(project_short_id, status_response))
            
            
            # Create and start the setup task that handles all async operations
            setup_task = asyncio.create_task(self._run_setup_worker_with_initialization(project_short_id, message_data))
            
            # Store the setup task for potential management
            if not hasattr(self, '_setup_tasks'):
                self._setup_tasks = {}
            self._setup_tasks[project_short_id] = setup_task
            
            logger.info(f"Setup background task started for project {project_short_id}")
            # Return immediately - main thread continues listening
            
        except Exception as e:
            logger.error(f"Error handling setup message: {e}")
            import traceback
            logger.error(f"Setup message handling error traceback: {traceback.format_exc()}")
            # Schedule error message to be sent asynchronously
            asyncio.create_task(self._send_error(project_short_id, f"Setup message processing error: {str(e)}"))
    
    async def _run_setup_worker_with_initialization(self, project_short_id: str, message_data: Dict[str, Any]):
        """Background worker that handles initial setup messages and then runs the actual setup process."""
        try:
            logger.info(f"Starting setup worker with initialization for project {project_short_id}")
            
            # Send setup start message (as per README)
            start_message = create_message(
                MessageType.SETUP_LOG,
                status=None,
                workflow_id=None,
                content="Setup start...",
                result=None
            )
            logger.info(f"Sending setup start message to project {project_short_id}")
            await self.socket_service.send_to_project(project_short_id, start_message)
            
            # Create WebSocket send function for setup process
            websocket_send_func = self.process_monitor.create_websocket_send_function(project_short_id)
            logger.info(f"Created WebSocket send function for project {project_short_id}")
            
            
            
            # Now run the actual setup worker
            await self._run_setup_worker(project_short_id, websocket_send_func)
            
        except Exception as e:
            logger.error(f"Error in setup worker initialization: {e}")
            import traceback
            logger.error(f"Setup worker initialization error traceback: {traceback.format_exc()}")
            await self._send_error(project_short_id, f"Setup worker initialization error: {str(e)}")

    async def _run_setup_worker(self, project_short_id: str, websocket_send_func):
        """Background worker that runs the actual setup process."""
        try:
            status_response = create_message(
                MessageType.SETUP_LOG,
                status="processing",
                workflow_id=None,
                content="In setup worker",
                result={"timestamp": datetime.now().isoformat()}
            )
            asyncio.create_task(self.socket_service.send_to_project(project_short_id, status_response))
            
            logger.info(f"Starting setup worker for project {project_short_id}")
            
            # Mark connection as actively processing to prevent timeout cleanup
            self.socket_service.mark_connection_active(project_short_id, True)
            
            # Extend timeout for long-running setup operations
            self.socket_service.extend_timeout_for_processing(project_short_id, additional_timeout=7200)  # Add 2 hours
            
            # Start periodic connection refresh task
            refresh_task = asyncio.create_task(self._periodic_connection_refresh(project_short_id))
            
            # Use the setup function with websocket logging
            workflow_results = await self.setup_project(project_short_id, websocket_send_func)
            
            # Cancel the refresh task
            refresh_task.cancel()
            
            logger.info(f"Setup completed for project {project_short_id}, processing {len(workflow_results)} results")
            
            # Send individual completion messages for each workflow (as per README)
            completion_count = 0
            workflow_graphs = []
            for workflow in workflow_results:
                if "workflow_graph" in workflow:
                    workflow_id = workflow.get("workflow_id", "unknown")
                    workflow_graph = workflow["workflow_graph"]
                    workflow_graphs.append(workflow_graph)
                    
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
            
            # Send overall setup complete message with all workflow graphs
            if workflow_graphs:
                overall_completion_message = create_message(
                    MessageType.SETUP_COMPLETE,
                    status=None,
                    workflow_id=None,
                    content=f"Project setup completed successfully with {len(workflow_graphs)} workflows",
                    result=workflow_graphs
                )
                await self.socket_service.send_to_project(project_short_id, overall_completion_message)
                logger.info(f"Sent overall setup completion message for project {project_short_id}")
            
            logger.info(f"Sent {completion_count} workflow completion messages for project {project_short_id}")
            
        except Exception as e:
            logger.error(f"Setup failed for project {project_short_id}: {e}")
            import traceback
            logger.error(f"Setup error traceback: {traceback.format_exc()}")
            # Send error message (as per README - uses setup-complete type for errors)
            error_message = create_message(
                MessageType.SETUP_COMPLETE,
                status="error",
                workflow_id=None,
                content=f"Setup failed: {str(e)}",
                result=None
            )
            await self.socket_service.send_to_project(project_short_id, error_message)
        finally:
            # Mark connection as no longer processing
            self.socket_service.mark_connection_active(project_short_id, False)
            
            # Clean up the setup task
            if hasattr(self, '_setup_tasks') and project_short_id in self._setup_tasks:
                del self._setup_tasks[project_short_id]
            logger.info(f"Setup worker completed for project {project_short_id}")
    
    async def _periodic_connection_refresh(self, project_short_id: str):
        """Periodically refresh connection state during long-running operations."""
        try:
            while True:
                await asyncio.sleep(60)  # Refresh every minute
                if self.socket_service.is_connection_processing(project_short_id):
                    self.socket_service.refresh_connection_state(project_short_id)
                    logger.debug(f"Refreshed connection state for {project_short_id}")
                else:
                    # Stop refreshing if no longer processing
                    break
        except asyncio.CancelledError:
            logger.debug(f"Connection refresh task cancelled for {project_short_id}")
        except Exception as e:
            logger.error(f"Error in connection refresh task for {project_short_id}: {e}")

    async def _send_error(self, project_short_id: str, error_message: str):
        """Send error message to client."""
        try:
            error_msg = create_error_message(error_message)
            await self.socket_service.send_to_project(project_short_id, error_msg)
        except Exception as e:
            logger.error(f"Failed to send error message: {e}")