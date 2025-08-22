"""
Main Socket Service for managing project-specific WebSocket connections.
Integrates with the existing FastAPI server to provide real-time communication.
"""

import asyncio
import json
import logging
import uuid
import websockets
from datetime import datetime
from typing import Dict, Any, Optional, Set, List
from fastapi import WebSocket, WebSocketDisconnect

from .protocols import create_error_message
from .message_store import message_store

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

class SocketService:
    """
    Main service for managing project-specific socket connections.
    
    Features:
    - One socket per project_short_id
    - Real-time message broadcasting
    - Connection health monitoring
    - Configurable socket timeouts (6000s default, -1 for no expiry)
    - Integration with existing services
    """
    
    def __init__(self, default_timeout: int = 18000):  # Increased to 5 hours for long-running setups
        """Initialize the SocketService."""
        self.active_connections = {}
        self.default_timeout = default_timeout  # 18000 seconds default (5 hours), -1 for no expiry
        
        # Connection handlers for each project
        self.connection_handlers: Dict[str, List[asyncio.Task]] = {}
        
        # Background tasks for each project
        self.background_tasks: Dict[str, List[asyncio.Task]] = {}
        
        logger.info(f"SocketService initialized with default timeout: {default_timeout}s")
    
    async def register_project_socket(self, project_short_id: str, websocket: WebSocket) -> bool:
        """Register a new WebSocket connection for a project."""
        try:
            logger.info(f"Starting socket registration for project {project_short_id}")
            # Note: websocket.accept() is already called in the API endpoint
            logger.info(f"WebSocket already accepted for project {project_short_id}")

            # Close existing connection if any
            if project_short_id in self.active_connections:
                logger.info(f"Closing existing connection for project {project_short_id}")
                await self.disconnect_project(project_short_id)

            # Store new connection with timeout configuration
            self.active_connections[project_short_id] = {
                "socket": websocket,
                "connected_at": datetime.now(),
                "timeout": self.default_timeout,
                "is_processing": False  # Track if connection is actively processing
            }
            logger.info(f"Stored connection for project {project_short_id}")

            logger.info(f"Project {project_short_id} registered successfully")

            # Send connection confirmation message
            try:
                from .protocols import create_message, MessageType
                connection_message = create_message(
                    MessageType.SETUP_LOG,
                    status=None,
                    workflow_id=None,
                    content="WebSocket connection established",
                    result=None
                )
                await self.send_to_project(project_short_id, connection_message)
                logger.info(f"Connection confirmation sent to {project_short_id}")
            except Exception as e:
                logger.warning(f"Could not send connection confirmation to {project_short_id}: {e}")

            # Wait a moment for WebSocket to be fully ready
            await asyncio.sleep(0.1)

            # Verify the WebSocket is still in a valid state before starting listener
            try:
                # Check if WebSocket is still valid
                if hasattr(websocket, 'client_state'):
                    if websocket.client_state.name == 'DISCONNECTED':
                        logger.error(f"WebSocket disconnected before starting listener for project {project_short_id}")
                        await self.disconnect_project(project_short_id)
                        return False
                
                logger.info(f"WebSocket state verified for project {project_short_id}")
            except Exception as e:
                logger.warning(f"Could not verify WebSocket state for project {project_short_id}: {e}")

            # Start continuous message listening on a separate thread
            logger.info(f"Starting message listener for project {project_short_id}")
            self._start_message_listener(project_short_id, websocket)
            logger.info(f"Message listener started for project {project_short_id}")

            return True

        except Exception as e:
            logger.error(f"Error registering project {project_short_id}: {e}")
            import traceback
            logger.error(f"Registration error traceback: {traceback.format_exc()}")
            return False
    
    def _start_message_listener(self, project_short_id: str, websocket: WebSocket):
        """Start a message listener for a specific project."""
        async def message_listener():
            """Listen for messages from the client."""
            try:
                logger.info(f"Starting message listener for project {project_short_id}")
                logger.debug(f"Waiting for message from client {project_short_id}")
                
                while True:
                    try:
                        # Wait for message with timeout
                        try:
                            message = await asyncio.wait_for(websocket.receive_text(), timeout=1.0)
                        except asyncio.TimeoutError:
                            # No message received, continue listening
                            continue
                        except Exception as e:
                            logger.error(f"Error receiving message from {project_short_id}: {e}")
                            break
                        
                        if message:
                            logger.debug(f"Raw message received from client {project_short_id}: {message}")
                            
                            # Parse and handle the message
                            try:
                                message_data = json.loads(message)
                                logger.info(f"Parsed message from client {project_short_id}: {message_data.get('type', 'unknown')}")
                                
                                # Route the message to the message handler
                                await self._handle_message_with_handler(project_short_id, message_data)
                                
                            except json.JSONDecodeError as e:
                                logger.error(f"Invalid JSON from client {project_short_id}: {e}")
                                error_message = {
                                    "type": "error",
                                    "data": {
                                        "status": "error",
                                        "workflow_id": None,
                                        "content": f"Invalid JSON message: {str(e)}",
                                        "result": None
                                    }
                                }
                                await self.send_to_project(project_short_id, error_message)
                            except Exception as e:
                                logger.error(f"Error handling message from client {project_short_id}: {e}")
                                import traceback
                                logger.error(f"Message handling error traceback: {traceback.format_exc()}")
                                
                                # Try to send error message
                                try:
                                    error_message = {
                                        "type": "error",
                                        "data": {
                                            "status": "error",
                                            "workflow_id": None,
                                            "content": f"Message processing error: {str(e)}",
                                            "result": None
                                        }
                                    }
                                    await self.send_to_project(project_short_id, error_message)
                                except:
                                    pass  # If we can't send error message, just log it
                                
                    except Exception as e:
                        logger.error(f"Error in message listener loop for {project_short_id}: {e}")
                        break
                        
            except Exception as e:
                logger.error(f"Fatal error in message listener for project {project_short_id}: {e}")
            finally:
                # Clean up connection when listener exits
                logger.info(f"Message listener exiting for project {project_short_id}")
                if project_short_id in self.active_connections:
                    await self.disconnect_project(project_short_id)
        
        # Create and store the listener task
        listener_task = asyncio.create_task(message_listener())
        
        if project_short_id not in self.connection_handlers:
            self.connection_handlers[project_short_id] = []
        self.connection_handlers[project_short_id].append(listener_task)
        
        logger.info(f"Message listener started for project {project_short_id}")
    
    def is_websocket_connected(self, websocket: WebSocket) -> bool:
        """Check if a WebSocket is still connected."""
        try:
            # Try to check the connection state - different approaches for different FastAPI versions
            if hasattr(websocket, 'client_state'):
                if websocket.client_state.name not in ['CONNECTED', 'CONNECTING']:
                    logger.debug(f"WebSocket state: {websocket.client_state.name}")
                    return False
            # If we can't check state, assume connected
            return True
        except Exception as e:
            logger.debug(f"Error checking WebSocket state: {e}")
            return True  # Assume connected if we can't check
    
    async def is_websocket_connected_async(self, websocket: WebSocket) -> bool:
        """Async check if a WebSocket is still connected by trying to send a ping."""
        try:
            # Try to send a ping to check if connection is alive
            await websocket.send_text('{"type": "ping", "data": {"content": "ping"}}')
            return True
        except Exception as e:
            logger.debug(f"WebSocket ping failed: {e}")
            return False
    
    async def disconnect_project(self, project_short_id: str):
        """Disconnect a project socket."""
        if project_short_id in self.active_connections:
            try:
                socket = self.active_connections[project_short_id]["socket"]
                await socket.close()
            except:
                pass  # Socket might already be closed
            
            # Clean up connection handlers
            if project_short_id in self.connection_handlers:
                for task in self.connection_handlers[project_short_id]:
                    if not task.done():
                        task.cancel()
                del self.connection_handlers[project_short_id]
            
            # Clean up background tasks
            if project_short_id in self.background_tasks:
                for task in self.background_tasks[project_short_id]:
                    if not task.done():
                        task.cancel()
                del self.background_tasks[project_short_id]
            
            del self.active_connections[project_short_id]
            logger.info(f"Project {project_short_id} disconnected")
    
    async def send_to_project(self, project_short_id: str, message: Dict[str, Any], max_retries: int = 3) -> bool:
        """Send message to a specific project via WebSocket with retry mechanism."""
        if project_short_id not in self.active_connections:
            logger.warning(f"No active connection for project {project_short_id}")
            return False
        
        
        connection_info = self.active_connections[project_short_id]
        websocket = connection_info["socket"]
        
        for attempt in range(max_retries):
            try:
                # Check if WebSocket is still in a valid state
                if hasattr(websocket, 'client_state'):
                    if websocket.client_state.name in ['DISCONNECTED', 'CLOSING', 'CLOSED']:
                        logger.warning(f"WebSocket {websocket.client_state.name} for {project_short_id}")
                        return False
                
                # Send the message
                
                await websocket.send_text(json.dumps(message))
                logger.debug(f"Successfully sent message to {project_short_id} (attempt {attempt + 1})")
                return True
                
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Failed to send message to {project_short_id} (attempt {attempt + 1}): {e}")
                    await asyncio.sleep(0.1 * (attempt + 1))  # Exponential backoff
                else:
                    logger.error(f"Failed to send message to {project_short_id} after {max_retries} attempts: {e}")
                    return False
        
        return False
    
    async def broadcast_to_all(self, message: Dict[str, Any]):
        """Broadcast message to all connected projects."""
        disconnected = []
        for project_short_id in list(self.active_connections.keys()):
            success = await self.send_to_project(project_short_id, message)
            if not success:
                disconnected.append(project_short_id)
        
        # Clean up disconnected projects
        for project_short_id in disconnected:
            await self.disconnect_project(project_short_id)
    
    def get_active_projects(self) -> Set[str]:
        """Get list of currently connected projects."""
        return set(self.active_connections.keys())
    
    def is_project_connected(self, project_short_id: str) -> bool:
        """Check if a project is currently connected."""
        return project_short_id in self.active_connections
    
    async def handle_message(self, project_short_id: str, message_data: Dict[str, Any]):
        """Handle incoming message from a project."""
        logger.info(f"Handling message from project {project_short_id}: {message_data.get('type', 'unknown')}")
        
        # Store incoming message
        message_store.store_incoming_message(project_short_id, message_data)
        
        # Import here to avoid circular imports
        from .message_handler import MessageHandler
        
        # Create handler instance if not exists
        if not hasattr(self, '_message_handler'):
            logger.info(f"Creating new MessageHandler instance for project {project_short_id}")
            self._message_handler = MessageHandler(self)
        else:
            logger.info(f"Using existing MessageHandler instance for project {project_short_id}")
        
        try:
            logger.info(f"Calling MessageHandler.handle_message for project {project_short_id}")
            await self._message_handler.handle_message(project_short_id, message_data)
            logger.info(f"MessageHandler.handle_message completed for project {project_short_id}")
        except Exception as e:
            logger.error(f"Error in MessageHandler.handle_message for project {project_short_id}: {e}")
            import traceback
            logger.error(f"MessageHandler error traceback: {traceback.format_exc()}")
            raise
    
    async def cleanup_inactive_connections(self):
        """Clean up inactive connections based on timeout."""
        current_time = datetime.now()
        inactive_projects = []
        
        for project_short_id, conn_info in self.active_connections.items():
            timeout = conn_info.get("timeout", self.default_timeout)
            
            # Skip cleanup if timeout is -1 (no expiry)
            if timeout == -1:
                continue
            
            # Skip cleanup if connection is actively processing
            if conn_info.get("is_processing", False):
                logger.debug(f"Skipping cleanup for {project_short_id} - actively processing")
                continue
                
        
        for project_short_id in inactive_projects:
            await self.disconnect_project(project_short_id)
            logger.info(f"Cleaned up inactive connection for {project_short_id} (timeout: {self.active_connections.get(project_short_id, {}).get('timeout', 'unknown')}s)")
    
    def update_socket_timeout(self, project_short_id: str, timeout: int) -> bool:
        """Update socket timeout for a specific project."""
        if project_short_id in self.active_connections:
            self.active_connections[project_short_id]["timeout"] = timeout
            logger.info(f"Updated timeout for project {project_short_id} to {timeout}s")
            return True
        return False
    
    def get_socket_timeout(self, project_short_id: str) -> Optional[int]:
        """Get socket timeout for a specific project."""
        if project_short_id in self.active_connections:
            return self.active_connections[project_short_id].get("timeout", self.default_timeout)
        return None
    
    async def get_project_by_workflow_id(self, workflow_id: str) -> Optional[str]:
        """Get project_short_id by workflow_id from database."""
        try:
            # Import here to avoid circular imports
            from ..database.db import database
            workflow = await database.find_one("workflows", {"id": workflow_id})
            return workflow.get("project_short_id") if workflow else None
        except Exception as e:
            logger.error(f"Error getting project by workflow ID {workflow_id}: {e}")
            return None
    
    async def send_to_workflow(self, workflow_id: str, message: Dict[str, Any]) -> bool:
        """Send message to socket associated with a workflow."""
        project_short_id = await self.get_project_by_workflow_id(workflow_id)
        if project_short_id:
            return await self.send_to_project(project_short_id, message)
        logger.warning(f"No project found for workflow {workflow_id}")
        return False
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get connection statistics."""
        connections_with_timeout = []
        for project_id, conn_info in self.active_connections.items():
            connections_with_timeout.append({
                "project_id": project_id,
                "connected_at": conn_info["connected_at"].isoformat(),
                "timeout": conn_info.get("timeout", self.default_timeout),
                "last_ping": None
            })
        
        return {
            "active_connections": len(self.active_connections),
            "connected_projects": list(self.active_connections.keys()),
            "connections_detail": connections_with_timeout,
            "default_timeout": self.default_timeout,
            "service_status": "healthy"
        }
    
    async def start_all_monitoring(self):
        """Start all monitoring background tasks."""
        logger.info("All socket monitoring services started")
    
    async def _start_cleanup_task(self):
        """Start background cleanup task for inactive connections."""
        async def cleanup_loop():
            while True:
                try:
                    await asyncio.sleep(300)  # Check every 5 minutes
                    await self.cleanup_inactive_connections()
                    logger.debug("Completed periodic socket cleanup check")
                except Exception as e:
                    logger.error(f"Error in cleanup loop: {e}")
        
    
    async def cleanup(self):
        """Clean up the service and stop background tasks."""
        
        # Clean up all connection handlers
        for project_short_id in list(self.connection_handlers.keys()):
            for task in self.connection_handlers[project_short_id]:
                if not task.done():
                    task.cancel()
            self.connection_handlers[project_short_id].clear()
        
        # Clean up all background tasks
        for project_short_id in list(self.background_tasks.keys()):
            for task in self.background_tasks[project_short_id]:
                if not task.done():
                    task.cancel()
            self.background_tasks[project_short_id].clear()
        
        # Disconnect all active connections
        for project_short_id in list(self.active_connections.keys()):
            await self.disconnect_project(project_short_id)
        
        logger.info("SocketService cleanup completed")

    def mark_connection_active(self, project_short_id: str, is_active: bool = True):
        """Mark a connection as actively processing (prevents timeout cleanup during operations)."""
        if project_short_id in self.active_connections:
            self.active_connections[project_short_id]["is_processing"] = is_active
            if is_active:
                # Reset the last ping time when starting an operation
                pass
            logger.debug(f"Marked connection {project_short_id} as {'active' if is_active else 'inactive'}")
    
    def is_connection_processing(self, project_short_id: str) -> bool:
        """Check if a connection is currently processing an operation."""
        if project_short_id in self.active_connections:
            return self.active_connections[project_short_id].get("is_processing", False)
        return False

    def is_connection_healthy(self, project_short_id: str) -> bool:
        """Check if a WebSocket connection is healthy and ready to send messages."""
        if project_short_id not in self.active_connections:
            return False
        
        connection_info = self.active_connections[project_short_id]
        websocket = connection_info["socket"]
        
        try:
            # Check WebSocket state
            if hasattr(websocket, 'client_state'):
                if websocket.client_state.name in ['DISCONNECTED', 'CLOSING', 'CLOSED']:
                    return False
            
            # Check if connection is marked as processing (active)
            if connection_info.get("is_processing", False):
                return True
            
            pass
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking connection health for {project_short_id}: {e}")
            return False

    def extend_timeout_for_processing(self, project_short_id: str, additional_timeout: int = 3600):
        """Extend timeout for connections that are actively processing long-running operations."""
        if project_short_id in self.active_connections:
            current_timeout = self.active_connections[project_short_id].get("timeout", self.default_timeout)
            if current_timeout != -1:  # Don't extend if already unlimited
                new_timeout = current_timeout + additional_timeout
                self.active_connections[project_short_id]["timeout"] = new_timeout
                logger.info(f"Extended timeout for {project_short_id} from {current_timeout}s to {new_timeout}s")
                return True
        return False

    def refresh_connection_state(self, project_short_id: str):
        """Refresh connection state to prevent timeout during long-running operations."""
        if project_short_id in self.active_connections:
            pass
            
            # If connection is processing, extend timeout if needed
            if self.active_connections[project_short_id].get("is_processing", False):
                current_timeout = self.active_connections[project_short_id].get("timeout", self.default_timeout)
                if current_timeout != -1:
                    # Add 30 minutes to timeout
                    new_timeout = current_timeout + 1800
                    self.active_connections[project_short_id]["timeout"] = new_timeout
                    logger.debug(f"Refreshed connection state for {project_short_id}, extended timeout to {new_timeout}s")
            
            logger.debug(f"Refreshed connection state for {project_short_id}")
            return True
        return False

    async def _handle_message_with_handler(self, project_short_id: str, message_data: Dict[str, Any]):
        """
        Helper to handle messages by delegating to the MessageHandler.
        This method is called by the message listener to process messages.
        """
        # Store incoming message
        message_store.store_incoming_message(project_short_id, message_data)
        
        # Import here to avoid circular imports
        from .message_handler import MessageHandler
        
        # Create handler instance if not exists
        if not hasattr(self, '_message_handler'):
            logger.info(f"Creating new MessageHandler instance for project {project_short_id}")
            self._message_handler = MessageHandler(self)
        else:
            logger.info(f"Using existing MessageHandler instance for project {project_short_id}")
        
        try:
            logger.info(f"Calling MessageHandler.handle_message for project {project_short_id}")
            await self._message_handler.handle_message(project_short_id, message_data)
            logger.info(f"MessageHandler.handle_message completed for project {project_short_id}")
        except Exception as e:
            logger.error(f"Error in MessageHandler.handle_message for project {project_short_id}: {e}")
            import traceback
            logger.error(f"MessageHandler error traceback: {traceback.format_exc()}")
            raise

    async def check_connection_health(self, project_short_id: str):
        """Check if a WebSocket connection is healthy and ready to send messages."""
        if project_short_id not in self.active_connections:
            return False
        
        connection_info = self.active_connections[project_short_id]
        websocket = connection_info["socket"]
        
        try:
            # Check WebSocket state
            if hasattr(websocket, 'client_state'):
                if websocket.client_state.name in ['DISCONNECTED', 'CLOSING', 'CLOSED']:
                    return False
            
            # Check if connection is marked as processing (active)
            if connection_info.get("is_processing", False):
                return True
            
            # Since we removed heartbeat, just check if connection is still valid
            return True
            
        except Exception as e:
            logger.error(f"Error checking connection health for {project_short_id}: {e}")
            return False

# Global socket service instance
socket_service = SocketService()
