"""
Main Socket Service for managing project-specific WebSocket connections.
Integrates with the existing FastAPI server to provide real-time communication.
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, Set
from fastapi import WebSocket, WebSocketDisconnect

from .protocols import create_error_message
from .message_store import message_store

logger = logging.getLogger(__name__)

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
    
    def __init__(self, default_timeout: int = 6000):
        # Active connections: {project_short_id: {"socket": websocket, "connected_at": datetime, "timeout": int}}
        self.active_connections: Dict[str, Dict[str, Any]] = {}
        self.default_timeout = default_timeout  # 6000 seconds default, -1 for no expiry
        
        # Heartbeat task (will be started when first connection is made)
        self._heartbeat_task = None
        
        logger.info(f"SocketService initialized with default timeout: {default_timeout}s")
    
    def _start_heartbeat_task(self):
        """Start background heartbeat task to keep connections alive."""
        async def heartbeat_loop():
            while True:
                try:
                    await asyncio.sleep(30)  # Send heartbeat every 30 seconds
                    await self._send_heartbeats()
                except Exception as e:
                    logger.error(f"Error in heartbeat loop: {e}")
        
        self._heartbeat_task = asyncio.create_task(heartbeat_loop())
    
    async def _send_heartbeats(self):
        """Send heartbeat to all active connections."""
        heartbeat_message = {
            "type": "heartbeat",
            "data": {
                "status": "alive",
                "workflow_id": None,
                "content": "heartbeat",
                "result": {"timestamp": datetime.now().isoformat()}
            }
        }
        
        disconnected = []
        for project_short_id in list(self.active_connections.keys()):
            success = await self.send_to_project(project_short_id, heartbeat_message)
            if not success:
                disconnected.append(project_short_id)
        
        # Clean up failed connections
        for project_short_id in disconnected:
            await self.disconnect_project(project_short_id)
    
    async def connect_project(self, project_short_id: str, websocket: WebSocket):
        """Connect a new project socket."""
        try:
            await websocket.accept()
            logger.info(f"WebSocket accepted for project {project_short_id}")
            
            # Close existing connection if any
            if project_short_id in self.active_connections:
                logger.info(f"Closing existing connection for project {project_short_id}")
                await self.disconnect_project(project_short_id)
            
            # Store new connection with timeout configuration
            self.active_connections[project_short_id] = {
                "socket": websocket,
                "connected_at": datetime.now(),
                "last_ping": datetime.now(),
                "timeout": self.default_timeout
            }
            
            # Start heartbeat task if this is the first connection
            if self._heartbeat_task is None:
                self._start_heartbeat_task()
                logger.info("Started heartbeat task for socket connections")
            
            logger.info(f"Project {project_short_id} connected successfully")
            
            # Wait a moment for WebSocket to be fully ready, then send connection confirmation
            await asyncio.sleep(0.1)  # Small delay to ensure connection is stable
            
            try:
                # Send connection confirmation as per README
                from .protocols import create_message, MessageType
                connection_message = create_message(
                    MessageType.SETUP_LOG,
                    status=None,
                    workflow_id=None,
                    content="WebSocket connection established",
                    result=None
                )
                success = await self.send_to_project(project_short_id, connection_message)
                
                if success:
                    logger.info(f"Connection confirmation sent to {project_short_id}")
                else:
                    logger.warning(f"Failed to send connection confirmation to {project_short_id}")
            except Exception as e:
                logger.warning(f"Could not send connection confirmation to {project_short_id}: {e}")
                
        except Exception as e:
            logger.error(f"Error connecting project {project_short_id}: {e}")
            raise
    
    async def disconnect_project(self, project_short_id: str):
        """Disconnect a project socket."""
        if project_short_id in self.active_connections:
            try:
                socket = self.active_connections[project_short_id]["socket"]
                await socket.close()
            except:
                pass  # Socket might already be closed
            
            del self.active_connections[project_short_id]
            logger.info(f"Project {project_short_id} disconnected")
    
    async def send_to_project(self, project_short_id: str, message: Dict[str, Any]) -> bool:
        """Send message to a specific project."""
        if project_short_id not in self.active_connections:
            logger.warning(f"No active connection for project {project_short_id}")
            return False
        
        try:
            # Store outgoing message
            message_store.store_outgoing_message(project_short_id, message)
            
            socket = self.active_connections[project_short_id]["socket"]
            
            # Check if socket is still connected
            try:
                # Try to check the connection state - different approaches for different FastAPI versions
                if hasattr(socket, 'client_state'):
                    if socket.client_state.name not in ['CONNECTED', 'CONNECTING']:
                        logger.warning(f"Socket not connected for project {project_short_id}, state: {socket.client_state.name}")
                        await self.disconnect_project(project_short_id)
                        return False
                # If we can't check state, we'll rely on the send operation to fail if disconnected
            except AttributeError:
                # Ignore state checking if not available
                pass
            
            await socket.send_text(json.dumps(message, default=str))
            logger.debug(f"Successfully sent {message.get('type', 'unknown')} message to {project_short_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to send message to {project_short_id}: {e}")
            await self.disconnect_project(project_short_id)
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
        # Store incoming message
        message_store.store_incoming_message(project_short_id, message_data)
        
        # Import here to avoid circular imports
        from .message_handler import MessageHandler
        
        # Create handler instance if not exists
        if not hasattr(self, '_message_handler'):
            self._message_handler = MessageHandler(self)
        
        await self._message_handler.handle_message(project_short_id, message_data)
    
    async def cleanup_inactive_connections(self):
        """Clean up inactive connections based on configured timeouts."""
        current_time = datetime.now()
        inactive_projects = []
        
        for project_short_id, conn_info in self.active_connections.items():
            timeout = conn_info.get("timeout", self.default_timeout)
            
            # Skip cleanup if timeout is -1 (no expiry)
            if timeout == -1:
                continue
                
            # Check if connection has exceeded its timeout
            time_since_ping = (current_time - conn_info["last_ping"]).total_seconds()
            if time_since_ping > timeout:
                inactive_projects.append(project_short_id)
        
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
                "last_ping": conn_info["last_ping"].isoformat()
            })
        
        return {
            "active_connections": len(self.active_connections),
            "connected_projects": list(self.active_connections.keys()),
            "connections_detail": connections_with_timeout,
            "default_timeout": self.default_timeout,
            "service_status": "healthy"
        }
    
    async def cleanup(self):
        """Clean up the service and stop background tasks."""
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
            self._heartbeat_task = None
            logger.info("Heartbeat task stopped")
        
        # Close all connections
        for project_id in list(self.active_connections.keys()):
            await self.disconnect_project(project_id)
        
        logger.info("SocketService cleanup completed")

# Global socket service instance
socket_service = SocketService()
