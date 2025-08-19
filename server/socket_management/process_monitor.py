"""
Process Monitor for real-time workflow execution monitoring.
Provides live updates to project sockets during workflow execution using existing message format.
"""

import asyncio
import json
import logging
import psutil
from datetime import datetime
from typing import Dict, Any, Optional, TYPE_CHECKING

from .protocols import (
    MessageType, create_message, create_workflow_status_message, 
    create_error_message, WorkflowStatus
)

if TYPE_CHECKING:
    from .socket_service import SocketService

logger = logging.getLogger(__name__)

class ProcessMonitor:
    """
    Monitors workflow processes and sends real-time updates via sockets.
    Uses the existing EvoAgentX message format.
    """
    
    def __init__(self, socket_service: 'SocketService'):
        self.socket_service = socket_service
        self.active_processes: Dict[str, Dict[str, Any]] = {}
        # Structure: {project_short_id: {"status": "running", "start_time": datetime, "progress": 0.0}}
    
    async def start_monitoring(self, project_short_id: str, process_info: Dict[str, Any]):
        """Start monitoring a process for a project."""
        self.active_processes[project_short_id] = {
            "status": "running",
            "start_time": datetime.now(),
            "progress": 0.0,
            "process_info": process_info
        }
        
        logger.info(f"Started monitoring process for project {project_short_id}")
        
        # Send monitoring started event using existing format
        start_message = create_message(
            MessageType.START,
            status="running",
            workflow_id=process_info.get("workflow_id"),
            content="Process monitoring started",
            result={"project_short_id": project_short_id}
        )
        
        await self.socket_service.send_to_project(project_short_id, start_message)
    
    async def update_progress(self, project_short_id: str, progress: float, message: str = ""):
        """Update progress for a project using existing progress message format."""
        if project_short_id in self.active_processes:
            self.active_processes[project_short_id]["progress"] = progress
            
            # Send progress update using existing format (matches workflow execution WebSocket)
            progress_message = create_message(
                MessageType.PROGRESS,
                status="running",
                workflow_id=self.active_processes[project_short_id]["process_info"].get("workflow_id"),
                content=f"{message} ({int(progress * 100)}% complete)" if message else f"{int(progress * 100)}% complete",
                result={"progress": progress}
            )
            
            await self.socket_service.send_to_project(project_short_id, progress_message)
    
    async def send_log_message(self, project_short_id: str, log_message: str, log_level: str = "info"):
        """Send log message to project socket using existing log format."""
        # Use existing log message format (matches workflow execution WebSocket)
        log_msg = create_message(
            MessageType.LOG,
            status="running",
            workflow_id=self.active_processes.get(project_short_id, {}).get("process_info", {}).get("workflow_id"),
            content=f"{log_level.upper()}: {log_message}",
            result=None
        )
        
        await self.socket_service.send_to_project(project_short_id, log_msg)
    
    async def send_output_message(self, project_short_id: str, output_message: str):
        """Send output message to project socket using existing output format."""
        # Use existing output message format (matches workflow execution WebSocket)
        output_msg = create_message(
            MessageType.OUTPUT,
            status="running", 
            workflow_id=self.active_processes.get(project_short_id, {}).get("process_info", {}).get("workflow_id"),
            content=output_message,
            result=None
        )
        
        await self.socket_service.send_to_project(project_short_id, output_msg)
    
    async def send_input_message(self, project_short_id: str, input_message: str):
        """Send input message to project socket using existing input format."""
        # Use existing input message format (matches workflow execution WebSocket)
        input_msg = create_message(
            MessageType.INPUT,
            status="running",
            workflow_id=self.active_processes.get(project_short_id, {}).get("process_info", {}).get("workflow_id"),
            content=input_message,
            result=None
        )
        
        await self.socket_service.send_to_project(project_short_id, input_msg)
    
    async def process_completed(self, project_short_id: str, result: Dict[str, Any]):
        """Mark process as completed and send completion event using existing format."""
        if project_short_id in self.active_processes:
            self.active_processes[project_short_id]["status"] = "completed"
            self.active_processes[project_short_id]["end_time"] = datetime.now()
            
            # Calculate duration
            duration = (datetime.now() - self.active_processes[project_short_id]["start_time"]).total_seconds()
            
            # Send completion event using existing complete format (matches workflow execution WebSocket)
            complete_message = create_message(
                MessageType.COMPLETE,
                status="complete",
                workflow_id=self.active_processes[project_short_id]["process_info"].get("workflow_id"),
                content="Process completed successfully",
                result={
                    **result,
                    "execution_time": f"{duration:.2f} seconds",
                    "duration_seconds": duration
                }
            )
            
            await self.socket_service.send_to_project(project_short_id, complete_message)
            
            # Clean up
            del self.active_processes[project_short_id]
    
    async def process_error(self, project_short_id: str, error: Exception):
        """Handle process error and send error event using existing format."""
        if project_short_id in self.active_processes:
            self.active_processes[project_short_id]["status"] = "error"
            
            # Send error event using existing error format
            error_message = create_error_message(
                error_message=f"Process error: {str(error)}",
                workflow_id=self.active_processes[project_short_id]["process_info"].get("workflow_id")
            )
            
            await self.socket_service.send_to_project(project_short_id, error_message)
            
            # Clean up
            del self.active_processes[project_short_id]
    
    async def send_workflow_status_update(self, project_short_id: str, status: str, workflow_id: str, content: str, result: Any = None):
        """Send workflow status update using existing workflow_status format."""
        status_message = create_workflow_status_message(
            status=status,
            workflow_id=workflow_id,
            content=content,
            result=result
        )
        
        await self.socket_service.send_to_project(project_short_id, status_message)
    
    def get_process_status(self, project_short_id: str) -> Optional[Dict[str, Any]]:
        """Get current process status for a project."""
        return self.active_processes.get(project_short_id)
    
    def get_system_resources(self) -> Dict[str, Any]:
        """Get current system resource usage."""
        try:
            return {
                "cpu_percent": psutil.cpu_percent(interval=1),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_usage": psutil.disk_usage('/').percent,
                "active_processes": len(self.active_processes),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Failed to get system resources: {e}")
            return {"error": str(e)}
    
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
                    logger.info(f"Sent message to {project_short_id}: {message_data.get('type', 'unknown')}")
                except json.JSONDecodeError:
                    # If not JSON, treat as a log message
                    logger.info(f"Sending plain text as log message to {project_short_id}: {message_str[:100]}...")
                    await self.send_log_message(project_short_id, message_str)
                    
            except Exception as e:
                logger.error(f"Error in websocket_send_func for {project_short_id}: {e}")
                # Don't let this error propagate and break the workflow
        
        return websocket_send_func
