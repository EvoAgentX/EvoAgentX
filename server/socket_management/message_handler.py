"""
Message Handler for processing socket commands.
Integrates with existing core modules and services using the existing message format.
"""

import json
import logging
import uuid
from datetime import datetime
from typing import Dict, Any, TYPE_CHECKING

from .protocols import (
    MessageType, CommandType, create_setup_complete_message, 
    create_execution_complete_message, create_error_message,
    create_message, WorkflowStatus
)
from .process_monitor import ProcessMonitor

if TYPE_CHECKING:
    from .socket_service import SocketService

logger = logging.getLogger(__name__)

class MessageHandler:
    """
    Handles all incoming socket messages and routes them to appropriate functions.
    Uses existing core modules and maintains compatibility with current message format.
    """
    
    def __init__(self, socket_service: 'SocketService'):
        self.socket_service = socket_service
        self.process_monitor = ProcessMonitor(socket_service)
        
        # Import existing core functions
        self._import_core_functions()
    
    def _import_core_functions(self):
        """Import existing core workflow functions."""
        try:
            from ..core.workflow_setup import setup_project_parallel_with_status_messages
            from ..core.workflow_execution import execute_workflow_with_websocket
            from ..services.database_service import DatabaseService
            
            self.setup_project = setup_project_parallel_with_status_messages
            self.execute_workflow = execute_workflow_with_websocket
            self.db_service = DatabaseService()
            
        except ImportError as e:
            logger.error(f"Failed to import core functions: {e}")
            # Set fallback functions
            self.setup_project = None
            self.execute_workflow = None
            self.db_service = None
    
    async def handle_message(self, project_short_id: str, message_data: Dict[str, Any]):
        """Main message handling entry point."""
        try:
            # Handle different message formats
            if "command" in message_data:
                # Command format: {"command": "project.setup", "parameters": {...}}
                await self._handle_command_message(project_short_id, message_data)
            elif "type" in message_data and message_data["type"] == "heartbeat":
                # Heartbeat format: {"type": "heartbeat"}
                await self._handle_heartbeat(project_short_id, message_data)
            elif "inputs" in message_data:
                # Direct execution format: {"inputs": {...}} - assume workflow execution
                await self._handle_direct_execution(project_short_id, message_data)
            else:
                # Unknown format
                await self._send_error(project_short_id, f"Unknown message format: {message_data}")
                
        except Exception as e:
            logger.error(f"Error handling message from {project_short_id}: {e}")
            await self._send_error(project_short_id, f"Message processing error: {str(e)}")
    
    async def _handle_command_message(self, project_short_id: str, message_data: Dict[str, Any]):
        """Handle command messages."""
        command = message_data.get("command")
        parameters = message_data.get("parameters", {})
        message_id = message_data.get("message_id", uuid.uuid4().hex)
        
        try:
            if command == CommandType.PROJECT_SETUP:
                await self._handle_project_setup(project_short_id, message_id, parameters)
            elif command == CommandType.WORKFLOW_EXECUTE:
                await self._handle_workflow_execute(project_short_id, message_id, parameters)
            elif command == CommandType.WORKFLOW_LIST:
                await self._handle_workflow_list(project_short_id, message_id, parameters)
            elif command == CommandType.WORKFLOW_STATUS:
                await self._handle_workflow_status(project_short_id, message_id, parameters)
            elif command == CommandType.QUERY_ANALYZE:
                await self._handle_query_analyze(project_short_id, message_id, parameters)
            elif command == CommandType.SYSTEM_HEALTH:
                await self._handle_system_health(project_short_id, message_id)
            else:
                await self._send_error(project_short_id, f"Unknown command: {command}")
                
        except Exception as e:
            logger.error(f"Error handling command {command}: {e}")
            await self._send_error(project_short_id, str(e))
    
    async def _handle_direct_execution(self, project_short_id: str, message_data: Dict[str, Any]):
        """Handle direct execution messages (like existing WebSocket workflow execution)."""
        # This mimics the existing /workflow/{workflow_id}/execute_ws endpoint
        inputs = message_data.get("inputs", {})
        workflow_id = message_data.get("workflow_id")
        
        if not workflow_id:
            await self._send_error(project_short_id, "workflow_id is required for execution")
            return
        
        await self._execute_workflow_with_monitoring(project_short_id, workflow_id, inputs)
    
    async def _handle_project_setup(self, project_short_id: str, message_id: str, parameters: Dict[str, Any]):
        """Handle project setup command using existing setup function."""
        try:
            if not self.setup_project:
                await self._send_error(project_short_id, "Project setup function not available")
                return
            
            # Send immediate acknowledgment
            ack_message = create_message(
                MessageType.WORKFLOW_STATUS,
                status="starting",
                workflow_id=None,
                content="Project setup started",
                result={"project_short_id": project_short_id}
            )
            await self.socket_service.send_to_project(project_short_id, ack_message)
            
            # Create WebSocket send function that works with existing setup code
            websocket_send_func = self.process_monitor.create_websocket_send_function(project_short_id)
            
            # Run setup in background task to prevent blocking socket connection
            async def run_setup():
                try:
                    logger.info(f"Starting setup for project {project_short_id}")
                    
                    # Use existing parallel setup function
                    workflow_results = await self.setup_project(project_short_id, websocket_send_func)
                    
                    logger.info(f"Setup completed for project {project_short_id}, processing {len(workflow_results)} results")
                    
                    # Extract workflow graphs (same logic as current WebSocket endpoint)
                    workflow_graphs = []
                    for workflow in workflow_results:
                        if "workflow_graph" in workflow:
                            workflow_graphs.append(workflow["workflow_graph"])
                    
                    logger.info(f"Extracted {len(workflow_graphs)} workflow graphs for project {project_short_id}")
                    
                    # Send completion message using existing format
                    setup_complete_message = create_setup_complete_message(workflow_graphs)
                    success = await self.socket_service.send_to_project(project_short_id, setup_complete_message)
                    
                    if success:
                        logger.info(f"Successfully sent setup completion for project {project_short_id}")
                    else:
                        logger.error(f"Failed to send setup completion for project {project_short_id}")
                    
                except Exception as e:
                    logger.error(f"Setup failed for project {project_short_id}: {e}")
                    await self._send_error(project_short_id, f"Project setup failed: {str(e)}")
            
            # Start setup as background task
            asyncio.create_task(run_setup())
            
        except Exception as e:
            await self._send_error(project_short_id, f"Project setup failed: {str(e)}")
    
    async def _handle_workflow_execute(self, project_short_id: str, message_id: str, parameters: Dict[str, Any]):
        """Handle workflow execution command."""
        try:
            workflow_id = parameters.get("workflow_id")
            inputs = parameters.get("inputs", {})
            
            if not workflow_id:
                await self._send_error(project_short_id, "workflow_id is required")
                return
            
            await self._execute_workflow_with_monitoring(project_short_id, workflow_id, inputs)
            
        except Exception as e:
            await self._send_error(project_short_id, f"Workflow execution failed: {str(e)}")
    
    async def _execute_workflow_with_monitoring(self, project_short_id: str, workflow_id: str, inputs: Dict[str, Any]):
        """Execute workflow with monitoring using existing execution function."""
        try:
            if not self.execute_workflow:
                await self._send_error(project_short_id, "Workflow execution function not available")
                return
            
            # Start process monitoring
            await self.process_monitor.start_monitoring(project_short_id, {"workflow_id": workflow_id})
            
            # Create WebSocket send function that works with existing execution code
            websocket_send_func = self.process_monitor.create_websocket_send_function(project_short_id)
            
            # Execute using existing workflow execution logic
            result = await self.execute_workflow(workflow_id, inputs, websocket_send_func)
            
            # Send completion message using existing format
            execution_complete_message = create_execution_complete_message(result)
            await self.socket_service.send_to_project(project_short_id, execution_complete_message)
            
        except Exception as e:
            await self.process_monitor.process_error(project_short_id, e)
    
    async def _handle_workflow_list(self, project_short_id: str, message_id: str, parameters: Dict[str, Any]):
        """Handle workflow list command."""
        try:
            if not self.db_service:
                await self._send_error(project_short_id, "Database service not available")
                return
            
            # Use database service to list workflows
            workflows = await self.db_service.list_workflows()
            
            # Send response using existing format
            list_message = create_message(
                MessageType.WORKFLOW_STATUS,
                status="complete",
                workflow_id=None,
                content="workflow list retrieved",
                result={"workflows": workflows}
            )
            
            await self.socket_service.send_to_project(project_short_id, list_message)
            
        except Exception as e:
            await self._send_error(project_short_id, f"Failed to list workflows: {str(e)}")
    
    async def _handle_workflow_status(self, project_short_id: str, message_id: str, parameters: Dict[str, Any]):
        """Handle workflow status command."""
        try:
            workflow_id = parameters.get("workflow_id")
            if not workflow_id:
                await self._send_error(project_short_id, "workflow_id is required")
                return
            
            if not self.db_service:
                await self._send_error(project_short_id, "Database service not available")
                return
            
            # Use database service to get workflow
            workflow = await self.db_service.get_workflow(workflow_id)
            if not workflow:
                await self._send_error(project_short_id, f"Workflow {workflow_id} not found")
                return
            
            # Send status using existing format
            status_message = create_message(
                MessageType.WORKFLOW_STATUS,
                status=workflow.get("status", "unknown"),
                workflow_id=workflow_id,
                content="workflow status retrieved",
                result=workflow
            )
            
            await self.socket_service.send_to_project(project_short_id, status_message)
            
        except Exception as e:
            await self._send_error(project_short_id, f"Failed to get workflow status: {str(e)}")
    
    async def _handle_query_analyze(self, project_short_id: str, message_id: str, parameters: Dict[str, Any]):
        """Handle user query analysis command."""
        try:
            query = parameters.get("query")
            if not query:
                await self._send_error(project_short_id, "query is required")
                return
            
            # Import and use existing query analysis
            try:
                from ..services.user_query_service import analyze_user_query
                result = await analyze_user_query(project_short_id, query)
                
                # Send result using existing format
                query_result_message = create_message(
                    MessageType.WORKFLOW_STATUS,
                    status="complete",
                    workflow_id=None,
                    content="query analysis completed",
                    result={"analysis_result": result}
                )
                
                await self.socket_service.send_to_project(project_short_id, query_result_message)
                
            except ImportError:
                await self._send_error(project_short_id, "Query analysis service not available")
            
        except Exception as e:
            await self._send_error(project_short_id, f"Query analysis failed: {str(e)}")
    
    async def _handle_system_health(self, project_short_id: str, message_id: str):
        """Handle system health check command."""
        health_info = {
            "status": "healthy",
            "active_projects": len(self.socket_service.active_connections),
            "system_resources": self.process_monitor.get_system_resources(),
            "timestamp": datetime.now().isoformat()
        }
        
        # Send health info using existing format
        health_message = create_message(
            MessageType.WORKFLOW_STATUS,
            status="complete",
            workflow_id=None,
            content="system health check completed",
            result=health_info
        )
        
        await self.socket_service.send_to_project(project_short_id, health_message)
    
    async def _handle_heartbeat(self, project_short_id: str, message_data: Dict[str, Any]):
        """Handle heartbeat messages."""
        # Update last ping time
        if project_short_id in self.socket_service.active_connections:
            self.socket_service.active_connections[project_short_id]["last_ping"] = datetime.now()
        
        # Send heartbeat response using existing format
        heartbeat_response = create_message(
            MessageType.CONNECTION,
            status="connected",
            workflow_id=None,
            content="heartbeat response",
            result={"timestamp": datetime.now().isoformat()}
        )
        
        await self.socket_service.send_to_project(project_short_id, heartbeat_response)
    
    async def _send_error(self, project_short_id: str, error_message: str):
        """Send error response using existing format."""
        error_response = create_error_message(error_message)
        await self.socket_service.send_to_project(project_short_id, error_response)
