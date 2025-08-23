"""
Workflow execution logic.
Phase 3: Execute workflow with provided inputs.
"""

import asyncio
import json
import os
import uuid
import queue
import sys
import threading
import time
import re
import requests
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable, Type
from io import StringIO
from contextlib import redirect_stdout, redirect_stderr

from dotenv import load_dotenv

from evoagentx.workflow import WorkFlowGenerator, WorkFlowGraph, WorkFlow
from evoagentx.models import LLMConfig
from evoagentx.models.model_configs import OpenAILLMConfig, OpenRouterConfig
from evoagentx.models.model_utils import create_llm_instance
from evoagentx.agents.agent_manager import AgentManager
from evoagentx.core.module_utils import parse_json_from_text
from evoagentx.tools import MCPToolkit

# Import loguru for enhanced logging
try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

from ..prompts import WORKFLOW_REQUIREMENT_PROMPT, CUSTOM_OUTPUT_EXTRACTION_PROMPT
from ..database.db import database

from ..utils.output_parser import parse_workflow_output
from ..utils.tool_creator import create_tools
from ..utils.websocket_utils import WebSocketEnhancedSink, WebSocketProgressTracker

load_dotenv(os.path.join(os.path.dirname(__file__), '../config/app.env'))

# Default LLM configuration
default_llm_config = {
    "model": "gpt-4o-mini",
    "openai_key": os.getenv("OPENAI_API_KEY"),
    # "stream": True,
    "output_response": True,
    "max_tokens": 16000
}

# Placeholder for sudo execution result
sudo_execution_result = None


def create_project_websocket_send_func(project_short_id: str) -> Optional[Callable]:
    """
    Create a websocket send function for a project if there's an active connection.
    
    Args:
        project_short_id: The project identifier
        
    Returns:
        Optional[Callable]: A function that sends messages to the project socket, or None if no connection
        
    Example:
        websocket_send_func = create_project_websocket_send_func("proj_123")
        if websocket_send_func:
            await websocket_send_func({"type": "log", "content": "Hello from workflow!"})
    """
    try:
        # Import socket service to check for active connections
        from ..socket_management.socket_service import socket_service
        
        if socket_service.is_project_connected(project_short_id):
            async def send_func(message):
                try:
                    # Ensure message is a dictionary
                    if not isinstance(message, dict):
                        if isinstance(message, str):
                            # Try to parse as JSON if it's a string
                            try:
                                import json
                                message = json.loads(message)
                            except json.JSONDecodeError:
                                print(f"❌ Failed to parse message string as JSON: {message}")
                                return False
                        else:
                            print(f"❌ Cannot send non-dict message: {message}")
                            return False
                    
                    # Send the message
                    result = await socket_service.send_to_project(project_short_id, message)
                    if not result:
                        print(f"⚠️  Failed to send message to project {project_short_id}")
                    return result
                except Exception as e:
                    print(f"❌ Error sending message to project {project_short_id}: {e}")
                    return False
            
            return send_func
        else:
            return None
    except Exception as e:
        print(f"⚠️  Error checking socket connection for project {project_short_id}: {e}")
        return None


def create_project_websocket_messenger(project_short_id: str) -> Optional[Dict[str, Callable]]:
    """
    Create a convenient messenger object with pre-configured message functions for a project.
    
    Args:
        project_short_id: The project identifier
        
    Returns:
        Optional[Dict[str, Callable]]: A dictionary with message functions, or None if no connection
        
    Example:
        messenger = create_project_websocket_messenger("proj_123")
        if messenger:
            await messenger["log"]("Workflow started")
            await messenger["progress"](0.5, "Halfway done")
            await messenger["error"]("Something went wrong")
    """
    try:
        # Import socket service and protocols
        from ..socket_management.socket_service import socket_service
        from ..socket_management.protocols import create_message, MessageType
        
        if socket_service.is_project_connected(project_short_id):
            async def send_func(message):
                return await socket_service.send_to_project(project_short_id, message)
            
            # Create convenience functions
            async def log_message(content: str, workflow_id: str = None):
                message = create_message(
                    MessageType.SETUP_LOG,
                    status=None,
                    workflow_id=workflow_id,
                    content=content,
                    result=None
                )
                return await send_func(message)
            
            async def progress_message(status: str, progress: float, content: str, workflow_id: str = None):
                message = create_message(
                    MessageType.SETUP_LOG,
                    status=status,
                    workflow_id=workflow_id,
                    content=content,
                    result={"progress": progress}
                )
                return await send_func(message)
            
            async def error_message(content: str, workflow_id: str = None):
                message = create_message(
                    MessageType.SETUP_LOG,
                    status="error",
                    workflow_id=workflow_id,
                    content=f"ERROR: {content}",
                    result=None
                )
                return await send_func(message)
            
            async def status_message(status: str, workflow_id: str = None):
                message = create_message(
                    MessageType.SETUP_LOG,
                    status=status,
                    workflow_id=workflow_id,
                    content=f"Status: {status}",
                    result=None
                )
                return await send_func(message)
            
            return {
                "send": send_func,
                "log": log_message,
                "progress": progress_message,
                "error": error_message,
                "status": status_message
            }
        else:
            return None
    except Exception as e:
        print(f"⚠️  Error creating messenger for project {project_short_id}: {e}")
        return None


def _create_supabase_public_url(project_short_id: str, file_path: str, execution_id: str = None) -> str:
    """Create Supabase signed URL for file access."""
    try:
        from ..utils.tool_creator import _create_storage_handler
        storage_handler = _create_storage_handler(project_short_id, execution_id)
        if storage_handler and hasattr(storage_handler, 'supabase'):
            # Combine base_path with file_path for full storage path
            full_path = storage_handler.translate_in(file_path.strip())
            # Create signed URL with 60 seconds expiry
            try:
                public_url = storage_handler.supabase.storage.from_(storage_handler.bucket_name).get_public_url(full_path, 60)
                return public_url["publicURL"]
            except Exception as e:
                print(f"⚠️  Failed to create signed URL for {file_path}: {e}")
                return file_path
    except Exception as e:
        print(f"⚠️  Failed to create signed URL for {file_path}: {e}")
    ## Not a path, return the original content
    return file_path

def _scan_and_replace_file_paths(data: Any, project_short_id: str, execution_id: str = None) -> Any:
    """Recursively scan data structure and replace file paths with Supabase signed URLs."""
    if isinstance(data, dict):
        return {k: _scan_and_replace_file_paths(v, project_short_id, execution_id) for k, v in data.items()}
    elif isinstance(data, list):
        return [_scan_and_replace_file_paths(item, project_short_id, execution_id) for item in data]
    elif isinstance(data, str):
        # Match common file path patterns: /path/to/file.ext, C:\path\to\file.ext, ./file.ext
        file_path_pattern = r'^([A-Za-z]:\\)?([\/\\]?[^\/\\]+[\/\\])*[^\/\\]+\.[a-zA-Z0-9]+$'
        if re.match(file_path_pattern, data.strip()):
            return _create_supabase_public_url(project_short_id, data.strip(), execution_id)
        return data
    return data


def _replace_urls_with_paths(data: Any, project_short_id: str, execution_id: str) -> Any:
    """
    Recursively scan data structure and replace URLs with Supabase storage paths.
    Downloads files from URLs and uploads them directly to Supabase storage.
    """
    if isinstance(data, dict):
        return {k: _replace_urls_with_paths(v, project_short_id, execution_id) for k, v in data.items()}
    elif isinstance(data, list):
        return [_replace_urls_with_paths(item, project_short_id, execution_id) for item in data]
    elif isinstance(data, str):
        # Match URL patterns: http://, https://, ftp://, etc.
        url_pattern = r'^https?://[^\s]+'
        if re.match(url_pattern, data.strip()):
            return _upload_url_to_supabase(data.strip(), project_short_id, execution_id)
        return data
    return data


def _upload_url_to_supabase(url: str, project_short_id: str, execution_id: str) -> str:
    """
    Download file from URL and upload directly to Supabase storage.
    Returns the Supabase storage path.
    """
    try:
        from ..utils.tool_creator import _create_storage_handler
        
        # Create storage handler for this execution
        storage_handler = _create_storage_handler(project_short_id, execution_id)
        if not storage_handler or not hasattr(storage_handler, 'supabase'):
            print(f"⚠️  No storage handler available for {project_short_id}, keeping original URL")
            return url
        
        # Download file content from URL
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        # Extract filename from URL or use a default
        filename = url.split('/')[-1] if '/' in url else 'downloaded_file'
        if '?' in filename:  # Remove query parameters
            filename = filename.split('?')[0]
        if not filename or '.' not in filename:
            filename = f"downloaded_file_{execution_id[:8]}.txt"
        
        # Create storage path in inputs folder
        storage_path = f"inputs/{filename}"
        
        # Upload directly to Supabase
        try:
            # Get file content as bytes
            file_content = response.content
            
            # Upload to Supabase storage
            upload_response = storage_handler.supabase.storage.from_(storage_handler.bucket_name).upload(
                path=storage_path,
                file=file_content,
                file_options={"upsert": True}
            )
            
            print(f"✅ Uploaded {url} to Supabase as {storage_path}")
            return storage_path
            
        except Exception as e:
            print(f"❌ Failed to upload {url} to Supabase: {e}")
            return url
            
    except Exception as e:
        print(f"❌ Failed to process URL {url}: {e}")
        return url


async def get_workflow(workflow_id: str) -> Dict[str, Any]:
    """Retrieve workflow information from the database"""
    workflow = await database.find_one("workflows", {"id": workflow_id})
    return workflow


async def list_workflows(skip: int = 0, limit: int = 100, status: str = None) -> List[Dict[str, Any]]:
    """List workflows with optional filtering"""
    query = {}
    if status:
        query["status"] = status
    
    workflows = await database.find_many("workflows", query, limit=limit)
    # Apply skip manually since find_many doesn't support skip parameter
    if skip > 0:
        workflows = workflows[skip:]
    return workflows


async def update_workflow_status(workflow_id: str, status: str, **kwargs):
    """Update workflow status and other fields"""
    # Update status and any additional fields
    updates = {"status": status, "updated_at": datetime.now(), **kwargs}
    await database.update(
        "workflows", 
        {"id": workflow_id}, 
        updates
    )


def create_llm_config(llm_config_dict: Dict[str, Any]) -> LLMConfig:
    """
    Convert a dictionary to the appropriate LLM config object based on the API key provided,
    then fallback to model type detection.
    """
    
    # Priority 1: Check which API key is provided (most explicit indicator of intent)
    if llm_config_dict.get("openrouter_key"):
        # If openrouter_key is provided, use OpenRouter regardless of model
        return OpenRouterConfig(**llm_config_dict)
    
    else:
        # If openai_key is provided, use OpenAI regardless of model
        return OpenAILLMConfig(**llm_config_dict)


async def execute_workflow_from_config(workflow: Dict[str, Any], llm_config_dict: Dict[str, Any], mcp_config: dict = None, inputs: Dict[str, Any] = None, database_information: Dict[str, Any] = None, task_info: Dict[str, Any] = None, websocket_send_func: Callable = None) -> Dict[str, Any]:
    """
    Execute a workflow with the given configuration.
    
    Args:
        workflow: The complete workflow document from database (must contain project_short_id and workflow_graph)
        llm_config_dict: LLM configuration dictionary
        mcp_config: Optional MCP configuration dictionary
        inputs: Optional inputs dictionary to pass to async_execute
        database_information: Optional database information for dynamic MongoDB toolkit creation
        task_info: Optional task info for socket messaging
        websocket_send_func: Optional WebSocket send function for real-time updates
        
    Returns:
        Dict containing only the essential execution results:
        - original_message: The raw output from workflow execution
        - parsed_json: Extracted JSON from the output (if found)
        
    Raises:
        Exception: Any error that occurs during workflow execution
    """
    try:
        if sudo_execution_result:
            # Parse the sudo execution result to extract only essential data
            parsed_output = parse_workflow_output(sudo_execution_result)
            return {
                "original_message": parsed_output["original_message"],
                "parsed_json": parsed_output["parsed_json"]
            }
        
        
        llm_config = create_llm_config(llm_config_dict)
        llm = create_llm_instance(llm_config)
        
        # Extract project_short_id from workflow data
        project_short_id = None
        if isinstance(workflow, dict) and "project_short_id" in workflow:
            project_short_id = workflow["project_short_id"]
        elif hasattr(workflow, "project_short_id"):
            project_short_id = workflow.project_short_id
        
        # Generate execution ID for consistent storage paths
        execution_id = str(uuid.uuid4())
        print(f"🚀 Starting workflow execution with execution ID {execution_id}")
        
        # Send execution start message via socket if available
        if websocket_send_func and task_info:
            try:
                workflow_name = task_info.get("workflow_name", "Unknown")
                start_message = {
                    "type": "SETUP_LOG",
                    "status": "executing",
                    "workflow_id": workflow.get("id", "unknown"),
                    "content": f"🚀 Starting workflow execution: {workflow_name}",
                    "result": {"execution_id": execution_id}
                }
                # Send message asynchronously without blocking
                asyncio.create_task(websocket_send_func(start_message))
            except Exception as e:
                print(f"⚠️  Failed to send execution start message via socket: {e}")
        
        # Process inputs: replace URLs with Supabase storage paths
        if inputs and project_short_id:
            print(f"🔄 Processing inputs for URL replacement...")
            if websocket_send_func:
                try:
                    process_message = {
                        "type": "SETUP_LOG",
                        "status": "processing",
                        "workflow_id": workflow.get("id", "unknown"),
                        "content": "🔄 Processing inputs for URL replacement...",
                        "result": None
                    }
                    # Send message asynchronously without blocking
                    asyncio.create_task(websocket_send_func(process_message))
                except Exception as e:
                    print(f"⚠️  Failed to send input processing message via socket: {e}")
            
            inputs = _replace_urls_with_paths(inputs, project_short_id, execution_id)
        
        # Create tools with proper storage handling first
        execution_tools = []
        if mcp_config:
            mcp_toolkit = MCPToolkit(config=mcp_config)
            execution_tools = mcp_toolkit.get_tools()
        
        # Create tools with project_short_id and execution_id for storage configuration
        if project_short_id:
            if websocket_send_func:
                try:
                    tools_message = {
                        "type": "SETUP_LOG",
                        "status": "preparing",
                        "workflow_id": workflow.get("id", "unknown"),
                        "content": "🔧 Creating execution tools...",
                        "result": None
                    }
                    # Send message asynchronously without blocking
                    asyncio.create_task(websocket_send_func(tools_message))
                except Exception as e:
                    print(f"⚠️  Failed to send tools creation message via socket: {e}")
            
            execution_tools += create_tools(project_short_id, database_information, execution_id)
            print(f"🔧 Created {len(execution_tools)} tools for project {project_short_id}")
            
            if websocket_send_func:
                try:
                    tools_ready_message = {
                        "type": "SETUP_LOG",
                        "status": "ready",
                        "workflow_id": workflow.get("id", "unknown"),
                        "content": f"🔧 Created {len(execution_tools)} execution tools",
                        "result": {"tool_count": len(execution_tools)}
                    }
                    # Send message asynchronously without blocking
                    asyncio.create_task(websocket_send_func(tools_ready_message))
                except Exception as e:
                    print(f"⚠️  Failed to send tools ready message via socket: {e}")
        else:
            print("⚠️  No project_short_id found, creating tools without storage support")
            execution_tools += create_tools("default", database_information, execution_id)
        
        # Extract and reconstruct workflow graph from the workflow document (after tools are ready)
        if isinstance(workflow, dict) and "workflow_graph" in workflow:
            workflow_graph_data = workflow["workflow_graph"]
            if isinstance(workflow_graph_data, WorkFlowGraph):
                workflow_graph = workflow_graph_data
            else:
                if websocket_send_func:
                    try:
                        init_message = {
                            "type": "SETUP_LOG",
                            "status": "initializing",
                            "workflow_id": workflow.get("id", "unknown"),
                            "content": "🏗️ Initializing workflow graph...",
                            "result": None
                        }
                        # Send message asynchronously without blocking
                        asyncio.create_task(websocket_send_func(init_message))
                    except Exception as e:
                        print(f"⚠️  Failed to send workflow init message via socket: {e}")
                
                workflow_graph: WorkFlowGraph = WorkFlowGraph.from_dict(workflow_graph_data, llm_config=llm_config, tools=execution_tools)
        elif isinstance(workflow, WorkFlowGraph):
            workflow_graph = workflow
        else:
            workflow_graph: WorkFlowGraph = WorkFlowGraph.from_dict(workflow, llm_config=llm_config, tools=execution_tools)
        
        # Create agent manager and add agents from workflow
        if websocket_send_func:
            try:
                agents_message = {
                    "type": "SETUP_LOG",
                    "status": "preparing",
                    "workflow_id": workflow.get("id", "unknown"),
                    "content": "🤖 Creating agent manager and adding agents...",
                    "result": None
                }
                # Send message asynchronously without blocking
                asyncio.create_task(websocket_send_func(agents_message))
            except Exception as e:
                print(f"⚠️  Failed to send agents creation message via socket: {e}")
        
        agent_manager = AgentManager(tools=execution_tools)
        agent_manager.add_agents_from_workflow(workflow_graph, llm_config=llm_config)

        # Create and execute workflow
        workflow_instance = WorkFlow(graph=workflow_graph, agent_manager=agent_manager, llm=llm)
        workflow_instance.init_module()
        
        if websocket_send_func:
            try:
                execute_message = {
                    "type": "SETUP_LOG",
                    "status": "executing",
                    "workflow_id": workflow.get("id", "unknown"),
                    "content": "⚡ Executing workflow...",
                    "result": None
                }
                # Send message asynchronously without blocking
                asyncio.create_task(websocket_send_func(execute_message))
            except Exception as e:
                print(f"⚠️  Failed to send execution start message via socket: {e}")
        
        # Execute workflow - it returns a structured dict with all output parameters
        output = await workflow_instance.async_execute(inputs=inputs, extract_output=False)
        
        # The workflow already returns structured output as a dict
        if isinstance(output, dict):
            parsed_json = output
        else:
            # Fallback for string output (shouldn't happen with extract_output=False)
            parsed_json = {"workflow_output": str(output)}
        
        if websocket_send_func:
            try:
                completion_message = {
                    "type": "SETUP_LOG",
                    "status": "completed",
                    "workflow_id": workflow.get("id", "unknown"),
                    "content": "✅ Workflow execution completed successfully",
                    "result": {"output_keys": list(parsed_json.keys()) if isinstance(parsed_json, dict) else ["output"]}
                }
                # Send message asynchronously without blocking
                asyncio.create_task(websocket_send_func(completion_message))
            except Exception as e:
                print(f"⚠️  Failed to send completion message via socket: {e}")
        
        # Scan and replace file paths with URLs if we have parsed_json and project_short_id
        if parsed_json and project_short_id:
            if websocket_send_func:
                try:
                    processing_message = {
                        "type": "SETUP_LOG",
                        "status": "processing",
                        "workflow_id": workflow.get("id", "unknown"),
                        "content": "🔄 Processing output for file path replacement...",
                        "result": None
                    }
                    # Send message asynchronously without blocking
                    asyncio.create_task(websocket_send_func(processing_message))
                except Exception as e:
                    print(f"⚠️  Failed to send processing message via socket: {e}")
            
            parsed_json = _scan_and_replace_file_paths(parsed_json, project_short_id, execution_id)
            
            if websocket_send_func:
                try:
                    final_message = {
                        "type": "SETUP_LOG",
                        "status": "finalizing",
                        "workflow_id": workflow.get("id", "unknown"),
                        "content": "✨ Finalizing workflow output...",
                        "result": None
                    }
                    # Send message asynchronously without blocking
                    asyncio.create_task(websocket_send_func(final_message))
                except Exception as e:
                    print(f"⚠️  Failed to send finalization message via socket: {e}")
        
        return {
            "original_message": output,
            "parsed_json": parsed_json
        }
        
    except Exception as e:
        # Send error message via socket if available
        if websocket_send_func:
            try:
                error_message = {
                    "type": "SETUP_LOG",
                    "status": "error",
                    "workflow_id": workflow.get("id", "unknown"),
                    "content": f"❌ Workflow execution failed: {str(e)}",
                    "result": {"error": str(e)}
                }
                # Send message asynchronously without blocking
                asyncio.create_task(websocket_send_func(error_message))
            except Exception as socket_error:
                print(f"⚠️  Failed to send error message via socket: {socket_error}")
        
        # Re-raise the exception instead of returning an error dictionary
        # This allows proper error handling in the calling functions
        raise e


async def execute_workflow(workflow_id: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Phase 3: Execute workflow with provided inputs.
    Updated to work with individual workflow records and correct database schema.
    Returns only essential execution data.
    """
    try:
        # Check if workflow exists
        workflow = await get_workflow(workflow_id)
        if not workflow:
            raise ValueError(f"Workflow with ID {workflow_id} not found")
        
        # Check if workflow generation was completed
        if workflow.get("workflow_graph") is None:
            raise ValueError(f"Workflow {workflow_id} has not completed generation phase")
        
        # Extract project_short_id to find active socket connection
        project_short_id = workflow.get("project_short_id")
        
        # Get workflow graph and task info early for socket messaging
        workflow_graph = workflow["workflow_graph"]
        task_info = workflow.get("task_info", {})
        workflow_name = task_info.get("workflow_name", workflow_id)
        
        if workflow_graph is None:
            print(f"⚠️ No workflow graph available for {workflow_name}")
            await update_workflow_status(workflow_id, "failed")
            raise ValueError(f"Workflow {workflow_id} is not found")
            
        print(f"🚀 Executing workflow: {workflow_name}")
        
        # Create websocket send function if there's an active connection for this project
        websocket_send_func = create_project_websocket_send_func(project_short_id) if project_short_id else None
        
        if websocket_send_func:
            print(f"🔌 Found active socket connection for project {project_short_id}, enabling real-time updates")
            # Test the socket connection with a simple message
            try:
                test_message = {
                    "type": "SETUP_LOG",
                    "status": None,
                    "workflow_id": workflow_id,
                    "content": "Starting workflow execution",
                    "result": None
                }
                # Send test message asynchronously without blocking
                asyncio.create_task(websocket_send_func(test_message))
                print(f"✅ Socket test message sent successfully")
            except Exception as e:
                print(f"⚠️  Socket test message failed: {e}")
                websocket_send_func = None  # Disable socket messaging if test fails
        elif project_short_id:
            print(f"⚠️  No active socket connection found for project {project_short_id}")
        else:
            print(f"ℹ️  No project_short_id found in workflow, proceeding without socket updates")
        
        # Update workflow status
        await update_workflow_status(workflow_id, "running")
        
        # Send status update via socket if available
        if websocket_send_func:
            try:
                status_message = {
                    "type": "SETUP_LOG",
                    "status": "running",
                    "workflow_id": workflow_id,
                    "content": f"Workflow {workflow_name} is now running",
                    "result": None
                }
                # Send message asynchronously without blocking
                asyncio.create_task(websocket_send_func(status_message))
            except Exception as e:
                print(f"⚠️  Failed to send running status via socket: {e}")
        
        # Get database information for dynamic MongoDB toolkit
        database_information = task_info.get("database_information")
        
        # If we have a websocket connection, use the enhanced execution with real-time updates
        if websocket_send_func:
            print(f"📡 Executing workflow with real-time socket updates")
            execution_result = await execute_workflow_with_websocket(
                workflow_id=workflow_id,
                inputs=inputs,
                websocket_send_func=websocket_send_func
            )
        else:
            print(f"📡 Executing workflow without socket updates")
            # Execute the workflow with database information
            execution_result = await execute_workflow_from_config(
                workflow,  # Pass the whole workflow document instead of just workflow_graph
                default_llm_config, 
                mcp_config={}, 
                inputs=inputs,
                database_information=database_information,
                task_info=task_info,
                websocket_send_func=websocket_send_func  # Pass the websocket_send_func here
            )
        
        if execution_result is None:
            print(f"❌ Failed to execute workflow: {workflow_name}")
            await update_workflow_status(workflow_id, "failed")
            raise ValueError(f"Execution is not successful: {execution_result}")
        
        # Update workflow storage with execution results
        await update_workflow_status(
            workflow_id, 
            "completed",
            execution_result=execution_result
        )
        
        # Send final completion message via socket if available
        if websocket_send_func:
            try:
                final_completion_message = {
                    "type": "SETUP_LOG",
                    "status": "completed",
                    "workflow_id": workflow_id,
                    "content": f"🎉 Workflow {workflow_name} execution completed and saved to database",
                    "result": {"summary": "completed", "workflow_id": workflow_id}
                }
                # Send message asynchronously without blocking
                asyncio.create_task(websocket_send_func(final_completion_message))
            except Exception as e:
                print(f"⚠️  Failed to send final completion message via socket: {e}")
        
        # Return only the essential data
        return execution_result["parsed_json"]
        
    except Exception as e:
        # Send error message via socket if available
        if websocket_send_func:
            try:
                error_message = {
                    "type": "SETUP_LOG",
                    "status": "error",
                    "workflow_id": workflow_id,
                    "content": f"❌ Workflow execution failed: {str(e)}",
                    "result": {"error": str(e)}
                }
                # Send message asynchronously without blocking
                asyncio.create_task(websocket_send_func(error_message))
            except Exception as socket_error:
                print(f"⚠️  Failed to send error message via socket: {socket_error}")
        
        await update_workflow_status(workflow_id, "failed")
        raise ValueError(f"Execution is not successful: {e}")


# WebSocket execution implementation - restored from old service
async def execute_workflow_with_websocket(
    workflow_id: str, 
    inputs: Dict[str, Any], 
    websocket_send_func: Callable
) -> Dict[str, Any]:
    """
    Execute workflow with WebSocket-based real-time progress updates.
    Enhanced to support all message types from the system diagram.
    
    Args:
        workflow_id: The workflow ID to execute
        inputs: Input parameters for workflow execution
        websocket_send_func: Function to send messages via WebSocket
        
    Returns:
        Dict containing execution results
    """
    progress_tracker = WebSocketProgressTracker(websocket_send_func, workflow_id, "workflow")
    websocket_sink = None
    execution_start_time = time.time()
    
    try:
        # Send connection confirmation
        await progress_tracker.send_connection_confirmation()
        
        # Send start notification
        await progress_tracker.send_start_notification("Workflow execution")
        
        # Send initial progress
        await progress_tracker.send_progress_update("initializing", 0.0, "Starting workflow execution...")
        
        # Check if workflow exists
        workflow = await get_workflow(workflow_id)
        if not workflow:
            await progress_tracker.send_error(f"Workflow with ID {workflow_id} not found")
            raise ValueError(f"Workflow with ID {workflow_id} not found")
        
        await progress_tracker.send_progress_update("validating", 0.1, "Validating workflow configuration...")
        
        # Check if workflow generation was completed
        if workflow.get("workflow_graph") is None:
            await progress_tracker.send_error(f"Workflow {workflow_id} has not completed generation phase")
            raise ValueError(f"Workflow {workflow_id} has not completed generation phase")
        
        await progress_tracker.send_progress_update("preparing", 0.2, "Preparing workflow execution...")
        
        # Update workflow status
        await update_workflow_status(workflow_id, "running")
        
        # Get workflow graph and task info
        workflow_graph = workflow["workflow_graph"]
        task_info = workflow.get("task_info", {})
        workflow_name = task_info.get("workflow_name", workflow_id)
        
        if workflow_graph is None:
            await progress_tracker.send_error("No workflow graph available")
            await update_workflow_status(workflow_id, "failed")
            return {
                "status": "failed",
                "error": "No workflow graph available"
            }
        
        # Generate execution ID
        execution_id = f"exec_{workflow_id}_{int(time.time())}"
        
        await progress_tracker.send_progress_update("executing", 0.3, f"Executing workflow: {workflow_name}")
        
        # Get database information (currently ignored - database tools disabled)
        database_information = task_info.get("database_information")
        
        # Setup isolated workflow execution logging using the new system
        from ..core.workflow_logging import IsolatedWorkflowLogger
        
        isolated_logger = IsolatedWorkflowLogger(workflow_id, "execution")
        bound_logger = isolated_logger.setup_isolated_logging(websocket_send_func)
        execution_id = isolated_logger.process_id
        
        # Keep WebSocket enhanced sink for backward compatibility
        websocket_sink = WebSocketEnhancedSink(websocket_send_func, workflow_id, "workflow")
        sink_id = None
        
        try:
            # Send progress for workflow initialization
            await progress_tracker.send_progress_update("initializing", 0.5, "Initializing workflow components...")
            
            # Log workflow execution start with bound logger for isolation
            bound_logger.info(f"Starting workflow execution: {workflow_name}")
            bound_logger.info(f"Workflow inputs: {inputs}")
            
            # Execute the workflow (database tools disabled)
            execution_result = await execute_workflow_from_config(
                workflow,  # Pass the whole workflow document instead of just workflow_graph
                default_llm_config, 
                mcp_config={}, 
                inputs=inputs,
                database_information=database_information,
                task_info=task_info,
                websocket_send_func=websocket_send_func # Pass the websocket_send_func here
            )
            
            # Log execution completion with bound logger
            bound_logger.info(f"Workflow execution completed successfully")
            bound_logger.info(f"Execution result keys: {list(execution_result.keys()) if execution_result else 'None'}")
            
            await progress_tracker.send_progress_update("finalizing", 0.9, "Finalizing execution results...")
            
            if execution_result is None:
                bound_logger.error(f"Failed to execute workflow: {workflow_name}")
                await progress_tracker.send_error(f"Failed to execute workflow: {workflow_name}")
                await update_workflow_status(workflow_id, "failed")
                return {
                    "original_message": "Failed to execute workflow",
                    "parsed_json": None
                }
            
            # execution_result now already contains the parsed format from execute_workflow_from_config
            # No need to parse again since it already returns {"original_message": "...", "parsed_json": ...}
            
            # Get captured output from the sink
            captured_output = websocket_sink.get_buffer_contents() if websocket_sink else {}
            
            # Update workflow storage with execution results
            try:
                await update_workflow_status(
                    workflow_id, 
                    "completed",
                    execution_result=execution_result
                )
                print(f"✅ Successfully saved execution result to database for workflow {workflow_id}")
            except Exception as e:
                print(f"❌ Error saving execution result to database: {e}")
                # Continue execution even if database save fails
            
            # Calculate total execution time
            total_execution_time = time.time() - execution_start_time
            
            # Return only the essential data for WebSocket clients
            final_result = {
                "original_message": execution_result.get("original_message", ""),
                "parsed_json": execution_result.get("parsed_json", None)
            }
            
            await progress_tracker.send_progress_update("completed", 1.0, "Workflow execution completed successfully")
            
            # Send completion message
            await progress_tracker.send_completion(final_result, "Workflow execution completed successfully")
            
            # Send workflow status update: complete
            if hasattr(progress_tracker, 'send_workflow_status'):
                await progress_tracker.send_workflow_status("complete", None)  # None for workflow_id when complete
            else:
                # Fallback to using the websocket_send_func directly
                try:
                    from ..utils.websocket_utils import send_workflow_status_message
                    await send_workflow_status_message(websocket_send_func, "complete", None)
                except Exception as e:
                    print(f"Warning: Could not send completion status: {e}")
            
            return final_result
            
        except Exception as e:
            # Log execution error with bound logger for isolation
            bound_logger.error(f"Workflow execution failed: {str(e)}")
            # Send execution error
            await progress_tracker.send_error(f"Workflow execution failed: {str(e)}")
            await update_workflow_status(workflow_id, "failed")
            raise e
            
        finally:
            # Cleanup
            if websocket_sink:
                websocket_sink.stop()
            if sink_id:
                try:
                    logger.remove(sink_id)
                except Exception as e:
                    print(f"Warning: Could not remove loguru sink: {e}")
            
            # Get captured workflow engine logs before cleanup
            captured_logs = isolated_logger.get_captured_logs()
            if captured_logs:
                print(f"📊 Workflow {workflow_id} captured {len(captured_logs)} engine logs during execution")
            
            # Cleanup isolated logging
            isolated_logger.cleanup()
    
    except Exception as e:
        # Send connection error for unexpected failures
        await progress_tracker.send_error(f"Unexpected error: {str(e)}")
        await update_workflow_status(workflow_id, "failed")
        return {
            "original_message": f"Failed to execute workflow: {str(e)}",
            "parsed_json": None
        }


# Example usage of the socket messaging pattern:
"""
# Example 1: Basic usage in any function
async def some_workflow_function(workflow_id: str):
    # Get the workflow to find project_short_id
    workflow = await get_workflow(workflow_id)
    project_short_id = workflow.get("project_short_id")
    
    # Create websocket send function
    websocket_send_func = create_project_websocket_send_func(project_short_id)
    
    if websocket_send_func:
        # Send real-time updates
        await websocket_send_func({
            "type": "SETUP_LOG",
            "status": None,
            "workflow_id": workflow_id,
            "content": "Starting workflow function",
            "result": None
        })
    
    # ... do work ...
    
    if websocket_send_func:
        await websocket_send_func({
            "type": "SETUP_LOG", 
            "status": None,
            "workflow_id": workflow_id,
            "content": "Workflow function completed",
            "result": None
        })

# Example 2: Using the convenience messenger
async def another_workflow_function(workflow_id: str):
    workflow = await get_workflow(workflow_id)
    project_short_id = workflow.get("project_short_id")
    
    # Create messenger with convenience functions
    messenger = create_project_websocket_messenger(project_short_id)
    
    if messenger:
        await messenger["log"]("Starting workflow function", workflow_id)
        await messenger["progress"]("starting", 0.0, "Initializing...", workflow_id)
        
        # ... do work ...
        
        await messenger["progress"]("processing", 0.5, "Halfway done", workflow_id)
        
        # ... more work ...
        
        await messenger["progress"]("completed", 1.0, "All done!", workflow_id)
        await messenger["log"]("Workflow function completed successfully", workflow_id)
    else:
        # Fallback to console logging
        print("No active socket connection, using console logging")

# Example 3: Integration with existing workflow setup pattern
async def setup_project_with_socket(project_short_id: str):
    # This is exactly what the workflow setup does
    websocket_send_func = create_project_websocket_send_func(project_short_id)
    
    if websocket_send_func:
        print(f"🔌 Found active socket for project {project_short_id}")
        # Use the same pattern as workflow setup
        await websocket_send_func({
            "type": "SETUP_LOG",
            "status": None,
            "workflow_id": None,
            "content": "Project setup started",
            "result": None
        })
    else:
        print(f"⚠️  No active socket for project {project_short_id}")
    
    # ... continue with setup logic ...
"""
