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
from ..db import database

from ..utils.output_parser import parse_workflow_output
from ..utils.tool_creator import create_tools
from ..utils.websocket_utils import WebSocketEnhancedSink, WebSocketProgressTracker

load_dotenv(os.path.join(os.path.dirname(__file__), '../config/app.env'))

# Default LLM configuration
default_llm_config = {
    "model": "gpt-4o",
    "openai_key": os.getenv("OPENAI_API_KEY"),
    # "stream": True,
    "output_response": True,
    "max_tokens": 16000
}

# Placeholder for sudo execution result
sudo_execution_result = None


def _create_supabase_public_url(project_short_id: str, file_path: str) -> str:
    """Create Supabase signed URL for file access."""
    try:
        from ..utils.tool_creator import _create_storage_handler
        storage_handler = _create_storage_handler(project_short_id)
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

def _scan_and_replace_file_paths(data: Any, project_short_id: str) -> Any:
    """Recursively scan data structure and replace file paths with Supabase signed URLs."""
    if isinstance(data, dict):
        return {k: _scan_and_replace_file_paths(v, project_short_id) for k, v in data.items()}
    elif isinstance(data, list):
        return [_scan_and_replace_file_paths(item, project_short_id) for item in data]
    elif isinstance(data, str):
        # Match common file path patterns: /path/to/file.ext, C:\path\to\file.ext, ./file.ext
        file_path_pattern = r'^([A-Za-z]:\\)?([\/\\]?[^\/\\]+[\/\\])*[^\/\\]+\.[a-zA-Z0-9]+$'
        if re.match(file_path_pattern, data.strip()):
            return _create_supabase_public_url(project_short_id, data.strip())
        return data
    return data


async def get_workflow(workflow_id: str) -> Dict[str, Any]:
    """Retrieve workflow information from the database"""
    workflow = await database.find_one("workflows", {"id": workflow_id})
    return workflow


async def list_workflows(skip: int = 0, limit: int = 100, status: str = None) -> List[Dict[str, Any]]:
    """List workflows with optional filtering"""
    query = {}
    if status:
        query["status"] = status
    
    workflows = await database.find("workflows", query, skip=skip, limit=limit)
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


async def execute_workflow_from_config(workflow: Dict[str, Any], llm_config_dict: Dict[str, Any], mcp_config: dict = None, inputs: Dict[str, Any] = None, database_information: Dict[str, Any] = None, task_info: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Execute a workflow with the given configuration.
    
    Args:
        workflow: The complete workflow document from database (must contain project_short_id and workflow_graph)
        llm_config_dict: LLM configuration dictionary
        mcp_config: Optional MCP configuration dictionary
        inputs: Optional inputs dictionary to pass to async_execute
        database_information: Optional database information for dynamic MongoDB toolkit creation
        
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
        
        # Handle workflow graph extraction from the workflow document
        if isinstance(workflow, dict) and "workflow_graph" in workflow:
            # Extract workflow graph from the workflow document
            workflow_graph_data = workflow["workflow_graph"]
            if isinstance(workflow_graph_data, WorkFlowGraph):
                workflow_graph = workflow_graph_data
            else:
                workflow_graph: WorkFlowGraph = WorkFlowGraph.from_dict(workflow_graph_data)
        elif isinstance(workflow, WorkFlowGraph):
            # Direct WorkFlowGraph object (fallback for backward compatibility)
            workflow_graph = workflow
        else:
            # Try to treat the entire workflow as a workflow graph (fallback)
            workflow_graph: WorkFlowGraph = WorkFlowGraph.from_dict(workflow)
        
        # Create tools with proper storage handling
        tools = []
        if mcp_config:
            mcp_toolkit = MCPToolkit(config=mcp_config)
            tools = mcp_toolkit.get_tools()
        
        # Create tools with project_short_id for storage configuration
        if project_short_id:
            tools += create_tools(project_short_id, database_information)
        else:
            print("⚠️  No project_short_id found, creating tools without storage support")
            tools += create_tools("default", database_information)
        
        agent_manager = AgentManager(tools=tools)
        agent_manager.add_agents_from_workflow(workflow_graph, llm_config=llm_config)

        workflow = WorkFlow(graph=workflow_graph, agent_manager=agent_manager, llm=llm)
        workflow.init_module()
        output = await workflow.async_execute(inputs=inputs)
        
        # Use custom prompt to process the output and generate structured results
        if task_info and "workflow_outputs" in task_info:
            # Get expected outputs for the prompt
            
            # Get workflow goal
            goal = workflow_graph.goal if hasattr(workflow_graph, 'goal') else "Process the workflow execution results"
            
            # Format expected outputs for the prompt
            expected_outputs_formatted = []
            for output_param in task_info["workflow_outputs"]:
                expected_outputs_formatted.append({
                    "name": output_param["name"],
                    "type": output_param["type"],
                    "description": output_param["description"]
                })
            
            # Use custom prompt to generate structured output
            custom_prompt = CUSTOM_OUTPUT_EXTRACTION_PROMPT.format(
                expected_outputs=json.dumps(expected_outputs_formatted, indent=2),
                workflow_execution_results=output
            )
            
            # Generate structured output using the custom prompt
            try:
                structured_output = await llm.async_generate(prompt=custom_prompt)
                parsed_json = None
                
                if hasattr(structured_output, 'content'):
                    output_content = structured_output.content
                else:
                    output_content = str(structured_output)
                
                # Try to parse the structured output as JSON
                try:
                    # First, try to extract JSON from code blocks if present
                    import re
                    code_block_pattern = r'```(?:json)?\s*\n(.*?)\n\s*```'
                    matches = re.findall(code_block_pattern, output_content, re.DOTALL)
                    
                    if matches:
                        # Use the first code block match
                        json_content = matches[0].strip()
                        parsed_json = json.loads(json_content)
                    else:
                        # Try to parse the entire content as JSON
                        parsed_json = json.loads(output_content)
                        
                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse JSON from structured output: {e}")
                    print(f"Raw output content: {output_content}")
                    # If JSON parsing fails, create a simple structure
                    parsed_json = {"workflow_output": output_content}
                    
            except Exception as e:
                print(f"Warning: Failed to generate structured output: {e}")
                parsed_json = None
        else:
            parsed_json = None
        
        # Scan and replace file paths with URLs if we have parsed_json and project_short_id
        if parsed_json and project_short_id:
            parsed_json = _scan_and_replace_file_paths(parsed_json, project_short_id)
        
        return {
            "original_message": output,
            "parsed_json": parsed_json
        }
        
    except Exception as e:
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
        
        # Update workflow status
        await update_workflow_status(workflow_id, "running")
        
        # Get workflow graph (now a single workflow, not a list)
        workflow_graph = workflow["workflow_graph"]
        task_info = workflow.get("task_info", {})
        workflow_name = task_info.get("workflow_name", workflow_id)
        
        if workflow_graph is None:
            print(f"⚠️ No workflow graph available for {workflow_name}")
            await update_workflow_status(workflow_id, "failed")
            raise ValueError(f"Workflow {workflow_id} is not found")
            
        print(f"🚀 Executing workflow: {workflow_name}")
        
        # Get database information for dynamic MongoDB toolkit
        database_information = task_info.get("database_information")
        
        # Execute the workflow with database information
        execution_result = await execute_workflow_from_config(
            workflow,  # Pass the whole workflow document instead of just workflow_graph
            default_llm_config, 
            mcp_config={}, 
            inputs=inputs,
            database_information=database_information,
            task_info=task_info
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
        
        # Return only the essential data
        return execution_result["parsed_json"]
        
    except Exception as e:
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
        
        # Setup WebSocket enhanced sink
        websocket_sink = WebSocketEnhancedSink(websocket_send_func, workflow_id, "workflow")
        
        # Add custom sink to loguru
        sink_id = None
        try:
            sink_id = logger.add(websocket_sink.write, level="INFO", format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")
        except Exception as e:
            print(f"Warning: Could not add loguru sink: {e}")
        
        try:
            # Send progress for workflow initialization
            await progress_tracker.send_progress_update("initializing", 0.5, "Initializing workflow components...")
            
            # Execute the workflow (database tools disabled)
            execution_result = await execute_workflow_from_config(
                workflow,  # Pass the whole workflow document instead of just workflow_graph
                default_llm_config, 
                mcp_config={}, 
                inputs=inputs,
                database_information=database_information,
                task_info=task_info
            )
            
            await progress_tracker.send_progress_update("finalizing", 0.9, "Finalizing execution results...")
            
            if execution_result is None:
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
    
    except Exception as e:
        # Send connection error for unexpected failures
        await progress_tracker.send_error(f"Unexpected error: {str(e)}")
        await update_workflow_status(workflow_id, "failed")
        return {
            "original_message": f"Failed to execute workflow: {str(e)}",
            "parsed_json": None
        }
