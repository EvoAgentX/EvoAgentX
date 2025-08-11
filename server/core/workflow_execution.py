"""
Workflow execution logic.
Phase 3: Execute workflow with provided inputs.
"""

import json
import os
from typing import Dict, Any, Callable, List
from dotenv import load_dotenv

from evoagentx.workflow import WorkFlowGenerator, WorkFlowGraph, WorkFlow
from evoagentx.models import LLMConfig
from evoagentx.models.model_configs import OpenAILLMConfig, OpenRouterConfig
from evoagentx.models.model_utils import create_llm_instance
from evoagentx.agents.agent_manager import AgentManager
from evoagentx.tools import MCPToolkit

from ..prompts import CUSTOM_OUTPUT_EXTRACTION_PROMPT
from ..database.db import database
from ..utils.output_parser import parse_workflow_output
from ..utils.tool_creator import create_tools_with_database

load_dotenv(os.path.join(os.path.dirname(__file__), '../config/app.env'))

# Default LLM configuration
default_llm_config = {
    "openai_key": os.getenv("OPENAI_API_KEY"),
    "model": "gpt-4o-mini",
    "temperature": 0.1
}

# Placeholder for sudo execution result
sudo_execution_result = None


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
    from datetime import datetime
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
        workflow: The workflow definition/configuration to execute
        llm_config_dict: LLM configuration dictionary
        mcp_config: Optional MCP configuration dictionary
        inputs: Optional inputs dictionary to pass to async_execute
        database_information: Optional database information for dynamic MongoDB toolkit creation
        
    Returns:
        Dict containing only the essential execution results:
        - original_message: The raw output from workflow execution
        - parsed_json: Extracted JSON from the output (if found)
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
        
        # Handle both WorkFlowGraph objects and dictionaries
        if isinstance(workflow, WorkFlowGraph):
            workflow_graph = workflow
        else:
            workflow_graph: WorkFlowGraph = WorkFlowGraph.from_dict(workflow)
        
        # Create tools (database tools disabled)
        tools = []
        if mcp_config:
            mcp_toolkit = MCPToolkit(config=mcp_config)
            tools = mcp_toolkit.get_tools()
        tools += create_tools_with_database(database_information)
        
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
        
        return {
            "original_message": output,
            "parsed_json": parsed_json
        }
        
    except Exception as e:
        # Return error in the same format
        error_message = f"In the execution process, got error:\n{e}"
        return {
            "original_message": error_message,
            "parsed_json": None
        }


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
            return {
                "original_message": "No workflow graph available",
                "parsed_json": None
            }
        
        print(f"🚀 Executing workflow: {workflow_name}")
        
        # Get database information for dynamic MongoDB toolkit
        database_information = task_info.get("database_information")
        
        # Execute the workflow with database information
        execution_result = await execute_workflow_from_config(
            workflow_graph, 
            default_llm_config, 
            mcp_config={}, 
            inputs=inputs,
            database_information=database_information,
            task_info=task_info
        )
        
        if execution_result is None:
            print(f"❌ Failed to execute workflow: {workflow_name}")
            await update_workflow_status(workflow_id, "failed")
            return {
                "original_message": "Failed to execute workflow",
                "parsed_json": None
            }
        
        # Update workflow storage with execution results
        await update_workflow_status(
            workflow_id, 
            "completed",
            execution_result=execution_result
        )
        
        # Return only the essential data
        return execution_result
        
    except Exception as e:
        await update_workflow_status(workflow_id, "failed")
        return {
            "original_message": f"Failed to execute workflow: {str(e)}",
            "parsed_json": None
        }


# WebSocket execution will be implemented here
async def execute_workflow_with_websocket(
    workflow_id: str, 
    inputs: Dict[str, Any], 
    websocket_send_func: Callable
) -> Dict[str, Any]:
    """
    Execute workflow with WebSocket support for real-time progress updates.
    """
    # For now, just call the regular execution function
    # WebSocket implementation will be added later
    return await execute_workflow(workflow_id, inputs)
