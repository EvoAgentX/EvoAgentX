"""
Workflow generation logic.
Phase 2: Generate workflow graph based on task_info.
"""

import os
from datetime import datetime
from typing import Dict, Any
from dotenv import load_dotenv

from evoagentx.workflow import WorkFlowGenerator, WorkFlowGraph, WorkFlow
from evoagentx.models import LLMConfig
from evoagentx.models.model_configs import OpenAILLMConfig, OpenRouterConfig
from evoagentx.models.model_utils import create_llm_instance
from evoagentx.core.module_utils import parse_json_from_text

from ..prompts import WORKFLOW_GENERATION_GOAL_PROMPT
from ..database.db import database

load_dotenv(os.path.join(os.path.dirname(__file__), '../config/app.env'))

# Default LLM configuration
default_llm_config = {
    "openai_key": os.getenv("OPENAI_API_KEY"),
    "model": "gpt-4o-mini",
    "temperature": 0.1
}


async def get_workflow(workflow_id: str) -> Dict[str, Any]:
    """Retrieve workflow information from the database"""
    workflow = await database.find_one("workflows", {"id": workflow_id})
    return workflow


async def update_workflow_status(workflow_id: str, status: str, **kwargs):
    """Update workflow status and other fields"""
    # Update status and any additional fields
    updates = {"status": status, "updated_at": datetime.now(), **kwargs}
    await database.update(
        "workflows", 
        {"id": workflow_id}, 
        updates
    )


async def generate_workflow_from_goal(goal: str, llm_config_dict: Dict[str, Any], mcp_config: dict = None) -> str:
    """
    Generate a workflow from a goal using the WorkFlowGenerator with tools.
    """
    try:
        # Start with the predefined generation tools
        from ..utils import generation_tools
        tools = generation_tools.copy()
        
        # Create LLM instance from config
        # Convert dict to proper LLMConfig object
        if llm_config_dict.get("openai_key"):
            from evoagentx.models.model_configs import OpenAILLMConfig
            llm_config = OpenAILLMConfig(**llm_config_dict)
        else:
            # Default to OpenAI if no specific config
            from evoagentx.models.model_configs import OpenAILLMConfig
            llm_config = OpenAILLMConfig(
                model="gpt-4o-mini",
                openai_key=os.getenv("OPENAI_API_KEY"),
                temperature=0.1
            )
        
        llm = create_llm_instance(llm_config)
        
        # Add MCP tools if config is provided
        if mcp_config:
            try:
                from evoagentx.tools import MCPToolkit
                mcp_toolkit = MCPToolkit(config=mcp_config)
                mcp_tools = mcp_toolkit.get_toolkits()
                tools.extend(mcp_tools)
                print(f"🔧 Added {len(mcp_tools)} MCP tools to generation tools")
            except Exception as e:
                print(f"⚠️  Failed to load MCP tools: {e}, proceeding with generation tools only")
        
        print(f"🔧 Using {len(tools)} total tools for workflow generation")
        
        # Initialize the WorkFlowGenerator with LLM and tools
        workflow_generator = WorkFlowGenerator(llm=llm, tools=tools)
        
        # Generate the actual workflow
        workflow_graph = workflow_generator.generate_workflow(goal=goal)
        
        # Return the workflow graph as a string (you might want to serialize this differently)
        return workflow_graph.get_config()
        
    except Exception as e:
        raise ValueError(f"Failed to generate workflow from goal: {str(e)}")


async def generate_workflow(workflow_id: str) -> Dict[str, Any]:
    """
    Phase 2: Generate workflow graph based on task_info.
    Updated to work with individual workflow records.
    """
    try:
        # Check if workflow exists
        workflow = await get_workflow(workflow_id)
        if not workflow:
            raise ValueError(f"Workflow with ID {workflow_id} not found")
        
        if workflow.get("task_info") is None or workflow.get("task_info").get("workflow_requirement") is None:
            raise ValueError(f"Workflow {workflow_id} has no workflow requirement")
        
        # Load MCP configuration for tools
        mcp_config = None
        try:
            # Try to load MCP config from the sample config file
            mcp_config_path = os.path.join(os.path.dirname(__file__), '../sample_mcp.config')
            if os.path.exists(mcp_config_path):
                import json
                with open(mcp_config_path, 'r') as f:
                    mcp_config = json.load(f)
                print(f"🔧 Loaded MCP configuration from {mcp_config_path}")
            else:
                print(f"⚠️  No MCP configuration found at {mcp_config_path}, proceeding without tools")
        except Exception as e:
            print(f"⚠️  Failed to load MCP configuration: {e}, proceeding without tools")
        
        formatted_goal = WORKFLOW_GENERATION_GOAL_PROMPT.format(
            workflow_inputs=workflow["task_info"]["workflow_inputs"],
            workflow_outputs=workflow["task_info"]["workflow_outputs"],
            requirement=workflow["task_info"]["workflow_requirement"]
        )
        
        # Generate the actual workflow using the WorkFlowGenerator with tools
        generated_workflow = await generate_workflow_from_goal(
                formatted_goal, 
                default_llm_config, 
                mcp_config=mcp_config  # Pass the actual MCP config instead of empty dict
            )
        
        # Use the generated workflow, not the existing one
        workflow_graph = generated_workflow
        
        await database.update(
            "workflows",
            {"id": workflow_id},
            {"workflow_graph": workflow_graph, "updated_at": datetime.now(), "status": "pending"}
        )
        return {
            "workflow_graph": workflow_graph,
            "status": "success"
        }
        
    except Exception as e:
        await update_workflow_status(workflow_id, "failed")
        raise ValueError(f"Failed to generate workflow: {str(e)}")
