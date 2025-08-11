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
    Generate a workflow from a goal.
    """
    # For now, return a placeholder - this will be implemented with actual workflow generation
    return f"Generated workflow for goal: {goal}"


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
        
        formatted_goal = WORKFLOW_GENERATION_GOAL_PROMPT.format(
            workflow_inputs=workflow["task_info"]["workflow_inputs"],
            workflow_outputs=workflow["task_info"]["workflow_outputs"],
            requirement=workflow["task_info"]["workflow_requirement"]
        )
        
        workflow_graph = await generate_workflow_from_goal(
                formatted_goal, 
                default_llm_config, 
                mcp_config={}
            )
        
        # The workflow_graph now contains the single generated workflow from setup
        workflow_graph = workflow["workflow_graph"]
        
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
