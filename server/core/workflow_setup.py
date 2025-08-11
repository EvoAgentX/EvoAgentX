"""
Workflow setup and project initialization logic.
Phase 1: Setup workflow with extraction AND generation.
"""

import asyncio
import json
import os
from typing import Dict, Any, List
from dotenv import load_dotenv

from evoagentx.workflow import WorkFlowGenerator, WorkFlowGraph, WorkFlow
from evoagentx.models import LLMConfig
from evoagentx.models.model_configs import OpenAILLMConfig, OpenRouterConfig
from evoagentx.models.model_utils import create_llm_instance
from evoagentx.core.module_utils import parse_json_from_text

from ..prompts import WORKFLOW_GENERATION_GOAL_PROMPT, WORKFLOW_REQUIREMENT_PROMPT
from ..database.db import database, requirement_database

load_dotenv(os.path.join(os.path.dirname(__file__), '../config/app.env'))

# Default LLM configuration
default_llm_config = {
    "openai_key": os.getenv("OPENAI_API_KEY"),
    "model": "gpt-4o-mini",
    "temperature": 0.1
}

# Supabase configuration
SUPABASE_BUCKET_REQUIREMENT = os.getenv("SUPABASE_BUCKET_REQUIREMENT", "requirements")

async def retrieve_requirement_from_storage(project_short_id: str) -> str:
    """
    Retrieve requirement document from Supabase storage using project_short_id.
    
    Args:
        project_short_id: The project identifier
        
    Returns:
        str: The requirement document content as a string
        
    Raises:
        Exception: If the requirement document cannot be retrieved
    """
    try:
        # Use the existing requirement_database client
        if not requirement_database.client:
            raise Exception("Requirement database client not connected")
        
        # Construct file path
        file_path = f"projects/{project_short_id}/requirement.md"
        
        # Download the requirement document using the existing client
        response = (
            requirement_database.client.storage
            .from_(SUPABASE_BUCKET_REQUIREMENT)
            .download(file_path)
        )
        
        # Convert bytes to string
        requirement_content = response.decode("utf-8")
        
        print(f"✅ Retrieved requirement document for project {project_short_id}")
        return requirement_content
        
    except Exception as e:
        print(f"❌ Error retrieving requirement document: {str(e)}")
        raise Exception(f"Failed to retrieve requirement document: {str(e)}")


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


async def extract_workflow_requirements(detailed_requirements: str) -> Dict[str, Any]:
    """
    Simple extraction function that:
    - Takes detailed requirements document
    - Uses WORKFLOW_REQUIREMENT_PROMPT to extract workflows and database info
    - Simple validation
    - Returns: {workflows: [...], database_information: {...}}
    """
    try:
        # Use LLM to extract workflows and database info
        llm_config = create_llm_config(default_llm_config)
        llm = create_llm_instance(llm_config)
        
        # Format the prompt with the requirements
        extraction_prompt = WORKFLOW_REQUIREMENT_PROMPT.format(requirement=detailed_requirements)
        
        # Get LLM response
        response = llm.single_generate([{"role": "user", "content": extraction_prompt}])
        
        # Parse JSON from response
        extracted_data = parse_json_from_text(response)
        
        if not extracted_data:
            raise ValueError(f"No JSON found in LLM response: {response}")
        
        try:
            result = json.loads(extracted_data[0])
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in LLM response: {extracted_data[0]}")
        
        # Simple validation
        if "workflows" not in result:
            raise ValueError("No workflows found in extracted data")
        
        if "database_information" not in result:
            raise ValueError("No database information found in extracted data")
        
        return result
        
    except Exception as e:
        raise ValueError(f"Error extracting workflow requirements: {str(e)}")


async def generate_workflow_from_goal(goal: str, llm_config_dict: Dict[str, Any], mcp_config: dict = None) -> str:
    """
    Generate a workflow from a goal.
    """
    # For now, return a placeholder - this will be implemented in workflow_generation.py
    return f"Generated workflow for goal: {goal}"


async def setup_project(project_short_id: str) -> List[Dict[str, Any]]:
    """
    Phase 1: Setup workflow with extraction AND generation.
    Returns a list of workflow configurations.
    """
    # Retrieve requirement document from storage
    print(f"📥 Retrieving requirement document for project {project_short_id}...")
    detailed_requirements = await retrieve_requirement_from_storage(project_short_id)
    
    # Extract workflows and database info
    print(f"🔍 Extracting workflows from detailed requirements...")
    extracted_data = await extract_workflow_requirements(detailed_requirements)
    
    print(f"✅ Extracted {len(extracted_data['workflows'])} workflows")
    
    # Generate workflows for each extracted workflow
    print(f"🏗️ Generating workflows...")
    generated_workflows = []
    for extracted_workflow in extracted_data["workflows"]:
        print(f"   Generating workflow: {extracted_workflow['workflow_name']}")
        
        # Use WORKFLOW_GENERATION_GOAL_PROMPT with proper structure
        formatted_goal = WORKFLOW_GENERATION_GOAL_PROMPT.format(
            workflow_inputs=extracted_workflow["workflow_inputs"],
            workflow_outputs=extracted_workflow["workflow_outputs"],
            requirement=extracted_workflow["workflow_requirement"]
        )
        
        # Generate workflow
        workflow_graph = await generate_workflow_from_goal(
            formatted_goal, 
            default_llm_config, 
            mcp_config={}
        )
        
        
        try:
            if hasattr(workflow_graph, 'get_config'):
                workflow_dict = workflow_graph.get_config()
            elif hasattr(workflow_graph, 'get_workflow_description'):
                workflow_dict = {
                    "goal": workflow_graph.goal,
                    "description": workflow_graph.get_workflow_description()
                }
            else:
                workflow_dict = str(workflow_graph)
        except Exception as e:
            workflow_dict = f"Workflow generated successfully (serialization error: {str(e)})"
        
        
        generated_workflows.append({
            "workflow_name": extracted_workflow["workflow_name"],
            "workflow_id": extracted_workflow["workflow_id"],
            "workflow_requirement": extracted_workflow["workflow_requirement"],
            "workflow_inputs": extracted_workflow["workflow_inputs"],
            "workflow_outputs": extracted_workflow["workflow_outputs"],
            "workflow_graph": workflow_dict
        })
    
    print(f"✅ Generated {len(generated_workflows)} workflows")
    
    # Insert each workflow as individual records and create workflow configs
    workflow_configs = []
    for workflow_data in generated_workflows:
        workflow_id = workflow_data["workflow_id"]
        # Create task_info with workflow details
        task_info = {
            "workflow_name": workflow_data["workflow_name"],
            "workflow_requirement": workflow_data["workflow_requirement"],
            "workflow_inputs": workflow_data["workflow_inputs"],
            "workflow_outputs": workflow_data["workflow_outputs"],
            "database_information": extracted_data["database_information"]
        }
        
        # Create individual workflow document according to database schema
        workflow_doc = {
            "id": workflow_id,
            "status": "pending",
            "task_info": task_info,
            "workflow_graph": workflow_data["workflow_graph"],
            "project_short_id": project_short_id,
            "execution_result": None
        }
        
        await database.insert("workflows", workflow_doc)
        print(f"✅ Created workflow record: {workflow_id}")
        
        # Create workflow config for response
        workflow_config = {
            "workflow_id": workflow_id,
            "workflow_name": workflow_data["workflow_name"],
            "workflow_inputs": workflow_data["workflow_inputs"],
            "workflow_outputs": workflow_data["workflow_outputs"],
            "workflow_graph": workflow_data["workflow_graph"],
        }
        
        workflow_configs.append(workflow_config)
    
    return workflow_configs
