"""
Workflow setup and project initialization logic.
Phase 1: Setup workflow with extraction AND generation.
"""

import asyncio
import json
import os
from typing import Dict, Any, List, Callable
from dotenv import load_dotenv
from datetime import datetime

from evoagentx.workflow import WorkFlowGenerator, WorkFlowGraph, WorkFlow
from evoagentx.models import LLMConfig
from evoagentx.models.model_configs import OpenAILLMConfig, OpenRouterConfig
from evoagentx.models.model_utils import create_llm_instance
from evoagentx.core.module_utils import parse_json_from_text
from evoagentx.tools import MCPToolkit
from evoagentx.tools import GoogleFreeSearchToolkit, DDGSSearchToolkit, WikipediaSearchToolkit, ArxivToolkit, StorageToolkit, CMDToolkit, RSSToolkit
from evoagentx.core.base_config import Parameter

from ..prompts import WORKFLOW_GENERATION_GOAL_PROMPT, WORKFLOW_REQUIREMENT_PROMPT
from ..database.db import database, requirement_database
from ..utils.websocket_utils import WebSocketProgressTracker, send_progress_message, send_log_message, send_error_message
from ..utils import generation_tools
from ..core.workflow_logging import isolated_workflow_process

load_dotenv(os.path.join(os.path.dirname(__file__), '../config/app.env'))

# Default LLM configuration
default_llm_config = {
    "openai_key": os.getenv("OPENAI_API_KEY"),
    "model": "gpt-4o-mini",
    "temperature": 0.1
}

# Supabase configuration
SUPABASE_BUCKET_STORAGE = os.getenv("SUPABASE_BUCKET_STORAGE", "requirements")

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
    sample_requirement_path = os.path.join('./debug/server/sample_requirement.md')
    with open(sample_requirement_path, 'r') as file:
        requirement_content = file.read()
    return requirement_content
    
    try:
        # Use the existing requirement_database client
        if not requirement_database.client:
            raise Exception("Requirement database client not connected")
        
        # Construct file path
        file_path = f"projects/{project_short_id}/requirement.md"
        
        # Download the requirement document using the existing client
        response = (
            requirement_database.client.storage
            .from_(SUPABASE_BUCKET_STORAGE)
            .download(file_path)
        )
        
        # Convert bytes to string
        requirement_content = response.decode("utf-8")
        
        print(f"✅ Retrieved requirement document for project {project_short_id}")
        return requirement_content
        
    except Exception as e:
        print(f"❌ Error retrieving requirement document: {str(e)}")
        raise Exception(f"Failed to retrieve requirement document: {str(e)}")


def _dicts_to_parameters(params: List[Dict[str, Any]]) -> List[Parameter]:
    """Convert list of dicts to Parameter objects."""
    if not params:
        return []
    result = []
    for p in params:
        result.append(Parameter(
            name=p.get("name", ""),
            type=p.get("type", "string"),
            description=p.get("description", ""),
            required=bool(p.get("required", True))
        ))
    return result


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


async def generate_workflow_from_goal_with_logging(payload, llm_config_dict: Dict[str, Any], mcp_config: dict = None, workflow_id: str = None, websocket_send_func: Callable = None) -> WorkFlowGraph:
    """
    Generate a workflow using the WorkFlowGenerator with tools and isolated logging.
    
    Args:
        payload: Dict with keys "goal", "workflow_inputs", "workflow_outputs" 
                 OR str (legacy support)
        llm_config_dict: LLM configuration dictionary
        mcp_config: Optional MCP configuration dictionary
        workflow_id: Workflow ID for isolated logging
        websocket_send_func: Optional WebSocket send function for logs
        
    Returns:
        WorkFlowGraph: The generated workflow graph
    """
    if websocket_send_func:
            from ..socket_management.protocols import create_message, MessageType
            status_message = create_message(
                MessageType.SETUP_LOG,
                status=None,
                workflow_id=workflow_id,
                content=f"Test loggings, starting workflow generation -1",
                result=None
            )
            await websocket_send_func(json.dumps(status_message))
    if workflow_id and websocket_send_func:
        # Use isolated logging
        with isolated_workflow_process(workflow_id, "generation", websocket_send_func) as (bound_logger, process_id):
            bound_logger.info(f"🏗️ Starting workflow generation with isolated logging")
            return await _generate_workflow_core(payload, llm_config_dict, mcp_config, bound_logger)
    else:
        # Fallback to original function without logging
        return await _generate_workflow_core(payload, llm_config_dict, mcp_config)

async def _generate_workflow_core(payload, llm_config_dict: Dict[str, Any], mcp_config: dict = None, logger_instance=None):
    """Core workflow generation logic."""
    # Use provided logger or print fallback
    def log_info(message):
        if logger_instance:
            logger_instance.info(message)
        else:
            print(message)
    
    def log_error(message):
        if logger_instance:
            logger_instance.error(message)
        else:
            print(f"ERROR: {message}")
    
    # Start with the predefined generation tools
    tools = generation_tools.copy()
    
    try:
        # Convert dictionary to appropriate LLM config object and create LLM instance
        llm_config = create_llm_config(llm_config_dict)
        llm = create_llm_instance(llm_config)
        
        # Add MCP tools if config is provided
        if mcp_config:
            try:
                from evoagentx.tools import MCPToolkit
                mcp_toolkit = MCPToolkit(config=mcp_config)
                mcp_tools = mcp_toolkit.get_toolkits()
                tools.extend(mcp_tools)
                log_info(f"🔧 Added {len(mcp_tools)} MCP tools to generation tools")
            except Exception as e:
                log_error(f"⚠️  Failed to load MCP tools: {e}, proceeding with generation tools only")
        
        log_info(f"🔧 Using {len(tools)} total tools for workflow generation")
        
    except Exception as e:
        log_error(f"Error initializing components: {e}")
        return None
    
    # Extract parameters from payload
    if isinstance(payload, str):
        # Legacy support - treat as goal only
        goal = payload
        workflow_inputs = []
        workflow_outputs = []
    else:
        # New dict format
        goal = payload.get("goal")
        workflow_inputs = _dicts_to_parameters(payload.get("workflow_inputs", []))
        workflow_outputs = _dicts_to_parameters(payload.get("workflow_outputs", []))
    
    log_info(f"📝 Generating workflow for goal: {goal[:100]}...")
    
    # Initialize and generate workflow
    workflow_generator = WorkFlowGenerator(llm=llm, tools=tools)
    
    # Generate the workflow with proper parameters
    workflow_graph: WorkFlowGraph = workflow_generator.generate_workflow(
        goal=goal,
        workflow_inputs=workflow_inputs,
        workflow_outputs=workflow_outputs
    )
    
    log_info(f"✅ Workflow generation completed successfully")
    return workflow_graph

async def generate_workflow_from_goal(payload, llm_config_dict: Dict[str, Any], mcp_config: dict = None) -> WorkFlowGraph:
    """
    Generate a workflow using the WorkFlowGenerator with tools.
    
    Args:
        payload: Dict with keys "goal", "workflow_inputs", "workflow_outputs" 
                 OR str (legacy support)
        llm_config_dict: LLM configuration dictionary
        mcp_config: Optional MCP configuration dictionary
        
    Returns:
        WorkFlowGraph: The generated workflow graph
    """
    # Start with the predefined generation tools
    tools = generation_tools.copy()
    
    try:
        # Convert dictionary to appropriate LLM config object and create LLM instance
        llm_config = create_llm_config(llm_config_dict)
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
        
    except Exception as e:
        print(f"Error initializing components: {e}")
        return None
    
    # Extract parameters from payload
    if isinstance(payload, str):
        # Legacy support - treat as goal only
        goal = payload
        workflow_inputs = []
        workflow_outputs = []
    else:
        # New dict format
        goal = payload.get("goal")
        workflow_inputs = _dicts_to_parameters(payload.get("workflow_inputs", []))
        workflow_outputs = _dicts_to_parameters(payload.get("workflow_outputs", []))
    
    # Initialize and generate workflow
    workflow_generator = WorkFlowGenerator(llm=llm, tools=tools)
    
    # Generate the workflow with proper parameters
    workflow_graph: WorkFlowGraph = workflow_generator.generate_workflow(
        goal=goal,
        workflow_inputs=workflow_inputs,
        workflow_outputs=workflow_outputs
    )
    return workflow_graph


async def setup_project(project_short_id: str, websocket_send_func: Callable = None) -> List[Dict[str, Any]]:
    """
    Phase 1: Setup workflow with extraction AND generation.
    Returns a list of workflow configurations.
    
    Args:
        project_short_id: The project identifier
        websocket_send_func: Optional WebSocket send function for real-time updates
    """
    # Retrieve requirement document from storage
    print(f"📥 Retrieving requirement document for project {project_short_id}...")
    detailed_requirements = await retrieve_requirement_from_storage(project_short_id)
    
    # Extract workflows and database info
    print(f"🔍 Extracting workflows from detailed requirements...")
    extracted_data = await extract_workflow_requirements(detailed_requirements)
    
    print(f"✅ Extracted {len(extracted_data['workflows'])} workflows")
    
    # Create initial workflow records with "uninitialized" status
    print(f"📝 Creating initial workflow records...")
    for extracted_workflow in extracted_data["workflows"]:
        workflow_id = extracted_workflow["workflow_id"]
        task_info = {
            "workflow_name": extracted_workflow["workflow_name"],
            "workflow_requirement": extracted_workflow["workflow_requirement"],
            "workflow_inputs": extracted_workflow["workflow_inputs"],
            "workflow_outputs": extracted_workflow["workflow_outputs"],
            "database_information": extracted_data["database_information"]
        }
        
        # Create initial workflow document with "uninitialized" status
        workflow_doc = {
            "id": workflow_id,
            "status": "uninitialized",  # Start with uninitialized
            "task_info": task_info,
            "workflow_graph": None,  # No workflow graph yet
            "project_short_id": project_short_id,
            "execution_result": None
        }
        
        await database.insert("workflows", workflow_doc)
        print(f"📝 Created initial workflow record: {workflow_id} (status: uninitialized)")
    
    # Generate workflows for each extracted workflow
    print(f"🏗️ Generating workflows...")
    generated_workflows = []
    for extracted_workflow in extracted_data["workflows"]:
        print(f"   Generating workflow: {extracted_workflow['workflow_name']}")
        
        # Create payload dict with goal and parameters
        payload = {
            "goal": extracted_workflow["workflow_requirement"],
            "workflow_inputs": extracted_workflow["workflow_inputs"],
            "workflow_outputs": extracted_workflow["workflow_outputs"]
        }
        
        # Generate workflow with isolated logging
        workflow_graph = await generate_workflow_from_goal_with_logging(
            payload, 
            default_llm_config, 
            mcp_config={},
            workflow_id=workflow_id,
            websocket_send_func=websocket_send_func
        )
        
        
        try:
            workflow_dict = workflow_graph.get_config()
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
    
    # Update workflow records with generated workflows and set status to pending
    workflow_configs = []
    for workflow_data in generated_workflows:
        workflow_id = workflow_data["workflow_id"]
        
        # Update the existing workflow record with generated data and set status to pending
        await database.update(
            "workflows",
            {"id": workflow_id},
            {
                "workflow_graph": workflow_data["workflow_graph"],
                "status": "pending",  # Set to pending after successful generation
                "updated_at": datetime.now()
            }
        )
        print(f"✅ Updated workflow record: {workflow_id} (status: pending)")
        
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

async def setup_project_parallel(project_short_id: str, websocket_send_func: Callable = None) -> List[Dict[str, Any]]:
    """
    Phase 1: Setup workflow with extraction AND parallel generation with retry logic.
    This is an enhanced version with parallel execution and automatic retries.
    
    Args:
        project_short_id: The project identifier
        websocket_send_func: Optional WebSocket send function for real-time updates
        
    Returns:
        List of workflow configurations.
    """
    # Retrieve requirement document from storage
    print(f"📥 Retrieving requirement document for project {project_short_id}...")
    detailed_requirements = await retrieve_requirement_from_storage(project_short_id)
    
    # Extract workflows and database info
    print(f"🔍 Extracting workflows from detailed requirements...")
    extracted_data = await extract_workflow_requirements(detailed_requirements)
    
    print(f"✅ Extracted {len(extracted_data['workflows'])} workflows")
    
    # Create initial workflow records with "uninitialized" status
    print(f"📝 Creating initial workflow records...")
    workflow_ids = []
    for extracted_workflow in extracted_data["workflows"]:
        workflow_id = extracted_workflow["workflow_id"]
        workflow_ids.append(workflow_id)
        task_info = {
            "workflow_name": extracted_workflow["workflow_name"],
            "workflow_requirement": extracted_workflow["workflow_requirement"],
            "workflow_inputs": extracted_workflow["workflow_inputs"],
            "workflow_outputs": extracted_workflow["workflow_outputs"],
            "database_information": extracted_data["database_information"]
        }
        
        # Create initial workflow document with "uninitialized" status
        workflow_doc = {
            "id": workflow_id,
            "status": "uninitialized",  # Start with uninitialized
            "task_info": task_info,
            "workflow_graph": None,  # No workflow graph yet
            "project_short_id": project_short_id,
            "execution_result": None
        }
        
        await database.insert("workflows", workflow_doc)
        print(f"📝 Created initial workflow record: {workflow_id} (status: uninitialized)")
    
    # Send status update: uninitialized after requirement extraction and before generation
    print(f"📤 Sending status update: uninitialized for {len(workflow_ids)} workflows")
    
    # Generate workflows for each extracted workflow in parallel with retry logic
    print(f"🏗️ Generating workflows in parallel with retry logic...")
    
    # Create semaphore to limit concurrent LLM API calls (rate limiting)
    concurrency_level = int(os.getenv("PARALLEL_WORKFLOW_CONCURRENCY", "5"))
    semaphore = asyncio.Semaphore(concurrency_level)  # Use configurable concurrency level
    print(f"   Using concurrency level: {concurrency_level}")
    
    async def generate_single_workflow_with_retry(extracted_workflow: Dict[str, Any], max_retries: int = 2) -> Dict[str, Any]:
        """Generate a single workflow with retry logic and rate limiting"""
        workflow_id = extracted_workflow["workflow_id"]
        workflow_name = extracted_workflow["workflow_name"]
        
        for attempt in range(max_retries + 1):
            try:
                async with semaphore:
                    if attempt > 0:
                        print(f"   🔄 Retry {attempt} for workflow: {workflow_name}")
                    else:
                        print(f"   Generating workflow: {workflow_name}")
                    
                    # Create payload dict with goal and parameters
                    payload = {
                        "goal": extracted_workflow["workflow_requirement"],
                        "workflow_inputs": extracted_workflow["workflow_inputs"],
                        "workflow_outputs": extracted_workflow["workflow_outputs"]
                    }
                    
                    # Generate workflow with isolated logging
                    workflow_graph = await generate_workflow_from_goal_with_logging(
                        payload, 
                        default_llm_config, 
                        mcp_config={},
                        workflow_id=workflow_id,
                        websocket_send_func=websocket_send_func
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
                    
                    return {
                        "workflow_name": workflow_name,
                        "workflow_id": workflow_id,
                        "workflow_requirement": extracted_workflow["workflow_requirement"],
                        "workflow_inputs": extracted_workflow["workflow_inputs"],
                        "workflow_outputs": extracted_workflow["workflow_outputs"],
                        "workflow_graph": workflow_dict,
                        "success": True
                    }
                    
            except Exception as e:
                error_msg = f"Attempt {attempt + 1} failed: {str(e)}"
                print(f"   ❌ {error_msg}")
                
                if attempt < max_retries:
                    # Wait before retry (exponential backoff)
                    wait_time = 2 ** attempt
                    print(f"   ⏳ Waiting {wait_time}s before retry...")
                    await asyncio.sleep(wait_time)
                else:
                    # Final failure - return error info instead of updating database
                    return {
                        "workflow_name": workflow_name,
                        "workflow_id": workflow_id,
                        "workflow_requirement": extracted_workflow["workflow_requirement"],
                        "workflow_inputs": extracted_workflow["workflow_inputs"],
                        "workflow_outputs": extracted_workflow["workflow_outputs"],
                        "workflow_graph": f"Workflow generation failed after {max_retries + 1} attempts. Last error: {str(e)}",
                        "success": False,
                        "error": str(e)
                    }
    
    # Create concurrent tasks for all workflow generations
    workflow_tasks = [
        generate_single_workflow_with_retry(extracted_workflow) 
        for extracted_workflow in extracted_data["workflows"]
    ]
    
    # Execute all workflow generations concurrently
    generated_workflows = await asyncio.gather(*workflow_tasks, return_exceptions=True)
    
    # Handle results from parallel execution
    processed_workflows = []
    for i, result in enumerate(generated_workflows):
        if isinstance(result, Exception):
            print(f"❌ Error generating workflow {extracted_data['workflows'][i]['workflow_name']}: {str(result)}")
            # Create a fallback workflow entry for failed generations
            fallback_workflow = {
                "workflow_name": extracted_data['workflows'][i]['workflow_name'],
                "workflow_id": extracted_data['workflows'][i]['workflow_id'],
                "workflow_requirement": extracted_data['workflows'][i]['workflow_requirement'],
                "workflow_inputs": extracted_data['workflows'][i]['workflow_inputs'],
                "workflow_outputs": extracted_data['workflows'][i]['workflow_outputs'],
                "workflow_graph": f"Workflow generation failed after retries: {str(result)}",
                "success": False,
                "error": str(result)
            }
            processed_workflows.append(fallback_workflow)
        else:
            # Check if the result has success flag
            if hasattr(result, 'get') and result.get("success") is False:
                print(f"❌ Workflow generation failed for {result['workflow_name']}: {result.get('error', 'Unknown error')}")
            else:
                print(f"✅ Workflow generation succeeded for {result['workflow_name']}")
            processed_workflows.append(result)
    
    print(f"✅ Generated {len(processed_workflows)} workflows (parallel execution with retry)")
    
    # Update workflow records with generated workflows and set appropriate status
    workflow_configs = []
    for workflow_data in processed_workflows:
        workflow_id = workflow_data["workflow_id"]
        
        # Set status based on generation success
        if workflow_data.get("success", True):
            status = "pending"  # Set to pending after successful generation
        else:
            status = "failed"   # Set to failed if generation failed
        
        # Update the existing workflow record with generated data and status
        await database.update(
            "workflows",
            {"id": workflow_id},
            {
                "workflow_graph": workflow_data["workflow_graph"],
                "status": status,
                "updated_at": datetime.now()
            }
        )
        print(f"✅ Updated workflow record: {workflow_id} (status: {status})")
        
        # Create workflow config for response
        workflow_config = {
            "workflow_id": workflow_id,
            "workflow_name": workflow_data["workflow_name"],
            "workflow_inputs": workflow_data["workflow_inputs"],
            "workflow_outputs": workflow_data["workflow_outputs"],
            "workflow_graph": workflow_data["workflow_graph"],
        }
        
        workflow_configs.append(workflow_config)
    
    # Send status update: pending after successful generation
    print(f"📤 Sending status update: pending for {len(workflow_configs)} workflows")
    
    return workflow_configs

async def setup_project_parallel_with_status_messages(project_short_id: str, websocket_send_func: Callable = None) -> List[Dict[str, Any]]:
    """
    Phase 1: Setup workflow with extraction AND parallel generation with retry logic.
    This version sends WebSocket status messages in the new format if a websocket_send_func is provided.
    """
    # Retrieve requirement document from storage
    print(f"📥 Retrieving requirement document for project {project_short_id}...")
    detailed_requirements = await retrieve_requirement_from_storage(project_short_id)
    
    # Extract workflows and database info
    print(f"🔍 Extracting workflows from detailed requirements...")
    extracted_data = await extract_workflow_requirements(detailed_requirements)
    
    print(f"✅ Extracted {len(extracted_data['workflows'])} workflows")
    
    # Create initial workflow records with "uninitialized" status
    print(f"📝 Creating initial workflow records...")
    workflow_ids = []
    for extracted_workflow in extracted_data["workflows"]:
        workflow_id = extracted_workflow["workflow_id"]
        workflow_ids.append(workflow_id)
        task_info = {
            "workflow_name": extracted_workflow["workflow_name"],
            "workflow_requirement": extracted_workflow["workflow_requirement"],
            "workflow_inputs": extracted_workflow["workflow_inputs"],
            "workflow_outputs": extracted_workflow["workflow_outputs"],
            "database_information": extracted_data["database_information"]
        }
        
        # Create initial workflow document with "uninitialized" status
        workflow_doc = {
            "id": workflow_id,
            "status": "uninitialized",  # Start with uninitialized
            "task_info": task_info,
            "workflow_graph": None,  # No workflow graph yet
            "project_short_id": project_short_id,
            "execution_result": None
        }
        
        await database.insert("workflows", workflow_doc)
        print(f"📝 Created initial workflow record: {workflow_id} (status: uninitialized)")
        
        # Send status update: uninitialized after workflow is extracted and added to database
        if websocket_send_func:
            from ..socket_management.protocols import create_message, MessageType
            status_message = create_message(
                MessageType.SETUP_LOG,
                status=None,
                workflow_id=workflow_id,
                content=f"{workflow_id} updates database status to: uninitialized",
                result=None
            )
            await websocket_send_func(json.dumps(status_message))
    
    # Generate workflows for each extracted workflow in parallel with retry logic
    print(f"🏗️ Generating workflows in parallel with retry logic...")
    
    # Create semaphore to limit concurrent LLM API calls (rate limiting)
    concurrency_level = int(os.getenv("PARALLEL_WORKFLOW_CONCURRENCY", "5"))
    semaphore = asyncio.Semaphore(concurrency_level)  # Use configurable concurrency level
    print(f"   Using concurrency level: {concurrency_level}")
    
    if websocket_send_func:
        from ..socket_management.protocols import create_message, MessageType
    status_message = create_message(
        MessageType.SETUP_LOG,
        status=None,
        workflow_id=workflow_id,
        content=f"Test loggings, starting workflow generation 12",
        result=None
    )
    await websocket_send_func(json.dumps(status_message))
    
    async def generate_single_workflow_with_retry(extracted_workflow: Dict[str, Any], max_retries: int = 2) -> Dict[str, Any]:
        """Generate a single workflow with retry logic and rate limiting"""
        workflow_id = extracted_workflow["workflow_id"]
        workflow_name = extracted_workflow["workflow_name"]
        
        status_message = create_message(
            MessageType.SETUP_LOG,
            status=None,
            workflow_id=workflow_id,
            content=f"Start function generate_single_workflow_with_retry",
            result=None
        )
        
        for attempt in range(max_retries + 1):
            try:
                async with semaphore:
                    if attempt > 0:
                        print(f"   🔄 Retry {attempt} for workflow: {workflow_name}")
                    else:
                        print(f"   Generating workflow: {workflow_name}")
                    
                    # Create payload dict with goal and parameters
                    payload = {
                        "goal": extracted_workflow["workflow_requirement"],
                        "workflow_inputs": extracted_workflow["workflow_inputs"],
                        "workflow_outputs": extracted_workflow["workflow_outputs"]
                    }
                    
                    # Generate workflow with isolated logging
                    workflow_graph = await generate_workflow_from_goal_with_logging(
                        payload, 
                        default_llm_config, 
                        mcp_config={},
                        workflow_id=workflow_id,
                        websocket_send_func=websocket_send_func
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
                    
                    return {
                        "workflow_name": workflow_name,
                        "workflow_id": workflow_id,
                        "workflow_requirement": extracted_workflow["workflow_requirement"],
                        "workflow_inputs": extracted_workflow["workflow_inputs"],
                        "workflow_outputs": extracted_workflow["workflow_outputs"],
                        "workflow_graph": workflow_dict,
                        "success": True
                    }
                    
            except Exception as e:
                error_msg = f"Attempt {attempt + 1} failed: {str(e)}"
                print(f"   ❌ {error_msg}")
                
                if attempt < max_retries:
                    # Wait before retry (exponential backoff)
                    wait_time = 2 ** attempt
                    print(f"   ⏳ Waiting {wait_time}s before retry...")
                    await asyncio.sleep(wait_time)
                else:
                    # Final failure - return error info instead of updating database
                    return {
                        "workflow_name": workflow_name,
                        "workflow_id": workflow_id,
                        "workflow_requirement": extracted_workflow["workflow_requirement"],
                        "workflow_inputs": extracted_workflow["workflow_inputs"],
                        "workflow_outputs": extracted_workflow["workflow_outputs"],
                        "workflow_graph": f"Workflow generation failed after {max_retries + 1} attempts. Last error: {str(e)}",
                        "success": False,
                        "error": str(e)
                    }
    
    # Create concurrent tasks for all workflow generations
    workflow_tasks = [
        generate_single_workflow_with_retry(extracted_workflow) 
        for extracted_workflow in extracted_data["workflows"]
    ]
    
    # Execute all workflow generations concurrently
    generated_workflows = await asyncio.gather(*workflow_tasks, return_exceptions=True)
    
    # Handle results from parallel execution
    processed_workflows = []
    for i, result in enumerate(generated_workflows):
        if isinstance(result, Exception):
            print(f"❌ Error generating workflow {extracted_data['workflows'][i]['workflow_name']}: {str(result)}")
            # Create a fallback workflow entry for failed generations
            fallback_workflow = {
                "workflow_name": extracted_data['workflows'][i]['workflow_name'],
                "workflow_id": extracted_data['workflows'][i]['workflow_id'],
                "workflow_requirement": extracted_data['workflows'][i]['workflow_requirement'],
                "workflow_inputs": extracted_data['workflows'][i]['workflow_inputs'],
                "workflow_outputs": extracted_data['workflows'][i]['workflow_outputs'],
                "workflow_graph": f"Workflow generation failed after retries: {str(result)}",
                "success": False,
                "error": str(result)
            }
            processed_workflows.append(fallback_workflow)
        else:
            # Check if the result has success flag
            if hasattr(result, 'get') and result.get("success") is False:
                print(f"❌ Workflow generation failed for {result['workflow_name']}: {result.get('error', 'Unknown error')}")
            else:
                print(f"✅ Workflow generation succeeded for {result['workflow_name']}")
            processed_workflows.append(result)
    
    print(f"✅ Generated {len(processed_workflows)} workflows (parallel execution with retry)")
    
    # Update workflow records with generated workflows and set appropriate status
    workflow_configs = []
    successful_workflows = []
    for workflow_data in processed_workflows:
        workflow_id = workflow_data["workflow_id"]
        
        # Set status based on generation success
        if workflow_data.get("success", True):
            status = "pending"  # Set to pending after successful generation
            successful_workflows.append(workflow_id)
        else:
            status = "failed"   # Set to failed if generation failed
        
        # Update the existing workflow record with generated data and status
        await database.update(
            "workflows",
            {"id": workflow_id},
            {
                "workflow_graph": workflow_data["workflow_graph"],
                "status": status,
                "updated_at": datetime.now()
            }
        )
        print(f"✅ Updated workflow record: {workflow_id} (status: {status})")
        
        # Send status update: pending after successful generation
        if websocket_send_func and workflow_data.get("success", True):
            status_message = create_message(
                MessageType.SETUP_LOG,
                status=None,
                workflow_id=workflow_id,
                content=f"{workflow_id} updates database status to: pending",
                result=None
            )
            await websocket_send_func(json.dumps(status_message))
        
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

async def get_project_workflow_status(project_short_id: str) -> Dict[str, Any]:
    """
    Get the status of parallel workflow generation for a project.
    Returns detailed status of all workflows being generated.
    """
    try:
        # Find all workflows for the project
        workflows = await database.find_many("workflows", {"project_short_id": project_short_id})
        
        if not workflows:
            raise ValueError(f"No workflows found for project {project_short_id}")
        
        # Calculate overall statistics
        total_workflows = len(workflows)
        uninitialized_workflows = sum(1 for w in workflows if w.get("status") == "uninitialized")
        completed_workflows = sum(1 for w in workflows if w.get("status") == "completed")
        failed_workflows = sum(1 for w in workflows if w.get("status") == "failed")
        pending_workflows = sum(1 for w in workflows if w.get("status") == "pending")
        
        # Determine overall status
        if failed_workflows == total_workflows:
            overall_status = "failed"
        elif uninitialized_workflows == total_workflows:
            overall_status = "uninitialized"
        elif pending_workflows == total_workflows:
            overall_status = "pending"
        elif completed_workflows == total_workflows:
            overall_status = "completed"
        elif failed_workflows > 0:
            overall_status = "completed_with_failures"
        else:
            overall_status = "running"
        
        # Build individual workflow statuses
        workflow_statuses = []
        for workflow in workflows:
            # Calculate progress based on status
            if workflow.get("status") == "completed":
                progress = 1.0
            elif workflow.get("status") == "failed":
                progress = 0.0
            elif workflow.get("status") == "pending":
                progress = 0.5  # Generation complete, ready for execution
            elif workflow.get("status") == "uninitialized":
                progress = 0.0  # Not started yet
            else:
                progress = 0.0
            
            # Extract error information from workflow_graph if it contains error message
            error_message = None
            workflow_graph = workflow.get("workflow_graph", "")
            if isinstance(workflow_graph, str) and "Workflow generation failed" in workflow_graph:
                error_message = workflow_graph
            
            # Handle datetime fields safely
            started_at = None
            completed_at = None
            
            created_at = workflow.get("created_at")
            if created_at:
                if hasattr(created_at, 'isoformat'):
                    started_at = created_at.isoformat()
                elif isinstance(created_at, str):
                    started_at = created_at
                else:
                    started_at = str(created_at)
            
            updated_at = workflow.get("updated_at")
            if updated_at and workflow.get("status") == "completed":
                if hasattr(updated_at, 'isoformat'):
                    completed_at = updated_at.isoformat()
                elif isinstance(updated_at, str):
                    completed_at = updated_at
                else:
                    completed_at = str(updated_at)
            
            status = {
                "workflow_id": workflow["id"],
                "workflow_name": workflow.get("task_info", {}).get("workflow_name", "Unknown"),
                "status": workflow.get("status", "pending"),
                "progress": progress,
                "error_message": error_message,
                "started_at": started_at,
                "completed_at": completed_at
            }
            workflow_statuses.append(status)
        
        # Calculate estimated completion time based on current progress
        estimated_completion_time = None
        if overall_status == "running" and completed_workflows > 0:
            # Simple estimation: assume remaining workflows take same time as completed ones
            remaining_workflows = total_workflows - completed_workflows - failed_workflows
            if remaining_workflows > 0:
                # This is a placeholder - in a real implementation, you'd track actual timing
                estimated_minutes = remaining_workflows * 2  # Assume 2 minutes per workflow
                estimated_completion_time = f"~{estimated_minutes} minutes"
        
        return {
            "project_short_id": project_short_id,
            "total_workflows": total_workflows,
            "uninitialized_workflows": uninitialized_workflows,
            "completed_workflows": completed_workflows,
            "failed_workflows": failed_workflows,
            "pending_workflows": pending_workflows,
            "workflows": workflow_statuses,
            "overall_status": overall_status,
            "estimated_completion_time": estimated_completion_time,
            "success_rate": f"{(completed_workflows / total_workflows * 100):.1f}%" if total_workflows > 0 else "0%"
        }
        
    except Exception as e:
        raise ValueError(f"Failed to get project workflow status: {str(e)}")


async def setup_project_parallel_with_websocket(
    project_short_id: str, 
    websocket_send_func: Callable
) -> List[Dict[str, Any]]:
    """
    Setup project with parallel workflow generation using WebSocket for real-time progress updates.
    This function demonstrates how to reuse the generalized websocket utilities.
    
    Args:
        project_short_id: The project identifier
        websocket_send_func: Function to send messages via WebSocket
        
    Returns:
        List of generated workflow configurations
    """
    progress_tracker = WebSocketProgressTracker(websocket_send_func, project_short_id, "project_setup")
    
    try:
        # Send connection confirmation
        await progress_tracker.send_connection_confirmation()
        
        # Send start notification
        await progress_tracker.send_start_notification("Parallel project setup")
        
        # Send initial progress
        await progress_tracker.send_progress_update("initializing", 0.0, "Starting parallel project setup...")
        
        # Step 1: Retrieve requirement document
        await progress_tracker.send_progress_update("retrieving", 0.1, "Retrieving requirement document...")
        await progress_tracker.send_log_message("INFO", f"Retrieving requirement document for project {project_short_id}")
        
        detailed_requirements = await retrieve_requirement_from_storage(project_short_id)
        
        await progress_tracker.send_progress_update("retrieving", 0.2, "Requirement document retrieved successfully")
        await progress_tracker.send_output_message(f"Retrieved requirement document: {len(detailed_requirements)} characters")
        
        # Step 2: Extract workflow requirements
        await progress_tracker.send_progress_update("extracting", 0.3, "Extracting workflow requirements...")
        await progress_tracker.send_log_message("INFO", "Extracting workflows from detailed requirements")
        
        extracted_data = await extract_workflow_requirements(detailed_requirements)
        workflows_to_generate = extracted_data["workflows"]
        
        await progress_tracker.send_progress_update("extracting", 0.4, f"Extracted {len(workflows_to_generate)} workflows")
        await progress_tracker.send_output_message(f"Extracted {len(workflows_to_generate)} workflows for parallel generation")
        
        # Step 3: Generate workflows in parallel
        await progress_tracker.send_progress_update("generating", 0.5, "Starting parallel workflow generation...")
        await progress_tracker.send_log_message("INFO", f"Starting parallel generation of {len(workflows_to_generate)} workflows")
        
        # Create tasks for parallel execution
        tasks = []
        for i, extracted_workflow in enumerate(workflows_to_generate):
            task = generate_single_workflow_with_websocket(
                extracted_workflow, 
                progress_tracker, 
                i, 
                len(workflows_to_generate)
            )
            tasks.append(task)
        
        # Execute all tasks in parallel
        generated_workflows = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        successful_workflows = []
        failed_workflows = []
        
        for i, result in enumerate(generated_workflows):
            if isinstance(result, Exception):
                failed_workflows.append({
                    "workflow_name": workflows_to_generate[i]["workflow_name"],
                    "error": str(result)
                })
                await progress_tracker.send_error(f"Workflow {workflows_to_generate[i]['workflow_name']} failed: {str(result)}")
            else:
                successful_workflows.append(result)
                await progress_tracker.send_log_message("INFO", f"Successfully generated workflow: {result['workflow_name']}")
        
        # Step 4: Save to database
        await progress_tracker.send_progress_update("saving", 0.9, "Saving workflows to database...")
        await progress_tracker.send_log_message("INFO", "Saving generated workflows to database")
        
        # Save successful workflows
        for workflow in successful_workflows:
            try:
                await database.insert("workflows", workflow)
                await progress_tracker.send_output_message(f"Saved workflow: {workflow['workflow_name']}")
            except Exception as e:
                await progress_tracker.send_error(f"Failed to save workflow {workflow['workflow_name']}: {str(e)}")
        
        # Final progress update
        await progress_tracker.send_progress_update("completed", 1.0, "Parallel project setup completed")
        
        # Send completion message
        completion_result = {
            "total_workflows": len(workflows_to_generate),
            "successful_workflows": len(successful_workflows),
            "failed_workflows": len(failed_workflows),
            "workflows": successful_workflows,
            "errors": failed_workflows
        }
        
        await progress_tracker.send_completion(completion_result, "Parallel project setup completed successfully")
        
        return successful_workflows
        
    except Exception as e:
        await progress_tracker.send_error(f"Parallel project setup failed: {str(e)}")
        raise e


async def generate_single_workflow_with_websocket(
    extracted_workflow: Dict[str, Any], 
    progress_tracker: WebSocketProgressTracker,
    workflow_index: int,
    total_workflows: int
) -> Dict[str, Any]:
    """
    Generate a single workflow with websocket progress updates.
    This function demonstrates how to use the convenience functions from websocket_utils.
    
    Args:
        extracted_workflow: The extracted workflow data
        progress_tracker: The websocket progress tracker
        workflow_index: Index of this workflow in the batch
        total_workflows: Total number of workflows being generated
        
    Returns:
        Generated workflow configuration
    """
    workflow_name = extracted_workflow["workflow_name"]
    
    try:
        # Calculate progress for this specific workflow
        base_progress = workflow_index / total_workflows
        workflow_progress = base_progress + (1.0 / total_workflows) * 0.5  # 50% of this workflow's progress
        
        await progress_tracker.send_progress_update(
            "generating", 
            workflow_progress, 
            f"Generating workflow: {workflow_name}"
        )
        
        # Create payload dict with goal and parameters
        payload = {
            "goal": extracted_workflow["workflow_requirement"],
            "workflow_inputs": extracted_workflow["workflow_inputs"],
            "workflow_outputs": extracted_workflow["workflow_outputs"]
        }
        
        workflow_graph = await generate_workflow_from_goal(
            payload, 
            default_llm_config, 
            mcp_config={}
        )
        
        # Update progress to completion
        completion_progress = base_progress + (1.0 / total_workflows)
        await progress_tracker.send_progress_update(
            "generating", 
            completion_progress, 
            f"Completed workflow: {workflow_name}"
        )
        
        # Format the result
        try:
            workflow_dict = workflow_graph.get_config()
        except Exception as e:
            workflow_dict = f"Workflow generated successfully (serialization error: {str(e)})"
        
        return {
            "workflow_name": extracted_workflow["workflow_name"],
            "workflow_id": extracted_workflow["workflow_id"],
            "workflow_requirement": extracted_workflow["workflow_requirement"],
            "workflow_inputs": extracted_workflow["workflow_inputs"],
            "workflow_outputs": extracted_workflow["workflow_outputs"],
            "workflow_graph": workflow_dict,
            "status": "pending",
            "created_at": datetime.now(),
            "updated_at": datetime.now()
        }
        
    except Exception as e:
        await progress_tracker.send_error(f"Failed to generate workflow {workflow_name}: {str(e)}")
        raise e
