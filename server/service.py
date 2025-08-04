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
from typing import Dict, Any, List, Optional, Callable
from io import StringIO
from dotenv import load_dotenv

from evoagentx.workflow import WorkFlowGenerator, WorkFlowGraph, WorkFlow
from evoagentx.models import LLMConfig
from evoagentx.models.model_configs import OpenAILLMConfig, OpenRouterConfig
from evoagentx.models.model_utils import create_llm_instance
from evoagentx.agents.agent_manager import AgentManager
from evoagentx.core.module_utils import parse_json_from_text
from evoagentx.tools import MCPToolkit

from .prompts import WORKFLOW_GENERATION_PROMPT, WORKFLOW_GENERATION_GOAL_PROMPT, WORKFLOW_REQUIREMENT_PROMPT, TASK_INFO_PROMPT_SUDO, CONNECTION_INSTRUCTION_PROMPT
from .db import database, requirement_database

import sys
import io
import threading
import time
from contextlib import redirect_stdout, redirect_stderr

load_dotenv(os.path.join(os.path.dirname(__file__), 'app.env'))
MONGODB_URL = os.getenv("MONGODB_URL", None)

def parse_workflow_output(output: str) -> Dict[str, Any]:
    """
    Parse workflow execution output to extract JSON and clean up markdown formatting.
    
    This function handles various output formats:
    1. Pure JSON: {"key": "value"}
    2. Markdown with JSON: ```json\n{"key": "value"}\n```
    3. Markdown with other content: ```markdown\ncontent\n```
    4. Mixed content with JSON blocks
    
    Args:
        output: The raw output string from workflow execution
        
    Returns:
        Dict containing parsed result with keys:
        - "parsed_json": Extracted JSON object (if found)
        - "cleaned_text": Text with markdown formatting removed
        - "original": Original output
        - "has_json": Boolean indicating if JSON was found
    """
    if not isinstance(output, str):
        return {
            "parsed_json": None,
            "cleaned_text": str(output),
            "original": output,
            "has_json": False
        }
    
    result = {
        "parsed_json": None,
        "cleaned_text": output,
        "original": output,
        "has_json": False
    }
    
    # Try to extract JSON from markdown code blocks
    json_patterns = [
        r'```json\s*\n(.*?)\n```',  # ```json\n...\n```
        r'```\s*\n(.*?)\n```',      # ```\n...\n``` (generic code block)
        r'`(.*?)`',                  # `...` (inline code)
    ]
    
    extracted_json = None
    
    for pattern in json_patterns:
        matches = re.findall(pattern, output, re.DOTALL)
        for match in matches:
            try:
                # Try to parse as JSON
                parsed = json.loads(match.strip())
                if isinstance(parsed, dict):
                    extracted_json = parsed
                    result["has_json"] = True
                    break
            except (json.JSONDecodeError, ValueError):
                continue
        
        if extracted_json:
            break
    
    # If no JSON found in code blocks, try to find JSON in the entire text
    if not extracted_json:
        # Look for JSON-like patterns in the text
        json_candidates = re.findall(r'\{[^{}]*"[^"]*"[^{}]*\}', output)
        for candidate in json_candidates:
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    extracted_json = parsed
                    result["has_json"] = True
                    break
            except (json.JSONDecodeError, ValueError):
                continue
    
    # Clean up markdown formatting
    cleaned_text = output
    
    # Remove markdown code blocks
    cleaned_text = re.sub(r'```[a-zA-Z]*\s*\n.*?\n```', '', cleaned_text, flags=re.DOTALL)
    
    # Remove inline code blocks
    cleaned_text = re.sub(r'`[^`]*`', '', cleaned_text)
    
    # Remove markdown headers
    cleaned_text = re.sub(r'^#{1,6}\s+.*$', '', cleaned_text, flags=re.MULTILINE)
    
    # Remove markdown formatting
    cleaned_text = re.sub(r'\*\*(.*?)\*\*', r'\1', cleaned_text)  # Bold
    cleaned_text = re.sub(r'\*(.*?)\*', r'\1', cleaned_text)      # Italic
    cleaned_text = re.sub(r'~~(.*?)~~', r'\1', cleaned_text)      # Strikethrough
    
    # Clean up extra whitespace
    cleaned_text = re.sub(r'\n\s*\n', '\n', cleaned_text)  # Multiple newlines
    cleaned_text = cleaned_text.strip()
    
    result["parsed_json"] = extracted_json
    result["cleaned_text"] = cleaned_text
    
    return result
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
SUPABASE_BUCKET_REQUIREMENT = os.getenv("SUPABASE_BUCKET_REQUIREMENT")
default_llm_config = {
    "model": "gpt-4o-mini",
    "openai_key": OPENAI_API_KEY,
    # "stream": True,
    "output_response": True,
    "max_tokens": 16000
}
# default_llm_config = {
#     "model": "openai/gpt-4o-mini",
#     "openrouter_key": OPENROUTER_API_KEY,
#     # "stream": True,
#     "output_response": True,
#     "max_tokens": 16000
# }

TUNNEL_INFO_PATH = "./server/tunnel_info.json"
MCP_CONFIG_PATH = "./server/mcp.config"
# sudo_workflow = WorkFlow.from_file("examples/output/jobs/jobs_demo_4o_mini.json")
sudo_workflow = None
# sudo_execution_result = "Sudo execution result for the given workflow."
sudo_execution_result = None


# Database will be initialized in main.py
# Use the global workflow_db instance from db.py

# default_tools = MCPToolkit(config_path=MCP_CONFIG_PATH).get_tools()
# default_tools += [FileTool()]
from evoagentx.tools import GoogleFreeSearchToolkit, DDGSSearchToolkit, WikipediaSearchToolkit, ArxivToolkit, MongoDBToolkit, StorageToolkit, CMDToolkit, RSSToolkit
default_tools = [GoogleFreeSearchToolkit(), DDGSSearchToolkit(), WikipediaSearchToolkit(), ArxivToolkit(), MongoDBToolkit(), StorageToolkit(), CMDToolkit(), RSSToolkit()]

def create_dynamic_mongodb_toolkit(database_name: str = None) -> MongoDBToolkit:
    """
    Create a MongoDB toolkit dynamically using the extracted database name.
    
    Args:
        database_name: The database name extracted from requirements
        
    Returns:
        MongoDBToolkit instance configured with the database
    """
    if not database_name:
        # If no database name provided, create a default toolkit
        return MongoDBToolkit()
    
    # Create toolkit with the specific database name
    return MongoDBToolkit(
        connection_string=MONGODB_URL,
        database_name=database_name
    )

def create_tools_with_database(database_information: Dict[str, Any] = None) -> list:
    """
    Create tools list with dynamic MongoDB toolkit based on database information.
    
    Args:
        database_information: Database information extracted from requirements
        
    Returns:
        List of tools including dynamic MongoDB toolkit
    """
    tools = [GoogleFreeSearchToolkit(), DDGSSearchToolkit(), WikipediaSearchToolkit(), ArxivToolkit(), StorageToolkit(), CMDToolkit(), RSSToolkit()]
    
    # Add dynamic MongoDB toolkit if database information is available
    if database_information and database_information.get("database_name"):
        database_name = database_information["database_name"]
        mongodb_toolkit = create_dynamic_mongodb_toolkit(database_name)
        tools.append(mongodb_toolkit)
        print(f"🔧 Added dynamic MongoDB toolkit for database: {database_name}")
    else:
        # Add default MongoDB toolkit if no specific database info
        tools.append(MongoDBToolkit())
        print("🔧 Added default MongoDB toolkit")
    
    return tools


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
    # from .sample_requirement import SAMPLE_REQUIREMENT
    # return SAMPLE_REQUIREMENT
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

async def list_workflows() -> Dict[str, Any]:
    """List all workflows in the database"""
    workflows = await database.find_many("workflows")
    total_count = await database.count("workflows")
    
    active_projects = [
        w["id"] for w in workflows 
        if w.get("status") not in ["completed", "failed"]
    ]
    
    return {
        "projects": [w["id"] for w in workflows],
        "total_count": total_count,
        "active_projects": active_projects
    }


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
    
    if sudo_workflow:
        return sudo_workflow
    
    try:
        # Convert dictionary to appropriate LLM config object and create LLM instance
        llm_config = create_llm_config(llm_config_dict)
        llm = create_llm_instance(llm_config)
        
        if mcp_config:
            tools = MCPToolkit(config=mcp_config)
        else:
            tools = default_tools
    except Exception as e:
        print(f"Error initializing components: {e}")
        return None
    
    workflow_generator = WorkFlowGenerator(llm=llm, tools=tools)
    
    # Generate the workflow
    workflow_graph: WorkFlowGraph = workflow_generator.generate_workflow(goal=goal)
    return workflow_graph

async def execute_workflow_from_config(workflow: Dict[str, Any], llm_config_dict: Dict[str, Any], mcp_config: dict = None, inputs: Dict[str, Any] = None, database_information: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Execute a workflow with the given configuration.
    
    Args:
        workflow: The workflow definition/configuration to execute
        llm_config_dict: LLM configuration dictionary
        mcp_config: Optional MCP configuration dictionary
        inputs: Optional inputs dictionary to pass to async_execute
        database_information: Optional database information for dynamic MongoDB toolkit creation
        
    Returns:
        Dict containing execution results and status
        
    """
    try:
        if sudo_execution_result:
            return {
            "status": "completed",
            "message": sudo_execution_result,
            "workflow_received": bool(workflow),
            "llm_config_received": bool(llm_config_dict),
            "mcp_config_received": bool(mcp_config)
        }
        
        
        llm_config = create_llm_config(llm_config_dict)
        llm = create_llm_instance(llm_config)
        
        # Handle both WorkFlowGraph objects and dictionaries
        if isinstance(workflow, WorkFlowGraph):
            workflow_graph = workflow
        else:
            workflow_graph: WorkFlowGraph = WorkFlowGraph.from_dict(workflow)
        
        # Create tools with dynamic MongoDB toolkit
        tools = []
        if mcp_config:
            mcp_toolkit = MCPToolkit(config=mcp_config)
            tools = mcp_toolkit.get_tools()
        tools += create_tools_with_database(database_information)
        
        agent_manager = AgentManager(tools=tools)
        agent_manager.add_agents_from_workflow(workflow_graph, llm_config=llm_config)
        # from pdb import set_trace; set_trace()

        workflow = WorkFlow(graph=workflow_graph, agent_manager=agent_manager, llm=llm)
        workflow.init_module()
        output = await workflow.async_execute(inputs=inputs)
        
        return {
            "status": "completed",
            "message": output,
            "workflow_received": bool(workflow),
            "llm_config_received": bool(llm_config_dict),
            "mcp_config_received": bool(mcp_config)
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"In the execution process, got error:\n{e}",
            "workflow_received": bool(workflow),
            "llm_config_received": bool(llm_config_dict),
            "mcp_config_received": bool(mcp_config)
        }



    









### _____________________________________________
### Workflow CRUD 
### _____________________________________________
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
        
        # # Create workflow config for response
        workflow_config = {
            "workflow_id": workflow_id,
            "workflow_name": workflow_data["workflow_name"],
            "workflow_inputs": workflow_data["workflow_inputs"],
            "workflow_outputs": workflow_data["workflow_outputs"],
            "workflow_graph": workflow_data["workflow_graph"],
        }
        
        workflow_configs.append(workflow_config)
    
    return workflow_configs

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

async def execute_workflow(workflow_id: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Phase 3: Execute workflow with provided inputs.
    Updated to work with individual workflow records and correct database schema.
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
                "status": "failed",
                "error": "No workflow graph available"
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
            database_information=database_information
        )
        
        if execution_result is None:
            print(f"❌ Failed to execute workflow: {workflow_name}")
            await update_workflow_status(workflow_id, "failed")
            return {
                "status": "failed",
                "error": "Failed to execute workflow"
            }
        
        # Process execution result
        if isinstance(execution_result, dict):
            execution_message = execution_result.get("message", "")
        else:
            execution_message = str(execution_result)
            
        # Parse and clean the execution output
        parsed_output = parse_workflow_output(execution_message)
        
        # Update execution_result with parsed output
        if isinstance(execution_result, dict):
            execution_result["message"] = parsed_output["cleaned_text"]
            execution_result["parsed_json"] = parsed_output["parsed_json"]
            execution_result["has_json"] = parsed_output["has_json"]
            execution_result["original_output"] = parsed_output["original"]
        else:
            execution_result = {
                "message": parsed_output["cleaned_text"],
                "parsed_json": parsed_output["parsed_json"],
                "has_json": parsed_output["has_json"],
                "original_output": parsed_output["original"]
            }
        
        # Update workflow storage with execution results
        await update_workflow_status(
            workflow_id, 
            "completed",
            execution_result=execution_result
        )
        
        return {
            "status": "completed",
            "workflow_name": workflow_name,
            "result": execution_result
        }
        
    except Exception as e:
        await update_workflow_status(workflow_id, "failed")
        raise ValueError(f"Failed to execute workflow: {str(e)}")


























### _____________________________________________
### WebSocket-based Workflow Execution with Real-time Logging
### _____________________________________________

import asyncio
import json
import queue
import threading
from typing import Optional, Dict, Any, Callable
from loguru import logger
from evoagentx.core.logging import save_logger, get_log_file

class WebSocketEnhancedSink:
    """
    Enhanced WebSocket sink that captures loguru messages, stdin/stdout, and provides periodic updates.
    """
    
    def __init__(self, websocket_send_func: Callable, workflow_id: str):
        self.websocket_send_func = websocket_send_func
        self.workflow_id = workflow_id
        self.message_queue = queue.Queue()
        self.running = True
        self.stdout_buffer = StringIO()
        self.stderr_buffer = StringIO()
        self.last_update_time = time.time()
        self.update_interval = 3.0  # Update every 3 seconds
        
        # Rate limiting for output messages
        self.last_output_time = {"stdout": 0, "stderr": 0}
        self.output_rate_limit = 0.1  # Minimum 100ms between output messages
        
        # Store original stdout/stderr
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        
        # Start message processing thread
        self.processing_thread = threading.Thread(target=self._process_messages, daemon=True)
        self.processing_thread.start()
        
        # Start periodic update thread
        self.update_thread = threading.Thread(target=self._periodic_updates, daemon=True)
        self.update_thread.start()
        
        # Redirect stdout/stderr to capture output
        self._redirect_output()
    
    def _redirect_output(self):
        """Redirect stdout and stderr to capture output."""
        class CapturingStream:
            def __init__(self, original_stream, buffer, sink):
                self.original_stream = original_stream
                self.buffer = buffer
                self.sink = sink
            
            def write(self, text):
                self.original_stream.write(text)
                self.buffer.write(text)
                # Flush to ensure immediate capture
                self.buffer.flush()
                
                # Only send stdout/stderr messages for substantial content to reduce spam
                if text.strip() and len(text.strip()) > 5:
                    self.sink._send_output_message("stdout" if self.original_stream == sys.stdout else "stderr", text)
            
            def flush(self):
                self.original_stream.flush()
                self.buffer.flush()
        
        # Create capturing streams
        self.stdout_capturer = CapturingStream(sys.stdout, self.stdout_buffer, self)
        self.stderr_capturer = CapturingStream(sys.stderr, self.stderr_buffer, self)
        
        # Redirect
        sys.stdout = self.stdout_capturer
        sys.stderr = self.stderr_capturer
    
    def capture_stdin_input(self, input_text: str):
        """Capture stdin input and send via WebSocket."""
        if input_text.strip():
            input_data = {
                "type": "input",
                "timestamp": datetime.now().isoformat(),
                "input_type": "stdin",
                "content": input_text.strip(),
                "workflow_id": self.workflow_id
            }
            
            # Send immediately in a thread-safe way
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.run_coroutine_threadsafe(
                        self._send_websocket_message(input_data), 
                        loop
                    )
            except RuntimeError:
                # No event loop available, just print
                print(f"Input message: stdin - {input_text.strip()}")
    
    def _restore_output(self):
        """Restore original stdout and stderr."""
        if hasattr(self, 'original_stdout'):
            sys.stdout = self.original_stdout
        if hasattr(self, 'original_stderr'):
            sys.stderr = self.original_stderr
    
    def _send_output_message(self, output_type: str, content: str):
        """Send stdout/stderr message via WebSocket."""
        if not content.strip():
            return
        
        # Rate limiting - only send if enough time has passed since last message of this type
        current_time = time.time()
        if current_time - self.last_output_time.get(output_type, 0) < self.output_rate_limit:
            return
        
        self.last_output_time[output_type] = current_time
            
        output_data = {
            "type": "output",
            "timestamp": datetime.now().isoformat(),
            "output_type": output_type,  # "stdout" or "stderr"
            "content": content.strip(),
            "workflow_id": self.workflow_id
        }
        
        # Send immediately in a thread-safe way
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.run_coroutine_threadsafe(
                    self._send_websocket_message(output_data), 
                    loop
                )
        except RuntimeError:
            # No event loop available - only log significant stderr messages to avoid spam
            if output_type == "stderr" and len(content.strip()) > 10:
                print(f"Stderr output: {content.strip()}")
            elif output_type == "stdout":
                # Only log stdout if it's substantial content
                if len(content.strip()) > 20:
                    print(f"Stdout output: {content.strip()}")
    
    def write(self, message):
        """Write method called by loguru for each log message."""
        if self.running:
            self.message_queue.put(message)
    
    def _periodic_updates(self):
        """Send periodic status updates every 3 seconds."""
        while self.running:
            try:
                time.sleep(self.update_interval)
                if self.running:
                    # Get current buffer contents
                    stdout_content = self.stdout_buffer.getvalue()
                    stderr_content = self.stderr_buffer.getvalue()
                    
                    # Send periodic update with comprehensive information
                    update_data = {
                        "type": "periodic_update",
                        "timestamp": datetime.now().isoformat(),
                        "workflow_id": self.workflow_id,
                        "stdout_buffer": stdout_content[-1000:],  # Last 1000 chars
                        "stderr_buffer": stderr_content[-1000:],  # Last 1000 chars
                        "buffer_sizes": {
                            "stdout": len(stdout_content),
                            "stderr": len(stderr_content)
                        },
                        "status": "running",
                        "message": f"Workflow execution in progress. Captured {len(stdout_content)} stdout chars, {len(stderr_content)} stderr chars"
                    }
                    
                    try:
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            asyncio.run_coroutine_threadsafe(
                                self._send_websocket_message(update_data), 
                                loop
                            )
                    except RuntimeError:
                        pass  # No event loop available
                        
            except Exception as e:
                print(f"Error in periodic updates: {e}")
    
    def _process_messages(self):
        """Process messages from queue and send via WebSocket."""
        while self.running:
            try:
                message = self.message_queue.get(timeout=1)
                if message:
                    # Parse loguru message format
                    log_data = self._parse_loguru_message(message)
                    if log_data:
                        # Send via WebSocket in a thread-safe way
                        try:
                            # Get the current event loop
                            loop = asyncio.get_event_loop()
                            if loop.is_running():
                                # Use run_coroutine_threadsafe to safely call async function from thread
                                future = asyncio.run_coroutine_threadsafe(
                                    self._send_websocket_message(log_data), 
                                    loop
                                )
                                # Don't wait for the result to avoid blocking
                                future.add_done_callback(self._handle_websocket_result)
                            else:
                                # Fallback: just print the message if no running loop
                                print(f"WebSocket log message: {log_data.get('message', 'Unknown')}")
                        except RuntimeError:
                            # No event loop available, just print the message
                            print(f"WebSocket log message: {log_data.get('message', 'Unknown')}")
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error processing log message: {e}")
    
    def _handle_websocket_result(self, future):
        """Handle the result of the WebSocket send operation."""
        try:
            future.result()  # This will raise any exception that occurred
        except Exception as e:
            print(f"Error in WebSocket send operation: {e}")
    
    def _parse_loguru_message(self, message: str) -> Optional[Dict[str, Any]]:
        """Parse loguru message format and extract structured data."""
        try:
            # Remove ANSI color codes and parse timestamp
            import re
            # Remove color codes
            message = re.sub(r'\x1b\[[0-9;]*m', '', message)
            
            # Try to parse timestamp and level from loguru format
            # Expected format: "2024-01-01 12:00:00.123 | LEVEL | message"
            parts = message.strip().split(' | ', 2)
            if len(parts) >= 3:
                timestamp_str, level, log_message = parts
                return {
                    "type": "log",
                    "timestamp": timestamp_str,
                    "level": level,
                    "message": log_message,
                    "workflow_id": self.workflow_id
                }
            else:
                # Fallback for unformatted messages
                return {
                    "type": "log",
                    "timestamp": None,
                    "level": "INFO",
                    "message": message.strip(),
                    "workflow_id": self.workflow_id
                }
        except Exception as e:
            print(f"Error parsing loguru message: {e}")
            return None
    
    async def _send_websocket_message(self, log_data: Dict[str, Any]):
        """Send message through WebSocket."""
        try:
            await self.websocket_send_func(json.dumps(log_data))
        except Exception as e:
            print(f"Error sending WebSocket message: {e}")
    
    def get_buffer_contents(self) -> Dict[str, Any]:
        """Get current buffer contents for final reporting."""
        return {
            "stdout": self.stdout_buffer.getvalue(),
            "stderr": self.stderr_buffer.getvalue(),
            "stdout_size": len(self.stdout_buffer.getvalue()),
            "stderr_size": len(self.stderr_buffer.getvalue())
        }
    
    def stop(self):
        """Stop the sink and cleanup."""
        self.running = False
        
        # Restore original output streams
        self._restore_output()
        
        # Wait for threads to finish
        if self.processing_thread.is_alive():
            self.processing_thread.join(timeout=2)
        if self.update_thread.is_alive():
            self.update_thread.join(timeout=2)

class WorkflowExecutionProgress:
    """
    Tracks workflow execution progress and sends updates via WebSocket.
    """
    
    def __init__(self, websocket_send_func: Callable, workflow_id: str):
        self.websocket_send_func = websocket_send_func
        self.workflow_id = workflow_id
        self.current_phase = "initializing"
        self.progress = 0.0
        self.total_tasks = 0
        self.completed_tasks = 0
    
    async def send_progress_update(self, phase: str, progress: float, message: str = None):
        """Send progress update via WebSocket."""
        self.current_phase = phase
        self.progress = progress
        
        progress_data = {
            "type": "progress",
            "timestamp": datetime.now().isoformat(),
            "phase": phase,
            "progress": progress,
            "message": message,
            "workflow_id": self.workflow_id
        }
        
        try:
            await self.websocket_send_func(json.dumps(progress_data))
        except Exception as e:
            print(f"Error sending progress update: {e}")
    
    async def send_completion(self, result: Dict[str, Any]):
        """Send completion message with final result."""
        completion_data = {
            "type": "complete",
            "timestamp": datetime.now().isoformat(),
            "result": result,
            "workflow_id": self.workflow_id
        }
        
        try:
            await self.websocket_send_func(json.dumps(completion_data))
        except Exception as e:
            print(f"Error sending completion: {e}")
    
    async def send_error(self, error_message: str):
        """Send error message."""
        error_data = {
            "type": "error",
            "timestamp": datetime.now().isoformat(),
            "error": error_message,
            "workflow_id": self.workflow_id
        }
        
        try:
            await self.websocket_send_func(json.dumps(error_data))
        except Exception as e:
            print(f"Error sending error message: {e}")
    
    async def send_final_result(self, result: Dict[str, Any], workflow_name: str = None):
        """Send detailed final result message."""
        final_result_data = {
            "type": "final_result",
            "timestamp": datetime.now().isoformat(),
            "workflow_id": self.workflow_id,
            "workflow_name": workflow_name or self.workflow_id,
            "status": "completed",
            "execution_result": result.get("result", {}),
            "captured_output": result.get("captured_output", {}),
            "message": "Workflow execution completed successfully"
        }
        
        try:
            await self.websocket_send_func(json.dumps(final_result_data))
        except Exception as e:
            print(f"Error sending final result: {e}")

async def execute_workflow_with_websocket(
    workflow_id: str, 
    inputs: Dict[str, Any], 
    websocket_send_func: Callable
) -> Dict[str, Any]:
    """
    Execute workflow with WebSocket-based real-time progress updates.
    
    Args:
        workflow_id: The workflow ID to execute
        inputs: Input parameters for workflow execution
        websocket_send_func: Function to send messages via WebSocket
        
    Returns:
        Dict containing execution results
    """
    progress_tracker = WorkflowExecutionProgress(websocket_send_func, workflow_id)
    websocket_sink = None
    
    try:
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
        
        await progress_tracker.send_progress_update("executing", 0.3, f"Executing workflow: {workflow_name}")
        
        # Get database information for dynamic MongoDB toolkit
        database_information = task_info.get("database_information")
        
        # Setup WebSocket enhanced sink
        websocket_sink = WebSocketEnhancedSink(websocket_send_func, workflow_id)
        
        # Add custom sink to loguru
        sink_id = logger.add(websocket_sink.write, level="INFO", format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")
        
        try:
            # Execute the workflow with database information
            execution_result = await execute_workflow_from_config(
                workflow_graph, 
                default_llm_config, 
                mcp_config={}, 
                inputs=inputs,
                database_information=database_information
            )
            
            await progress_tracker.send_progress_update("finalizing", 0.9, "Finalizing execution results...")
            
            if execution_result is None:
                await progress_tracker.send_error(f"Failed to execute workflow: {workflow_name}")
                await update_workflow_status(workflow_id, "failed")
                return {
                    "status": "failed",
                    "error": "Failed to execute workflow"
                }
            
            # Process execution result
            if isinstance(execution_result, dict):
                execution_message = execution_result.get("message", "")
            else:
                execution_message = str(execution_result)
                
            # Parse and clean the execution output
            parsed_output = parse_workflow_output(execution_message)
            
            # Update execution_result with parsed output
            if isinstance(execution_result, dict):
                execution_result["message"] = parsed_output["cleaned_text"]
                execution_result["parsed_json"] = parsed_output["parsed_json"]
                execution_result["has_json"] = parsed_output["has_json"]
                execution_result["original_output"] = parsed_output["original"]
            else:
                execution_result = {
                    "message": parsed_output["cleaned_text"],
                    "parsed_json": parsed_output["parsed_json"],
                    "has_json": parsed_output["has_json"],
                    "original_output": parsed_output["original"]
                }
            
            # Get captured output from the sink
            captured_output = websocket_sink.get_buffer_contents() if websocket_sink else {}
            
            # Update workflow storage with execution results
            try:
                await update_workflow_status(
                    workflow_id, 
                    "completed",
                    execution_result=execution_result,
                    captured_output=captured_output
                )
                print(f"✅ Successfully saved execution result to database for workflow {workflow_id}")
            except Exception as e:
                print(f"❌ Error saving execution result to database: {e}")
                # Continue execution even if database save fails
            
            final_result = {
                "status": "completed",
                "workflow_name": workflow_name,
                "result": execution_result,
                "captured_output": captured_output
            }
            
            await progress_tracker.send_progress_update("completed", 1.0, "Workflow execution completed successfully")
            
            # Send detailed final result message
            await progress_tracker.send_final_result(final_result, workflow_name)
            
            # Also send the standard completion message
            await progress_tracker.send_completion(final_result)
            
            # Verify the result was saved by checking the database
            try:
                saved_workflow = await get_workflow(workflow_id)
                if saved_workflow and saved_workflow.get("execution_result"):
                    print(f"✅ Verified execution result saved in database for workflow {workflow_id}")
                else:
                    print(f"⚠️ Warning: Execution result not found in database for workflow {workflow_id}")
            except Exception as e:
                print(f"⚠️ Warning: Could not verify database save: {e}")
            
            return final_result
            
        finally:
            # Remove custom sink
            logger.remove(sink_id)
            if websocket_sink:
                websocket_sink.stop()
        
    except Exception as e:
        error_message = f"Error executing workflow: {str(e)}"
        await progress_tracker.send_error(error_message)
        await update_workflow_status(workflow_id, "failed")
        
        # Cleanup
        if websocket_sink:
            websocket_sink.stop()
        
        raise ValueError(error_message)


























### _____________________________________________
### User Query Router Service
### _____________________________________________

async def analyze_user_query(project_short_id: str, query: str) -> Dict[str, Any]:
    """
    Analyze user query using UserQueryRouter.
    
    This function:
    1. Collects all workflows for the given project_short_id
    2. Builds workflow context dictionary
    3. Initializes UserQueryRouter and processes the query
    4. Returns structured analysis result
    
    Args:
        project_short_id: The project identifier
        query: The user's query string to analyze
        
    Returns:
        Dictionary containing the query analysis result
        
    Raises:
        ValueError: If project not found or UserQueryRouter fails
    """
    try:
        print(f"🔍 Analyzing user query for project {project_short_id}")
        print(f"Query: {query[:100]}...")
        
        # Step 1: Collect all workflows for the project
        workflows = await database.find_many("workflows", {"project_short_id": project_short_id})
        
        if not workflows:
            print(f"⚠️ No workflows found for project {project_short_id}")
            # Return empty result but don't fail
            return {
                "original_query": query,
                "classified_operations": [],
                "is_composite": False,
                "workflow_context": {},
                "total_operations": 0,
                "has_frontend": False,
                "has_backend": False
            }
        
        print(f"📋 Found {len(workflows)} workflows for project {project_short_id}")
        
        # Step 2: Build workflow context dictionary
        workflow_context = {}
        for workflow in workflows:
            workflow_id = workflow["id"]
            workflow_context[workflow_id] = {
                "workflow_graph": workflow.get("workflow_graph"),
                "task_info": workflow.get("task_info"),
                "status": workflow.get("status"),
                "project_short_id": workflow.get("project_short_id")
            }
        
        print(f"🏗️ Built workflow context with {len(workflow_context)} workflows")
        
        # Step 3: Initialize UserQueryRouter
        from .components.user_query_router.user_query_router import UserQueryRouter
        user_query_router = UserQueryRouter()
        
        # Step 4: Process the query
        print("🤖 Processing query with UserQueryRouter...")
        result = user_query_router.route_query(
            user_query=query,
            workflow_context=workflow_context
        )
        
        # Step 5: Convert result to dictionary format
        response_data = {
            "original_query": result.original_query,
            "classified_operations": result.classified_operations,
            "is_composite": result.is_composite,
            "total_operations": result.total_operations,
            "has_frontend": result.has_frontend,
            "has_backend": result.has_backend
        }
        
        print(f"✅ Query analysis completed successfully")
        print(f"   - Total operations: {result.total_operations}")
        print(f"   - Composite query: {result.is_composite}")
        print(f"   - Has frontend operations: {result.has_frontend}")
        print(f"   - Has backend operations: {result.has_backend}")
        
        return response_data
        
    except Exception as e:
        print(f"❌ Error analyzing user query: {str(e)}")
        raise ValueError(f"Failed to analyze user query: {str(e)}")

























