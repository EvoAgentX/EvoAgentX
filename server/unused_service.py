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
# Environment variables are loaded in main.py from config/app.env

from evoagentx.workflow import WorkFlowGenerator, WorkFlowGraph, WorkFlow
from evoagentx.models import LLMConfig
from evoagentx.models.model_configs import OpenAILLMConfig, OpenRouterConfig
from evoagentx.models.model_utils import create_llm_instance
from evoagentx.agents.agent_manager import AgentManager
from evoagentx.core.module_utils import parse_json_from_text
from evoagentx.tools import MCPToolkit

from .prompts import WORKFLOW_GENERATION_PROMPT, WORKFLOW_GENERATION_GOAL_PROMPT, WORKFLOW_REQUIREMENT_PROMPT, TASK_INFO_PROMPT_SUDO, CONNECTION_INSTRUCTION_PROMPT, CUSTOM_OUTPUT_EXTRACTION_PROMPT
from .db import database, requirement_database

import sys
import io
import threading
import time
from contextlib import redirect_stdout, redirect_stderr

# Environment variables are loaded in main.py from config/app.env

from .utils.output_parser import parse_workflow_output


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
SUPABASE_BUCKET_STORAGE = os.getenv("SUPABASE_BUCKET_STORAGE")
default_llm_config = {
    "model": "gpt-4o",
    "openai_key": OPENAI_API_KEY,
    # "stream": True,
    "output_response": True,
    "max_tokens": 16000
}
# default_llm_config = {
#     "model": "openai/gpt-5-mini",
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
from evoagentx.tools import GoogleFreeSearchToolkit, DDGSSearchToolkit, WikipediaSearchToolkit, ArxivToolkit, StorageToolkit, CMDToolkit, RSSToolkit
default_tools = [GoogleFreeSearchToolkit(), DDGSSearchToolkit(), WikipediaSearchToolkit(), ArxivToolkit(), StorageToolkit(), CMDToolkit(), RSSToolkit()]

def create_tools_with_database(database_information: Dict[str, Any] = None) -> list:
    """
    Create tools list (database tools disabled).
    
    Args:
        database_information: Database information (ignored)
        
    Returns:
        List of tools without database toolkit
    """
    print("🔧 Database tools disabled")
    return []


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

async def execute_workflow_from_config(workflow: Dict[str, Any], llm_config_dict: Dict[str, Any], mcp_config: dict = None, inputs: Dict[str, Any] = None, database_information: Dict[str, Any] = None, task_info: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Execute a workflow with the given configuration.
    
    Args:
        workflow: The workflow definition/configuration to execute
        llm_config_dict: LLM configuration dictionary
        mcp_config: Optional MCP configuration dictionary
        inputs: Optional inputs dictionary to pass to async_execute
        database_information: Optional database information (currently ignored - database tools disabled)
        
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
        # from pdb import set_trace; set_trace()

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
        
        # Get database information (currently ignored - database tools disabled)
        database_information = task_info.get("database_information")
        
        # Execute the workflow (database tools disabled)
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
            "data": {
                "output_type": output_type,
                "content": content.strip(),
                "timestamp": datetime.now().isoformat()
            }
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
                        "data": {
                            "stdout_buffer": stdout_content,
                            "stderr_buffer": stderr_content,
                            "buffer_sizes": {
                                "stdout": len(stdout_content),
                                "stderr": len(stderr_content)
                            },
                            "status": "running",
                            "message": f"Workflow execution in progress. Captured {len(stdout_content)} stdout chars, {len(stderr_content)} stderr chars"
                        },
                        "timestamp": datetime.now().isoformat()
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
                                print(f"WebSocket log message: {log_data.get('content', 'Unknown')}")
                        except RuntimeError:
                            # No event loop available, just print the message
                            print(f"WebSocket log message: {log_data.get('content', 'Unknown')}")
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error processing log message: {e}")
    
    def _handle_websocket_result(self, future):
        """Handle WebSocket send result."""
        try:
            future.result()
        except Exception as e:
            print(f"Error in WebSocket send: {e}")
    
    def _parse_loguru_message(self, message: str) -> Optional[Dict[str, Any]]:
        """Parse loguru message format and convert to WebSocket message."""
        try:
            # Parse loguru message format: "YYYY-MM-DD HH:mm:ss | LEVEL | MESSAGE"
            parts = message.strip().split(" | ", 2)
            if len(parts) >= 3:
                timestamp_str, level, content = parts
                
                # Parse timestamp
                try:
                    timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
                    timestamp_iso = timestamp.isoformat()
                except ValueError:
                    timestamp_iso = datetime.now().isoformat()
                
                # Determine log level
                level_upper = level.upper()
                if level_upper in ["ERROR", "CRITICAL"]:
                    message_type = "error"
                elif level_upper in ["WARNING"]:
                    message_type = "warning"
                elif level_upper in ["INFO"]:
                    message_type = "info"
                else:
                    message_type = "debug"
                
                return {
                    "type": "log",
                    "data": {
                        "level": level_upper,
                        "message": content,
                        "timestamp": timestamp_iso
                    }
                }
            else:
                # Fallback for malformed messages
                return {
                    "type": "log",
                    "data": {
                        "level": "INFO",
                        "message": message.strip(),
                        "timestamp": datetime.now().isoformat()
                    }
                }
        except Exception as e:
            print(f"Error parsing loguru message: {e}")
            return None
    
    async def _send_websocket_message(self, log_data: Dict[str, Any]):
        """Send log message via WebSocket."""
        try:
            await self.websocket_send_func(json.dumps(log_data))
        except Exception as e:
            print(f"Error sending WebSocket message: {e}")
    
    def get_buffer_contents(self) -> Dict[str, Any]:
        """Get current buffer contents."""
        return {
            "stdout": self.stdout_buffer.getvalue(),
            "stderr": self.stderr_buffer.getvalue(),
            "stdout_size": len(self.stdout_buffer.getvalue()),
            "stderr_size": len(self.stderr_buffer.getvalue())
        }
    
    def stop(self):
        """Stop the sink and cleanup."""
        self.running = False
        self._restore_output()
        
        # Stop background threads
        if hasattr(self, '_periodic_thread') and self._periodic_thread:
            self._periodic_thread.join(timeout=1)
        if hasattr(self, '_process_thread') and self._process_thread:
            self._process_thread.join(timeout=1)

class WorkflowExecutionProgress:
    """
    Tracks workflow execution progress and sends updates via WebSocket.
    Enhanced to support all message types from the system diagram.
    """
    
    def __init__(self, websocket_send_func: Callable, workflow_id: str):
        self.websocket_send_func = websocket_send_func
        self.workflow_id = workflow_id
        self.current_phase = "initializing"
        self.progress = 0.0
        self.total_tasks = 0
        self.completed_tasks = 0
        self.execution_id = None
        self.workflow_name = None
    
    async def send_connection_confirmation(self):
        """Send WebSocket connection confirmation."""
        connection_data = {
            "type": "connection",
            "message": "WebSocket connected successfully",
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            await self.websocket_send_func(json.dumps(connection_data))
        except Exception as e:
            print(f"Error sending connection confirmation: {e}")
    
    async def send_heartbeat_response(self):
        """Send heartbeat response."""
        heartbeat_data = {
            "type": "pong",
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            await self.websocket_send_func(json.dumps(heartbeat_data))
        except Exception as e:
            print(f"Error sending heartbeat response: {e}")
    
    async def send_setup_progress(self, step: str, progress: float, message: str = None):
        """Send setup progress update."""
        setup_data = {
            "type": "setup-progress",
            "data": {
                "step": step,
                "progress": progress,
                "message": message or f"Setup step: {step}"
            },
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            await self.websocket_send_func(json.dumps(setup_data))
        except Exception as e:
            print(f"Error sending setup progress: {e}")
    
    async def send_setup_finished(self, status: str, workflow_id: str = None, workflow_graph: dict = None, error: dict = None):
        """Send setup completion message."""
        setup_finished_data = {
            "type": "setup-finished",
            "data": {
                "status": status,
                "workflowId": workflow_id or self.workflow_id,
                "workflowGraph": workflow_graph,
                "error": error
            },
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            await self.websocket_send_func(json.dumps(setup_finished_data))
        except Exception as e:
            print(f"Error sending setup finished: {e}")
    
    async def send_execute_started(self, execution_id: str, workflow_name: str = None):
        """Send execution start message."""
        self.execution_id = execution_id
        self.workflow_name = workflow_name
        
        execute_started_data = {
            "type": "execute-started",
            "data": {
                "executionId": execution_id,
                "startedAt": int(datetime.now().timestamp() * 1000),
                "workflowId": self.workflow_id,
                "workflowName": workflow_name
            },
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            await self.websocket_send_func(json.dumps(execute_started_data))
        except Exception as e:
            print(f"Error sending execute started: {e}")
    
    async def send_status_update(self, status: str, progress: float = None, current_node: str = None):
        """Send status update message."""
        status_data = {
            "type": "status",
            "data": {
                "executionId": self.execution_id,
                "status": status,
                "progress": progress,
                "currentNode": current_node,
                "workflowId": self.workflow_id
            },
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            await self.websocket_send_func(json.dumps(status_data))
        except Exception as e:
            print(f"Error sending status update: {e}")
    
    async def send_node_progress(self, node_id: str, status: str, progress: float = None, message: str = None):
        """Send node progress update."""
        node_progress_data = {
            "type": "node-progress",
            "data": {
                "nodeId": node_id,
                "status": status,
                "progress": progress,
                "message": message,
                "executionId": self.execution_id
            },
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            await self.websocket_send_func(json.dumps(node_progress_data))
        except Exception as e:
            print(f"Error sending node progress: {e}")
    
    async def send_run_detail(self, node_id: str, status: str, outputs: dict = None, execution_time: float = None):
        """Send detailed node execution result."""
        run_detail_data = {
            "type": "run-detail",
            "data": {
                "nodeId": node_id,
                "status": status,
                "outputs": outputs or {},
                "executionTime": execution_time,
                "executionId": self.execution_id
            },
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            await self.websocket_send_func(json.dumps(run_detail_data))
        except Exception as e:
            print(f"Error sending run detail: {e}")
    
    async def send_execute_finished(self, status: str, results: dict = None, error: dict = None, total_execution_time: float = None):
        """Send execution completion message."""
        execute_finished_data = {
            "type": "execute-finished",
            "data": {
                "status": status,
                "executionId": self.execution_id,
                "results": results,
                "error": error,
                "totalExecutionTime": total_execution_time,
                "workflowId": self.workflow_id,
                "workflowName": self.workflow_name
            },
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            await self.websocket_send_func(json.dumps(execute_finished_data))
        except Exception as e:
            print(f"Error sending execute finished: {e}")
    
    async def send_control_response(self, control_type: str, status: str, message: str = None):
        """Send control operation response (pause/resume/cancel)."""
        control_data = {
            "type": "status",
            "data": {
                "status": status,
                "executionId": self.execution_id,
                "controlType": control_type,
                "message": message or f"Execution {control_type}d"
            },
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            await self.websocket_send_func(json.dumps(control_data))
        except Exception as e:
            print(f"Error sending control response: {e}")
    
    async def send_connection_error(self, error_code: str, message: str, reconnect: bool = True):
        """Send connection error message."""
        error_data = {
            "type": "error",
            "data": {
                "code": error_code,
                "message": message,
                "reconnect": reconnect
            },
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            await self.websocket_send_func(json.dumps(error_data))
        except Exception as e:
            print(f"Error sending connection error: {e}")
    
    async def send_execution_error(self, error_code: str, message: str, node_id: str = None):
        """Send execution error message."""
        error_data = {
            "type": "error",
            "data": {
                "code": error_code,
                "message": message,
                "nodeId": node_id,
                "executionId": self.execution_id,
                "workflowId": self.workflow_id
            },
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            await self.websocket_send_func(json.dumps(error_data))
        except Exception as e:
            print(f"Error sending execution error: {e}")
    
    async def send_progress_update(self, phase: str, progress: float, message: str = None):
        """Send progress update via WebSocket (legacy method for backward compatibility)."""
        self.current_phase = phase
        self.progress = progress
        
        # Format content with progress information
        if message:
            content = f"{message} ({int(progress * 100)}% complete)"
        else:
            content = f"Phase: {phase} ({int(progress * 100)}% complete)"
        
        progress_data = {
            "type": "progress",
            "content": content,
            "result": None,
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            await self.websocket_send_func(json.dumps(progress_data))
        except Exception as e:
            print(f"Error sending progress update: {e}")
    
    async def send_completion(self, result: Dict[str, Any]):
        """Send completion message with final result (legacy method for backward compatibility)."""
        completion_data = {
            "type": "complete",
            "content": "Workflow execution completed successfully",
            "result": result,
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            await self.websocket_send_func(json.dumps(completion_data))
        except Exception as e:
            print(f"Error sending completion: {e}")
    
    async def send_error(self, error_message: str):
        """Send error message (legacy method for backward compatibility)."""
        error_data = {
            "type": "error",
            "content": error_message,
            "result": None,
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            await self.websocket_send_func(json.dumps(error_data))
        except Exception as e:
            print(f"Error sending error message: {e}")
    
    async def send_final_result(self, result: Dict[str, Any], workflow_name: str = None):
        """Send detailed final result message (legacy method for backward compatibility)."""
        workflow_display_name = workflow_name or self.workflow_id
        final_result_data = {
            "type": "complete",
            "content": f"Workflow '{workflow_display_name}' execution completed successfully",
            "result": result,
            "timestamp": datetime.now().isoformat()
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
    Enhanced to support all message types from the system diagram.
    
    Args:
        workflow_id: The workflow ID to execute
        inputs: Input parameters for workflow execution
        websocket_send_func: Function to send messages via WebSocket
        
    Returns:
        Dict containing execution results
    """
    progress_tracker = WorkflowExecutionProgress(websocket_send_func, workflow_id)
    websocket_sink = None
    execution_start_time = time.time()
    
    try:
        # Send connection confirmation
        await progress_tracker.send_connection_confirmation()
        
        # Send initial progress
        await progress_tracker.send_progress_update("initializing", 0.0, "Starting workflow execution...")
        
        # Check if workflow exists
        workflow = await get_workflow(workflow_id)
        if not workflow:
            await progress_tracker.send_execution_error("WORKFLOW_NOT_FOUND", f"Workflow with ID {workflow_id} not found")
            raise ValueError(f"Workflow with ID {workflow_id} not found")
        
        await progress_tracker.send_progress_update("validating", 0.1, "Validating workflow configuration...")
        
        # Check if workflow generation was completed
        if workflow.get("workflow_graph") is None:
            await progress_tracker.send_execution_error("WORKFLOW_NOT_GENERATED", f"Workflow {workflow_id} has not completed generation phase")
            raise ValueError(f"Workflow {workflow_id} has not completed generation phase")
        
        await progress_tracker.send_progress_update("preparing", 0.2, "Preparing workflow execution...")
        
        # Update workflow status
        await update_workflow_status(workflow_id, "running")
        
        # Get workflow graph and task info
        workflow_graph = workflow["workflow_graph"]
        task_info = workflow.get("task_info", {})
        workflow_name = task_info.get("workflow_name", workflow_id)
        
        if workflow_graph is None:
            await progress_tracker.send_execution_error("NO_WORKFLOW_GRAPH", "No workflow graph available")
            await update_workflow_status(workflow_id, "failed")
            return {
                "status": "failed",
                "error": "No workflow graph available"
            }
        
        # Generate execution ID
        execution_id = f"exec_{workflow_id}_{int(time.time())}"
        
        # Send execution started message
        await progress_tracker.send_execute_started(execution_id, workflow_name)
        
        await progress_tracker.send_progress_update("executing", 0.3, f"Executing workflow: {workflow_name}")
        await progress_tracker.send_status_update("running", 0.3, "initializing")
        
        # Get database information (currently ignored - database tools disabled)
        database_information = task_info.get("database_information")
        
        # Setup WebSocket enhanced sink
        websocket_sink = WebSocketEnhancedSink(websocket_send_func, workflow_id)
        
        # Add custom sink to loguru
        sink_id = logger.add(websocket_sink.write, level="INFO", format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")
        
        try:
            # Send node progress for workflow initialization
            await progress_tracker.send_node_progress("workflow_init", "running", 0.5, "Initializing workflow components...")
            
            # Execute the workflow (database tools disabled)
            execution_result = await execute_workflow_from_config(
                workflow_graph, 
                default_llm_config, 
                mcp_config={}, 
                inputs=inputs,
                database_information=database_information,
                task_info=task_info
            )
            
            # Send node completion for workflow execution
            await progress_tracker.send_run_detail("workflow_execution", "completed", 
                                                {"status": "success"}, time.time() - execution_start_time)
            
            await progress_tracker.send_progress_update("finalizing", 0.9, "Finalizing execution results...")
            await progress_tracker.send_status_update("finalizing", 0.9, "finalizing")
            
            if execution_result is None:
                await progress_tracker.send_execution_error("EXECUTION_FAILED", f"Failed to execute workflow: {workflow_name}")
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
            await progress_tracker.send_status_update("completed", 1.0, "completed")
            
            # Send detailed final result message
            await progress_tracker.send_final_result(final_result, workflow_name)
            
            # Send execution finished message
            await progress_tracker.send_execute_finished("completed", final_result, None, total_execution_time)
            
            # Also send the standard completion message for backward compatibility
            await progress_tracker.send_completion(final_result)
            
            return final_result
            
        except Exception as e:
            # Send execution error
            await progress_tracker.send_execution_error("EXECUTION_ERROR", f"Workflow execution failed: {str(e)}")
            await update_workflow_status(workflow_id, "failed")
            raise e
            
        finally:
            # Cleanup
            if websocket_sink:
                websocket_sink.stop()
            if sink_id:
                logger.remove(sink_id)
    
    except Exception as e:
        # Send connection error for unexpected failures
        await progress_tracker.send_connection_error("EXECUTION_ERROR", f"Unexpected error: {str(e)}", False)
        await update_workflow_status(workflow_id, "failed")
        return {
            "original_message": f"Failed to execute workflow: {str(e)}",
            "parsed_json": None
        }


























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


























