import asyncio
import json
import os
import uuid
from datetime import datetime
from typing import Dict, Any
from dotenv import load_dotenv

from evoagentx.workflow import WorkFlowGenerator, WorkFlowGraph, WorkFlow
from evoagentx.models import LLMConfig
from evoagentx.models.model_configs import OpenAILLMConfig, OpenRouterConfig
from evoagentx.models.model_utils import create_llm_instance
from evoagentx.agents.agent_manager import AgentManager
from evoagentx.core.module_utils import parse_json_from_text
from evoagentx.tools import MCPToolkit

from .prompts import WORKFLOW_GENERATION_PROMPT, TASK_INFO_PROMPT_SUDO, CONNECTION_INSTRUCTION_PROMPT
from .db import database

from .task_manager import (
    store_task_result,
    create_stream_task,
    update_stream_task,
    complete_stream_task,
    send_to_client,
    remove_task_from_client
)
from .models import ProcessResponse

import sys
import io
import threading
import time
from contextlib import redirect_stdout, redirect_stderr

load_dotenv(os.path.join(os.path.dirname(__file__), "server", 'app.env'))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
# default_llm_config = {
#     "model": "gpt-4o-mini",
#     "openai_key": OPENAI_API_KEY,
#     # "stream": True,
#     "output_response": True,
#     "max_tokens": 16000
# }
default_llm_config = {
    "model": "openai/gpt-4o-mini",
    "openrouter_key": OPENROUTER_API_KEY,
    # "stream": True,
    "output_response": True,
    "max_tokens": 16000
}

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
default_tools = []

class OutputCapture:
    """Capture stdout and stderr output for streaming"""
    def __init__(self):
        self.stdout_buffer = io.StringIO()
        self.stderr_buffer = io.StringIO()
        self.combined_output = []
        self.lock = threading.Lock()
        
    def write_stdout(self, text):
        """Write to stdout buffer"""
        with self.lock:
            self.stdout_buffer.write(text)
            self.combined_output.append(('stdout', text, time.time()))
            
    def write_stderr(self, text):
        """Write to stderr buffer"""
        with self.lock:
            self.stderr_buffer.write(text)
            self.combined_output.append(('stderr', text, time.time()))
            
    def get_new_output(self, last_index=0):
        """Get new output since last_index"""
        with self.lock:
            new_output = self.combined_output[last_index:]
            return new_output
            
    def get_total_output(self):
        """Get all captured output"""
        with self.lock:
            return self.combined_output.copy()

class StreamingStdout:
    """Custom stdout that captures output for streaming"""
    def __init__(self, output_capture):
        self.output_capture = output_capture
        self.original_stdout = sys.stdout
        
    def write(self, text):
        self.original_stdout.write(text)
        self.output_capture.write_stdout(text)
        
    def flush(self):
        self.original_stdout.flush()

class StreamingStderr:
    """Custom stderr that captures output for streaming"""
    def __init__(self, output_capture):
        self.output_capture = output_capture
        self.original_stderr = sys.stderr
        
    def write(self, text):
        self.original_stderr.write(text)
        self.output_capture.write_stderr(text)
        
    def flush(self):
        self.original_stderr.flush()

def read_tunnel_info():
    """Read tunnel information from JSON file"""
    try:
        if os.path.exists(TUNNEL_INFO_PATH):
            with open(TUNNEL_INFO_PATH, "r") as f:
                return json.load(f)
        return None
    except Exception:
        return None

def create_workflow_info(config: Dict[str, Any], execution_result: Dict[str, Any]) -> str:
    """Create comprehensive workflow info string including public URL and other details"""
    tunnel_info = read_tunnel_info()
    
    # Extract key information
    public_url = tunnel_info.get("public_url", "Not available") if tunnel_info else "Not available"
    local_url = tunnel_info.get("local_url", "Not available") if tunnel_info else "Not available"
    
    workflow_dict = config.get("workflow", {})
    
    # Build comprehensive workflow info string
    workflow_info = f"""
=== WORKFLOW EXECUTION INFORMATION ===
Timestamp: {datetime.now().isoformat()}

=== SERVER ACCESS INFORMATION ===
Public URL: {public_url}
Local URL: {local_url}

=== WORKFLOW CONFIGURATION ===
Workflow Status: {execution_result.get('status', 'Unknown')}
LLM Configuration: {config.get('llm_config', {}).get('model', 'Unknown')}
MCP Configuration: {'Enabled' if config.get('mcp_config') else 'Disabled'}

=== EXECUTION RESULTS ===
Execution Message: {execution_result.get('message', 'No message available')}
Workflow Received: {execution_result.get('workflow_received', False)}
LLM Config Received: {execution_result.get('llm_config_received', False)}
MCP Config Received: {execution_result.get('mcp_config_received', False)}

=== INPUTS PROVIDED ===
{json.dumps(config.get('inputs', {}), indent=2)}

""".strip()
    
    return workflow_info

def create_task_info(project_id: str, goal: str, additional_info: Dict[str, Any] = None, public_url: str = None) -> dict:
    """Generate comprehensive task info string for a project"""
    task_prompt = TASK_INFO_PROMPT_SUDO.format(
        goal=goal,
        additional_info=additional_info
    )
    
    llm_config = create_llm_config(additional_info.get("llm_config", default_llm_config))
    llm = create_llm_instance(llm_config)
    response = llm.single_generate([{"role": "user", "content": task_prompt}])
    
    # Debug: Print the LLM response
    print(f"LLM Response: {response}")
    
    task_info = parse_json_from_text(response)
    
    # Debug: Print what parse_json_from_text extracted
    print(f"Parsed JSON list: {task_info}")
    
    if not task_info:
        raise ValueError(f"No JSON found in LLM response: {response}")
    
    try:
        task_info = json.loads(task_info[0])
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        print(f"Failed to parse: {task_info[0]}")
        raise ValueError(f"Invalid JSON in LLM response: {task_info[0]}")
    
    # Add connection_instruction field using the template
    connection_instruction = CONNECTION_INSTRUCTION_PROMPT.format(
        project_id=project_id,
        public_url=public_url or "Not available",
        workflow_name=task_info.get("workflow_name", "Not available"),
        workflow_description=task_info.get("workflow_description", "Not available"),
        workflow_inputs=task_info.get("workflow_inputs", "Not available"),
        workflow_outputs=task_info.get("workflow_outputs", "Not available")
    )
    task_info["connection_instruction"] = connection_instruction
    
    return task_info

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
    

async def process_task(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Placeholder for the actual processing logic.
    This is where you'll implement your specific processing functionality.
    """
    # Simulate some processing time
    await asyncio.sleep(2)
    
    # Example processing - replace this with your actual logic
    return {
        "processed": True,
        "input_parameters": config,
        "sample_output": "This is a sample result"
    }

async def handle_process_request(config: Dict[str, Any]) -> ProcessResponse:
    """Handle a processing request and return a response"""
    task_id = str(uuid.uuid4())
    
    # Process the task
    result = await process_task(config)
    
    # Create response
    response = ProcessResponse(
        task_id=task_id,
        status="completed",
        result=result
    )
    
    # Store the result
    store_task_result(task_id, response)
    
    return response

async def process_stream_task(task_id: str, config: Dict[str, Any]):
    """
    Process a streaming task and generate updates.
    """
    total_steps = 5
    for step in range(total_steps):
        # Simulate processing time for each step
        await asyncio.sleep(1)
        
        # Update progress
        progress = {
            "step": step + 1,
            "total_steps": total_steps,
            "timestamp": datetime.now().isoformat(),
            "status": "processing",
            "progress": ((step + 1) / total_steps) * 100,
            "current_state": f"Processing step {step + 1}/{total_steps}"
        }
        
        update_stream_task(task_id, progress)
    
    # Final result
    final_result = {
        "status": "completed",
        "timestamp": datetime.now().isoformat(),
        "result": {
            "processed": True,
            "input_parameters": config,
            "final_output": "Streaming task completed successfully"
        }
    }
    
    update_stream_task(task_id, final_result)
    complete_stream_task(task_id)

async def start_streaming_task(config: Dict[str, Any]) -> Dict[str, Any]:
    """Start a new streaming task"""
    task_id = str(uuid.uuid4())
    
    # Initialize the stream task
    create_stream_task(task_id, config)
    
    # Determine task type and start appropriate processing
    task_type = config.get("task_type", "default")
    
    # Default processing task
    asyncio.create_task(process_stream_task(task_id, config["parameters"]))
    
    return {
        "task_id": task_id,
        "status": "started",
        "stream_url": f"/stream/{task_id}",
        "task_type": task_type
    } 


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

async def execute_workflow_from_config(workflow: Dict[str, Any], llm_config_dict: Dict[str, Any], mcp_config: dict = None, inputs: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Execute a workflow with the given configuration.
    
    Args:
        workflow: The workflow definition/configuration to execute
        llm_config_dict: LLM configuration dictionary
        mcp_config: Optional MCP configuration dictionary
        inputs: Optional inputs dictionary to pass to async_execute
        
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
        workflow_graph: WorkFlowGraph = WorkFlowGraph.from_dict(workflow)
        if mcp_config:
            mcp_toolkit = MCPToolkit(config=mcp_config)
            tools = mcp_toolkit.get_tools()
        else:
            tools = default_tools
        
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

async def execute_workflow_from_config_with_capture(workflow: Dict[str, Any], llm_config_dict: Dict[str, Any], mcp_config: dict = None, inputs: Dict[str, Any] = None, output_capture: OutputCapture = None) -> Dict[str, Any]:
    """
    Execute a workflow with output capture for streaming.
    Modified version of execute_workflow_from_config that captures stdout/stderr.
    """
    try:
        if sudo_execution_result:
            # Even for sudo, we can simulate some output
            if output_capture:
                output_capture.write_stdout("Starting sudo workflow execution...\n")
                await asyncio.sleep(0.5)
                output_capture.write_stdout("Initializing workflow components...\n")
                await asyncio.sleep(0.5)
                output_capture.write_stdout("Executing workflow logic...\n")
                await asyncio.sleep(1)
                output_capture.write_stdout("Workflow execution completed successfully!\n")
                
            return {
                "status": "completed",
                "message": sudo_execution_result,
                "workflow_received": bool(workflow),
                "llm_config_received": bool(llm_config_dict),
                "mcp_config_received": bool(mcp_config)
            }
        
        # Setup output capture
        if output_capture:
            streaming_stdout = StreamingStdout(output_capture)
            streaming_stderr = StreamingStderr(output_capture)
            
            # Redirect stdout and stderr
            original_stdout = sys.stdout
            original_stderr = sys.stderr
            sys.stdout = streaming_stdout
            sys.stderr = streaming_stderr
            
            try:
                output_capture.write_stdout("🚀 Starting workflow execution...\n")
                
                # Create LLM config and instance
                output_capture.write_stdout("🔧 Creating LLM configuration...\n")
                llm_config = create_llm_config(llm_config_dict)
                llm = create_llm_instance(llm_config)
                
                # Create workflow graph
                output_capture.write_stdout("📊 Loading workflow graph...\n")
                workflow_graph: WorkFlowGraph = WorkFlowGraph.from_dict(workflow)
                
                # Setup tools
                output_capture.write_stdout("🛠️ Setting up tools...\n")
                if mcp_config:
                    mcp_toolkit = MCPToolkit(config=mcp_config)
                    tools = mcp_toolkit.get_tools()
                    output_capture.write_stdout(f"   • Loaded {len(tools)} MCP tools\n")
                else:
                    tools = default_tools
                    output_capture.write_stdout(f"   • Using {len(tools)} default tools\n")
                
                # Setup agent manager
                output_capture.write_stdout("🤖 Setting up agent manager...\n")
                agent_manager = AgentManager(tools=tools)
                agent_manager.add_agents_from_workflow(workflow_graph, llm_config=llm_config)
                
                # Create and initialize workflow
                output_capture.write_stdout("⚙️ Initializing workflow...\n")
                workflow_instance = WorkFlow(graph=workflow_graph, agent_manager=agent_manager, llm=llm)
                workflow_instance.init_module()
                
                # Execute workflow
                output_capture.write_stdout("▶️ Executing workflow...\n")
                if inputs:
                    output_capture.write_stdout(f"   • Inputs: {inputs}\n")
                
                output = await workflow_instance.async_execute(inputs=inputs)
                
                output_capture.write_stdout("✅ Workflow execution completed!\n")
                output_capture.write_stdout(f"   • Output: {str(output)[:200]}{'...' if len(str(output)) > 200 else ''}\n")
                
                return {
                    "status": "completed",
                    "message": output,
                    "workflow_received": bool(workflow),
                    "llm_config_received": bool(llm_config_dict),
                    "mcp_config_received": bool(mcp_config)
                }
                
            finally:
                # Restore original stdout and stderr
                sys.stdout = original_stdout
                sys.stderr = original_stderr
                
        else:
            # Fallback to original implementation if no output capture
            return await execute_workflow_from_config(workflow, llm_config_dict, mcp_config, inputs)
            
    except Exception as e:
        if output_capture:
            output_capture.write_stderr(f"❌ Error in workflow execution: {str(e)}\n")
            
        return {
            "status": "error",
            "message": f"In the execution process, got error:\n{e}",
            "workflow_received": bool(workflow),
            "llm_config_received": bool(llm_config_dict),
            "mcp_config_received": bool(mcp_config)
        }



def generate_project_id() -> str:
    """Generate a unique project ID"""
    return f"proj_{uuid.uuid4().hex[:12]}"

async def setup_project(workflow_id: str, requirement_id: str, user_id: str) -> Dict[str, Any]:
    """
    Phase 1: Setup workflow and generate task_info.
    Updated to use generalized database and retrieve requirement information.
    """
    # Check if workflow already exists (for logging purposes)
    existing_workflow = await database.find_one("workflows", {"id": workflow_id})
    workflow_exists = existing_workflow is not None
    
    if workflow_exists:
        print(f"🔄 Workflow {workflow_id} already exists, will update with new setup")
    
    # Retrieve requirement information from database
    requirement = await database.find_one("requirements", {"id": requirement_id})
    if not requirement:
        raise ValueError(f"Requirement with ID {requirement_id} not found")
    
    # Extract goal from requirement description field
    goal = requirement.get("description", "")
    if not goal:
        # Fallback to goal field if description is empty
        goal = requirement.get("goal", f"Complete task for requirement {requirement_id}")
    
    # Prepare additional info from requirement
    additional_info = {
        "title": requirement.get("title", ""),
        "category": requirement.get("category", ""),
        "status": requirement.get("status", ""),
        "llm_config": default_llm_config
    }
    
    # Get tunnel information
    tunnel_info = read_tunnel_info()
    public_url = tunnel_info.get("public_url") if tunnel_info else None
    
    # Generate comprehensive task info (reusing existing logic)
    task_info = create_task_info(
        workflow_id, 
        goal, 
        additional_info, 
        public_url
    )
    
    if workflow_exists:
        # Update existing workflow with complete new setup
        await database.update(
            "workflows",
            {"id": workflow_id},
            {
                "user_id": user_id,
                "requirement_id": requirement_id,
                "task_info": task_info,
                "workflow_graph": None,
                "execution_result": None,
                "status": "uninitialized"
            }
        )
        print(f"✅ Updated existing workflow {workflow_id} with new setup")
    else:
        # Create new workflow document
        workflow_doc = {
            "id": workflow_id,
            "user_id": user_id,
            "requirement_id": requirement_id,
            "task_info": task_info,
            "workflow_graph": None,
            "execution_result": None,
            "status": "uninitialized"
        }
        await database.insert("workflows", workflow_doc)
        print(f"✅ Created new workflow {workflow_id}")
    
    return task_info

async def get_project(workflow_id: str) -> Dict[str, Any]:
    """Retrieve workflow information from the database"""
    return await database.find_one("workflows", {"id": workflow_id})

async def update_project_status(workflow_id: str, status: str, **kwargs):
    """Update workflow status and other fields"""
    # Update status and any additional fields
    updates = {"status": status, **kwargs}
    await database.update(
        "workflows", 
        {"id": workflow_id}, 
        updates
    )

async def list_projects() -> Dict[str, Any]:
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

async def generate_workflow_for_project(workflow_id: str) -> Dict[str, Any]:
    """
    Phase 2: Generate workflow graph based on task_info.
    Updated to use new database structure.
    """
    # Check if workflow exists
    workflow = await get_project(workflow_id)
    if not workflow:
        raise ValueError(f"Workflow with ID {workflow_id} not found")
    
    # Check if setup was completed
    if workflow.get("task_info") is None:
        raise ValueError(f"Workflow {workflow_id} has not completed setup phase")
    
    task_info = workflow["task_info"]
    
    # Extract information for workflow generation
    goal = task_info.get("workflow_description", "")
    workflow_inputs = task_info.get("workflow_inputs", [])
    workflow_outputs = task_info.get("workflow_outputs", [])
    
    # Format the prompt with goal, inputs, and outputs
    formatted_goal = WORKFLOW_GENERATION_PROMPT.format(
        goal=goal, 
        inputs=workflow_inputs, 
        outputs=workflow_outputs
    )
    
    try:
        # Update workflow status
        await update_project_status(workflow_id, "running")
        
        # Generate workflow graph
        workflow_graph = await generate_workflow_from_goal(
            formatted_goal, 
            default_llm_config, 
            mcp_config={}
        )
        
        if workflow_graph is None:
            await update_project_status(workflow_id, "failed")
            return {
                "workflow_graph": {},
                "status": "failed"
            }
        
        # Convert workflow_graph to serializable format
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
        
        # Update workflow storage with generated graph
        await update_project_status(
            workflow_id, 
            "pending",
            workflow_graph=workflow_dict
        )
        
        return {
            "workflow_graph": workflow_dict,
            "status": "success"
        }
        
    except Exception as e:
        await update_project_status(workflow_id, "failed")
        raise ValueError(f"Error generating workflow: {str(e)}")

async def execute_workflow_for_project(workflow_id: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Phase 3: Execute workflow with provided inputs.
    Updated to use new database structure.
    """
    # Check if workflow exists
    workflow = await get_project(workflow_id)
    if not workflow:
        raise ValueError(f"Workflow with ID {workflow_id} not found")
    
    # Check if workflow generation was completed
    if workflow.get("workflow_graph") is None:
        raise ValueError(f"Workflow {workflow_id} has not completed generation phase")
    
    try:
        # Update workflow status
        await update_project_status(workflow_id, "running")
        
        # Get workflow graph
        workflow_graph = workflow["workflow_graph"]
        
        # Execute the workflow
        execution_result = await execute_workflow_from_config(
            workflow_graph, 
            default_llm_config, 
            mcp_config={}, 
            inputs=inputs
        )
        
        if execution_result is None:
            await update_project_status(workflow_id, "failed")
            return {
                "status": "failed",
                "error": "Failed to execute workflow"
            }
        
        # Process execution result
        if isinstance(execution_result, dict):
            execution_message = execution_result.get("message", "")
        else:
            execution_message = str(execution_result)
            
        # Clean up markdown formatting from the message
        if isinstance(execution_message, str):
            if execution_message.startswith("```markdown"):
                execution_message = execution_message[11:]
            if execution_message.endswith("```"):
                execution_message = execution_message[:-3]
        
        # Update execution_result with cleaned message
        if isinstance(execution_result, dict):
            execution_result["message"] = execution_message
        else:
            execution_result = execution_message
        
        # Update workflow storage with execution results
        await update_project_status(
            workflow_id, 
            "completed",
            execution_result=execution_result
        )
        
        return execution_result
        
    except Exception as e:
        await update_project_status(workflow_id, "failed")
        raise ValueError(f"Error executing workflow: {str(e)}")

async def execute_workflow_for_project_stream(task_id: str, workflow_id: str, inputs: Dict[str, Any]):
    """
    Streaming version of execute_workflow_for_project.
    Streams real command output during workflow execution.
    """
    output_capture = OutputCapture()
    output_index = 0
    
    try:
        # Initial setup message
        update_stream_task(task_id, {
            "timestamp": datetime.now().isoformat(),
            "status": "starting",
            "output_type": "setup",
            "message": "Starting workflow execution...",
            "command_output": "Initializing workflow execution stream...\n"
        })
        
        # Validate workflow exists
        workflow = await get_project(workflow_id)
        if not workflow:
            error_result = {
                "status": "error",
                "timestamp": datetime.now().isoformat(),
                "output_type": "error",
                "error": f"Workflow with ID {workflow_id} not found",
                "command_output": f"ERROR: Workflow {workflow_id} not found\n"
            }
            update_stream_task(task_id, error_result)
            complete_stream_task(task_id)
            return

        # Check if workflow generation was completed
        if workflow.get("workflow_graph") is None:
            error_result = {
                "status": "error",
                "timestamp": datetime.now().isoformat(),
                "output_type": "error",
                "error": f"Workflow {workflow_id} has not completed generation phase",
                "command_output": f"ERROR: Workflow {workflow_id} not ready for execution\n"
            }
            update_stream_task(task_id, error_result)
            complete_stream_task(task_id)
            return

        # Update workflow status
        await update_project_status(workflow_id, "running")
        
        # Start output monitoring task
        async def monitor_output():
            nonlocal output_index
            while True:
                try:
                    # Get new output every 1 second
                    new_output = output_capture.get_new_output(output_index)
                    
                    if new_output:
                        # Format the output for streaming
                        command_output = ""
                        for output_type, text, timestamp in new_output:
                            command_output += text
                            
                        # Send update with new command output
                        update_stream_task(task_id, {
                            "timestamp": datetime.now().isoformat(),
                            "status": "running",
                            "output_type": "command",
                            "command_output": command_output,
                            "total_output_lines": len(output_capture.get_total_output())
                        })
                        
                        output_index += len(new_output)
                    
                    await asyncio.sleep(1)  # Check every 1 second
                    
                except Exception as e:
                    print(f"Error monitoring output: {e}")
                    break
        
        # Start monitoring task
        monitor_task = asyncio.create_task(monitor_output())
        
        try:
            # Execute the workflow with output capture
            workflow_graph = workflow["workflow_graph"]
            
            execution_result = await execute_workflow_from_config_with_capture(
                workflow_graph, 
                default_llm_config, 
                mcp_config={}, 
                inputs=inputs,
                output_capture=output_capture
            )
            
            # Cancel monitoring task
            monitor_task.cancel()
            
            # Send any remaining output
            remaining_output = output_capture.get_new_output(output_index)
            if remaining_output:
                command_output = ""
                for output_type, text, timestamp in remaining_output:
                    command_output += text
                    
                update_stream_task(task_id, {
                    "timestamp": datetime.now().isoformat(),
                    "status": "running",
                    "output_type": "command",
                    "command_output": command_output,
                    "total_output_lines": len(output_capture.get_total_output())
                })
            
            if execution_result is None:
                await update_project_status(workflow_id, "failed")
                error_result = {
                    "status": "error",
                    "timestamp": datetime.now().isoformat(),
                    "output_type": "error",
                    "error": "Failed to execute workflow",
                    "command_output": "ERROR: Workflow execution failed\n"
                }
                update_stream_task(task_id, error_result)
                complete_stream_task(task_id)
                return

            # Process execution result (same logic as original function)
            if isinstance(execution_result, dict):
                execution_message = execution_result.get("message", "")
            else:
                execution_message = str(execution_result)
                
            # Clean up markdown formatting from the message
            if isinstance(execution_message, str):
                if execution_message.startswith("```markdown"):
                    execution_message = execution_message[11:]
                if execution_message.endswith("```"):
                    execution_message = execution_message[:-3]
            
            # Update execution_result with cleaned message
            if isinstance(execution_result, dict):
                execution_result["message"] = execution_message
            else:
                execution_result = execution_message
            
            # Update workflow storage with execution results
            await update_project_status(
                workflow_id, 
                "completed",
                execution_result=execution_result
            )
            
            # Final result with complete output
            total_output = output_capture.get_total_output()
            complete_command_output = ""
            for output_type, text, timestamp in total_output:
                complete_command_output += text
                
            final_result = {
                "status": "completed",
                "timestamp": datetime.now().isoformat(),
                "output_type": "completion",
                "workflow_id": workflow_id,
                "execution_result": execution_result,
                "command_output": complete_command_output,
                "total_output_lines": len(total_output),
                "message": "Workflow execution completed successfully"
            }
            
            update_stream_task(task_id, final_result)
            complete_stream_task(task_id)
            
        except Exception as e:
            # Cancel monitoring task
            monitor_task.cancel()
            raise e
            
    except Exception as e:
        # Handle errors
        await update_project_status(workflow_id, "failed")
        error_result = {
            "status": "error",
            "timestamp": datetime.now().isoformat(),
            "output_type": "error",
            "error": f"Error executing workflow: {str(e)}",
            "workflow_id": workflow_id,
            "command_output": f"ERROR: {str(e)}\n"
        }
        update_stream_task(task_id, error_result)
        complete_stream_task(task_id)

async def start_streaming_workflow_execution(workflow_id: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Start a new streaming workflow execution task"""
    task_id = str(uuid.uuid4())
    
    # Initialize the stream task
    config = {
        "task_type": "workflow_execution",
        "workflow_id": workflow_id,
        "inputs": inputs,
        "timeout": 300  # 1 hour timeout by default
    }
    create_stream_task(task_id, config)
    
    # Start the streaming workflow execution
    asyncio.create_task(execute_workflow_for_project_stream(task_id, workflow_id, inputs))
    
    return {
        "task_id": task_id,
        "status": "started",
        "stream_url": f"/stream/{task_id}",
        "task_type": "workflow_execution",
        "workflow_id": workflow_id
    }
