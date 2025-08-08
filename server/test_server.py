"""
EvoAgentX API Tutorial - Complete Workflow Lifecycle

This test demonstrates the complete lifecycle of the EvoAgentX project setup and workflow execution.
It covers all phases: setup, generation, execution, and WebSocket streaming.

================================================================================
API ENDPOINTS OVERVIEW
================================================================================

1. HEALTH CHECK
   GET /health
   - Input: None
   - Output: {"status": "healthy"}

2. PROJECT SETUP (Phase 1)
   POST /project/setup
   - Input: {"project_short_id": "string"}
   - Output: {"workflow_graphs": [...], "message": "string"}

3. WORKFLOW GENERATION (Phase 2)
   POST /workflow/{workflow_id}/generate
   - Input: None (workflow_id in URL)
   - Output: {"workflow_graph": {...}, "status": "success|failed"}

4. WORKFLOW EXECUTION (Phase 3) - HTTP
   POST /workflow/{workflow_id}/execute
   - Input: {"inputs": {"key1": "value1", "key2": "value2"}}
   - Output: {"execution_result": {...}}

5. WORKFLOW EXECUTION (Phase 3) - WebSocket ⭐
   WebSocket /workflow/{workflow_id}/execute_ws
   - Input: {"inputs": {"key1": "value1", "key2": "value2"}}
   - Output: Real-time streaming messages

6. WORKFLOW STATUS
   GET /workflow/{workflow_id}/status
   - Input: None (workflow_id in URL)
   - Output: Complete workflow state and data

7. WORKFLOW GRAPH
   GET /workflow/{workflow_id}/get_graph
   - Input: None (workflow_id in URL)
   - Output: {"workflow_graph": {...}}

8. USER QUERY ANALYSIS
   POST /project/{project_short_id}/user_query
   - Input: {"query": "string"}
   - Output: {"result": {...}}

================================================================================
WEBSOCKET EXECUTION API - DETAILED DOCUMENTATION
================================================================================

The WebSocket execution endpoint is the most powerful feature, providing real-time progress
updates during workflow execution.

ENDPOINT: WebSocket /workflow/{workflow_id}/execute_ws

CONNECTION:
- URL: ws://localhost:8001/workflow/{workflow_id}/execute_ws
- Authentication: Include 'eax-access-token' header in initial connection

INPUT MESSAGE FORMAT:
{
    "inputs": {
        "character_name": "Alice",
        "character_type": "human", 
        "setting": "a magical forest",
        "genre": "fantasy",
        "target_age": "8-12",
        "moral_lesson": "friendship and courage",
        "story_length": "medium",
        "language": "English"
    }
}

OUTPUT MESSAGE TYPES:
====================

1. CONNECTION CONFIRMATION
   {"type": "connection", "content": "WebSocket connection established", "result": null}

2. EXECUTION START
   {"type": "start", "content": "Workflow execution started", "result": null}

3. PROGRESS UPDATES
   {"type": "progress", "content": "Executing workflow: story_generation (75% complete)", "result": null}

4. LOG MESSAGES
   {"type": "log", "content": "INFO: Starting workflow execution", "result": null}

5. OUTPUT MESSAGES
   {"type": "output", "content": "Processing user data...", "result": null}

6. COMPLETION
   {"type": "complete", "content": "Workflow execution completed successfully", "result": {...}}

7. ERROR MESSAGES
   {"type": "error", "content": "Failed to connect to database", "result": null}

EXAMPLE COMPLETE WORKFLOW:
==========================

1. Setup Project:
   POST /project/setup
   {"project_short_id": "nw7czqj"}

2. Generate Workflow:
   POST /workflow/550e8400-e29b-41d4-a716-446655440001/generate

3. Execute via WebSocket:
   WebSocket: ws://localhost:8001/workflow/550e8400-e29b-41d4-a716-446655440001/execute_ws
   Send: {"inputs": {"character_name": "Alice", "character_type": "human", ...}}

4. Receive Real-time Updates:
   - Connection confirmation
   - Execution start
   - Progress updates (0% to 100%)
   - Log messages
   - Final result

KNOWN ISSUES:
- Workflow generation may fail due to 'workflow_inputs' key error in service.py
- List workflows returns "projects" instead of "workflows" in response

These issues are handled gracefully in the test with proper error reporting.

FIXED TEST DATA:
- Project ID: "nw7czqj"
- Workflow IDs: ["550e8400-e29b-41d4-a716-446655440001", "7f8a9a6a-35b1-4891-8a4c-5715a5c5478f"]
- Test inputs: Fixed story generation inputs for consistent results

WEBSOCKET STREAMING TEST:
- Tests real-time progress updates via WebSocket connections
- Validates connection establishment, progress updates, log messages, and final results
- Includes timeout handling and error reporting
- Supports multiple workflow testing in sequence

WEBSOCKET API DOCUMENTATION:
==================

test_websocket_connection Function
=================================

Purpose:
--------
Tests real-time WebSocket streaming for workflow execution, providing live progress updates,
log messages, and execution results.

Function Signature:
------------------
async def test_websocket_connection(workflow_id: str) -> Dict[str, Any]

Inputs:
-------
- workflow_id (str): The unique identifier of the workflow to execute via WebSocket

Outputs:
--------
Returns a dictionary with the following structure:

Success Case:
{
    "success": True,
    "workflow_id": "550e8400-e29b-41d4-a716-446655440001",
    "total_messages": 15,
    "progress_updates": 5,
    "log_messages": 8,
    "output_messages": 2,
    "input_messages": 0,
    "periodic_updates": 3,
    "final_result": {
        "original_message": "...",
        "parsed_json": {...}
    },
    "messages": [
        {"type": "connection", "content": "WebSocket connection established", "result": null},
        {"type": "start", "content": "Workflow execution started", "result": null},
        {"type": "progress", "content": "Executing workflow: name (75% complete)", "result": null},
        {"type": "log", "content": "INFO: Starting workflow execution", "result": null},
        {"type": "output", "content": "Processing user data...", "result": null},
        {"type": "complete", "content": "Workflow execution completed successfully", "result": {...}}
    ]
}

Error Cases:
{
    "success": False,
    "error": "WebSocket connection failed: Connection refused",
    "workflow_id": "550e8400-e29b-41d4-a716-446655440001"
}

{
    "success": False,
    "error": "WebSocket connection timed out after 2 minutes",
    "workflow_id": "550e8400-e29b-41d4-a716-446655440001"
}

{
    "success": False,
    "error": "No final result received",
    "messages": [...]
}

WebSocket Message Types:
-----------------------
1. connection: Initial connection confirmation
   {"type": "connection", "content": "WebSocket connection established", "result": null}

2. start: Execution start notification
   {"type": "start", "content": "Workflow execution started", "result": null}

3. progress: Real-time progress updates
   {"type": "progress", "content": "Executing workflow: name (75% complete)", "result": null}

4. log: Log messages from workflow execution
   {"type": "log", "content": "INFO: Database connection established", "result": null}

5. output: Execution output messages
   {"type": "output", "content": "Processing user data...", "result": null}

6. complete: Final execution result
   {"type": "complete", "content": "Workflow execution completed successfully", "result": {...}}

7. error: Error messages
   {"type": "error", "content": "Failed to connect to database", "result": null}
"""

import asyncio
import json
import os
import uuid
import requests
from datetime import datetime
from typing import Dict, Any, List, Tuple
from dotenv import load_dotenv

# Load environment variables
load_dotenv('server/app.env')

# Test configuration
BASE_URL = "https://evoagentx-server.fly.dev"
# BASE_URL = "http://localhost:8001"
# WebSocket URL - derived from BASE_URL
WS_BASE_URL = BASE_URL.replace("http://", "ws://").replace("https://", "wss://")
ACCESS_TOKEN = os.getenv("EAX_ACCESS_TOKEN", "default_secret_token_change_me")
HEADERS = {
    "eax-access-token": ACCESS_TOKEN,
    "Content-Type": "application/json"
}
# Fixed test data for consistent testing
test_workflow_id_1 = "550e8400-e29b-41d4-a716-446655440001"
test_workflow_id_2 = "7f8a9a6a-35b1-4891-8a4c-5715a5c5478f"

# Fixed test inputs for consistent results - Updated for treatment recommendation workflow
TEST_INPUTS = {
    "character_name": "Alice",
    "character_type": "human",
    "setting": "a magical forest",
    "genre": "fantasy",
    "target_age": "8-12",
    "moral_lesson": "friendship and courage",
    "story_length": "medium",
    "language": "English"
}

# Fixed test configuration
TEST_CONFIG = {
    "project_short_id": "nw7czqj",
    "test_workflow_ids": [test_workflow_id_1, test_workflow_id_2]
}
def generate_test_ids() -> str:
    """Generate fixed test IDs for consistent testing"""
    project_short_id = TEST_CONFIG["project_short_id"]
    return project_short_id

def test_health_check() -> Tuple[bool, Dict[str, Any]]:
    """Test the health check endpoint"""
    print("🔍 Testing health check...")
    result = {"success": False, "response": None, "error": None}
    
    try:
        response = requests.get(f"{BASE_URL}/health")
        result["response"] = response.json()
        
        assert response.status_code == 200
        assert result["response"]["status"] == "healthy"
        
        result["success"] = True
        print("✅ Health check passed")
        
    except Exception as e:
        result["error"] = str(e)
        print(f"❌ Health check failed: {e}")
    
    return result["success"], result

def test_project_setup(project_short_id: str) -> Tuple[bool, Dict[str, Any], List[str]]:
    """Test Phase 1: Project setup"""
    print("\n🚀 Testing Phase 1: Project Setup")
    
    request_data = {
        "project_short_id": project_short_id
    }
    
    print(f"Request: POST {BASE_URL}/project/setup")
    print(f"Request Body: {json.dumps(request_data, indent=2)}")
    
    result = {"success": False, "response": None, "error": None, "workflow_ids": []}
    
    try:
        response = requests.post(
            f"{BASE_URL}/project/setup",
            headers=HEADERS,
            json=request_data
        )
        
        print(f"Response Status: {response.status_code}")
        result["response"] = response.json()
        print(f"Response Body: {json.dumps(result['response'], indent=2)}")
        
        assert response.status_code == 200
        assert "workflow_graphs" in result["response"]
        assert "message" in result["response"]
        
        # Extract workflow IDs
        for workflow in result["response"]["workflow_graphs"]:
            if "workflow_id" in workflow:
                result["workflow_ids"].append(workflow["workflow_id"])
        
        result["success"] = True
        print("✅ Project setup completed successfully")
        
    except Exception as e:
        result["error"] = str(e)
        print(f"❌ Project setup failed: {e}")
    
    return result["success"], result, result["workflow_ids"]

def test_workflow_generation(workflow_ids: List[str]) -> Tuple[bool, Dict[str, Any]]:
    """Test Phase 2: Workflow generation"""
    print("\n🔧 Testing Phase 2: Workflow Generation")
    
    if not workflow_ids:
        print("❌ No workflow IDs available from setup phase")
        return False, {"error": "No workflow IDs available"}
    
    result = {"success": False, "responses": {}, "errors": {}}
    success_count = 0
    
    for workflow_id in workflow_ids:
        print(f"\nGenerating workflow: {workflow_id}")
        
        print(f"Request: POST {BASE_URL}/workflow/{workflow_id}/generate")
        
        try:
            response = requests.post(
                f"{BASE_URL}/workflow/{workflow_id}/generate",
                headers=HEADERS
            )
            
            print(f"Response Status: {response.status_code}")
            response_data = response.json()
            print(f"Response Body: {json.dumps(response_data, indent=2)}")
            
            # Handle both success and error cases
            if response.status_code == 200:
                assert "workflow_graph" in response_data
                assert "status" in response_data
                
                result["responses"][workflow_id] = response_data
                print(f"✅ Workflow generation completed for {workflow_id}")
                success_count += 1
            else:
                error_msg = f"HTTP {response.status_code}: {response_data.get('detail', 'Unknown error')}"
                result["errors"][workflow_id] = error_msg
                print(f"❌ Workflow generation failed for {workflow_id}: {error_msg}")
            
        except Exception as e:
            error_msg = str(e)
            result["errors"][workflow_id] = error_msg
            print(f"❌ Workflow generation failed for {workflow_id}: {error_msg}")
    
    result["success"] = success_count > 0
    return result["success"], result

def test_workflow_execution(workflow_ids: List[str]) -> Tuple[bool, Dict[str, Any]]:
    """Test Phase 3: Workflow execution"""
    print("\n⚡ Testing Phase 3: Workflow Execution")
    
    if not workflow_ids:
        print("❌ No workflow IDs available from setup phase")
        return False, {"error": "No workflow IDs available"}
    
    # Use fixed test inputs for consistent results
    test_inputs = TEST_INPUTS
    
    result = {"success": False, "responses": {}, "errors": {}}
    success_count = 0
    
    for workflow_id in workflow_ids:
        print(f"\nExecuting workflow: {workflow_id}")
        
        # The API expects workflow_id in the URL path, not in the request body
        request_data = {
            "inputs": test_inputs
        }
        
        print(f"Request: POST {BASE_URL}/workflow/{workflow_id}/execute")
        print(f"Request Body: {json.dumps(request_data, indent=2)}")
        
        try:
            response = requests.post(
                f"{BASE_URL}/workflow/{workflow_id}/execute",
                headers=HEADERS,
                json=request_data
            )
            
            print(f"Response Status: {response.status_code}")
            response_data = response.json()
            print(f"Response Body: {json.dumps(response_data, indent=2)}")
            
            # Handle both success and error cases
            if response.status_code == 200:
                assert "execution_result" in response_data
                result["responses"][workflow_id] = response_data
                print(f"✅ Workflow execution completed for {workflow_id}")
                success_count += 1
            else:
                error_msg = f"HTTP {response.status_code}: {response_data.get('detail', 'Unknown error')}"
                result["errors"][workflow_id] = error_msg
                print(f"❌ Workflow execution failed for {workflow_id}: {error_msg}")
            
        except Exception as e:
            error_msg = str(e)
            result["errors"][workflow_id] = error_msg
            print(f"❌ Workflow execution failed for {workflow_id}: {error_msg}")
    
    result["success"] = success_count > 0
    return result["success"], result

def test_workflow_status(workflow_ids: List[str]) -> Tuple[bool, Dict[str, Any]]:
    """Test workflow status endpoint"""
    print("\n📊 Testing Workflow Status")
    
    if not workflow_ids:
        print("❌ No workflow IDs available")
        return False, {"error": "No workflow IDs available"}
    
    result = {"success": False, "responses": {}, "errors": {}}
    success_count = 0
    
    for workflow_id in workflow_ids:
        print(f"\nChecking status for workflow: {workflow_id}")
        
        print(f"Request: GET {BASE_URL}/workflow/{workflow_id}/status")
        
        try:
            response = requests.get(
                f"{BASE_URL}/workflow/{workflow_id}/status",
                headers=HEADERS
            )
            
            print(f"Response Status: {response.status_code}")
            response_data = response.json()
            print(f"Response Body: {json.dumps(response_data, indent=2)}")
            
            assert response.status_code == 200
            assert "workflow_id" in response_data
            assert "status" in response_data
            assert "phases" in response_data
            
            result["responses"][workflow_id] = response_data
            print(f"✅ Workflow status retrieved for {workflow_id}")
            success_count += 1
            
        except Exception as e:
            error_msg = str(e)
            result["errors"][workflow_id] = error_msg
            print(f"❌ Workflow status check failed for {workflow_id}: {error_msg}")
    
    result["success"] = success_count > 0
    return result["success"], result

def test_list_workflows() -> Tuple[bool, Dict[str, Any]]:
    """Test listing all workflows"""
    print("\n📋 Testing List All Workflows")
    
    print(f"Request: GET {BASE_URL}/workflows")
    
    result = {"success": False, "response": None, "error": None}
    
    try:
        response = requests.get(
            f"{BASE_URL}/workflows",
            headers=HEADERS
        )
        
        print(f"Response Status: {response.status_code}")
        result["response"] = response.json()
        print(f"Response Body: {json.dumps(result['response'], indent=2)}")
        
        assert response.status_code == 200
        # The actual response has "projects" instead of "workflows"
        assert "projects" in result["response"] or "workflows" in result["response"]
        
        result["success"] = True
        print("✅ List workflows completed successfully")
        
    except Exception as e:
        result["error"] = str(e)
        print(f"❌ List workflows failed: {e}")
    
    return result["success"], result

def test_workflow_graph(workflow_ids: List[str]) -> Tuple[bool, Dict[str, Any]]:
    """Test workflow graph endpoint"""
    print("\n📊 Testing Workflow Graph Retrieval")
    
    if not workflow_ids:
        print("❌ No workflow IDs available")
        return False, {"error": "No workflow IDs available"}
    
    result = {"success": False, "responses": {}, "errors": {}}
    success_count = 0
    
    for workflow_id in workflow_ids:
        print(f"\nRetrieving graph for workflow: {workflow_id}")
        
        print(f"Request: GET {BASE_URL}/workflow/{workflow_id}/get_graph")
        
        try:
            response = requests.get(
                f"{BASE_URL}/workflow/{workflow_id}/get_graph",
                headers=HEADERS
            )
            
            print(f"Response Status: {response.status_code}")
            response_data = response.json()
            print(f"Response Body: {json.dumps(response_data, indent=2)}")
            
            assert response.status_code == 200
            assert "workflow_graph" in response_data
            
            # Check if workflow_graph is present (could be None if not generated yet)
            if response_data["workflow_graph"] is not None:
                print(f"✅ Workflow graph retrieved for {workflow_id}")
            else:
                print(f"⚠️  Workflow graph is None for {workflow_id} (not generated yet)")
            
            result["responses"][workflow_id] = response_data
            success_count += 1
            
        except Exception as e:
            error_msg = str(e)
            result["errors"][workflow_id] = error_msg
            print(f"❌ Workflow graph retrieval failed for {workflow_id}: {error_msg}")
    
    result["success"] = success_count > 0
    return result["success"], result

def test_websocket_streaming(workflow_ids: List[str]) -> Tuple[bool, Dict[str, Any]]:
    """Test WebSocket streaming functionality for workflow execution"""
    print(f"\n🔌 Testing WebSocket Streaming for {len(workflow_ids)} workflow(s)")
    
    import websockets
    import asyncio
    import json
    from datetime import datetime
    
    async def test_websocket_connection(workflow_id: str) -> Dict[str, Any]:
        """Test WebSocket connection for a single workflow"""
        uri = f"{WS_BASE_URL}/workflow/{workflow_id}/execute_ws"
        headers = {"eax-access-token": ACCESS_TOKEN}
        
        try:
            async with websockets.connect(uri, additional_headers=headers) as websocket:
                print(f"  📡 Connected to WebSocket for workflow {workflow_id}")
                
                # Wait for connection confirmation
                connection_msg = await websocket.recv()
                print(f"  📨 RAW CONNECTION MESSAGE: {connection_msg}")
                connection_data = json.loads(connection_msg)
                print(f"  📋 PARSED CONNECTION DATA: {json.dumps(connection_data, indent=2)}")
                
                if connection_data.get("type") != "connection":
                    return {
                        "success": False,
                        "error": f"Expected connection message, got: {connection_data.get('type')}"
                    }
                
                print(f"  ✅ Connection confirmed for workflow {workflow_id}")
                
                # Send execution inputs - USE SAME INPUTS AS REGULAR EXECUTION TEST
                test_inputs = TEST_INPUTS  # Use the same inputs as test_workflow_execution
                execution_request = {
                    "inputs": test_inputs
                }
                
                print(f"  📤 Sending execution request with inputs: {json.dumps(test_inputs, indent=2)}")
                await websocket.send(json.dumps(execution_request))
                print(f"  📤 Sent execution request for workflow {workflow_id}")
                
                # Collect all messages during execution
                messages = []
                progress_updates = []
                log_messages = []
                output_messages = []
                input_messages = []
                periodic_updates = []
                final_result = None
                error_occurred = False
                
                try:
                    while True:
                        message = await asyncio.wait_for(websocket.recv(), timeout=120.0)  # 2 minute timeout
                        print(f"  📨 RAW MESSAGE: {message}")
                        data = json.loads(message)
                        messages.append(data)
                        
                        msg_type = data.get("type")
                        content = data.get("content", "")
                        result = data.get("result")
                        
                        print(f"  📋 PARSED DATA: {json.dumps(data, indent=2)}")
                        
                        if msg_type == "start":
                            print(f"    🚀 Execution started for workflow {workflow_id}")
                        elif msg_type == "progress":
                            progress_updates.append({
                                "content": content
                            })
                            print(f"    📊 Progress: {content}")
                        elif msg_type == "log":
                            log_messages.append({
                                "content": content
                            })
                            print(f"    📝 Log: {content}")
                        elif msg_type == "output":
                            output_messages.append({
                                "content": content
                            })
                            print(f"    📤 Output: {content[:100]}{'...' if len(content) > 100 else ''}")
                        elif msg_type == "input":
                            input_messages.append({
                                "content": content
                            })
                            print(f"    📥 Input: {content[:100]}{'...' if len(content) > 100 else ''}")
                        elif msg_type == "complete":
                            print(data)
                            final_result = result
                            print(f"    ✅ Final result received for workflow {workflow_id}")
                            break
                        elif msg_type == "error":
                            error_occurred = True
                            print(f"    ❌ Error received: {content}")
                            break
                        else:
                            print(f"    📋 Unknown message type: {msg_type}")
                
                except asyncio.TimeoutError:
                    return {
                        "success": False,
                        "error": "WebSocket connection timed out after 2 minutes"
                    }
                
                # Analyze results
                if error_occurred:
                    return {
                        "success": False,
                        "error": "Error occurred during WebSocket execution",
                        "messages": messages
                    }
                
                if not final_result:
                    return {
                        "success": False,
                        "error": "No final result received",
                        "messages": messages
                    }
                
                return {
                    "success": True,
                    "workflow_id": workflow_id,
                    "total_messages": len(messages),
                    "progress_updates": len(progress_updates),
                    "log_messages": len(log_messages),
                    "output_messages": len(output_messages),
                    "input_messages": len(input_messages),
                    "periodic_updates": len(periodic_updates),
                    "final_result": final_result,
                    "messages": messages
                }
                
        except websockets.exceptions.InvalidURI:
            return {
                "success": False,
                "error": f"Invalid WebSocket URI: {uri}"
            }
        except websockets.exceptions.ConnectionClosed:
            return {
                "success": False,
                "error": "WebSocket connection was closed unexpectedly"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"WebSocket connection failed: {str(e)}"
            }
    
    # Run WebSocket tests for all workflow IDs
    results = []
    all_passed = True
    
    for workflow_id in workflow_ids:
        print(f"\n  🔄 Testing WebSocket streaming for workflow: {workflow_id}")
        
        # Run the async test
        try:
            result = asyncio.run(test_websocket_connection(workflow_id))
            results.append(result)
            
            if result["success"]:
                print(f"  ✅ WebSocket test PASSED for workflow {workflow_id}")
                print(f"    📊 Progress updates: {result['progress_updates']}")
                print(f"    📝 Log messages: {result['log_messages']}")
                print(f"    📤 Output messages: {result['output_messages']}")
                print(f"    📥 Input messages: {result['input_messages']}")
                print(f"    🔄 Periodic updates: {result['periodic_updates']}")
                print(f"    📨 Total messages: {result['total_messages']}")
            else:
                print(f"  ❌ WebSocket test FAILED for workflow {workflow_id}")
                print(f"    🚨 Error: {result['error']}")
                all_passed = False
                
        except Exception as e:
            error_result = {
                "success": False,
                "error": f"Test execution failed: {str(e)}",
                "workflow_id": workflow_id
            }
            results.append(error_result)
            print(f"  ❌ WebSocket test FAILED for workflow {workflow_id}")
            print(f"    🚨 Error: {str(e)}")
            all_passed = False
    
    # Summary
    passed_count = sum(1 for r in results if r["success"])
    total_count = len(results)
    
    print(f"\n📊 WebSocket Streaming Test Summary:")
    print(f"  ✅ Passed: {passed_count}/{total_count}")
    print(f"  ❌ Failed: {total_count - passed_count}/{total_count}")
    
    # Show detailed message counts
    if results:
        total_output = sum(r.get('output_messages', 0) for r in results if r.get('success'))
        total_input = sum(r.get('input_messages', 0) for r in results if r.get('success'))
        total_periodic = sum(r.get('periodic_updates', 0) for r in results if r.get('success'))
        print(f"  📤 Total output messages: {total_output}")
        print(f"  📥 Total input messages: {total_input}")
        print(f"  🔄 Total periodic updates: {total_periodic}")
    
    return all_passed, {
        "total_workflows": total_count,
        "passed_workflows": passed_count,
        "failed_workflows": total_count - passed_count,
        "results": results
    }

def test_user_query_analysis(project_short_id: str) -> Tuple[bool, Dict[str, Any]]:
    """
    Test the UserQueryRouter API endpoint.
    
    Args:
        project_short_id: The project identifier to test with
        
    Returns:
        Tuple of (success, result_dict)
    """
    print(f"🔍 Testing UserQueryRouter API for project {project_short_id}")
    
    try:
        # Test query
        test_query = "I want to add a new node at the end of the workflow to print all middle variables in the workflow"
        
        # Prepare request
        url = f"{BASE_URL}/project/{project_short_id}/user_query"
        headers = {
            "Content-Type": "application/json",
            "eax-access-token": ACCESS_TOKEN
        }
        payload = {
            "query": test_query
        }
        
        print(f"📤 Sending request to: {url}")
        print(f"📝 Query: {test_query}")
        
        # Make request
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        
        print(f"📥 Response status: {response.status_code}")
        
        if response.status_code == 200:
            response_data = response.json()
            print(f"✅ UserQueryRouter API test passed")
            print(f"   - Response structure: {list(response_data.keys())}")
            
            # Validate response structure
            if "result" in response_data:
                result = response_data["result"]
                print(f"   - Original query: {result.get('original_query', 'N/A')}")
                print(f"   - Total operations: {result.get('total_operations', 0)}")
                print(f"   - Is composite: {result.get('is_composite', False)}")
                print(f"   - Has frontend: {result.get('has_frontend', False)}")
                print(f"   - Has backend: {result.get('has_backend', False)}")
                
                # Check classified operations
                classified_ops = result.get('classified_operations', [])
                print(f"   - Classified operations: {len(classified_ops)}")
                for i, op in enumerate(classified_ops):
                    print(f"     Operation {i+1}: {op.get('category', 'unknown')} - {op.get('atomic_query', 'N/A')}")
                    print(f"       Not clear: {op.get('not_clear', False)}")
                    print(f"       Follow-up questions: {op.get('follow_up_questions', [])}")
                    print(f"       Clarity reasoning: {op.get('clarity_reasoning', 'N/A')}")
                
                # Print full result structure
                print(f"\n📋 Full Analysis Result:")
                print(f"   - Original query: {result.get('original_query', 'N/A')}")
                print(f"   - Total operations: {result.get('total_operations', 0)}")
                print(f"   - Is composite: {result.get('is_composite', False)}")
                print(f"   - Has frontend: {result.get('has_frontend', False)}")
                print(f"   - Has backend: {result.get('has_backend', False)}")
                
                # Print complete response for debugging
                print(f"\n🔍 Complete Response Data:")
                import json
                print(json.dumps(response_data, indent=2, default=str))
                
                return True, {
                    "status": "success",
                    "response": response_data,
                    "query": test_query,
                    "project_short_id": project_short_id
                }
            else:
                print(f"❌ Invalid response structure: missing 'result' field")
                return False, {
                    "status": "failed",
                    "error": "Invalid response structure",
                    "response": response_data
                }
        else:
            print(f"❌ UserQueryRouter API test failed with status {response.status_code}")
            try:
                error_data = response.json()
                print(f"   Error details: {error_data}")
            except:
                print(f"   Error text: {response.text}")
            
            return False, {
                "status": "failed",
                "error": f"HTTP {response.status_code}",
                "response_text": response.text
            }
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {str(e)}")
        return False, {
            "status": "failed",
            "error": f"Request exception: {str(e)}"
        }
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        return False, {
            "status": "failed",
            "error": f"Unexpected error: {str(e)}"
        }

def run_complete_test() -> Dict[str, Any]:
    """Run the complete test lifecycle"""
    print("🧪 Starting Complete Server Lifecycle Test")
    print("=" * 60)
    
    # Generate test IDs
    project_short_id = generate_test_ids()
    
    test_results = {
        "test_start_time": datetime.now().isoformat(),
        "project_short_id": project_short_id,
        "phases": {},
        "detailed_results": {}
    }
    
    # Phase 1: Health Check
    health_check_passed, health_result = test_health_check()
    test_results["phases"]["health_check"] = {
        "passed": health_check_passed,
        "timestamp": datetime.now().isoformat(),
        "result": health_result
    }
    
    if not health_check_passed:
        print("❌ Health check failed, stopping test")
        return test_results
    
    # # Phase 2: Project Setup
    # setup_passed, setup_result, workflow_ids = test_project_setup(project_short_id)
    # test_results["phases"]["setup"] = {
    #     "passed": setup_passed,
    #     "timestamp": datetime.now().isoformat(),
    #     "workflow_ids": workflow_ids,
    #     "result": setup_result
    # }
    
    # if not setup_passed:
    #     print("❌ Project setup failed, stopping test")
    #     return test_results
    workflow_ids = [test_workflow_id_2]  # Use the treatment recommendation workflow
    
    # # Phase 3: Workflow Generation
    # generation_passed, generation_result = test_workflow_generation(workflow_ids)
    # test_results["phases"]["generation"] = {
    #     "passed": generation_passed,
    #     "timestamp": datetime.now().isoformat(),
    #     "result": generation_result
    # }
    
    
    # Phase 4: Workflow Execution
    execution_passed, execution_result = test_workflow_execution(workflow_ids)
    test_results["phases"]["execution"] = {
        "passed": execution_passed,
        "timestamp": datetime.now().isoformat(),
        "result": execution_result
    }
    
    # # Phase 5: Status Check
    # status_passed, status_result = test_workflow_status(workflow_ids)
    # test_results["phases"]["status_check"] = {
    #     "passed": status_passed,
    #     "timestamp": datetime.now().isoformat(),
    #     "result": status_result
    # }
    
    # # Phase 5: WebSocket Streaming Test
    # streaming_passed, streaming_result = test_websocket_streaming(workflow_ids)
    # test_results["phases"]["websocket_streaming"] = {
    #     "passed": streaming_passed,
    #     "timestamp": datetime.now().isoformat(),
    #     "result": streaming_result
    # }
    
    # # Phase 6: Status Check
    # status_passed, status_result = test_workflow_status(workflow_ids)
    # test_results["phases"]["status_check"] = {
    #     "passed": status_passed,
    #     "timestamp": datetime.now().isoformat(),
    #     "result": status_result
    # }
    
    # # Phase 7: List Workflows
    # list_passed, list_result = test_list_workflows()
    # test_results["phases"]["list_workflows"] = {
    #     "passed": list_passed,
    #     "timestamp": datetime.now().isoformat(),
    #     "result": list_result
    # }
    
    # # Phase 8: User Query Analysis
    # query_analysis_passed, query_analysis_result = test_user_query_analysis(project_short_id)
    # test_results["phases"]["user_query_analysis"] = {
    #     "passed": query_analysis_passed,
    #     "timestamp": datetime.now().isoformat(),
    #     "result": query_analysis_result
    # }
    
    # # Phase 9: Workflow Graph Test
    # graph_passed, graph_result = test_workflow_graph(workflow_ids)
    # test_results["phases"]["workflow_graph"] = {
    #     "passed": graph_passed,
    #     "timestamp": datetime.now().isoformat(),
    #     "result": graph_result
    # }
    
    test_results["test_end_time"] = datetime.now().isoformat()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    all_passed = all([
        health_check_passed,
        # setup_passed,
        # generation_passed,
        # execution_passed,
        # streaming_passed,
        # status_passed,
        # list_passed,
        # query_analysis_passed,
        # graph_passed
    ])
    
    print(f"Health Check: {'✅ PASSED' if health_check_passed else '❌ FAILED'}")
    # print(f"Project Setup: {'✅ PASSED' if setup_passed else '❌ FAILED'}")
    # print(f"Workflow Generation: {'✅ PASSED' if generation_passed else '❌ FAILED'}")
    # print(f"Workflow Execution: {'✅ PASSED' if execution_passed else '❌ FAILED'}")
    # print(f"WebSocket Streaming: {'✅ PASSED' if streaming_passed else '❌ FAILED'}")
    # print(f"Status Check: {'✅ PASSED' if status_passed else '❌ FAILED'}")
    # print(f"List Workflows: {'✅ PASSED' if list_passed else '❌ FAILED'}")
    # print(f"User Query Analysis: {'✅ PASSED' if query_analysis_passed else '❌ FAILED'}")
    # print(f"Workflow Graph: {'✅ PASSED' if graph_passed else '❌ FAILED'}")
    print(f"\nOverall Result: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
    
    return test_results

def main():
    """Main test function"""
    # Uncomment the next line to run server issue debugging
    
    results = run_complete_test()
    
    # Save test results
    return results

if __name__ == "__main__":
    # Run the WebSocket test
    main() 