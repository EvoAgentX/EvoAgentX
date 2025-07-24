#!/usr/bin/env python3
"""
WebSocket Workflow Test - Real Workflow Testing via Server API

Tests the complete EvoAgentX workflow lifecycle through server endpoints,
using WebSocket for streaming execution updates:
1. Project Setup
2. Workflow Generation 
3. WebSocket Workflow Execution

Configure the parameters below and run to test against real database.
"""

import os
import json
import asyncio
import websockets
import requests
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
server_dir = os.path.dirname(__file__)
env_file = os.path.join(server_dir, 'app.env')
load_dotenv(env_file, override=True)

# Get access token
ACCESS_TOKEN = os.getenv("EAX_ACCESS_TOKEN", "default_secret_token_change_me")

# =============================================================================
# CONFIGURATION - Modify these values for your test
# =============================================================================

# Test Parameters (MODIFY THESE)
TEST_CONFIG = {
    # Server Configuration
    "server_url": "http://localhost:8001",  # Change if server runs on different port
    "ws_url": "ws://localhost:8001",  # WebSocket URL
    
    # Test Data - Configure your test values here
    "workflow_id": "042705af-ed39-4589-8a7f-00f13d5e6b03",  # This value goes into 'id' field in real DB
    "user_id": "417b4875-e095-46d9-a46d-802dfef99d74",
    "requirement_id": "04233f59-4670-452f-b823-c9d5560542bf",
    
    # Test Metadata
    "test_name": "EvoAgentX WebSocket Workflow Lifecycle Test via Server API",
    "test_description": "Testing project setup, workflow generation, and WebSocket execution phases through HTTP endpoints"
}

# Common headers for all requests
HEADERS = {
    "Content-Type": "application/json",
    "eax-access-token": ACCESS_TOKEN
}

# =============================================================================
# TEST FUNCTIONS
# =============================================================================

def test_phase_1_project_setup(config):
    """
    Phase 1: Project Setup
    Creates initial workflow record with task_info for setup phase via API.
    """
    print("\n📋 PHASE 1: Project Setup")
    print("=" * 40)
    
    setup_request = {
        "workflow_id": config["workflow_id"],
        "user_id": config["user_id"], 
        "requirement_id": config["requirement_id"]
    }
    
    try:
        print(f"🚀 Setting up workflow via API...")
        print(f"   Workflow ID: {config['workflow_id']}")
        print(f"   User ID: {config['user_id']}")
        print(f"   Requirement ID: {config['requirement_id']}")
        
        # Call setup API endpoint
        response = requests.post(
            f"{config['server_url']}/project/setup",
            json=setup_request,
            headers=HEADERS
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Created workflow record via API")
            print(f"   Response: Setup completed successfully")
            print(f"   Task info generated: {result.get('task_info') is not None}")
            
            # Verify creation by checking status
            status_response = requests.get(f"{config['server_url']}/workflow/{config['workflow_id']}/status", headers={"eax-access-token": ACCESS_TOKEN})
            if status_response.status_code == 200:
                workflow_status = status_response.json()
                print(f"✅ Verification: Found workflow in database")
                print(f"   Status: {workflow_status.get('status', 'unknown')}")
                print(f"   Setup Complete: {workflow_status.get('phases', {}).get('setup_complete', False)}")
                return True
            else:
                print("❌ Verification failed: Could not retrieve workflow status")
                return False
        else:
            print(f"❌ Setup failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Phase 1 failed: {e}")
        return False

def test_phase_2_workflow_generation(config):
    """
    Phase 2: Workflow Generation
    Generates workflow graph via API based on task_info from setup phase.
    """
    print("\n🏗️ PHASE 2: Workflow Generation")
    print("=" * 40)
    
    generation_request = {
        "workflow_id": config["workflow_id"]
    }
    
    try:
        print(f"🚀 Generating workflow via API...")
        print(f"   Workflow ID: {config['workflow_id']}")
        print(f"   Using task_info from setup phase")
        
        # Call generation API endpoint
        response = requests.post(
            f"{config['server_url']}/workflow/generate",
            json=generation_request,
            headers=HEADERS
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Workflow generated via API")
            print(f"   Status: {result.get('status', 'unknown')}")
            print(f"   Workflow graph available: {result.get('workflow_graph') is not None}")
            
            # Verify generation by checking status
            status_response = requests.get(f"{config['server_url']}/workflow/{config['workflow_id']}/status", headers={"eax-access-token": ACCESS_TOKEN})
            if status_response.status_code == 200:
                workflow_status = status_response.json()
                print(f"✅ Verification: Workflow generation confirmed")
                print(f"   Generation Complete: {workflow_status.get('phases', {}).get('generation_complete', False)}")
                
                # Check if workflow_graph exists
                if workflow_status.get('workflow_graph'):
                    graph = workflow_status['workflow_graph']
                    nodes_count = len(graph.get('nodes', []))
                    edges_count = len(graph.get('edges', []))
                    print(f"   Graph nodes: {nodes_count}")
                    print(f"   Graph edges: {edges_count}")
                    print(f"   Response: {result}")
                
                return True
            else:
                print("❌ Verification failed: Could not retrieve workflow status")
                return False
        else:
            print(f"❌ Generation failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Phase 2 failed: {e}")
        return False

async def test_phase_3_websocket_workflow_execution(config):
    """
    Phase 3: WebSocket Workflow Execution
    Executes workflow via WebSocket API with test inputs.
    """
    print("\n⚡ PHASE 3: WebSocket Workflow Execution")
    print("=" * 40)
    
    execution_request = {
        "workflow_id": config["workflow_id"],
        "inputs": {
            "goal": "Analyze data and provide insights",
        }
    }
    
    try:
        print(f"🚀 Starting WebSocket workflow execution via API...")
        print(f"   Workflow ID: {config['workflow_id']}")
        print(f"   Input keys: {list(execution_request['inputs'].keys())}")
        print(f"   Using workflow_graph from generation phase")
        
        # Step 1: Start the workflow execution
        response = requests.post(
            f"{config['server_url']}/workflow/execute_ws",
            json=execution_request,
            headers=HEADERS
        )
        
        if response.status_code != 200:
            print(f"❌ Failed to start WebSocket execution: {response.text}")
            return False
            
        result = response.json()
        print(f"✅ WebSocket workflow execution started!")
        print(f"   Task ID: {result['task_id']}")
        print(f"   WebSocket URL: {result['ws_url']}")
        print(f"   Message: {result['message']}")
        print("\n🔄 Connecting to WebSocket...\n")
        
        # Step 2: Connect to WebSocket and handle updates
        ws_url = f"{config['ws_url']}{result['ws_url']}"
        execution_completed = False
        execution_success = False
        last_update_time = datetime.now()
        timeout_seconds = 300  # 5 minute timeout
        
        try:
            async with websockets.connect(ws_url) as websocket:
                while True:
                    try:
                        # Set timeout for receiving messages
                        message = await asyncio.wait_for(
                            websocket.recv(),
                            timeout=30  # 30 second timeout for receiving messages
                        )
                        
                        # Update last activity time
                        last_update_time = datetime.now()
                        
                        # Parse the message
                        event_data = json.loads(message)
                        event_type = event_data.get("event")
                        data = event_data.get("data", {})
                        
                        if event_type == "update":
                            step = data.get("step", "?")
                            total_steps = data.get("total_steps", "?")
                            progress = data.get("progress", 0)
                            current_state = data.get("current_state", "Processing...")
                            message = data.get("message", "")
                            command_output = data.get("command_output", "")
                            
                            print(f"📊 Step {step}/{total_steps} ({progress}%): {current_state}")
                            if message:
                                print(f"   📝 {message}")
                            if command_output:
                                print(f"command_output.strip()")
                            print()
                            
                        elif event_type == "complete":
                            print("🎉 Workflow execution completed!")
                            print(f"   Status: {data.get('status', 'unknown')}")
                            print(f"   Workflow ID: {data.get('workflow_id', 'unknown')}")
                            
                            execution_result = data.get("execution_result")
                            if execution_result:
                                if isinstance(execution_result, dict):
                                    print(f"   Execution Result: {json.dumps(execution_result, indent=2)}")
                                else:
                                    print(f"   Execution Result: {execution_result}")
                            
                            execution_completed = True
                            execution_success = True
                            break
                            
                        elif event_type == "error":
                            print("❌ Workflow execution failed!")
                            print(f"   Error: {data.get('error', 'Unknown error')}")
                            print(f"   Workflow ID: {data.get('workflow_id', 'unknown')}")
                            
                            execution_completed = True
                            execution_success = False
                            break
                            
                    except asyncio.TimeoutError:
                        # Check overall timeout
                        if (datetime.now() - last_update_time).seconds > timeout_seconds:
                            print("❌ WebSocket timeout: No updates received for too long")
                            return False
                        continue
                        
        except Exception as e:
            print(f"❌ WebSocket error: {e}")
            return False
            
        if not execution_completed:
            print("❌ WebSocket connection ended without completion or error event")
            return False
            
        # Verify final execution state
        status_response = requests.get(f"{config['server_url']}/workflow/{config['workflow_id']}/status", headers={"eax-access-token": ACCESS_TOKEN})
        if status_response.status_code == 200:
            workflow_status = status_response.json()
            print(f"\n✅ Verification: Final workflow state")
            print(f"   Execution Complete: {workflow_status.get('phases', {}).get('execution_complete', False)}")
            print(f"   Final Status: {workflow_status.get('status', 'unknown')}")
            
            return execution_success and workflow_status.get('phases', {}).get('execution_complete', False)
        else:
            print("\n❌ Verification failed: Could not retrieve final workflow status")
            return False
            
    except Exception as e:
        print(f"❌ Phase 3 failed: {e}")
        return False

def display_final_workflow_state(config):
    """Display the final state of the workflow after all phases"""
    print("\n📊 FINAL WORKFLOW STATE")
    print("=" * 40)
    
    try:
        response = requests.get(f"{config['server_url']}/workflow/{config['workflow_id']}/status", headers={"eax-access-token": ACCESS_TOKEN})
        if response.status_code == 200:
            workflow = response.json()
            
            print(f"Workflow ID: {workflow.get('workflow_id', 'unknown')}")
            print(f"Status: {workflow.get('status', 'unknown')}")
            print("\nPhase Completion:")
            phases = workflow.get('phases', {})
            print(f"✓ Setup: {phases.get('setup_complete', False)}")
            print(f"✓ Generation: {phases.get('generation_complete', False)}")
            print(f"✓ Execution: {phases.get('execution_complete', False)}")
            
            if workflow.get('error'):
                print(f"\n❌ Error: {workflow['error']}")
        else:
            print(f"❌ Could not retrieve final workflow state: {response.text}")
            
    except Exception as e:
        print(f"❌ Error displaying final state: {e}")

def test_server_health(config):
    """Test if the server is running and healthy"""
    print("\n🏥 Testing Server Health")
    print("=" * 40)
    
    try:
        response = requests.get(f"{config['server_url']}/health", headers={"eax-access-token": ACCESS_TOKEN}, timeout=5)
        if response.status_code == 200:
            print("✅ Server is healthy!")
            return True
        else:
            print(f"❌ Server health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Server health check failed: {e}")
        print(f"💡 Make sure the server is running at {config['server_url']}")
        return False

async def run_workflow_lifecycle_test():
    """
    Run the complete workflow lifecycle test using WebSocket execution
    """
    print("\n🌟 EvoAgentX WebSocket Workflow Lifecycle Test")
    print("=" * 60)
    print(f"Test Name: {TEST_CONFIG['test_name']}")
    print(f"Description: {TEST_CONFIG['test_description']}")
    print(f"Server URL: {TEST_CONFIG['server_url']}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Test server health first
    if not test_server_health(TEST_CONFIG):
        return False
        
    # Run each phase
    success = True
    try:
        # Phase 1: Project Setup
        if not test_phase_1_project_setup(TEST_CONFIG):
            print("\n❌ Phase 1 (Setup) failed - stopping test")
            return False
            
        # Phase 2: Workflow Generation
        if not test_phase_2_workflow_generation(TEST_CONFIG):
            print("\n❌ Phase 2 (Generation) failed - stopping test")
            return False
            
        # Phase 3: WebSocket Workflow Execution
        if not await test_phase_3_websocket_workflow_execution(TEST_CONFIG):
            print("\n❌ Phase 3 (WebSocket Execution) failed")
            success = False
            
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        success = False
        
    # Display final state
    display_final_workflow_state(TEST_CONFIG)
    
    # Print final test result
    print("\n" + "=" * 60)
    print(f"Test completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if success:
        print("✅ All phases completed successfully!")
    else:
        print("❌ Test failed - see logs above for details")
    print("=" * 60)
    
    return success

async def main():
    """Main entry point"""
    try:
        success = await run_workflow_lifecycle_test()
        if not success:
            exit(1)
    except KeyboardInterrupt:
        print("\n👋 Test interrupted by user")
        exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        exit(1)

if __name__ == "__main__":
    asyncio.run(main()) 