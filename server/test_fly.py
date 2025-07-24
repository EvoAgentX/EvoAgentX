#!/usr/bin/env python3
"""
Fly.io Deployment Test - Real Workflow Testing via Fly.io Server API

Tests the complete EvoAgentX workflow lifecycle through fly.io server endpoints,
using WebSocket for streaming execution updates:
1. Project Setup
2. Workflow Generation 
3. WebSocket Workflow Execution

This test connects to the deployed fly.io server to verify the deployment is working.
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
print(f"ACCESS_TOKEN: {ACCESS_TOKEN}")

# =============================================================================
# CONFIGURATION - Modify these values for your test
# =============================================================================

# Test Parameters (MODIFY THESE)
TEST_CONFIG = {
    # Fly.io Server Configuration
    "server_url": "https://evoagentx-server.fly.dev",  # Fly.io server URL
    "ws_url": "wss://evoagentx-server.fly.dev",  # WebSocket URL (note: wss for HTTPS)
    
    # Test Data - Configure your test values here
    "workflow_id": "042705af-ed39-4589-8a7f-00f13d5e6b03",  # This value goes into 'id' field in real DB
    "user_id": "417b4875-e095-46d9-a46d-802dfef99d74",
    "requirement_id": "04233f59-4670-452f-b823-c9d5560542bf",
    
    # Test Metadata
    "test_name": "EvoAgentX Fly.io Deployment Test",
    "test_description": "Testing project setup, workflow generation, and WebSocket execution phases through fly.io server"
}

# Common headers for all requests
HEADERS = {
    "Content-Type": "application/json",
    "eax-access-token": ACCESS_TOKEN
}

# =============================================================================
# TEST FUNCTIONS
# =============================================================================

def test_server_health(config):
    """
    Test basic server connectivity and health endpoint.
    """
    print("\n🏥 SERVER HEALTH CHECK")
    print("=" * 40)
    
    try:
        print(f"🚀 Testing server connectivity...")
        print(f"   Server URL: {config['server_url']}")
        
        # Test health endpoint
        response = requests.get(f"{config['server_url']}/health", headers=HEADERS, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Server is healthy!")
            print(f"   Status: {result.get('status', 'unknown')}")
            print(f"   Response time: {response.elapsed.total_seconds():.2f}s")
            return True
        else:
            print(f"❌ Server health check failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print(f"❌ Connection failed: Could not connect to {config['server_url']}")
        print(f"   Make sure the fly.io app is running and accessible")
        return False
    except requests.exceptions.Timeout:
        print(f"❌ Timeout: Server took too long to respond")
        return False
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

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
        print(f"🚀 Setting up workflow via fly.io API...")
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
            print(f"✅ Created workflow record via fly.io API")
            print(f"   Response: Setup completed successfully")
            print(f"   Task info generated: {result.get('task_info') is not None}")
            
            # Verify creation by checking status
            status_response = requests.get(f"{config['server_url']}/workflow/{config['workflow_id']}/status", headers={"eax-access-token": ACCESS_TOKEN})
            if status_response.status_code == 200:
                workflow_status = status_response.json()
                print(f"✅ Verification: Found workflow in fly.io database")
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
        print(f"🚀 Generating workflow via fly.io API...")
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
            print(f"✅ Workflow generated via fly.io API")
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
        print(f"🚀 Starting WebSocket workflow execution via fly.io API...")
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
        print(f"✅ WebSocket workflow execution started on fly.io!")
        print(f"   Task ID: {result['task_id']}")
        print(f"   WebSocket URL: {result['ws_url']}")
        print(f"   Message: {result['message']}")
        print("\n🔄 Connecting to WebSocket...\n")
        
        # Step 2: Connect to WebSocket and handle updates
        ws_url = f"{config['ws_url']}{result['ws_url']}"
        execution_completed = False
        execution_success = False
        last_update_time = datetime.now()
        
        try:
            async with websockets.connect(ws_url) as websocket:
                print(f"✅ Connected to WebSocket: {ws_url}")
                print(f"   Waiting for execution updates...\n")
                
                # Listen for WebSocket messages
                while not execution_completed:
                    try:
                        message = await asyncio.wait_for(websocket.recv(), timeout=30.0)
                        data = json.loads(message)
                        
                        event_type = data.get("event", "update")
                        event_data = data.get("data", {})
                        
                        current_time = datetime.now()
                        time_diff = (current_time - last_update_time).total_seconds()
                        last_update_time = current_time
                        
                        print(f"📡 [{current_time.strftime('%H:%M:%S')}] {event_type.upper()}: {event_data}")
                        
                        if event_type == "complete":
                            execution_completed = True
                            execution_success = True
                            print(f"\n✅ Workflow execution completed successfully!")
                        elif event_type == "error":
                            execution_completed = True
                            execution_success = False
                            print(f"\n❌ Workflow execution failed!")
                            
                    except asyncio.TimeoutError:
                        print(f"⏰ Timeout waiting for WebSocket message")
                        break
                    except websockets.exceptions.ConnectionClosed:
                        print(f"🔌 WebSocket connection closed")
                        break
                        
        except Exception as e:
            print(f"❌ WebSocket connection failed: {e}")
            return False
            
        return execution_success
        
    except Exception as e:
        print(f"❌ Phase 3 failed: {e}")
        return False

def display_final_workflow_state(config):
    """
    Display the final state of the workflow after all phases.
    """
    print("\n📊 FINAL WORKFLOW STATE")
    print("=" * 40)
    
    try:
        response = requests.get(f"{config['server_url']}/workflow/{config['workflow_id']}/status", headers={"eax-access-token": ACCESS_TOKEN})
        if response.status_code == 200:
            workflow = response.json()
            print(f"✅ Final workflow state retrieved from fly.io")
            print(f"   Workflow ID: {workflow.get('workflow_id')}")
            print(f"   Status: {workflow.get('status')}")
            print(f"   User ID: {workflow.get('user_id')}")
            print(f"   Created: {workflow.get('created_at')}")
            print(f"   Updated: {workflow.get('updated_at')}")
            
            phases = workflow.get('phases', {})
            print(f"   Phases:")
            print(f"     - Setup Complete: {phases.get('setup_complete', False)}")
            print(f"     - Generation Complete: {phases.get('generation_complete', False)}")
            print(f"     - Execution Complete: {phases.get('execution_complete', False)}")
            
            if workflow.get('execution_result'):
                print(f"   Execution Result: Available")
            else:
                print(f"   Execution Result: Not available")
                
        else:
            print(f"❌ Could not retrieve final workflow state: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error retrieving final state: {e}")

async def run_workflow_lifecycle_test():
    """
    Run the complete workflow lifecycle test on fly.io.
    """
    print("🚀 FLY.IO DEPLOYMENT TEST")
    print("=" * 50)
    print(f"Test: {TEST_CONFIG['test_name']}")
    print(f"Description: {TEST_CONFIG['test_description']}")
    print(f"Server: {TEST_CONFIG['server_url']}")
    print(f"WebSocket: {TEST_CONFIG['ws_url']}")
    print(f"Workflow ID: {TEST_CONFIG['workflow_id']}")
    print("=" * 50)
    
    # Test server health first
    if not test_server_health(TEST_CONFIG):
        print("\n❌ Server health check failed. Cannot proceed with workflow tests.")
        return False
    
    # Run all phases
    phase1_success = test_phase_1_project_setup(TEST_CONFIG)
    if not phase1_success:
        print("\n❌ Phase 1 failed. Cannot proceed to Phase 2.")
        return False
    
    phase2_success = test_phase_2_workflow_generation(TEST_CONFIG)
    if not phase2_success:
        print("\n❌ Phase 2 failed. Cannot proceed to Phase 3.")
        return False
    
    phase3_success = await test_phase_3_websocket_workflow_execution(TEST_CONFIG)
    
    # Display final state
    display_final_workflow_state(TEST_CONFIG)
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 TEST SUMMARY")
    print("=" * 50)
    print(f"✅ Server Health: {'PASS' if True else 'FAIL'}")
    print(f"✅ Phase 1 (Setup): {'PASS' if phase1_success else 'FAIL'}")
    print(f"✅ Phase 2 (Generation): {'PASS' if phase2_success else 'FAIL'}")
    print(f"✅ Phase 3 (Execution): {'PASS' if phase3_success else 'FAIL'}")
    
    overall_success = phase1_success and phase2_success and phase3_success
    print(f"\n🎯 Overall Result: {'PASS' if overall_success else 'FAIL'}")
    
    if overall_success:
        print("🎉 Fly.io deployment is working correctly!")
    else:
        print("⚠️  Some issues detected with the fly.io deployment.")
    
    return overall_success

async def main():
    """
    Main function to run the fly.io deployment test.
    """
    try:
        success = await run_workflow_lifecycle_test()
        exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
        exit(1)
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        exit(1)

if __name__ == "__main__":
    asyncio.run(main()) 