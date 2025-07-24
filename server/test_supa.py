#!/usr/bin/env python3
"""
Supabase Database Test - Real Workflow Testing via Server API

Tests the complete EvoAgentX workflow lifecycle through server endpoints:
1. Project Setup
2. Workflow Generation 
3. Workflow Execution

Configure the parameters below and run to test against real database.
"""

import os
import requests
import json
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
    
    # Test Data - Configure your test values here
    "workflow_id": "042705af-ed39-4589-8a7f-00f13d5e6b03",  # This value goes into 'id' field in real DB
    "user_id": "417b4875-e095-46d9-a46d-802dfef99d74",
    "requirement_id": "04233f59-4670-452f-b823-c9d5560542bf",
    
    # Test Metadata
    "test_name": "EvoAgentX Workflow Lifecycle Test via Server API",
    "test_description": "Testing project setup, workflow generation, and execution phases through HTTP endpoints"
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

def test_phase_3_workflow_execution(config):
    """
    Phase 3: Workflow Execution
    Executes the generated workflow with provided inputs via API.
    """
    print("\n⚡ PHASE 3: Workflow Execution")
    print("=" * 40)
    
    execution_request = {
        "workflow_id": config["workflow_id"],
        "inputs": {
            "goal": "Create a comprehensive stock analysis report for AAPL",
            "analysis_type": "technical_and_fundamental",
            "timeframe": "5_years"
        }
    }
    
    try:
        print(f"🚀 Executing workflow via API...")
        print(f"   Workflow ID: {config['workflow_id']}")
        print(f"   Input goal: {execution_request['inputs']['goal']}")
        print(f"   Using workflow_graph from generation phase")
        
        # Call execution API endpoint
        response = requests.post(
            f"{config['server_url']}/workflow/execute",
            json=execution_request,
            headers=HEADERS
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Workflow executed via API")
            print(f"   Execution result available: {result.get('execution_result') is not None}")
            
            # Show brief summary of execution result
            exec_result = result.get('execution_result')
            if exec_result:
                print(f"   Execution result type: {type(exec_result)}")
                if isinstance(exec_result, dict) and 'message' in exec_result:
                    print(f"   Result preview: {str(exec_result['message'])[:100]}...")
                elif isinstance(exec_result, str):
                    print(f"   Result preview: {exec_result[:100]}...")
            
            # Verify execution by checking status
            status_response = requests.get(f"{config['server_url']}/workflow/{config['workflow_id']}/status", headers={"eax-access-token": ACCESS_TOKEN})
            if status_response.status_code == 200:
                workflow_status = status_response.json()
                print(f"✅ Verification: Workflow execution confirmed")
                print(f"   Execution Complete: {workflow_status.get('phases', {}).get('execution_complete', False)}")
                return True
            else:
                print("❌ Verification failed: Could not retrieve workflow status")
                return False
        else:
            print(f"❌ Execution failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Phase 3 failed: {e}")
        return False

def display_final_workflow_state(config):
    """Display the complete final state of the workflow for verification via API."""
    print("\n📊 FINAL WORKFLOW STATE")
    print("=" * 40)
    
    try:
        # Get workflow status via API
        response = requests.get(f"{config['server_url']}/workflow/{config['workflow_id']}/status", headers={"eax-access-token": ACCESS_TOKEN})
        
        if response.status_code != 200:
            print("❌ Workflow not found")
            return
            
        workflow = response.json()
        
        print(f"🔄 Workflow ID: {workflow.get('workflow_id', config['workflow_id'])}")
        print(f"👤 User ID: {workflow.get('user_id', 'unknown')}")
        print(f"📝 Requirement ID: {workflow.get('requirement_id', 'unknown')}")
        print(f"📊 Status: {workflow.get('status', 'unknown')}")
        
        # Phases Summary
        phases = workflow.get('phases', {})
        print(f"\n📋 Workflow Phases:")
        print(f"   Setup Complete: {phases.get('setup_complete', False)}")
        print(f"   Generation Complete: {phases.get('generation_complete', False)}")
        print(f"   Execution Complete: {phases.get('execution_complete', False)}")
        
        # Task Info Summary
        if workflow.get('task_info'):
            print(f"\n📋 Task Info Available: ✅")
        else:
            print(f"\n📋 Task Info Available: ❌")
        
        # Workflow Graph Summary
        if workflow.get('workflow_graph'):
            graph = workflow['workflow_graph']
            print(f"\n🏗️ Workflow Graph:")
            print(f"   Nodes: {len(graph.get('nodes', []))}")
            print(f"   Edges: {len(graph.get('edges', []))}")
        else:
            print(f"\n🏗️ Workflow Graph: ❌ Not generated")
            
        # Execution Results Summary
        if workflow.get('execution_result'):
            print(f"\n⚡ Execution Results: ✅ Available")
            result = workflow['execution_result']
            if isinstance(result, dict) and 'message' in result:
                print(f"   Result preview: {str(result['message'])[:100]}...")
        else:
            print(f"\n⚡ Execution Results: ❌ Not executed")
            
        print(f"\n📅 Timestamps:")
        print(f"   Created: {workflow.get('created_at', 'unknown')}")
        print(f"   Updated: {workflow.get('updated_at', 'unknown')}")
        
    except Exception as e:
        print(f"❌ Error displaying workflow state: {e}")

def cleanup_test_data(config):
    """Clean up test data (optional - not implemented for API test)."""
    print("\n🧹 CLEANUP (Optional)")
    print("=" * 25)
    
    print("ℹ️ Cleanup not implemented for API test")
    print(f"   Test workflow preserved: {config['workflow_id']}")
    print(f"   You can manually delete from Supabase dashboard if needed")

def test_server_health(config):
    """Test server health before running workflow tests."""
    print("\n🔍 TESTING SERVER HEALTH")
    print("=" * 30)
    
    try:
        response = requests.get(f"{config['server_url']}/health", headers=HEADERS)
        if response.status_code == 200:
            print("✅ Server is healthy and responding")
            return True
        else:
            print(f"❌ Server health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to server: {e}")
        print(f"   Make sure server is running on {config['server_url']}")
        return False

# =============================================================================
# MAIN TEST RUNNER
# =============================================================================

def run_workflow_lifecycle_test():
    """Main test function that runs all three phases via API."""
    
    print("🚀 SUPABASE WORKFLOW LIFECYCLE TEST VIA SERVER API")
    print("=" * 60)
    print(f"Test: {TEST_CONFIG['test_name']}")
    print(f"Description: {TEST_CONFIG['test_description']}")
    print(f"Server URL: {TEST_CONFIG['server_url']}")
    print(f"Workflow ID: {TEST_CONFIG['workflow_id']}")
    print(f"User ID: {TEST_CONFIG['user_id']}")
    print(f"Requirement ID: {TEST_CONFIG['requirement_id']}")
    
    try:
        # Test server health first
        if not test_server_health(TEST_CONFIG):
            print("❌ Server health check failed - stopping test")
            return
            
        # Run all three phases
        results = []
        
        # Phase 1: Project Setup
        result1 = test_phase_1_project_setup(TEST_CONFIG)
        results.append(("Phase 1 - Setup", result1))
        
        if not result1:
            print("❌ Phase 1 failed - stopping test")
            return
            
        # Phase 2: Workflow Generation
        result2 = test_phase_2_workflow_generation(TEST_CONFIG)
        results.append(("Phase 2 - Generation", result2))
        
        if not result2:
            print("❌ Phase 2 failed - stopping test")
            return
            
        # Phase 3: Workflow Execution
        result3 = test_phase_3_workflow_execution(TEST_CONFIG)
        results.append(("Phase 3 - Execution", result3))
        
        # Display final state
        display_final_workflow_state(TEST_CONFIG)
        
        # Optional cleanup
        cleanup_test_data(TEST_CONFIG)
        
        # Test Summary
        print(f"\n🎯 TEST SUMMARY")
        print("=" * 20)
        for phase_name, success in results:
            status = "✅ PASSED" if success else "❌ FAILED"
            print(f"{phase_name}: {status}")
            
        overall_success = all(result[1] for result in results)
        print(f"\nOverall Result: {'✅ ALL TESTS PASSED' if overall_success else '❌ SOME TESTS FAILED'}")
        
        print("\n🔌 Test completed - no database cleanup needed")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        print("\n💡 Make sure:")
        print("1. EvoAgentX server is running")
        print(f"2. Server is accessible at {TEST_CONFIG['server_url']}")
        print("3. Supabase is properly configured in server environment")
        print("4. The workflow table exists in your Supabase database")

def main():
    """Entry point - modify TEST_CONFIG above to customize your test."""
    
    print("🔧 Starting API-based workflow test...")
    print(f"⚙️  Server URL: {TEST_CONFIG['server_url']}")
    print(f"📊 Note: This test uses the server API, not direct database access")
    print(f"🔑 Ensure your server is configured with proper Supabase credentials")
    
    run_workflow_lifecycle_test()

if __name__ == "__main__":
    main()