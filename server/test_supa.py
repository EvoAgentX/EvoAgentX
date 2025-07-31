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
    "workflow_id": "550e8400-e29b-41d4-a716-446655440001",  # Proper UUID format for traditional workflow testing
    
    # New Setup Test Configuration - Using project_id and requirement_id
    "project_id": "zw7nnyv",  # Project identifier for requirement retrieval
    "requirement_id": "t78kwdm",  # Requirement identifier for requirement retrieval
    
    # Expected workflow IDs for verification
    "expected_workflow_ids": [
        "550e8400-e29b-41d4-a716-446655440001",  # Disease Analysis Workflow
        "550e8400-e29b-41d4-a716-446655440002"   # Treatment Recommendation Workflow
    ],
    
    # Test Metadata
    "test_name": "EvoAgentX Workflow Lifecycle Test via Server API",
    "test_description": "Testing new setup process and workflow lifecycle through HTTP endpoints"
}

# Common headers for all requests
HEADERS = {
    "Content-Type": "application/json",
    "eax-access-token": ACCESS_TOKEN
}

# =============================================================================
# TEST FUNCTIONS
# =============================================================================

def test_new_setup_process(config):
    """
    Test the new setup process with project_id and requirement_id.
    This uses the new setup approach that retrieves requirements from storage.
    """
    print("\n🧪 NEW SETUP PROCESS TEST")
    print("=" * 40)
    
    setup_request = {
        "project_id": config["project_id"],
        "requirement_id": config["requirement_id"]
    }
    
    try:
        print(f"🚀 Setting up workflow with project_id and requirement_id via API...")
        print(f"   Project ID: {config['project_id']}")
        print(f"   Requirement ID: {config['requirement_id']}")
        
        # Call setup API endpoint
        response = requests.post(
            f"{config['server_url']}/project/setup",
            json=setup_request,
            headers=HEADERS
        )
        
        print(response.text)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ New setup completed successfully via API!")
            print(f"   Message: {result.get('message', '')}")
            
            # Print workflow details and verify IDs
            workflows = result.get('workflows', [])
            print(f"   Total workflows: {len(workflows)}")
            
            expected_ids = config.get('expected_workflow_ids', [])
            
            print(f"\n📋 WORKFLOW ID VERIFICATION")
            print("=" * 30)
            
            for i, workflow in enumerate(workflows, 1):
                workflow_name = workflow.get('workflow_name', f'Workflow {i}')
                workflow_id = workflow.get('workflow_id', 'Unknown')
                project_id = workflow.get('project_id', 'Unknown')
                requirement_id = workflow.get('requirement_id', 'Unknown')
                
                print(f"\n   Workflow {i}: {workflow_name}")
                print(f"     - Workflow ID: {workflow_id}")
                print(f"     - Project ID: {project_id}")
                print(f"     - Requirement ID: {requirement_id}")
                print(f"     - Inputs: {len(workflow.get('workflow_inputs', []))}")
                print(f"     - Outputs: {len(workflow.get('workflow_outputs', []))}")
                
                # Verify ID matches expected
                if i <= len(expected_ids):
                    expected_id = expected_ids[i-1]
                    if workflow_id == expected_id:
                        print(f"     ✅ ID MATCHES EXPECTED: {expected_id}")
                    else:
                        print(f"     ❌ ID MISMATCH - Expected: {expected_id}, Got: {workflow_id}")
                else:
                    print(f"     ⚠️  No expected ID for workflow {i}")
            
            # Verify creation by checking status of the first workflow
            if workflows:
                workflow_id = workflows[0].get('workflow_id', 'Unknown')
            status_response = requests.get(
                f"{config['server_url']}/workflow/{workflow_id}/status", 
                headers={"eax-access-token": ACCESS_TOKEN}
            )
            if status_response.status_code == 200:
                workflow_status = status_response.json()
                print(f"\n✅ Verification: Found workflow in database")
                print(f"   Status: {workflow_status.get('status', 'unknown')}")
                print(f"   Setup Complete: {workflow_status.get('phases', {}).get('setup_complete', False)}")
                return True
            else:
                print(f"❌ Verification failed: {status_response.text}")
                return False
        else:
            print(f"❌ New setup failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ New setup test failed: {e}")
        return False


def test_phase_3_workflow_execution(config):
    """
    Phase 3: Workflow Execution
    Executes the generated workflow with provided inputs via API.
    """
    print("\n⚡ PHASE 3: Workflow Execution")
    print("=" * 40)
    
    # For the new structure, use the first workflow ID and appropriate inputs for disease analysis
    execution_request = {
        "workflow_id": "550e8400-e29b-41d4-a716-446655440001",  # Use the first workflow ID
        "inputs": {
            "symptoms": "The pet is showing signs of lethargy, loss of appetite, and occasional vomiting",
            "pet_info": {
                "name": "Buddy",
                "species": "dog",
                "breed": "Golden Retriever",
                "age": "5",
                "weight": "65"
            }
        }
    }
    
    try:
        print(f"🚀 Executing workflow via API...")
        print(f"   Workflow ID: {execution_request['workflow_id']}")
        print(f"   Input symptoms: {execution_request['inputs']['symptoms']}")
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
            status_response = requests.get(f"{config['server_url']}/workflow/{execution_request['workflow_id']}/status", headers={"eax-access-token": ACCESS_TOKEN})
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
        # Get workflow status via API - use the first workflow ID from setup
        workflow_id = "550e8400-e29b-41d4-a716-446655440001"
        response = requests.get(f"{config['server_url']}/workflow/{workflow_id}/status", headers={"eax-access-token": ACCESS_TOKEN})
        
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
    print(f"   Test workflow preserved: 550e8400-e29b-41d4-a716-446655440001")
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
    
    try:
        # Test server health first
        if not test_server_health(TEST_CONFIG):
            print("❌ Server health check failed - stopping test")
            return
            
        # Test new setup process first
        print("\n🧪 TESTING NEW SETUP PROCESS")
        new_setup_success = test_new_setup_process(TEST_CONFIG)
        if not new_setup_success:
            print("\n❌ New setup process failed. Proceeding with traditional workflow test...")
        
        # Run Phase 3 (Execution) - Setup and Generation are handled by test_new_setup_process
        results = []
        
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
        print(f"🧪 New Setup Process: {'✅ PASSED' if new_setup_success else '❌ FAILED'}")
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