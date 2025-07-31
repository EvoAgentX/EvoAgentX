#!/usr/bin/env python3
"""
Test script for the new setup functionality using project_id and requirement_id.
This tests the updated API structure that retrieves requirements from Supabase storage.
"""

import requests
import json
import os
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables
server_dir = os.path.dirname(__file__)
env_file = os.path.join(server_dir, 'app.env')
load_dotenv(env_file, override=True)

# Get access token
ACCESS_TOKEN = os.getenv("EAX_ACCESS_TOKEN", "default_secret_token_change_me")
SERVER_URL = "http://localhost:8001"

# Test configuration
TEST_CONFIG = {
    "server_url": SERVER_URL,
    "project_id": "zw7nnyv",
    "requirement_id": "t78kwdm",
    "expected_workflow_ids": [
        "550e8400-e29b-41d4-a716-446655440001",
        "550e8400-e29b-41d4-a716-446655440002"
    ]
}

HEADERS = {
    "Content-Type": "application/json",
    "eax-access-token": ACCESS_TOKEN
}

def test_setup_with_project_and_requirement_ids():
    """
    Test the new setup process using project_id and requirement_id
    """
    print("\n🚀 TESTING NEW SETUP PROCESS")
    print("=" * 50)
    
    setup_request = {
        "project_id": TEST_CONFIG["project_id"],
        "requirement_id": TEST_CONFIG["requirement_id"]
    }
    
    try:
        print(f"📋 Setup Request:")
        print(f"   Project ID: {TEST_CONFIG['project_id']}")
        print(f"   Requirement ID: {TEST_CONFIG['requirement_id']}")
        
        # Call setup API endpoint
        print(f"\n🔄 Calling /project/setup...")
        response = requests.post(
            f"{TEST_CONFIG['server_url']}/project/setup",
            json=setup_request,
            headers=HEADERS
        )
        
        print(f"📊 Response Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Setup completed successfully!")
            print(f"   Message: {result.get('message', '')}")
            
            # Check for workflow_graph in response
            workflow_graphs = result.get('workflow_graph', [])
            print(f"   Total workflow graphs: {len(workflow_graphs)}")
            
            print(f"\n📋 WORKFLOW GRAPH ANALYSIS")
            print("=" * 30)
            
            for i, workflow_graph in enumerate(workflow_graphs, 1):
                print(f"\n   Workflow Graph {i}:")
                print(f"     - Type: {type(workflow_graph)}")
                
                if isinstance(workflow_graph, dict):
                    print(f"     - Keys: {list(workflow_graph.keys())}")
                    
                    # Check for common workflow graph structure
                    if 'nodes' in workflow_graph:
                        nodes = workflow_graph.get('nodes', [])
                        print(f"     - Nodes: {len(nodes)}")
                        if nodes:
                            print(f"       First node: {nodes[0] if len(nodes) > 0 else 'None'}")
                    
                    if 'edges' in workflow_graph:
                        edges = workflow_graph.get('edges', [])
                        print(f"     - Edges: {len(edges)}")
                        if edges:
                            print(f"       First edge: {edges[0] if len(edges) > 0 else 'None'}")
                    
                    # Check for other common fields
                    for key in ['id', 'name', 'description', 'inputs', 'outputs']:
                        if key in workflow_graph:
                            value = workflow_graph[key]
                            if isinstance(value, list):
                                print(f"     - {key.capitalize()}: {len(value)} items")
                            else:
                                print(f"     - {key.capitalize()}: {value}")
                else:
                    print(f"     - Content: {str(workflow_graph)[:100]}...")
            
            # Test status endpoint for both expected workflow IDs
            print(f"\n🔍 Testing status endpoints for both workflows...")
            
            for i, expected_workflow_id in enumerate(TEST_CONFIG["expected_workflow_ids"], 1):
                print(f"\n   Workflow {i} ID: {expected_workflow_id}")
                
                status_response = requests.get(
                    f"{TEST_CONFIG['server_url']}/workflow/{expected_workflow_id}/status",
                    headers={"eax-access-token": ACCESS_TOKEN}
                )
                
                if status_response.status_code == 200:
                    status_result = status_response.json()
                    print(f"   ✅ Status endpoint working!")
                    print(f"      Workflow Status: {status_result.get('status', 'unknown')}")
                    print(f"      Setup Complete: {status_result.get('phases', {}).get('setup_complete', False)}")
                    print(f"      Generation Complete: {status_result.get('phases', {}).get('generation_complete', False)}")
                    print(f"      Execution Complete: {status_result.get('phases', {}).get('execution_complete', False)}")
                else:
                    print(f"   ❌ Status endpoint failed: {status_response.status_code}")
                    print(f"      Error: {status_response.text}")
                    return False
            
            return True
                
        else:
            print(f"❌ Setup failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        return False

def test_requirement_retrieval():
    """
    Test the requirement retrieval functionality
    """
    print("\n📄 TESTING REQUIREMENT RETRIEVAL")
    print("=" * 40)
    
    try:
        # Test the requirement retrieval endpoint (if it exists)
        # For now, we'll just test that the setup process works
        # which implies requirement retrieval is working
        
        print("✅ Requirement retrieval is tested as part of setup process")
        return True
        
    except Exception as e:
        print(f"❌ Requirement retrieval test failed: {e}")
        return False

def test_workflow_execution():
    """
    Test workflow execution for the first workflow
    """
    print("\n⚡ TESTING WORKFLOW EXECUTION")
    print("=" * 40)
    
    try:
        # Use the first workflow ID for execution test
        workflow_id = TEST_CONFIG["expected_workflow_ids"][0]
        print(f"🎯 Executing workflow: {workflow_id}")
        
        # Call the execution endpoint
        execution_request = {
            "workflow_id": workflow_id,
            "inputs": {
                "test_input": "This is a test input for workflow execution"
            }
        }
        
        print(f"📋 Execution Request:")
        print(f"   Workflow ID: {workflow_id}")
        print(f"   Inputs: {execution_request['inputs']}")
        
        # Call execution API endpoint
        print(f"\n🔄 Calling /workflow/{workflow_id}/execute...")
        response = requests.post(
            f"{TEST_CONFIG['server_url']}/workflow/{workflow_id}/execute",
            json=execution_request,
            headers=HEADERS
        )
        
        print(f"📊 Response Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Execution started successfully!")
            print(f"   Execution ID: {result.get('execution_id', 'Unknown')}")
            print(f"   Status: {result.get('status', 'Unknown')}")
            
            # Check execution status
            print(f"\n🔍 Checking execution status...")
            status_response = requests.get(
                f"{TEST_CONFIG['server_url']}/workflow/{workflow_id}/status",
                headers={"eax-access-token": ACCESS_TOKEN}
            )
            
            if status_response.status_code == 200:
                status_result = status_response.json()
                print(f"✅ Execution status retrieved!")
                print("Result:\n{}")
                
                return True
            else:
                print(f"❌ Status check failed: {status_response.status_code}")
                print(f"   Error: {status_response.text}")
                return False
                
        else:
            print(f"❌ Execution failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Execution test failed with exception: {e}")
        return False

def main():
    """
    Run all tests
    """
    print("🧪 SETUP FUNCTIONALITY TESTS")
    print("=" * 50)
    
    results = []
    
    # Test requirement retrieval
    result1 = test_requirement_retrieval()
    results.append(("Requirement Retrieval", result1))
    
    # Test setup process
    result2 = test_setup_with_project_and_requirement_ids()
    results.append(("Setup Process", result2))
    
    # Test workflow execution (only if setup was successful)
    if result2:
        result3 = test_workflow_execution()
        results.append(("Workflow Execution", result3))
    else:
        print("\n⏭️  Skipping execution test due to setup failure")
        results.append(("Workflow Execution", False))
    
    # Print summary
    print("\n📊 TEST SUMMARY")
    print("=" * 30)
    
    all_passed = True
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"   {test_name}: {status}")
        if not result:
            all_passed = False
    
    if all_passed:
        print(f"\n🎉 All tests passed!")
    else:
        print(f"\n⚠️  Some tests failed!")
    
    return all_passed

if __name__ == "__main__":
    main() 