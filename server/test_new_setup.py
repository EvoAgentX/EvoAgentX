#!/usr/bin/env python3
"""
Test the new setup process with detailed requirements.
"""

import os
import json
import requests
from dotenv import load_dotenv

# Load environment variables
server_dir = os.path.dirname(__file__)
env_file = os.path.join(server_dir, 'app.env')
load_dotenv(env_file, override=True)

# Get access token
ACCESS_TOKEN = os.getenv("EAX_ACCESS_TOKEN", "default_secret_token_change_me")

# Test Configuration
TEST_CONFIG = {
    "server_url": "http://localhost:8001",
    "workflow_id": "test-new-setup-001",
    "user_id": "test-user-123",
    "detailed_requirements": """
    # Pet Management System Requirements
    
    ## System Overview
    Create a comprehensive pet management system with AI-powered disease analysis and treatment recommendations.
    
    ## AI Workflows Required
    
    ### 1. Disease Analysis Workflow
    - **Purpose**: Analyze pet symptoms and medical history to identify potential diseases
    - **Inputs**: 
      - Pet symptoms (text)
      - Pet medical history (text)
      - Pet age and breed (text)
    - **Outputs**:
      - Disease analysis report (markdown)
      - Confidence score (number)
      - Recommended tests (list)
    
    ### 2. Treatment Recommendation Workflow
    - **Purpose**: Generate treatment recommendations based on disease analysis
    - **Inputs**:
      - Disease analysis results (text)
      - Pet age and weight (text)
      - Available medications (text)
    - **Outputs**:
      - Treatment plan (markdown)
      - Medication recommendations (list)
      - Follow-up schedule (text)
    
    ## Database Entities
    
    ### Pets
    - pet_id (string, required)
    - name (string, required)
    - species (string, required)
    - breed (string, required)
    - age (number, required)
    - weight (number, required)
    - owner_id (string, required)
    
    ### Medical Records
    - record_id (string, required)
    - pet_id (string, required)
    - symptoms (text, required)
    - diagnosis (text, required)
    - treatment (text, required)
    - date (date, required)
    
    ### Diseases
    - disease_id (string, required)
    - name (string, required)
    - symptoms (text, required)
    - treatments (text, required)
    """
}

# Common headers for all requests
HEADERS = {
    "Content-Type": "application/json",
    "eax-access-token": ACCESS_TOKEN
}

def test_new_setup_process():
    """
    Test the new setup process with detailed requirements.
    """
    print("\n🧪 TESTING NEW SETUP PROCESS")
    print("=" * 50)
    
    setup_request = {
        "detailed_requirements": TEST_CONFIG["detailed_requirements"]
    }
    
    try:
        print(f"🚀 Setting up workflow with detailed requirements...")
        print(f"   Using detailed requirements only")
        
        # Call setup API endpoint
        response = requests.post(
            f"{TEST_CONFIG['server_url']}/project/setup",
            json=setup_request,
            headers=HEADERS
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Setup completed successfully!")
            print(f"   Workflow ID: {result.get('workflow_id', 'Unknown')}")
            print(f"   User ID: {result.get('user_id', 'Unknown')}")
            print(f"   Total workflows: {result.get('total_workflows', 0)}")
            print(f"   Database information: {result.get('database_information') is not None}")
            print(f"   Message: {result.get('message', '')}")
            
            # Print workflow details
            workflows = result.get('workflows', [])
            for i, workflow in enumerate(workflows, 1):
                print(f"\n   Workflow {i}: {workflow.get('workflow_name', 'Unknown')}")
                print(f"     - ID: {workflow.get('workflow_id', 'Unknown')}")
                print(f"     - Inputs: {len(workflow.get('workflow_inputs', []))}")
                print(f"     - Outputs: {len(workflow.get('workflow_outputs', []))}")
            
            # Verify creation by checking status
            workflow_id = result.get('workflow_id', 'Unknown')
            status_response = requests.get(
                f"{TEST_CONFIG['server_url']}/workflow/{workflow_id}/status", 
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
            print(f"❌ Setup failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    test_new_setup_process() 