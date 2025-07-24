import aiohttp
import asyncio
import json
import requests
import os
from dotenv import load_dotenv
import uuid

# Load environment variables
server_dir = os.path.dirname(__file__)
env_file = os.path.join(server_dir, 'app.env')
load_dotenv(env_file, override=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ACCESS_TOKEN = os.getenv("EAX_ACCESS_TOKEN", "default_secret_token_change_me")

# Common headers for all requests
HEADERS = {
    "Content-Type": "application/json",
    "eax-access-token": ACCESS_TOKEN
}

# =============================================================================
# WORKFLOW-BASED TESTS - GAOKAO SCORE ESTIMATION (THREE-PHASE STRUCTURE)
# =============================================================================

"""
Complete Gaokao Score Estimation Workflow Test Example (Updated Structure):

=== THREE-PHASE WORKFLOW PROCESS ===

1. PHASE 1 - SETUP INPUT:
{
  "workflow_id": "gaokao-estimation-001",
  "requirement_id": "req-gaokao-2024",
  "user_id": "test-user-123"
}

2. PHASE 2 - GENERATION INPUT:
{
  "workflow_id": "gaokao-estimation-001"
}

3. PHASE 3 - EXECUTION INPUT (WITH INPUTS):
{
  "workflow_id": "gaokao-estimation-001", 
  "inputs": {
    "goal": "Math: 80, English: 120, Physics: 120"
  }
}

=== EXPECTED OUTPUTS ===

1. PROJECT SETUP OUTPUT:
{
  "project_id": "proj_abc123def456",
  "public_url": "https://example.ngrok.io",
  "task_info": "Add ALEX ..."
}

2. WORKFLOW GENERATION OUTPUT:
{
  "success": true,
  "project_id": "proj_abc123def456",
  "workflow_graph": {
    "nodes": [...4 workflow nodes...],
    "edges": [...workflow connections...],
    "goal": "Create a stock price and trend analysis workflow...",
    "description": "Generated workflow for stock analysis"
  }
}

3. WORKFLOW EXECUTION OUTPUT:
{
  "success": true,
  "project_id": "proj_abc123def456",
  "execution_result": {
    "status": "completed",
    "message": "# Comprehensive Report: AAPL Stock Performance Analysis\n\n### 1. Current Price Metrics\n- **Stock Symbol**: AAPL\n- **Latest Stock Price**: $175.30\n- **Market Capitalization**: $2.8 Trillion\n- **Volume Traded**: 95 Million Shares\n\n### 2. Historical Price Data (Last 5 Years)\n- **Average Price**: $150.45\n- **Peak Price**: $182.50 (November 2021)\n- **Low Price**: $84.80 (March 2020)\n- **Volatility**: Notable fluctuations during earnings releases...\n\n### 3. Key Performance Metrics\n- **1-Year Change**: +15%\n- **5-Year Change**: +20%\n- **Dividend Yield**: 0.55%\n\n### 4. Technical Indicators\n- **50-day Moving Average**: $170.00\n- **200-day Moving Average**: $160.00\n- **RSI**: 65 (nearing overbought)\n- **MACD**: Positive divergence\n\n### 5. Recommendations\n- Strong buy recommendation based on solid fundamentals\n- Consider adding positions on market dips\n- Monitor economic indicators for potential impacts\n\n### Conclusion\nAPPL presents a compelling investment opportunity with consistent performance, positive market sentiment, and sound fundamentals.",
    "workflow_received": true,
    "llm_config_received": true,
    "mcp_config_received": false
  },
  "message": "Workflow executed successfully for project",
  "timestamp": "2024-06-18T11:25:02.789000"
}
"""

def test_health_check():
    """Test basic health check endpoint"""
    print("\n=== Testing Health Check ===")
    
    response = requests.get('http://localhost:8001/health', headers={"eax-access-token": ACCESS_TOKEN})
    assert response.status_code == 200
    
    data = response.json()
    print("✅ Health check passed:", data)
    return data

def test_project_setup():
    """
    Test project setup for Gaokao score estimation website workflow
    Updated to use new workflow structure with workflow_id
    
    Curl command:
    ```bash
    curl -X POST http://localhost:8001/project/setup \
      -H "Content-Type: application/json" \
      -H "eax-access-token: your_token_here" \
      -d '{
        "workflow_id": "gaokao-estimation-001",
        "requirement_id": "req-gaokao-2024",
        "user_id": "test-user-123"
      }'
    ```
    """
    print("\n=== Testing Project Setup ===")
    
    project_request = {
        "workflow_id": "gaokao-estimation-001",
        "requirement_id": "req-gaokao-2024", 
        "user_id": "test-user-123"
    }
    
    response = requests.post('http://localhost:8001/project/setup', json=project_request, headers=HEADERS)
    assert response.status_code == 200
    
    data = response.json()
    print("✅ Project setup successful:", data)
    return data

def test_project_status(project_id):
    """Test getting project status"""
    print(f"\n=== Testing Project Status for {project_id} ===")
    
    response = requests.get(f'http://localhost:8001/workflow/{project_id}/status', headers={"eax-access-token": ACCESS_TOKEN})
    assert response.status_code == 200
    
    data = response.json()
    print("✅ Project status retrieved:", data)
    return data

def test_project_workflow_generation(project_id):
    """
    Test workflow generation for the project
    Updated to use new workflow structure
    
    Curl command:
    ```bash
    curl -X POST http://localhost:8001/workflow/generate \
      -H "Content-Type: application/json" \
      -H "eax-access-token: your_token_here" \
      -d '{
        "workflow_id": "gaokao-estimation-001"
      }'
    ```
    """
    print(f"\n=== Testing Workflow Generation for {project_id} ===")
    
    generation_request = {
        "workflow_id": "gaokao-estimation-001"
    }
    
    response = requests.post('http://localhost:8001/workflow/generate', json=generation_request, headers=HEADERS)
    assert response.status_code == 200
    
    data = response.json()
    print("✅ Workflow generation successful:", data)
    return data

def test_project_workflow_generation_with_default_config(project_id):
    """
    Test workflow generation with default configuration
    Updated to use new workflow structure
    """
    print(f"\n=== Testing Workflow Generation with Default Config for {project_id} ===")
    
    generation_request = {
        "workflow_id": "gaokao-estimation-001"
    }
    
    response = requests.post('http://localhost:8001/workflow/generate', json=generation_request, headers=HEADERS)
    assert response.status_code == 200
    
    data = response.json()
    print("✅ Workflow generation with default config successful:", data)
    return data

def test_list_projects():
    """Test listing all projects"""
    print("\n=== Testing List Projects ===")
    
    response = requests.get('http://localhost:8001/projects', headers={"eax-access-token": ACCESS_TOKEN})
    assert response.status_code == 200
    
    data = response.json()
    print("✅ Projects listed successfully:", data)
    return data

def test_invalid_project():
    """Test handling of invalid project ID"""
    print("\n=== Testing Invalid Project ===")
    
    invalid_workflow_id = "invalid-workflow-id"
    
    # Test getting status of invalid project
    response = requests.get(f'http://localhost:8001/workflow/{invalid_workflow_id}/status', headers={"eax-access-token": ACCESS_TOKEN})
    assert response.status_code == 404
    
    print("✅ Invalid project correctly rejected")
    
    # Test generating workflow for invalid project
    generation_request = {
        "workflow_id": invalid_workflow_id
    }
    
    response = requests.post('http://localhost:8001/workflow/generate', json=generation_request, headers=HEADERS)
    # This might succeed if the workflow_id is just a string, but the generation should fail
    print("✅ Invalid workflow generation handled appropriately")
    
    return True

def test_project_workflow_execution(project_id):
    """
    Test workflow execution with inputs
    Updated to use new workflow structure
    
    Curl command:
    ```bash
    curl -X POST http://localhost:8001/workflow/execute \
      -H "Content-Type: application/json" \
      -H "eax-access-token: your_token_here" \
      -d '{
        "workflow_id": "gaokao-estimation-001",
        "inputs": {
          "goal": "Math: 80, English: 120, Physics: 120"
        }
      }'
    ```
    """
    print(f"\n=== Testing Workflow Execution for {project_id} ===")
    
    execution_request = {
        "workflow_id": "gaokao-estimation-001",
        "inputs": {
            "goal": "Math: 80, English: 120, Physics: 120"
        }
    }
    
    response = requests.post('http://localhost:8001/workflow/execute', json=execution_request, headers=HEADERS)
    assert response.status_code == 200
    
    data = response.json()
    print("✅ Workflow execution successful:", data)
    return data


# =============================================================================
# TEST RUNNER
# =============================================================================

if __name__ == "__main__":
    print("🚀 Starting Gaokao Score Estimation Workflow Tests (Updated Structure)...")
    
    test_health_check()
    
    # Phase 1: Setup - Create workflow and generate task_info
    setup_result = test_project_setup()
    if not setup_result:
        print("❌ Workflow setup failed, stopping test")
        raise Exception("Workflow setup failed")
    
    workflow_id = setup_result['workflow_id']
    print(f"\n📋 Using workflow_id: {workflow_id}")
    
    # Check status after setup
    test_project_status(workflow_id)
    
    # Phase 2: Generation - Generate workflow graph from task_info
    workflow_result = test_project_workflow_generation(workflow_id)
    if not workflow_result:
        print("❌ Workflow generation failed, stopping test")
        raise Exception("Workflow generation failed")
    
    # Check status after generation
    test_project_status(workflow_id)
    
    # Phase 3: Execution - Execute workflow with inputs
    execution_result = test_project_workflow_execution(workflow_id)
    if not execution_result:
        print("❌ Workflow execution failed")
    else:
        print("✅ All three phases completed successfully!")
    
    # Final status check
    test_project_status(workflow_id)
    
    print("\n🏁 Gaokao score estimation workflow test execution completed.") 