import aiohttp
import asyncio
import json
from sseclient import SSEClient
import requests
import os
from dotenv import load_dotenv
from evoagentx.models.model_configs import LLMConfig, OpenAILLMConfig

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# =============================================================================
# BASIC HTTP API TESTS
# =============================================================================

async def test_processing():
    """
    Test the basic synchronous processing endpoint.
    
    This demonstrates:
    - Simple HTTP request/response pattern
    - No streaming, just direct result
    - Good for quick operations that don't need progress updates
    """
    async with aiohttp.ClientSession() as session:
        # Test health check
        async with session.get('http://localhost:8001/health') as response:
            assert response.status == 200
            data = await response.json()
            print("Health check response:", data)

        # Test processing endpoint
        test_config = {
            "parameters": {
                "sample_param": "test_value",
                "another_param": 42
            },
            "timeout": 10
        }

        async with session.post(
            'http://localhost:8001/process',
            json=test_config
        ) as response:
            assert response.status == 200
            result = await response.json()
            print("\nProcessing response:", json.dumps(result, indent=2))

            # Verify the response structure
            assert "task_id" in result
            assert "status" in result
            assert "result" in result

# =============================================================================
# TASK-BASED STREAMING TESTS
# =============================================================================

def test_streaming():
    """
    Test the task-based streaming pattern.
    
    This demonstrates:
    - Starting a task and getting a task_id
    - Connecting to a task-specific SSE stream
    - Receiving real-time progress updates
    - Good for long operations where you want progress feedback
    - Each task gets its own stream URL
    
    Flow: POST /stream/process → get task_id → connect to /stream/{task_id}
    """
    # Start a streaming process
    test_config = {
        "parameters": {
            "stream_param": "test_stream",
            "iterations": 5
        },
        "timeout": 30
    }
    
    # Start the streaming process
    response = requests.post(
        'http://localhost:8001/stream/process',
        json=test_config
    )
    assert response.status_code == 200
    process_data = response.json()
    print("\nStream process started:", json.dumps(process_data, indent=2))
    
    # Connect to the SSE stream
    response = requests.get(f'http://localhost:8001{process_data["stream_url"]}', stream=True)
    messages = SSEClient(response)
    
    print("\nStreaming updates:")
    for msg in messages.events():
        if msg.event == "update":
            data = json.loads(msg.data)
            print(f"Progress: {data['progress']}% - {data['current_state']}")
        elif msg.event == "complete":
            data = json.loads(msg.data)
            print("\nTask completed:", json.dumps(data, indent=2))
            break
        elif msg.event == "error":
            data = json.loads(msg.data)
            print("\nError:", data["error"])
            break

def test_workflow_generation():
    """
    Test task-based streaming for workflow generation.
    
    This demonstrates:
    - Using the streaming pattern specifically for workflow generation
    - How LLM configurations are passed
    - Receiving the generated workflow through SSE
    - Task-specific streaming (one task = one stream)
    
    Use this pattern when:
    - You want progress updates during generation
    - You might start the task and check back later
    - You're building user-facing apps that need real-time feedback
    """
    llm_config = {
        "model": "gpt-4o-mini",
        "openai_key": OPENAI_API_KEY,
        "stream": True,
        "output_response": True,
        "max_tokens": 16000
    }
    print(OpenAILLMConfig(**llm_config))
    
    
    # Test workflow generation
    workflow_config = {
        "goal": "Create a simple data processing workflow that reads a CSV file and generates a summary report",
        "llm_config": llm_config,
        "timeout": 180
    }
    
    # Start the workflow generation process
    response = requests.post(
        'http://localhost:8001/stream/workflow/generate',
        json=workflow_config
    )
    assert response.status_code == 200
    process_data = response.json()
    print("\nWorkflow generation started:", json.dumps(process_data, indent=2))
    
    # Connect to the SSE stream
    response = requests.get(f'http://localhost:8001{process_data["stream_url"]}', stream=True)
    messages = SSEClient(response)
    
    print("\nWorkflow generation updates:")
    for msg in messages.events():
        if msg.event == "update":
            data = json.loads(msg.data)
            print(f"Step {data['step']}/{data['total_steps']} - Progress: {data['progress']}% - {data['current_state']}")
        elif msg.event == "complete":
            data = json.loads(msg.data)
            print("\nWorkflow generation completed:", json.dumps(data, indent=2))
            break
        elif msg.event == "error":
            data = json.loads(msg.data)
            print("\nError:", data["error"])
            break

# =============================================================================
# CLIENT-SESSION STREAMING TESTS
# =============================================================================

def test_client_session():
    """
    Test client session creation - the foundation of persistent streaming.
    
    This demonstrates:
    - Creating a persistent client session
    - Getting a client_id and persistent stream URL
    - Setting up for multiple tasks on one connection
    
    Client sessions enable:
    - Running multiple tasks without reconnecting
    - Persistent SSE connections
    - Better resource management
    - Interactive application patterns
    """
    print("\n=== Testing Client Session ===")
    
    # Step 1: Connect and get client session
    response = requests.post('http://localhost:8001/connect')
    assert response.status_code == 200
    client_data = response.json()
    print(f"Connected with client_id: {client_data['client_id']}")
    print(client_data)
    
    return client_data

def start_task_and_get_result(client_id, stream_url, task_config, timeout=180):
    """
    Helper function: Start a single task and wait for its specific result.
    
    This demonstrates:
    - How to extract results from a multi-task stream
    - Task ID matching to get specific results
    - Thread-based SSE listening with result capture
    - Building reusable workflow generation functions
    
    This pattern is perfect for:
    - Building libraries or SDKs
    - When you need the workflow result programmatically
    - Sequential workflow generation
    """
    import threading
    import time
    
    # Start the task
    response = requests.post(
        f'http://localhost:8001/client/{client_id}/workflow/generate',
        json=task_config
    )
    task_data = response.json()
    task_id = task_data['task_id']
    
    print(f"🚀 Started task {task_id[:8]} - waiting for result...")
    
    # Variable to store the result
    task_result = {"completed": False, "result": None, "error": None}
    
    def listen_for_task_result():
        response = requests.get(f'http://localhost:8001{stream_url}', stream=True)
        messages = SSEClient(response)
        
        for msg in messages.events():
            try:
                data = json.loads(msg.data)
                event_type = data.get("event_type", msg.event)
                msg_task_id = data.get("task_id")
                
                # Only process events for our specific task
                if msg_task_id == task_id:
                    if event_type == "task_completed":
                        print(f"✅ Task {task_id[:8]} completed!")
                        task_result["completed"] = True
                        task_result["result"] = data.get("result", {})
                        break
                    elif event_type == "task_error":
                        print(f"❌ Task {task_id[:8]} failed!")
                        task_result["completed"] = True  
                        task_result["error"] = data.get("error", "Unknown error")
                        break
                    elif event_type == "task_processing":
                        print(f"⚙️  Task {task_id[:8]} processing...")
                        
            except Exception as e:
                print(f"Error parsing message: {e}")
    
    # Start listener thread
    listener_thread = threading.Thread(target=listen_for_task_result)
    listener_thread.start()
    
    # Wait for completion
    listener_thread.join(timeout=timeout)
    
    if task_result["completed"]:
        if task_result["error"]:
            print(f"❌ Task failed: {task_result['error']}")
            return None
        else:
            print(f"🎯 Got workflow result for task {task_id[:8]}!")
            return task_result["result"]
    else:
        print(f"⏰ Task {task_id[:8]} timed out!")
        return None

def test_single_task_with_result():
    """
    Test getting a single workflow result using client sessions.
    
    This demonstrates:
    - How to use client sessions for single workflow generation
    - Capturing and using the workflow result programmatically
    - The helper function pattern for clean code
    
    Use this pattern when:
    - You need the workflow result for further processing
    - Building scripts or automation tools
    - You want the simplicity of "start task, get result"
    """
    print("\n=== Testing Single Task with Result Capture ===")
    
    # Connect
    client_data = test_client_session()
    client_id = client_data["client_id"]
    
    # Task configuration
    task_config = {
        "goal": "Create a simple email notification workflow",
        "llm_config": {
            "model": "gpt-4o-mini",
            "openai_key": OPENAI_API_KEY,
            "stream": True,
            "output_response": True,
            "max_tokens": 8000
        },
        "timeout": 120
    }
    
    # Get the workflow result
    workflow_result = start_task_and_get_result(
        client_id, 
        client_data["stream_url"], 
        task_config
    )
    
    if workflow_result:
        print(f"\n🎯 WORKFLOW RESULT:")
        print(f"   Goal: {workflow_result.get('goal', 'N/A')}")
        print(f"   Message: {workflow_result.get('message', 'N/A')}")
        
        # Now you can use the workflow result!
        workflow_graph = workflow_result.get("workflow_graph")
        if workflow_graph:
            print(f"   🔧 Workflow has {len(str(workflow_graph))} characters")
            # Do something with the workflow...
    
    return workflow_result

def test_client_workflow_generation():
    """
    Test basic client-session workflow generation with event monitoring.
    
    This demonstrates:
    - Client session workflow generation
    - Real-time event monitoring through SSE
    - How events flow through a persistent connection
    - Proper thread cleanup after task completion
    
    This is the foundation pattern that enables:
    - Interactive applications (like chat interfaces)
    - Real-time progress monitoring
    - Multi-task workflows on persistent connections
    """
    print("\n=== Testing Client-Session Workflow Generation ===")
    
    # Connect to get client session
    client_data = test_client_session()
    client_id = client_data["client_id"]
    
    # LLM configuration
    llm_config = {
        "model": "gpt-4o-mini",
        "openai_key": OPENAI_API_KEY,
        "stream": True,
        "output_response": True,
        "max_tokens": 16000
    }
    
    # Start SSE connection in a separate thread
    import threading
    
    def listen_to_events():
        response = requests.get(f'http://localhost:8001{client_data["stream_url"]}', stream=True)
        messages = SSEClient(response)
        
        print(f"\n🔄 Listening to client {client_id} events...")
        for msg in messages.events():
            try:
                data = json.loads(msg.data)
                event_type = data.get("event_type", msg.event)
                
                print(f"📡 Event: {event_type} - {data}")
                
                # Exit when task is completed or failed
                if event_type in ["task_completed", "task_error"]:
                    print(f"🎯 Task finished with event: {event_type}")
                    break
                    
            except Exception as e:
                print(f"Error parsing message: {e}")
                print(f"Raw message: {msg.data}")
    
    # Start SSE listener in background
    listener_thread = threading.Thread(target=listen_to_events)
    listener_thread.start()
    
    # Give SSE connection time to establish
    import time
    time.sleep(1)
    
    # Start workflow generation
    workflow_config = {
        "goal": "Create a workflow for processing customer feedback data and generating insights",
        "llm_config": llm_config,
        "timeout": 180
    }
    
    response = requests.post(
        f'http://localhost:8001/client/{client_id}/workflow/generate',
        json=workflow_config
    )
    
    print(response.json())
    assert response.status_code == 200
    task_data = response.json()
    print(f"Started workflow generation task: {task_data['task_id']}")
    
    # Wait for the listener thread to complete
    listener_thread.join(timeout=200)  # 200 second timeout
    
    return client_id

def test_multiple_tasks_single_client():
    """
    Test multiple concurrent workflow generations on a single client session.
    
    This demonstrates:
    - The power of persistent client sessions
    - Handling multiple concurrent tasks through one SSE connection
    - Task result capture and organization by task_id
    - Real-world multi-task workflow scenarios
    
    This pattern enables:
    - Batch workflow generation
    - Interactive workflow building sessions
    - Dashboard-like applications with multiple concurrent operations
    - Efficient resource usage (one connection, many tasks)
    
    Key concepts shown:
    - Task ID matching for result routing
    - Shared data structures for result capture
    - Proper thread synchronization
    - File-based result persistence
    """
    print("\n=== Testing Multiple Tasks on Single Client ===")
    
    # Connect
    client_data = test_client_session()
    client_id = client_data["client_id"]
    
    # LLM configuration
    llm_config = {
        "model": "gpt-4o-mini",
        "openai_key": OPENAI_API_KEY,
        "stream": True,
        "output_response": True,
        "max_tokens": 8000
    }
    
    # Start SSE listener
    import threading
    import time
    
    # Shared data structure to capture results
    task_results = {}  # task_id -> workflow result
    completed_tasks = []
    
    def listen_to_events():
        response = requests.get(f'http://localhost:8001{client_data["stream_url"]}', stream=True)
        messages = SSEClient(response)
        
        print(f"\n🔄 Listening to client {client_id} events...")
        for msg in messages.events():
            try:
                data = json.loads(msg.data)
                event_type = data.get("event_type", msg.event)
                task_id = data.get("task_id")
                
                if event_type == "task_started":
                    print(f"📋 Task {task_id[:8]} started: {data.get('goal', 'N/A')[:50]}...")
                elif event_type == "task_processing":
                    print(f"⚙️  Task {task_id[:8]} processing...")
                elif event_type == "task_completed":
                    print(f"✅ Task {task_id[:8]} completed!")
                    
                    # 🎯 CAPTURE THE WORKFLOW RESULT HERE!
                    task_results[task_id] = data.get("result", {})
                    completed_tasks.append(task_id)
                    
                    # Print the workflow result
                    workflow = data.get("result", {}).get("workflow_graph", "No workflow")
                    goal = data.get("result", {}).get("goal", "No goal")
                    print(f"   📊 Task {task_id[:8]} Result:")
                    print(f"      Goal: {goal}")
                    print(f"      Workflow: {str(workflow)[:100]}...")
                    
                    if len(completed_tasks) >= 2:  # Wait for both tasks
                        break
                elif event_type == "task_error":
                    print(f"❌ Task {task_id[:8]} error: {data.get('error', 'Unknown')}")
                    completed_tasks.append(task_id)  # Count errors as completed for loop exit
                    if len(completed_tasks) >= 2:
                        break
            except Exception as e:
                print(f"Error parsing message: {e}")
    
    # Start listener
    listener_thread = threading.Thread(target=listen_to_events)
    listener_thread.start()
    time.sleep(1)  # Let SSE connect
    
    # Start first task
    task1_config = {
        "goal": "Create a data validation workflow for incoming user data",
        "llm_config": llm_config,
        "timeout": 120
    }
    
    response1 = requests.post(
        f'http://localhost:8001/client/{client_id}/workflow/generate',
        json=task1_config
    )
    
    task1_data = response1.json()
    task1_id = task1_data['task_id']
    print(f"Started task 1: {task1_id}")
    
    # Start second task (should be queued and processed)
    task2_config = {
        "goal": "Create a reporting workflow for monthly sales analysis",
        "llm_config": llm_config,
        "timeout": 120
    }
    
    response2 = requests.post(
        f'http://localhost:8001/client/{client_id}/workflow/generate',
        json=task2_config
    )
    task2_data = response2.json()
    task2_id = task2_data['task_id']
    print(f"Started task 2: {task2_id}")
    
    # Wait for completion
    listener_thread.join(timeout=300)  # 5 minute timeout
    
    print(f"✅ Completed {len(completed_tasks)} tasks")
    
    # 🎯 NOW YOU CAN ACCESS THE WORKFLOW RESULTS!
    print("\n🎯 CAPTURED WORKFLOW RESULTS:")
    
    if task1_id in task_results:
        print(f"📋 Task 1 Result:")
        print(f"   Goal: {task_results[task1_id].get('goal', 'N/A')}")
        print(f"   Workflow: {task_results[task1_id].get('workflow_graph', 'N/A')}")
        
        # Save Task 1 workflow to file
        with open(f"workflow_task1_{task1_id[:8]}.json", "w") as f:
            json.dump(task_results[task1_id], f, indent=2)
        print(f"   💾 Saved to workflow_task1_{task1_id[:8]}.json")
    
    if task2_id in task_results:
        print(f"📋 Task 2 Result:")
        print(f"   Goal: {task_results[task2_id].get('goal', 'N/A')}")
        print(f"   Workflow: {task_results[task2_id].get('workflow_graph', 'N/A')}")
        
        # Save Task 2 workflow to file
        with open(f"workflow_task2_{task2_id[:8]}.json", "w") as f:
            json.dump(task_results[task2_id], f, indent=2)
        print(f"   💾 Saved to workflow_task2_{task2_id[:8]}.json")
    
    # Return the results so calling code can use them
    return {
        "client_id": client_id,
        "task_results": task_results,
        "task1_id": task1_id,
        "task2_id": task2_id
    }

# =============================================================================
# UTILITY AND DEBUG TESTS
# =============================================================================

def test_list_clients():
    """
    Test the client listing endpoint for debugging and monitoring.
    
    This demonstrates:
    - Server-side session management
    - Debugging capabilities for active sessions
    - Session lifecycle tracking
    
    Useful for:
    - Development and debugging
    - Monitoring server load
    - Understanding session lifecycles
    """
    print("\n=== Testing Client Listing ===")
    
    response = requests.get('http://localhost:8001/clients')
    assert response.status_code == 200
    clients_data = response.json()
    
    print(f"Active clients: {clients_data['total']}")
    for client in clients_data['active_clients']:
        print(f"  - Client {client['client_id']}")
        print(f"    Created: {client['created_at']}")
        print(f"    Last Activity: {client['last_activity']}")
        print(f"    Active Tasks: {len(client['active_tasks'])}")

def test_stream_workflow_generation():
    """
    Test streaming workflow generation (task-based pattern).
    
    This demonstrates:
    - Traditional task-based streaming for workflow generation
    - How to handle streaming events and capture final results
    - Comparison with client-session approach
    
    Use this when:
    - You need a simple one-task-one-stream pattern
    - Building simple workflow generation tools
    - You don't need persistent connections
    """
    print("\n=== Testing One-Off Workflow Generation ===")
    
    llm_config = {
        "model": "gpt-4o-mini",
        "openai_key": OPENAI_API_KEY,
        "stream": True,
        "output_response": True,
        "max_tokens": 16000
    }
    
    workflow_config = {
        "goal": "Create a simple data processing workflow that reads a CSV file",
        "llm_config": llm_config,
        "timeout": 180
    }
    
    stream_response = requests.post('http://localhost:8001/stream/workflow/generate', json=workflow_config)
    print(stream_response.json())
    
    stream_url = stream_response.json()["stream_url"]
    
    response = requests.get(f'http://localhost:8001{stream_url}', stream=True)
    messages = SSEClient(response)
    
    for msg in messages.events():
        print(msg.data)
    
    return msg.data

def test_simple_workflow_generation():
    """
    Test the simple synchronous workflow generation endpoint.
    
    This demonstrates:
    - The simplest possible workflow generation pattern
    - Synchronous HTTP request/response (no streaming)
    - Direct workflow result in the response body
    - Perfect for scripts, CLIs, and simple integrations
    
    Use this pattern when:
    - You want the simplest possible integration
    - Building command-line tools or scripts
    - You don't need progress updates
    - One-off workflow generation is sufficient
    
    Benefits:
    - No SSE complexity
    - Standard HTTP patterns
    - Easy to integrate into existing codebases
    - Immediate results
    
    Trade-offs:
    - Blocks HTTP connection during generation
    - No progress updates
    - Risk of timeouts for very long workflows
    """
    print("\n=== Testing Simple Synchronous Workflow Generation ===")
    
    # LLM configuration
    llm_config = {
        "model": "gpt-4o-mini",
        "openai_key": OPENAI_API_KEY,
        "stream": True,
        "output_response": True,
        "max_tokens": 8000
    }
    
    # Workflow configuration
    workflow_config = {
        "goal": "Create a simple email notification workflow for new user registrations",
        "llm_config": llm_config,
        "timeout": 120
    }
    
    print(f"🚀 Requesting workflow generation...")
    print(f"   Goal: {workflow_config['goal']}")
    
    # Make the synchronous request (this will block until workflow is generated)
    response = requests.post('http://localhost:8001/workflow/generate', json=workflow_config)
    
    if response.status_code == 200:
        result = response.json()
        
        print(f"✅ Workflow generated successfully!")
        print(f"   Success: {result.get('success', False)}")
        print(f"   Goal: {result.get('goal', 'N/A')}")
        print(f"   Message: {result.get('message', 'N/A')}")
        print(f"   Timestamp: {result.get('timestamp', 'N/A')}")
        
        # The workflow is right here in the response!
        workflow_graph = result.get('workflow_graph')
        if workflow_graph:
            print(f"   📊 Workflow preview: {str(workflow_graph)[:100]}...")
        
        return result
    else:
        print(f"❌ Request failed: {response.status_code}")
        print(f"   Error: {response.text}")
        return None

# =============================================================================
# TEST RUNNER
# =============================================================================

if __name__ == "__main__":
    print("Running test client...")
    
    # Test new client-session functionality
    print("\n" + "="*50)
    print("TESTING CLIENT-SESSION FUNCTIONALITY")
    print("="*50)
    
    # Test single client session
    test_client_workflow_generation()
    
    # Test multiple tasks on single client
    test_multiple_tasks_single_client()
    
    # Test client listing
    test_list_clients()
    
    # Test simple synchronous workflow generation
    test_simple_workflow_generation()
    
    # Test streaming workflow generation
    test_stream_workflow_generation()
    
    print("\n✅ All tests completed!")
