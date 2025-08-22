#!/usr/bin/env python3
"""
Enhanced test client for EvoAgentX Socket Management Service.
Design: Send setup message, execute workflows in background thread, and keep receiving/printing messages via WebSocket.

Features:
- WebSocket connection for real-time log streaming
- Background workflow execution (same as test_server.py)
- Concurrent setup and execution testing
- Real-time log monitoring during execution

Usage:
1. Run this script while the server is running
2. It will connect via WebSocket and send setup message
3. After 3 seconds, it starts workflow execution in background thread
4. All execution logs stream in real-time via WebSocket
5. Press Ctrl+C to stop

The send_execution() method runs in a separate thread so WebSocket listening continues uninterrupted.
"""

import asyncio
import json
import websockets
import threading
import requests
import time
from datetime import datetime

project_short_id = "9mshbju"

# Test configuration (same as test_server.py)
BASE_URL = "http://localhost:8001"
ACCESS_TOKEN = "default_secret_token_change_me"  # Change this to your actual token
HEADERS = {
    "eax-access-token": ACCESS_TOKEN,
    "Content-Type": "application/json"
}
test_workflow_id = "1c184cc8-4495-471c-b3fe-0ffe0c7ee315"

# Test inputs for workflow execution
TEST_INPUTS = {
    "theme": "A beautiful sunset over a calm ocean, with a small boat in the foreground. The sky is painted with soft, warm colors, and the water reflects the sunset's glow. The boat is a simple wooden vessel with a sail, gently rocking on the water. The scene is peaceful and serene, with a sense of tranquility and calm.",
    "characters": ["A beautiful sunset over a calm ocean, with a small boat in the foreground. The sky is painted with soft, warm colors, and the water reflects the sunset's glow. The boat is a simple wooden vessel with a sail, gently rocking on the water. The scene is peaceful and serene, with a sense of tranquility and calm."],
    "age_group": "3-5",
    "story_length": "short",
    "moral_lesson": "A beautiful sunset over a calm ocean, with a small boat in the foreground. The sky is painted with soft, warm colors, and the water reflects the sunset's glow. The boat is a simple wooden vessel with a sail, gently rocking on the water. The scene is peaceful and serene, with a sense of tranquility and calm.",
    "setting": "A beautiful sunset over a calm ocean, with a small boat in the foreground. The sky is painted with soft, warm colors, and the water reflects the sunset's glow. The boat is a simple wooden vessel with a sail, gently rocking on the water. The scene is peaceful and serene, with a sense of tranquility and calm."
}

class SimpleEvoAgentXClient:
    """Simple test client that sends a message and keeps receiving messages."""
    
    def __init__(self, server_url: str = "ws://localhost:8001", project_short_id: str = project_short_id):
        self.server_url = server_url
        self.project_short_id = project_short_id
        self.websocket = None
        self.message_count = 0
        self.start_time = datetime.now()
        
    async def connect(self):
        """Connect to the socket server."""
        socket_url = f"{self.server_url}/project/{self.project_short_id}/regist"
        print(f"🔌 Connecting to {socket_url}")
        
        # Add timeout settings to prevent premature disconnections
        self.websocket = await websockets.connect(
            socket_url,
            ping_interval=30,      # Send ping every 30 seconds
            ping_timeout=100,       # Wait 10 seconds for pong response
            close_timeout=100,      # Wait 10 seconds when closing
            max_size=2**20,        # 1MB max message size
            max_queue=2**10        # 1024 message queue size
        )
        print("✅ Connected successfully!")
        
    async def disconnect(self):
        """Disconnect from the socket server."""
        if self.websocket:
            await self.websocket.close()
            print("🔌 Disconnected")
    
    async def send_message(self):
        """Send a simple message to the server."""
        message = {
            "type": "setup",
            "data": {
                "project_short_id": self.project_short_id
            }
        }
        
        print(f"\n📤 Sending message to server:")
        print(json.dumps(message, indent=2))
        
        await self.websocket.send(json.dumps(message))
        print("✅ Message sent!")
    
    def send_execution(self, workflow_id: list):
        """Send workflow execution requests in a separate thread."""
        def _execute_workflows():
            print(f"\n⚡ Starting workflow execution in background thread...")
            print(f"   📋 Workflow IDs: {workflow_id}")
            print(f"   🔧 Test inputs: {json.dumps(TEST_INPUTS, indent=2)}")
            
            print(f"\n   🚀 Executing workflow: {workflow_id}")
            
            request_data = {"inputs": TEST_INPUTS}
            url = f"{BASE_URL}/workflow/{workflow_id}/execute"
            
            try:
                print(f"   📤 POST {url}")
                response = requests.post(url, headers=HEADERS, json=request_data, timeout=30)
                
                print(f"   📊 Status: {response.status_code}")
                if response.status_code == 200:
                    result = response.json()
                    print(f"   ✅ Success: {result.get('message', 'Execution started')}")
                else:
                    print(f"   ❌ Error: {response.text}")
            except Exception as e:
                print(f"   ❌ Error: {e}")
                        
            print(f"\n   🎯 All workflow executions completed!")
        
        # Start execution in background thread
        execution_thread = threading.Thread(target=_execute_workflows, daemon=True)
        execution_thread.start()
        print(f"🔄 Workflow execution started in background thread")
        return execution_thread
    
    async def listen_for_messages(self):
        """Keep listening for and printing messages from the server."""
        print("\n🎧 Listening for messages from server...")
        print("Press Ctrl+C to stop\n")
        
        try:
            async for message in self.websocket:
                self.message_count += 1
                elapsed_time = (datetime.now() - self.start_time).total_seconds()
                
                print(f"\n📨 Message #{self.message_count} (t={elapsed_time:.1f}s):")
                print(f"{'='*60}")
                
                try:
                    # Try to parse as JSON for pretty printing
                    data = json.loads(message)
                    print(json.dumps(data, indent=2))
                except json.JSONDecodeError:
                    # If not JSON, print as raw text
                    print(message)
                
                print(f"{'='*60}")
                
        except websockets.exceptions.ConnectionClosed:
            print("\n🔌 Connection closed by server")
        except Exception as e:
            print(f"\n❌ Error listening for messages: {e}")

async def main():
    """Main test function."""
    client = SimpleEvoAgentXClient()
    
    try:
        # Connect to server
        await client.connect()
        

        
        # # Send setup message
        # await client.send_message()
        
        # Keep listening for messages (execution logs will come through WebSocket)
        asyncio.create_task(client.listen_for_messages())
        
        # Wait a bit for setup to complete
        print("\n⏳ Waiting 3 seconds for setup to complete...")
        await asyncio.sleep(150)
        
        # Start workflow execution in background thread
        # Use example workflow IDs (you can replace with actual ones from setup response)
        
        print(f"\n🎯 Starting workflow execution with IDs: {test_workflow_id}")
        print(f"   💡 Tip: Replace these with actual workflow IDs from your setup response")
        print(f"   🔧 Using test inputs: {list(TEST_INPUTS.keys())}")
        execution_thread = client.send_execution(test_workflow_id)
        
        
        
    except KeyboardInterrupt:
        print("\n⏹️  Stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
    finally:
        elapsed_time = (datetime.now() - client.start_time).total_seconds()
        print(f"\n📊 Summary:")
        print(f"   Messages received: {client.message_count}")
        print(f"   Total time: {elapsed_time:.1f} seconds")
        
        await client.disconnect()

if __name__ == "__main__":
    print("EvoAgentX Enhanced Socket Client Test")
    print("=====================================")
    print("Make sure the server is running on localhost:8001")
    print("This client will:")
    print("  1. Send setup message")
    print("  2. Execute workflows in background thread")
    print("  3. Stream real-time logs via WebSocket")
    print("Press Ctrl+C to stop\n")
    
    asyncio.run(main())