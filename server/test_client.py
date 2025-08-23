#!/usr/bin/env python3
"""
Enhanced test client for EvoAgentX Socket Management Service.
Design: Test setup and execution phases separately with proper socket output capture.

Features:
- WebSocket connection for real-time log streaming
- Separate setup and execution testing phases
- Real-time log monitoring during both phases
- Proper phase separation for individual testing

Usage:
1. Run this script while the server is running
2. It will connect via WebSocket and test setup phase
3. After setup completion, it will test execution phase
4. All logs stream in real-time via WebSocket
5. Press Ctrl+C to stop

The client can test setup and execution phases independently.
"""

import asyncio
import json
import websockets
import threading
import requests
import time
from datetime import datetime
from typing import Optional, Dict, Any

project_short_id = "9mshbju"

# Test configuration
BASE_URL = "http://localhost:8001"
ACCESS_TOKEN = "default_secret_token_change_me"  # Change this to your actual token
HEADERS = {
    "eax-access-token": ACCESS_TOKEN,
    "Content-Type": "application/json"
}
test_workflow_id = "e2d8f077-7392-4dde-8694-6133a3c155e8"


# Test inputs for workflow execution
TEST_INPUTS = {
    "character_type": "hero",
    "basic_description": "A brave little mouse named Whiskers who loves to explore",
    "story_context": "A magical forest with ancient trees and hidden pathways"
}

class EvoAgentXTestClient:
    """Test client that can test setup and execution phases separately."""
    
    def __init__(self, server_url: str = "ws://localhost:8001", project_short_id: str = project_short_id):
        self.server_url = server_url
        self.project_short_id = project_short_id
        self.websocket = None
        self.message_count = 0
        self.start_time = datetime.now()
        self.setup_completed = False
        self.workflow_ids = []
        self.execution_completed = False
        
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
    
    async def send_setup_message(self):
        """Send setup message to trigger the setup phase."""
        message = {
            "type": "setup",
            "data": {
                "project_short_id": self.project_short_id
            }
        }
        
        print(f"\n📤 Sending SETUP message to server:")
        print(json.dumps(message, indent=2))
        
        await self.websocket.send(json.dumps(message))
        print("✅ Setup message sent! Starting setup phase...")
    
    def send_execution_request(self, workflow_id: str):
        """Send workflow execution request via HTTP API."""
        print(f"\n⚡ Sending EXECUTION request for workflow: {workflow_id}")
        print(f"   🔧 Test inputs: {list(TEST_INPUTS.keys())}")
        
        request_data = {"inputs": TEST_INPUTS}
        url = f"{BASE_URL}/workflow/{workflow_id}/execute"
        
        try:
            print(f"   📤 POST {url}")
            response = requests.post(url, headers=HEADERS, json=request_data, timeout=300)
            
            print(f"   📊 Status: {response.status_code}")
            if response.status_code == 200:
                result = response.json()
                print(f"   ✅ Success: {result.get('message', 'Execution started')}")
                return True
            else:
                print(f"   ❌ Error: {response.text}")
                return False
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return False
    
    async def process_message(self, message: str) -> bool:
        """Process incoming message and determine if setup is complete."""
        # Always print the raw message first
        print(f"\n📨 Raw message: {message}")
        
        try:
            data = json.loads(message)
            message_type = data.get("type")
            
            if message_type == "setup-complete":
                # Setup phase completed
                workflow_id = data.get("data", {}).get("workflow_id")
                if workflow_id:
                    self.workflow_ids.append(workflow_id)
                    print(f"🎯 Setup completed for workflow: {workflow_id}")
                self.setup_completed = True
                return True
            elif message_type == "setup-log":
                # Setup phase log message
                workflow_id = data.get("data", {}).get("workflow_id")
                content = data.get("data", {}).get("content", "")
                if "Setup start" in content:
                    print(f"🚀 Setup phase started...")
                elif "updates database status" in content:
                    print(f"💾 Database update: {content}")
                elif "generated" in content:
                    print(f"✅ Workflow generation: {content}")
                return False
            elif message_type == "runtime-log":
                # Execution phase log message
                workflow_id = data.get("data", {}).get("workflow_id")
                content = data.get("data", {}).get("content", "")
                print(f"⚡ Execution log for {workflow_id}: {content}")
                return False
            elif message_type == "setup-complete" and data.get("data", {}).get("result") is None:
                # Setup error
                error_content = data.get("data", {}).get("content", "")
                print(f"❌ Setup error: {error_content}")
                self.setup_completed = True
                return True
                
        except json.JSONDecodeError:
            print(f"⚠️  Message is not valid JSON")
            
        return False
    
    async def listen_for_messages(self):
        """Listen for messages and process them to track phase completion."""
        print("\n🎧 Listening for messages from server...")
        print("Press Ctrl+C to stop\n")
        
        try:
            async for message in self.websocket:
                self.message_count += 1
                elapsed_time = (datetime.now() - self.start_time).total_seconds()
                
                print(f"\n📨 Message #{self.message_count} (t={elapsed_time:.1f}s):")
                print(f"{'='*60}")
                
                print(message)
                
                print(f"{'='*60}")
                
        except websockets.exceptions.ConnectionClosed:
            print("\n🔌 Connection closed by server")
        except Exception as e:
            print(f"\n❌ Error listening for messages: {e}")

async def test_setup_only():
    """Test only the setup phase."""
    print("\n🧪 TESTING SETUP PHASE ONLY")
    print("=" * 50)
    
    client = EvoAgentXTestClient()
    
    try:
        await client.connect()
        await client.send_setup_message()
        
        # Listen for messages until setup completes or timeout
        timeout = 300  # 5 minutes timeout
        start_time = time.time()
        
        while not client.setup_completed and (time.time() - start_time) < timeout:
            try:
                # Process one message at a time
                message = await asyncio.wait_for(client.websocket.recv(), timeout=1.0)
                client.message_count += 1
                elapsed_time = (datetime.now() - client.start_time).total_seconds()
                
                print(f"\n📨 Message #{client.message_count} (t={elapsed_time:.1f}s):")
                print(f"{'='*60}")
                
                await client.process_message(message)
                print(f"{'='*60}")
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                print(f"Error processing message: {e}")
                break
        
        if client.setup_completed:
            print(f"\n✅ Setup phase completed successfully!")
            print(f"   📋 Workflow IDs: {client.workflow_ids}")
        else:
            print(f"\n⏰ Setup phase timed out after {timeout} seconds")
            
    except Exception as e:
        print(f"\n❌ Setup test error: {e}")
    finally:
        await client.disconnect()

async def test_execution_only(workflow_id: str):
    """Test only the execution phase."""
    print(f"\n🧪 TESTING EXECUTION PHASE ONLY for workflow: {workflow_id}")
    print("=" * 50)
    
    client = EvoAgentXTestClient()
    
    try:
        await client.connect()
        
        # Start listening for messages
        listen_task = asyncio.create_task(client.listen_for_messages())
        
        print(f"Listening for messages...")
        
        # Send execution request in background thread (non-blocking)
        loop = asyncio.get_event_loop()
        execution_task = loop.run_in_executor(None, client.send_execution_request, workflow_id)
        
        print(f"⚡ Execution request sent to background thread...")
        print(f"\n⏳ Execution starting in background. Listening for runtime logs...")
        print("Press Ctrl+C to stop\n")
        
        # Keep listening for execution logs (don't wait for execution to complete)
        await listen_task
            
    except KeyboardInterrupt:
        print("\n⏹️  Stopped by user")
    except Exception as e:
        print(f"\n❌ Execution test error: {e}")
    finally:
        await client.disconnect()

async def test_full_workflow():
    """Test the complete workflow: setup followed by execution."""
    print("\n🧪 TESTING FULL WORKFLOW: Setup + Execution")
    print("=" * 50)
    
    client = EvoAgentXTestClient()
    
    try:
        await client.connect()
        await client.send_setup_message()
        
        # Start listening for messages
        await client.listen_for_messages()
        
    except KeyboardInterrupt:
        print("\n⏹️  Stopped by user")
    except Exception as e:
        print(f"\n❌ Full workflow test error: {e}")
    finally:
        elapsed_time = (datetime.now() - client.start_time).total_seconds()
        print(f"\n📊 Summary:")
        print(f"   Messages received: {client.message_count}")
        print(f"   Total time: {elapsed_time:.1f} seconds")
        print(f"   Setup completed: {client.setup_completed}")
        print(f"   Execution completed: {client.execution_completed}")
        print(f"   Workflow IDs: {client.workflow_ids}")
        
        await client.disconnect()

async def main():
    """Main test function with menu selection."""
    print("EvoAgentX Socket Management Test Client")
    print("=======================================")
    print("Make sure the server is running on localhost:8001")
    print("\nSelect test mode:")
    print("1. Setup phase only")
    print("2. Execution phase only")
    print("3. Full workflow (setup + execution)")
    print("4. Exit")
    
    while True:
        try:
            choice = input("\nEnter your choice (1-4): ").strip()
            
            if choice == "1":
                await test_setup_only()
                break
            elif choice == "2":
                workflow_id = input(f"Enter workflow ID to test (or press Enter for default {test_workflow_id}): ").strip()
                if not workflow_id:
                    workflow_id = test_workflow_id
                await test_execution_only(workflow_id)
                break
            elif choice == "3":
                await test_full_workflow()
                break
            elif choice == "4":
                print("Exiting...")
                break
            else:
                print("Invalid choice. Please enter 1-4.")
                
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())