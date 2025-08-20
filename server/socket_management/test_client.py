#!/usr/bin/env python3
"""
Test client for EvoAgentX Socket Management Service.
Demonstrates how to connect and interact with the socket endpoints.
"""

import asyncio
import json
import websockets
import uuid
from datetime import datetime

project_short_id = "9mshbju"

class EvoAgentXSocketClient:
    """Test client for EvoAgentX socket connections."""
    
    def __init__(self, server_url: str = "ws://localhost:8001", project_short_id: str = project_short_id):
        self.server_url = server_url
        self.project_short_id = project_short_id
        self.websocket = None
        self.message_count = 0
        self.start_time = datetime.now()
        
        # Event tracking for waiting on completion
        self.setup_complete = asyncio.Event()
        self.execution_complete = asyncio.Event()
        self.current_operation = None
        self.operation_results = {}
        
    async def connect(self):
        """Connect to the socket server."""
        socket_url = f"{self.server_url}/project/{self.project_short_id}/parallel-setup"
        print(f"Connecting to {socket_url}")
        
        # Connect with longer timeouts for long-running operations
        self.websocket = await websockets.connect(
            socket_url,
            ping_interval=30,    # Send ping every 30 seconds
            ping_timeout=20,     # Wait 20 seconds for pong
            close_timeout=60     # Wait 60 seconds before force closing
        )
        print("Connected successfully!")
        
        # Start listening for messages and store the task
        self.listen_task = asyncio.create_task(self.listen_for_messages())
        print("🎧 Started listening for messages...")
        
    async def disconnect(self):
        """Disconnect from the socket server."""
        if hasattr(self, 'listen_task') and self.listen_task:
            self.listen_task.cancel()
            try:
                await self.listen_task
            except asyncio.CancelledError:
                pass
        if self.websocket:
            await self.websocket.close()
            print("Disconnected")
    
    async def listen_for_messages(self):
        """Listen for incoming messages from the server."""
        print("🎧 Starting to listen for messages...")
        try:
            async for message in self.websocket:
                print(f"📨 RAW MESSAGE RECEIVED: {type(message)}")
                print(f"\n🔍 RAW MESSAGE RECEIVED:")
                print(f"{'='*60}")
                print(message)
                print(f"{'='*60}")
                
                try:
                    data = json.loads(message)
                    await self.handle_message(data, message)
                except json.JSONDecodeError as e:
                    print(f"❌ Failed to parse JSON: {e}")
                    print(f"Raw message: {message}")
        except websockets.exceptions.ConnectionClosed:
            print("\n🔌 Connection closed by server")
        except Exception as e:
            print(f"\n❌ Error listening for messages: {e}")
            import traceback
            traceback.print_exc()
    
    async def handle_message(self, message: dict, raw_message: str):
        """Handle incoming messages from the server."""
        self.message_count += 1
        elapsed_time = (datetime.now() - self.start_time).total_seconds()
        
        message_type = message.get("type")
        data = message.get("data", {})
        
        print(f"\n📨 PARSED MESSAGE #{self.message_count} (t={elapsed_time:.1f}s):")
        print(f"   Type: {message_type}")
        print(f"   Status: {data.get('status')}")
        print(f"   Content: {data.get('content')}")
        print(f"   Workflow ID: {data.get('workflow_id')}")
        
        if data.get("result"):
            result = data['result']
            if isinstance(result, (dict, list)):
                result_preview = json.dumps(result, indent=2)
                if len(result_preview) > 300:
                    result_preview = result_preview[:300] + "..."
                print(f"   Result: {result_preview}")
            else:
                print(f"   Result: {result}")
        
        # Check for completion signals
        if message_type == "setup-complete":
            print(f"🎉 SETUP COMPLETED! Received {len(data.get('result', []))} workflow graphs")
            self.operation_results["setup"] = data.get('result')
            self.setup_complete.set()
        elif message_type == "execution-complete":
            print(f"🎉 EXECUTION COMPLETED!")
            self.operation_results["execution"] = data.get('result')
            self.execution_complete.set()
        elif message_type == "complete":
            print(f"🎉 WORKFLOW EXECUTION COMPLETED!")
            self.operation_results["workflow_execution"] = data.get('result')
            self.execution_complete.set()
        elif message_type == "error":
            print(f"❌ ERROR RECEIVED: {data.get('content')}")
            # Set completion events even on error to avoid hanging
            if self.current_operation == "setup":
                self.setup_complete.set()
            elif self.current_operation == "execution":
                self.execution_complete.set()
        elif message_type == "setup-log":
            workflow_id = data.get('workflow_id')
            if workflow_id:
                print(f"🏗️ SETUP LOG with WORKFLOW ID: {workflow_id}")
                print(f"   Content: {data.get('content')}")
            else:
                print(f"🏗️ SETUP LOG (no workflow ID)")
        elif message_type == "runtime-log":
            workflow_id = data.get('workflow_id')
            if workflow_id:
                print(f"🔄 RUNTIME LOG with WORKFLOW ID: {workflow_id}")
                print(f"   Content: {data.get('content')}")
            else:
                print(f"🔄 RUNTIME LOG (no workflow ID)")
        elif message_type == "heartbeat":
            print(f"💓 HEARTBEAT received (keeping connection alive)")
            # Don't print full structure for heartbeats to reduce noise
            return
        # Print full message structure for debugging
        print(f"\n🔬 FULL MESSAGE STRUCTURE:")
        print(json.dumps(message, indent=2, default=str))
    
    async def send_command(self, command: str, parameters: dict = None):
        """Send a command to the server."""
        message = {
            "command": command,
            "parameters": parameters or {},
            "message_id": uuid.uuid4().hex,
            "timestamp": datetime.now().isoformat()
        }
        
        message_json = json.dumps(message, indent=2)
        print(f"\n📤 SENDING COMMAND: {command}")
        print(f"{'='*50}")
        print(f"Raw JSON being sent:")
        print(message_json)
        print(f"{'='*50}")
        
        await self.websocket.send(json.dumps(message))
    
    async def send_execution_request(self, workflow_id: str, inputs: dict):
        """Send a workflow execution request (direct format)."""
        message = {
            "workflow_id": workflow_id,
            "inputs": inputs,
            "message_id": uuid.uuid4().hex
        }
        
        message_json = json.dumps(message, indent=2)
        print(f"\n🚀 SENDING EXECUTION REQUEST: {workflow_id}")
        print(f"{'='*50}")
        print(f"Raw JSON being sent:")
        print(message_json)
        print(f"{'='*50}")
        
        await self.websocket.send(json.dumps(message))
    
    async def send_heartbeat(self):
        """Send a heartbeat message."""
        message = {
            "type": "heartbeat",
            "timestamp": datetime.now().isoformat()
        }
        
        message_json = json.dumps(message, indent=2)
        print(f"\n💓 SENDING HEARTBEAT")
        print(f"{'='*30}")
        print(f"Raw JSON being sent:")
        print(message_json)
        print(f"{'='*30}")
        
        await self.websocket.send(json.dumps(message))
    
    def is_connected(self) -> bool:
        """Check if the WebSocket is still connected."""
        if not self.websocket:
            return False
        
        # Handle different websockets library versions
        if hasattr(self.websocket, 'closed'):
            # Older versions of websockets library
            return not self.websocket.closed
        elif hasattr(self.websocket, 'state'):
            # Newer versions (14.0+) use state attribute
            from websockets.protocol import State
            return self.websocket.state == State.OPEN
        else:
            # Fallback: assume connected if websocket object exists
            return True
    
    async def send_project_setup_and_wait(self, timeout: float = 120.0):
        """Send project setup command and wait for completion."""
        self.current_operation = "setup"
        self.setup_complete.clear()
        
        print(f"\n🚀 Starting project setup (timeout: {timeout}s)")
        # Send setup message as per README specification
        setup_message = {
            "type": "setup",
            "data": {
                "project_short_id": self.project_short_id
            }
        }
        await self.websocket.send(json.dumps(setup_message))
        
        print("⏳ Waiting for setup completion...")
        try:
            # Check connection periodically while waiting
            async def wait_with_connection_check():
                while not self.setup_complete.is_set():
                    if not self.is_connected():
                        raise ConnectionError("WebSocket connection lost during setup")
                    try:
                        # Give the server much more time - setup can be very slow
                        await asyncio.wait_for(self.setup_complete.wait(), timeout=600.0)
                        break
                    except asyncio.TimeoutError:
                        print(f"⏳ Still waiting for setup... (connection: {'✅' if self.is_connected() else '❌'})")
                        continue
            
            await asyncio.wait_for(wait_with_connection_check(), timeout=timeout)
            print("✅ Setup completed successfully!")
            return self.operation_results.get("setup")
        except asyncio.TimeoutError:
            print(f"⏰ Setup timed out after {timeout} seconds")
            return None
        except ConnectionError as e:
            print(f"🔌 Connection error during setup: {e}")
            return None
    
    async def send_workflow_execution_and_wait(self, workflow_id: str, inputs: dict, timeout: float = 300.0):
        """Send workflow execution command and wait for completion."""
        self.current_operation = "execution"
        self.execution_complete.clear()
        
        print(f"\n🚀 Starting workflow execution (timeout: {timeout}s)")
        await self.send_command("workflow.execute", {
            "workflow_id": workflow_id,
            "inputs": inputs
        })
        
        print("⏳ Waiting for execution completion...")
        try:
            await asyncio.wait_for(self.execution_complete.wait(), timeout=timeout)
            print("✅ Execution completed successfully!")
            return self.operation_results.get("execution") or self.operation_results.get("workflow_execution")
        except asyncio.TimeoutError:
            print(f"⏰ Execution timed out after {timeout} seconds")
            return None

async def main():
    """Main test function with proper completion waiting."""
    client = EvoAgentXSocketClient()
    
    try:
        # Connect to server
        await client.connect()
        
        # Wait for connection message
        print("⏳ Waiting for initial connection...")
        await asyncio.sleep(2)
        
        # Test 1: Project Setup with proper waiting
        print("\n" + "="*60)
        print("=== Testing Project Setup (with completion waiting) ===")
        print("="*60)
        print("⏰ Note: Project setup can take 5-10 minutes for complex workflows")
        setup_result = await client.send_project_setup_and_wait(timeout=600.0)  # 10 minutes timeout
        
        if setup_result:
            print(f"📊 Setup Result: {len(setup_result)} workflow graphs generated")
            
            # Test 2: Try to execute first workflow if available
            if setup_result and len(setup_result) > 0:
                # Look for workflow_id in the first workflow graph
                first_workflow = setup_result[0]
                workflow_id = None
                
                # Try to extract workflow_id from the workflow graph
                if isinstance(first_workflow, dict):
                    # Check different possible locations for workflow_id
                    workflow_id = (first_workflow.get('workflow_id') or 
                                 first_workflow.get('id') or
                                 first_workflow.get('nodes', {}).get('workflow_id'))
                
                if workflow_id:
                    print(f"\n" + "="*60)
                    print(f"=== Testing Workflow Execution: {workflow_id} ===")
                    print("="*60)
                    
                    test_inputs = {
                        "prompt": "Create a short story about a robot",
                        "style": "sci-fi",
                        "length": "short"
                    }
                    
                    execution_result = await client.send_workflow_execution_and_wait(
                        workflow_id=workflow_id,
                        inputs=test_inputs,
                        timeout=300.0
                    )
                    
                    if execution_result:
                        print(f"📊 Execution Result: {json.dumps(execution_result, indent=2)[:200]}...")
                else:
                    print("⚠️  No workflow_id found in setup result, skipping execution test")
        else:
            print("❌ Setup failed or timed out, skipping execution test")
        
        # Test 3: Test heartbeat (only other supported message type)
        print(f"\n" + "="*60)
        print("=== Testing Heartbeat ===")
        print("="*60)
        
        print(f"\n" + "="*60)
        print("=== Message Storage Information ===")
        print("="*60)
        print("Check message storage via HTTP endpoints:")
        print(f"GET http://localhost:8001/socket/{client.project_short_id}/messages")
        print(f"GET http://localhost:8001/socket/{client.project_short_id}/stats")
        print("GET http://localhost:8001/socket/messages/stats")
        
        # Keep connection alive briefly to see any final messages
        print("\n⏳ Keeping connection alive for final messages...")
        await asyncio.sleep(3)
        
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        elapsed_time = (datetime.now() - client.start_time).total_seconds()
        print(f"\n📊 TEST SUMMARY:")
        print(f"   Total messages received: {client.message_count}")
        print(f"   Total test time: {elapsed_time:.1f} seconds")
        print(f"   Average message rate: {client.message_count/elapsed_time:.2f} msg/sec" if elapsed_time > 0 else "   Average message rate: N/A")
        
        await client.disconnect()

if __name__ == "__main__":
    print("EvoAgentX Socket Client Test")
    print("============================")
    print("Make sure the server is running on localhost:8001")
    print("Press Ctrl+C to stop\n")
    
    asyncio.run(main())
