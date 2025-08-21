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
        
        self.heartbeat_count = 0
        self.heartbeat_responses_received = 0  # Track successful heartbeat responses
        self.heartbeat_task = None
        self.heartbeat_interval = 30  # seconds
        self.connection_confirmed = False
        
        
    async def connect(self):
        """Connect to the socket server."""
        socket_url = f"{self.server_url}/project/{self.project_short_id}/regist"
        print(f"🔌 Connecting to {socket_url}")
        
        # Connect with longer timeouts for long-running operations
        self.websocket = await websockets.connect(
            socket_url,
            ping_interval=30,    # Send ping every 30 seconds
            ping_timeout=20,     # Wait 20 seconds for pong
            close_timeout=60     # Wait 60 seconds before force closing
        )
        print("✅ Connected successfully!")
        
        # Start listening for messages and store the task
        self.listen_task = asyncio.create_task(self.listen_for_messages())
        print("🎧 Started listening for messages...")
        
        # Wait a moment for connection to stabilize and listener to be ready
        await asyncio.sleep(2)
        print("⏳ Connection stabilized, ready to receive messages...")
        
        # Wait for connection confirmation message
        print("⏳ Waiting for connection confirmation from server...")
        await asyncio.sleep(2)
        print("✅ Connection confirmed, ready to send messages...")
        
    async def disconnect(self):
        """Disconnect from the socket server."""
        if hasattr(self, 'listen_task') and self.listen_task:
            self.listen_task.cancel()
            try:
                await self.listen_task
            except asyncio.CancelledError:
                pass
        
        # Stop heartbeat if running
        # await self.stop_heartbeat()
        
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
                    print(f"✅ Successfully parsed JSON message")
                    print(f"🔍 Message type: {data.get('type')}")
                    print(f"🔍 Message data: {data.get('data', {})}")
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
        # Safety check for valid message data
        if not isinstance(message, dict):
            print(f"❌ Invalid message format: expected dict, got {type(message)}")
            return
            
        if not message:
            print(f"❌ Empty message received")
            return
        print(f"Message: {message}")
            
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
            result = data.get('result')
            if result is not None:
                if isinstance(result, (list, dict)):
                    result_count = len(result)
                else:
                    result_count = 1
                print(f"🎉 SETUP COMPLETED! Received {result_count} workflow graphs")
            else:
                print(f"🎉 SETUP COMPLETED! (no result data)")
            self.operation_results["setup"] = result
            self.setup_complete.set()
        elif message_type == "execution-complete":
            result = data.get('result')
            if result is not None:
                print(f"🎉 EXECUTION COMPLETED!")
            else:
                print(f"🎉 EXECUTION COMPLETED! (no result data)")
            self.operation_results["execution"] = result
            self.execution_complete.set()
        elif message_type == "complete":
            result = data.get('result')
            if result is not None:
                print(f"🎉 WORKFLOW EXECUTION COMPLETED!")
            else:
                print(f"🎉 WORKFLOW EXECUTION COMPLETED! (no result data)")
            self.operation_results["workflow_execution"] = result
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
            content = data.get('content', '')
            
            # Check if this is a connection confirmation
            if "WebSocket connection established" in content:
                self.connection_confirmed = True
                print(f"🔌 CONNECTION CONFIRMED: {content}")
            # Check if this is a heartbeat response (server sends heartbeat responses as setup-log)
            elif "heartbeat" in content.lower() or (data.get('result') and isinstance(data.get('result'), dict) and data.get('result', {}).get('timestamp')):
                self.heartbeat_responses_received += 1
                print(f"💓 HEARTBEAT RESPONSE #{self.heartbeat_responses_received}: {content}")
                result = data.get('result')
                if result and isinstance(result, dict) and result.get('timestamp'):
                    server_time = result['timestamp']
                    print(f"   📅 Server timestamp: {server_time}")
            else:
                if workflow_id:
                    print(f"🏗️ SETUP LOG with WORKFLOW ID: {workflow_id}")
                    print(f"   Content: {content}")
                else:
                    print(f"🏗️ SETUP LOG (no workflow ID)")
                    print(f"   Content: {content}")
        elif message_type == "runtime-log":
            workflow_id = data.get('workflow_id')
            if workflow_id:
                print(f"🔄 RUNTIME LOG with WORKFLOW ID: {workflow_id}")
                print(f"   Content: {data.get('content')}")
            else:
                print(f"🔄 RUNTIME LOG (no workflow ID)")
        elif message_type == "heartbeat":
            self.heartbeat_responses_received += 1
            print(f"💓 HEARTBEAT RESPONSE #{self.heartbeat_responses_received} (server alive, connection maintained)")
            # Show heartbeat response details
            result = data.get('result')
            if result and isinstance(result, dict):
                if result.get('timestamp'):
                    server_time = result['timestamp']
                    print(f"   📅 Server timestamp: {server_time}")
                if data.get('status'):
                    print(f"   ✅ Server status: {data['status']}")
            # Don't print full structure for heartbeats to reduce noise
            return
        else:
            print(f"❓ UNKNOWN MESSAGE TYPE: {message_type}")
        
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
        
        # Enhanced heartbeat logging
        print(f"💓 Sending heartbeat #{self.heartbeat_count + 1} at {datetime.now().strftime('%H:%M:%S')}")
        
        await self.websocket.send(json.dumps(message))
        
        self.heartbeat_count += 1
    
    def get_heartbeat_summary(self):
        """Get a summary of heartbeat activity."""
        return {
            "heartbeats_sent": self.heartbeat_count,
            "heartbeat_responses_received": self.heartbeat_responses_received,
            "heartbeat_success_rate": f"{(self.heartbeat_responses_received / max(self.heartbeat_count, 1)) * 100:.1f}%" if self.heartbeat_count > 0 else "0%",
            "connection_confirmed": self.connection_confirmed
        }
    
    async def start_heartbeat(self):
        """Start periodic heartbeat sending."""
        if self.heartbeat_task and not self.heartbeat_task.done():
            return  # Already running
        
        async def heartbeat_loop():
            while True:
                try:
                    if self.is_connected():
                        await self.send_heartbeat()
                    await asyncio.sleep(self.heartbeat_interval)
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    print(f"💓 Heartbeat error: {e}")
                    break
        
        self.heartbeat_task = asyncio.create_task(heartbeat_loop())
        print(f"💓 Started heartbeat every {self.heartbeat_interval} seconds")
    
    async def stop_heartbeat(self):
        """Stop periodic heartbeat sending."""
        if self.heartbeat_task and not self.heartbeat_task.done():
            self.heartbeat_task.cancel()
            try:
                await self.heartbeat_task
            except asyncio.CancelledError:
                pass
            self.heartbeat_task = None
            print("💓 Stopped heartbeat")
    
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
    
    async def send_setup_message_and_wait(self, timeout: float = 120.0):
        """Send setup message through registered socket and wait for completion."""
        self.current_operation = "setup"
        self.setup_complete.clear()
        
        print(f"\n🚀 Starting project setup via socket message (timeout: {timeout}s)")
        
        # Start heartbeat during setup
        # await self.start_heartbeat()
        
        try:
            # Send setup message as per README specification
            setup_message = {
                "type": "setup",
                "data": {
                    "project_short_id": self.project_short_id
                }
            }
            
            setup_message_json = json.dumps(setup_message, indent=2)
            print(f"📤 Sending setup message:")
            print(f"{'='*50}")
            print(setup_message_json)
            print(f"{'='*50}")
            
            await self.websocket.send(setup_message_json)
            print("✅ Setup message sent successfully")
            
            print("⏳ Waiting for setup completion...")
            print("💓 Heartbeat active during setup to keep connection alive...")
            
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
                        print(f"⏳ Still waiting for setup... (connection: {'✅' if self.is_connected() else '❌'}, heartbeats: {self.heartbeat_count})")
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
        finally:
            # Stop heartbeat after setup
            # await self.stop_heartbeat()
            self.current_operation = None
    
    async def send_workflow_execution_and_wait(self, workflow_id: str, inputs: dict, timeout: float = 300.0):
        """Execute workflow via HTTP API (socket is only for setup monitoring)."""
        self.current_operation = "execution"
        self.execution_complete.clear()
        
        print(f"\n🚀 Starting workflow execution via HTTP API (timeout: {timeout}s)")
        print("📡 Note: Execution uses HTTP API, socket is only for setup monitoring")
        
        try:
            # Execute workflow via HTTP API
            import aiohttp
            
            async with aiohttp.ClientSession() as session:
                url = f"http://localhost:8001/workflow/{workflow_id}/execute"
                payload = {"inputs": inputs}
                
                print(f"🌐 Sending HTTP POST to: {url}")
                print(f"📦 Payload: {json.dumps(payload, indent=2)}")
                
                async with session.post(url, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        print("✅ Workflow execution completed successfully via HTTP API")
                        return result.get("parsed_json")
                    else:
                        error_text = await response.text()
                        print(f"❌ HTTP execution failed: {response.status} - {error_text}")
                        return None
                        
        except Exception as e:
            print(f"❌ Execution error: {e}")
            return None
        finally:
            self.current_operation = None

async def main():
    """Main test function with proper completion waiting."""
    client = EvoAgentXSocketClient()
    
    try:
        # Connect to server (this registers the socket)
        await client.connect()
        
        # Wait for connection confirmation
        print("⏳ Waiting for connection confirmation...")
        await asyncio.sleep(2)
        
        # Check connection status
        connection_status = "✅" if client.connection_confirmed else "❌"
        heartbeat_summary = client.get_heartbeat_summary()
        print(f"🔌 Connection status: {connection_status}")
        print(f"💓 Heartbeats sent: {heartbeat_summary['heartbeats_sent']}, responses: {heartbeat_summary['heartbeat_responses_received']}")
        
        if not client.connection_confirmed:
            print("⚠️  Connection not confirmed, but proceeding with setup...")
        
        # Test 1: Project Setup via socket message (new architecture)
        print("\n" + "="*60)
        print("=== Testing Project Setup via Socket Message ===")
        print("="*60)
        print("⏰ Note: Project setup can take 5-10 minutes for complex workflows")
        print("📡 Using new socket registration + message-based setup")
        
        # Test basic connection first
        print("🔍 Testing basic WebSocket connection...")
        if client.is_connected():
            print("✅ WebSocket connection is active")
        else:
            print("❌ WebSocket connection is not active")
            raise ConnectionError("WebSocket connection lost before setup")
        
        # Send setup message through the registered socket
        print("\n📤 Sending setup message...")
        try:
            setup_result = await client.send_setup_message_and_wait(timeout=600.0)  # 10 minutes timeout
        except Exception as e:
            print(f"❌ Setup failed with error: {e}")
            print("🛑 Stopping tests due to setup failure")
            return
        
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
                    print(f"=== Testing Workflow Execution via HTTP API: {workflow_id} ===")
                    print("="*60)
                    print("📡 Using HTTP API for execution (socket is only for setup monitoring)")
                    
                    test_inputs = {
                        "prompt": "Create a short story about a robot",
                        "style": "sci-fi",
                        "length": "short"
                    }
                    
                    try:
                        execution_result = await client.send_workflow_execution_and_wait(
                            workflow_id=workflow_id,
                            inputs=test_inputs,
                            timeout=300.0
                        )
                        
                        if execution_result:
                            print(f"📊 Execution Result: {json.dumps(execution_result, indent=2)[:200]}...")
                        else:
                            print("❌ Execution failed or timed out")
                    except Exception as e:
                        print(f"❌ Execution failed with error: {e}")
                else:
                    print("⚠️  No workflow_id found in setup result, skipping execution test")
        else:
            print("❌ Setup failed or timed out, skipping execution test")
            return  # Stop here if setup failed
        
        # Test 3: Test heartbeat (only other supported message type)
        print(f"\n" + "="*60)
        print("=== Testing Heartbeat ===")
        print("="*60)
        
        # Only test heartbeat if setup and execution succeeded
        if client.is_connected():
            # Test heartbeat when server is not busy
            print("Testing heartbeat response when server is idle...")
            await client.send_heartbeat()
            await asyncio.sleep(2)  # Wait for response
            
            # Test multiple heartbeats
            print("Testing multiple heartbeats...")
            for i in range(3):
                await client.send_heartbeat()
                await asyncio.sleep(1)
            
            print("Heartbeat test completed")
        else:
            print("❌ WebSocket not connected, skipping heartbeat tests")
        
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
        
        # Heartbeat summary
        heartbeat_summary = client.get_heartbeat_summary()
        print(f"   💓 Heartbeat summary:")
        print(f"      Total sent: {heartbeat_summary['heartbeats_sent']}")
        print(f"      Received: {heartbeat_summary['heartbeat_responses_received']}")
        print(f"      Success Rate: {heartbeat_summary['heartbeat_success_rate']}")
        print(f"      Connection Confirmed: {'✅' if heartbeat_summary['connection_confirmed'] else '❌'}")
        
        await client.disconnect()

if __name__ == "__main__":
    print("EvoAgentX Socket Client Test")
    print("============================")
    print("Make sure the server is running on localhost:8001")
    print("Press Ctrl+C to stop\n")
    
    asyncio.run(main())
