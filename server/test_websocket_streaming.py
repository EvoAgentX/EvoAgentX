#!/usr/bin/env python3
"""
Standalone WebSocket Streaming Test

This script tests the WebSocket streaming functionality of the EvoAgentX server.
It connects to the WebSocket endpoint and validates real-time progress updates.

Usage:
    python test_websocket_streaming.py
"""

import asyncio
import json
import websockets
import os
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv('server/app.env', override=True)

# Configuration
BASE_URL = "ws://localhost:8001"
ACCESS_TOKEN = os.getenv("EAX_ACCESS_TOKEN", "default_secret_token_change_me")
TEST_WORKFLOW_ID = "550e8400-e29b-41d4-a716-446655440001"  # Use the fixed test workflow ID

async def test_websocket_streaming():
    """Test WebSocket streaming functionality"""
    print("🔌 Testing WebSocket Streaming")
    print("=" * 50)
    
    uri = f"{BASE_URL}/workflow/{TEST_WORKFLOW_ID}/execute_ws"
    
    headers = {"eax-access-token": ACCESS_TOKEN}
    
    try:
        print(f"📡 Connecting to WebSocket: {uri}")
        async with websockets.connect(uri, additional_headers=headers) as websocket:
            print("✅ WebSocket connection established")
            
            # Wait for connection confirmation
            connection_msg = await websocket.recv()
            connection_data = json.loads(connection_msg)
            
            if connection_data.get("type") != "connection":
                print(f"❌ Expected connection message, got: {connection_data.get('type')}")
                return False
            
            print(f"✅ Connection confirmed: {connection_data.get('message')}")
            
            # Send execution inputs
            test_inputs = {
                "symptoms": "Pet is lethargic, not eating, and has a fever",
                "pet_info": "Dog, 3 years old, Golden Retriever"
            }
            execution_request = {
                "inputs": test_inputs
            }
            
            await websocket.send(json.dumps(execution_request))
            print(f"📤 Sent execution request with inputs: {test_inputs}")
            
            # Collect and display all messages during execution
            messages = []
            progress_updates = []
            log_messages = []
            final_result = None
            error_occurred = False
            
            print("\n📨 Receiving real-time updates...")
            print("-" * 50)
            
            try:
                while True:
                    message = await asyncio.wait_for(websocket.recv(), timeout=120.0)  # 2 minute timeout
                    data = json.loads(message)
                    messages.append(data)
                    
                    msg_type = data.get("type")
                    timestamp = data.get("timestamp")
                    
                    print(f"📨 [{timestamp}] {msg_type.upper()}")
                    
                    if msg_type == "start":
                        print(f"    🚀 Execution started")
                        print(f"    📋 Workflow ID: {data.get('workflow_id')}")
                        print(f"    📥 Inputs: {data.get('inputs')}")
                        
                    elif msg_type == "progress":
                        progress = data.get("progress", 0)
                        phase = data.get("phase", "unknown")
                        message_text = data.get("message", "")
                        progress_updates.append({
                            "phase": phase,
                            "progress": progress,
                            "message": message_text
                        })
                        print(f"    📊 Phase: {phase}")
                        print(f"    📈 Progress: {progress:.1%}")
                        if message_text:
                            print(f"    💬 Message: {message_text}")
                            
                    elif msg_type == "log":
                        level = data.get("level", "INFO")
                        log_message = data.get("message", "")
                        log_messages.append({
                            "level": level,
                            "message": log_message,
                            "timestamp": timestamp
                        })
                        print(f"    📝 [{level}] {log_message}")
                        
                    elif msg_type == "final_result":
                        final_result = data.get("result", {})
                        print(f"    ✅ Final result received")
                        print(f"    📄 Result: {json.dumps(final_result, indent=2)}")
                        break
                        
                    elif msg_type == "error":
                        error_occurred = True
                        error_msg = data.get("error", "Unknown error")
                        print(f"    ❌ Error: {error_msg}")
                        break
                        
                    elif msg_type == "completion":
                        print(f"    🎉 Execution completed successfully")
                        break
                        
                    else:
                        print(f"    📋 Unknown message type: {msg_type}")
                        print(f"    📄 Data: {json.dumps(data, indent=2)}")
                    
                    print()  # Empty line for readability
                
            except asyncio.TimeoutError:
                print("❌ WebSocket connection timed out after 2 minutes")
                return False
            
            # Summary
            print("\n" + "=" * 50)
            print("📊 STREAMING TEST SUMMARY")
            print("=" * 50)
            print(f"Total messages received: {len(messages)}")
            print(f"Progress updates: {len(progress_updates)}")
            print(f"Log messages: {len(log_messages)}")
            print(f"Final result received: {'✅ Yes' if final_result else '❌ No'}")
            print(f"Error occurred: {'❌ Yes' if error_occurred else '✅ No'}")
            
            if error_occurred:
                print("\n❌ TEST FAILED: Error occurred during execution")
                return False
            elif not final_result:
                print("\n❌ TEST FAILED: No final result received")
                return False
            else:
                print("\n✅ TEST PASSED: WebSocket streaming working correctly")
                return True
                
    except websockets.exceptions.InvalidURI:
        print(f"❌ Invalid WebSocket URI: {uri}")
        return False
    except websockets.exceptions.ConnectionClosed:
        print("❌ WebSocket connection was closed unexpectedly")
        return False
    except Exception as e:
        print(f"❌ WebSocket connection failed: {str(e)}")
        return False

async def main():
    """Main test function"""
    print("🧪 WebSocket Streaming Test")
    print("=" * 50)
    print(f"Server: {BASE_URL}")
    print(f"Workflow ID: {TEST_WORKFLOW_ID}")
    print(f"Access Token: {ACCESS_TOKEN[:10]}...")
    print()
    
    success = await test_websocket_streaming()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 ALL TESTS PASSED")
    else:
        print("💥 SOME TESTS FAILED")
    print("=" * 50)
    
    return success

if __name__ == "__main__":
    asyncio.run(main()) 