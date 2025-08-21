#!/usr/bin/env python3
"""
Simple test client for EvoAgentX Socket Management Service.
Design: Send a message to the server and keep receiving/printing messages.
"""

import asyncio
import json
import websockets
from datetime import datetime

project_short_id = "9mshbju"

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
        
        self.websocket = await websockets.connect(socket_url)
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
        
        # Send a message
        await client.send_message()
        
        # Keep listening for messages
        await client.listen_for_messages()
        
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
    print("EvoAgentX Simple Socket Client Test")
    print("===================================")
    print("Make sure the server is running on localhost:8001")
    print("This client will send a message and keep receiving messages")
    print("Press Ctrl+C to stop\n")
    
    asyncio.run(main())