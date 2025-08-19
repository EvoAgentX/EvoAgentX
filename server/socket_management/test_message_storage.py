#!/usr/bin/env python3
"""
Test script to demonstrate message storage functionality.
Shows how to retrieve and analyze stored socket messages via HTTP endpoints.
"""

import requests
import json
import time
from datetime import datetime

class MessageStorageTest:
    """Test client for message storage HTTP endpoints."""
    
    def __init__(self, server_url: str = "http://localhost:8001", project_short_id: str = "test_project"):
        self.server_url = server_url
        self.project_short_id = project_short_id
    
    def get_project_messages(self, limit: int = 50, message_type: str = None, since_hours: int = None):
        """Get stored messages for the project."""
        url = f"{self.server_url}/socket/{self.project_short_id}/messages"
        params = {"limit": limit}
        
        if message_type:
            params["message_type"] = message_type
        if since_hours:
            params["since_hours"] = since_hours
            
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Error getting messages: {e}")
            return None
    
    def get_project_stats(self):
        """Get message statistics for the project."""
        url = f"{self.server_url}/socket/{self.project_short_id}/stats"
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Error getting stats: {e}")
            return None
    
    def get_all_stats(self):
        """Get message statistics for all projects."""
        url = f"{self.server_url}/socket/messages/stats"
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Error getting all stats: {e}")
            return None
    
    def export_messages(self, format: str = "json"):
        """Export project messages in specified format."""
        url = f"{self.server_url}/socket/{self.project_short_id}/export"
        params = {"format": format}
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            print(f"Error exporting messages: {e}")
            return None
    
    def cleanup_messages(self):
        """Trigger cleanup of old messages."""
        url = f"{self.server_url}/socket/messages/cleanup"
        
        try:
            response = requests.post(url)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Error cleaning up messages: {e}")
            return None
    
    def analyze_message_flow(self):
        """Analyze the message flow and patterns."""
        print("📊 Analyzing Message Flow")
        print("=" * 50)
        
        # Get project stats
        stats = self.get_project_stats()
        if stats and "error" not in stats:
            print(f"Project: {stats['project_short_id']}")
            print(f"Total messages: {stats['total_messages']}")
            print(f"Messages per minute: {stats['messages_per_minute']:.2f}")
            print(f"Average message size: {stats['average_message_size']:.0f} bytes")
            print(f"Total size: {stats['total_size_bytes']:,} bytes")
            
            if stats['first_message'] and stats['last_message']:
                print(f"First message: {stats['first_message']}")
                print(f"Last message: {stats['last_message']}")
            
            print("\nMessage Types:")
            for msg_type, count in stats['message_types'].items():
                print(f"  {msg_type}: {count}")
        else:
            print("No statistics available (project may not have any messages yet)")
        
        print("\n" + "=" * 50)
        
        # Get recent messages
        messages = self.get_project_messages(limit=10)
        if messages and messages['messages']:
            print(f"\n📨 Recent Messages (last {len(messages['messages'])})")
            print("-" * 30)
            
            for msg in messages['messages'][:5]:  # Show first 5
                timestamp = msg['timestamp'][:19]  # Remove microseconds
                direction = "⬅️" if msg['direction'] == 'incoming' else "➡️"
                msg_type = msg['message']['type']
                content = msg['message'].get('data', {}).get('content', 'N/A')[:50]
                
                print(f"{timestamp} {direction} {msg_type}: {content}")
        else:
            print("\n📨 No recent messages found")
        
        print("\n" + "=" * 50)
    
    def demonstrate_filtering(self):
        """Demonstrate message filtering capabilities."""
        print("\n🔍 Demonstrating Message Filtering")
        print("=" * 40)
        
        # Filter by message type
        for msg_type in ['connection', 'workflow_status', 'setup_complete', 'error']:
            filtered = self.get_project_messages(limit=5, message_type=msg_type)
            if filtered and filtered['messages']:
                print(f"\n{msg_type} messages: {len(filtered['messages'])} found")
                for msg in filtered['messages'][:2]:  # Show first 2
                    content = msg['message'].get('data', {}).get('content', 'N/A')[:40]
                    print(f"  - {content}")
            else:
                print(f"\n{msg_type} messages: None found")
        
        # Filter by time (last hour)
        recent = self.get_project_messages(limit=20, since_hours=1)
        if recent and recent['messages']:
            print(f"\nMessages from last hour: {len(recent['messages'])}")
        else:
            print(f"\nMessages from last hour: None found")
    
    def demonstrate_export(self):
        """Demonstrate message export functionality."""
        print("\n📤 Demonstrating Export Functionality")
        print("=" * 40)
        
        for format in ['json', 'csv', 'txt']:
            print(f"\nExporting as {format.upper()}:")
            exported = self.export_messages(format)
            if exported:
                # Show first 200 characters
                preview = exported[:200].replace('\n', '\\n')
                print(f"  Preview: {preview}...")
                print(f"  Total length: {len(exported)} characters")
            else:
                print(f"  Export failed")

def main():
    """Main test function."""
    print("EvoAgentX Message Storage Test")
    print("==============================")
    print("This script tests the message storage functionality")
    print("Make sure the server is running and you've sent some socket messages\n")
    
    tester = MessageStorageTest()
    
    try:
        # Test 1: Basic message analysis
        tester.analyze_message_flow()
        
        # Test 2: Message filtering
        tester.demonstrate_filtering()
        
        # Test 3: Export functionality
        tester.demonstrate_export()
        
        # Test 4: Global statistics
        print("\n🌍 Global Statistics")
        print("=" * 20)
        global_stats = tester.get_all_stats()
        if global_stats:
            print(f"Total projects: {global_stats['total_projects']}")
            print(f"Active projects: {len(global_stats['active_projects'])}")
            print(f"Total messages: {global_stats['total_messages']}")
            
            if global_stats['active_projects']:
                print(f"Active: {', '.join(global_stats['active_projects'])}")
        
        # Test 5: Cleanup (optional)
        print(f"\n🧹 Message Cleanup")
        print("=" * 20)
        print("Triggering cleanup of old messages...")
        cleanup_result = tester.cleanup_messages()
        if cleanup_result:
            print(f"Cleanup completed at: {cleanup_result['timestamp']}")
        
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"Test error: {e}")

if __name__ == "__main__":
    main()
