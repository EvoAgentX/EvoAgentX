#!/usr/bin/env python3
"""
Test script for the new tool creation system with centralized configuration.
This script tests the create_tools function with different configurations.
"""

import os
import sys
from dotenv import load_dotenv

# Add the parent directory to sys.path to import from server
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.tool_creator import create_tools, SERVER_CONFIG

def test_tool_creation():
    """Test the tool creation system"""
    print("🧪 Testing Tool Creation System with Centralized Configuration")
    print("=" * 70)
    
    # Display server configuration
    print("\n📋 Server Configuration:")
    print(f"   Search defaults: {SERVER_CONFIG['search_defaults']}")
    print(f"   Database defaults: {SERVER_CONFIG['database_defaults']}")
    print(f"   File defaults: {SERVER_CONFIG['file_defaults']}")
    print(f"   Tool names: {len(SERVER_CONFIG['tool_names'])} tools configured")
    
    # Test 1: Create tools with a test project ID
    print("\n1. Testing tool creation with test project ID...")
    try:
        tools = create_tools("test_project_123")
        print(f"✅ Successfully created {len(tools)} tools")
        
        # Categorize tools by type
        default_tools = []
        storage_tools = []
        
        for tool in tools:
            tool_name = getattr(tool, 'name', f'Tool_{type(tool).__name__}')
            if hasattr(tool, 'storage_handler') and tool.storage_handler:
                storage_tools.append(tool_name)
            else:
                default_tools.append(tool_name)
        
        print(f"   Default tools ({len(default_tools)}): {', '.join(default_tools[:5])}{'...' if len(default_tools) > 5 else ''}")
        print(f"   Storage tools ({len(storage_tools)}): {', '.join(storage_tools[:5])}{'...' if len(storage_tools) > 5 else ''}")
        
        # Test specific tool configurations
        print("\n   🔍 Tool Configuration Verification:")
        for tool in tools[:3]:  # Check first 3 tools
            if hasattr(tool, 'name'):
                print(f"      {tool.name}: {type(tool).__name__}")
                # Check if search tools have server defaults
                if hasattr(tool, 'search_wiki') and hasattr(tool.search_wiki, 'num_search_pages'):
                    print(f"         Search pages: {tool.search_wiki.num_search_pages}")
                if hasattr(tool, 'search_wiki') and hasattr(tool.search_wiki, 'max_content_words'):
                    print(f"         Max words: {tool.search_wiki.max_content_words}")
            
    except Exception as e:
        print(f"❌ Failed to create tools: {e}")
    
    # Test 2: Create tools with database information
    print("\n2. Testing tool creation with database information...")
    try:
        database_info = {
            "database_name": "test_db",
            "database_entities": []
        }
        tools = create_tools("test_project_456", database_info)
        print(f"✅ Successfully created {len(tools)} tools with database info")
        
    except Exception as e:
        print(f"❌ Failed to create tools with database info: {e}")
    
    # Test 3: Test with missing environment variables
    print("\n3. Testing tool creation with missing storage config...")
    try:
        # Temporarily unset storage environment variables
        original_storage_url = os.environ.get("SUPABASE_URL_STORAGE")
        original_storage_key = os.environ.get("SUPABASE_KEY_STORAGE")
        original_storage_bucket = os.environ.get("SUPABASE_BUCKET_STORAGE")
        
        if "SUPABASE_URL_STORAGE" in os.environ:
            del os.environ["SUPABASE_URL_STORAGE"]
        if "SUPABASE_KEY_STORAGE" in os.environ:
            del os.environ["SUPABASE_KEY_STORAGE"]
        if "SUPABASE_BUCKET_STORAGE" in os.environ:
            del os.environ["SUPABASE_BUCKET_STORAGE"]
        
        tools = create_tools("test_project_missing_config")
        print(f"✅ Successfully created {len(tools)} tools (storage tools disabled)")
        
        # Restore environment variables
        if original_storage_url:
            os.environ["SUPABASE_URL_STORAGE"] = original_storage_url
        if original_storage_key:
            os.environ["SUPABASE_KEY_STORAGE"] = original_storage_key
        if original_storage_bucket:
            os.environ["SUPABASE_BUCKET_STORAGE"] = original_storage_bucket
            
    except Exception as e:
        print(f"❌ Failed to create tools with missing config: {e}")
    
    print("\n" + "=" * 70)
    print("🎉 Tool creation tests completed!")
    print("\n📋 Tool Categorization Summary:")
    print("   Default tools (no storage): MCP, Search, RSS, Request, Database, File")
    print("   Storage tools: ArXiv, FAISS, CMD, Storage, Image Generation, Image Analysis")
    print("\n⚙️  Server Configuration Applied:")
    print(f"   Search limits: {SERVER_CONFIG['search_defaults']['num_search_pages']} pages, {SERVER_CONFIG['search_defaults']['max_content_words']} words max")
    print(f"   Database behavior: auto_save={SERVER_CONFIG['database_defaults']['auto_save']}, read_only={SERVER_CONFIG['database_defaults']['read_only']}")
    print(f"   Tool naming: {len(SERVER_CONFIG['tool_names'])} tools with consistent naming")

if __name__ == "__main__":
    # Load environment variables
    load_dotenv(override=True)
    
    test_tool_creation()
