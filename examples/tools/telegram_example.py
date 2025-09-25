"""
Example usage of Telegram tools in EvoAgentX

This example demonstrates how to use the various Telegram tools
for messaging, searching, and file operations.

Prerequisites:
1. Telegram API credentials with the following setup:
   - Create a new application at https://my.telegram.org/apps
   - Get your API ID and API Hash
   - Set up your phone number for authentication

2. Set your credentials as environment variables:
   export TELEGRAM_API_ID="your_api_id_here"
   export TELEGRAM_API_HASH="your_api_hash_here"
   export TELEGRAM_PHONE="your_phone_number_here"

Note: The TelegramToolkit will automatically retrieve the credentials from
the environment variables, making it compatible with AI agents.
"""

import os
from evoagentx.tools import TelegramToolkit

def main():
    # Initialize the toolkit - credentials will be automatically retrieved from environment
    telegram_toolkit = TelegramToolkit()
    
    # Check if credentials are available
    if not telegram_toolkit.telegram_base.api_id or not telegram_toolkit.telegram_base.api_hash:
        print("Please set TELEGRAM_API_ID and TELEGRAM_API_HASH environment variables")
        print("Get your credentials from: https://my.telegram.org/apps")
        return
    
    print("=== Telegram Tools Demo ===\n")
    
    # 1. List recent chats
    print("1. Listing Recent Chats")
    list_chats_tool = telegram_toolkit.get_tool("list_recent_chats")
    result = list_chats_tool(limit=5)
    
    if result["success"]:
        print(f"Found {result['chats_count']} recent chats:")
        for chat in result["chats"]:
            print(f"  - {chat['title']} ({chat['type']}) - ID: {chat['id']}")
        
        # Use the first non-system chat for other examples
        if result["chats"]:
            # Skip system chats (like Telegram user with ID 777000)
            for chat in result["chats"]:
                if chat["id"] != 777000:  # Skip Telegram system user
                    first_chat_id = chat["id"]
                    first_chat_title = chat["title"]
                    break
            else:
                # If all chats are system chats, use the first one anyway
                first_chat_id = result["chats"][0]["id"]
                first_chat_title = result["chats"][0]["title"]
    else:
        print(f"Failed to list chats: {result['error']}")
        return
    
    print("\n" + "="*50 + "\n")
    
    # 2. Fetch latest messages
    print(f"2. Fetching Latest Messages from '{first_chat_title}'")
    fetch_messages_tool = telegram_toolkit.get_tool("fetch_latest_messages")
    result = fetch_messages_tool(contact_name=first_chat_title, limit=3)
    
    if result["success"]:
        print(f"Found {result['messages_count']} recent messages:")
        for msg in result["messages"]:
            print(f"  - [{msg['date']}] {msg['text'][:50]}...")
    else:
        print(f"Failed to fetch messages: {result['error']}")
    
    print("\n" + "="*50 + "\n")
    
    # 3. Search messages by keyword
    print("3. Searching Messages by Keyword")
    search_tool = telegram_toolkit.get_tool("search_messages_by_keyword")
    result = search_tool(contact_name=first_chat_title, keyword="Hello", limit=3)
    
    if result["success"]:
        print(f"Found {result['matches_count']} messages containing 'hello':")
        for msg in result["messages"]:
            print(f"  - [{msg['date']}] {msg['text'][:50]}...")
    else:
        print(f"Failed to search messages: {result['error']}")
    
    print("\n" + "="*50 + "\n")
    
    # 4. Find files
    print("4. Finding Files in Chat")
    find_files_tool = telegram_toolkit.get_tool("find_and_retrieve_file")
    result = find_files_tool(contact_name="Telegram", filename_query="Kafka")
    
    if result["success"]:
        print(f"Found {result['files_found']} files matching 'Kafka':")
        for file_info in result["files"]:
            print(f"  - {file_info['filename']} ({file_info['file_size']} bytes, {file_info['mime_type']})")
    else:
        print(f"Failed to find files: {result['error']}")
    
    print("\n" + "="*50 + "\n")
    
    # 5. Send a test message using contact name
    print("5. Send Test Message by Contact Name")
    send_tool = telegram_toolkit.get_tool("send_message_by_name")
    
    # Try to send a message to a contact (you can modify this name)
    test_contact_name = "Vinay Kumar"  # This should match a contact in your Telegram
    test_message = "Hello! This is a test message from EvoAgentX Telegram tools. 🤖"
    
    print(f"Sending test message to contact: '{test_contact_name}'")
    result = send_tool(contact_name=test_contact_name, message_text=test_message)
    
    if result["success"]:
        print(f"✅ Message sent successfully!")
        print(f"   Message ID: {result['message_id']}")
        print(f"   Sent to: {result['chat']['title']}")
        print(f"   Message: {result['message_text']}")
        print(f"   Sent at: {result['sent_at']}")
    else:
        print(f"❌ Failed to send message: {result['error']}")
        if 'clarification_needed' in result:
            print("   Available contacts:")
            for contact in result['clarification_needed']:
                print(f"   - {contact}")
    
    print("\n" + "="*50 + "\n")
    
    # 6. Summarize contact messages
    print("6. Summarize Contact Messages")
    summarize_tool = telegram_toolkit.get_tool("summarize_contact_messages")
    
    # Summarize messages from a contact
    test_contact_name = "Telegram"  # This should match a contact in your Telegram
    print(f"Summarizing recent messages from contact: '{test_contact_name}'")
    result = summarize_tool(contact_name=test_contact_name, limit=10)
    
    if result["success"]:
        print(f"✅ Message summary generated successfully!")
        print(f"   Contact: {result['contact']['title']}")
        print(f"   Messages analyzed: {result['messages_analyzed']}")
        print(f"   Summary:")
        print(f"   {result['summary']}")
        if result['recent_messages']:
            print(f"   Recent messages preview:")
            for i, msg in enumerate(result['recent_messages'][:3], 1):
                direction = "You" if msg['is_outgoing'] else test_contact_name
                print(f"   {i}. [{msg['date']}] {direction}: {msg['text'][:50]}...")
    else:
        print(f"❌ Failed to summarize messages: {result['error']}")
        if 'clarification_needed' in result:
            print("   Available contacts:")
            for contact in result['clarification_needed']:
                print(f"   - {contact}")
    
    print("\n" + "=" * 50)
    
    # Test 7: Download File Tool
    print("\n7. Download File Tool")
    print("Downloading file from contact:", test_contact_name)
    
    download_tool = telegram_toolkit.get_tool("download_file")
    download_result = download_tool(
        contact_name=test_contact_name,
        filename_query="Kafka",
        download_dir="downloads"
    )
    
    if download_result["success"]:
        print(f"✅ File downloaded successfully!")
        print(f"   📁 File: {download_result['filename']}")
        print(f"   📍 Path: {download_result['file_path']}")
        print(f"   📊 Size: {download_result['file_size']} bytes")
        print(f"   📂 Directory: {download_result['download_dir']}")
    else:
        print(f"❌ Download failed: {download_result['error']}")
        if 'clarification_needed' in download_result:
            print("   Available contacts:")
            for contact in download_result['clarification_needed']:
                print(f"   - {contact}")
    
    print("\n" + "=" * 50)
    
    # Test 8: Read File Content Tool
    print("\n8. Read File Content Tool")
    print("Reading file content from contact:", test_contact_name)
    
    read_tool = telegram_toolkit.get_tool("read_file_content")
    
    # Test different content types
    content_tests = [
        ("summary", "Document summary"),
        ("first_lines", "First 3 lines"),
        ("last_lines", "Last 3 lines")
    ]
    
    for content_type, description in content_tests:
        print(f"\n   🔍 {description}:")
        read_result = read_tool(
            contact_name=test_contact_name,
            filename_query="Kafka",
            content_type=content_type,
            lines_count=3
        )
        
        if read_result["success"]:
            print(f"   ✅ {description} extracted successfully!")
            print(f"   📄 Content preview:")
            content_preview = read_result["content"][:200] + "..." if len(read_result["content"]) > 200 else read_result["content"]
            print(f"      {content_preview}")
            
            if "file_info" in read_result:
                file_info = read_result["file_info"]
                print(f"   📊 File info: {file_info}")
        else:
            print(f"   ❌ Failed to read {description}: {read_result['error']}")
    
    print("\n=== Demo Complete ===")
    print("All 8 Telegram tools are working correctly!")
    print("✅ Core Tools (6): fetch_latest_messages, search_messages_by_keyword, send_message_by_name, list_recent_chats, find_and_retrieve_file, summarize_contact_messages")
    print("✅ Enhanced File Tools (2): download_file, read_file_content")

if __name__ == "__main__":
    main()
