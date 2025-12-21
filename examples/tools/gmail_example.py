"""
Example usage of Gmail tools in EvoAgentX

This example demonstrates how to use the various Gmail tools
for reading messages, threads, and managing drafts.

Prerequisites:
1. Google Cloud Console setup:
   - Create a new project in Google Cloud Console
   - Enable Gmail API
   - Configure OAuth Consent Screen
   - Create OAuth 2.0 Desktop app credentials
   - Download client_secret.json and place it in the project root

2. First-time authentication:
   - Run this script once to authenticate
   - A browser will open for Google OAuth consent
   - After consent, token.json will be saved for future use

Note: The GmailToolkit will automatically handle authentication using
client_secret.json and token.json files.
"""

import os
from evoagentx.tools import GmailToolkit

def main():
    # Initialize the toolkit
    # It will look for client_secret.json in the current directory by default
    gmail_toolkit = GmailToolkit()
    
    # Check if client_secret.json exists
    if not os.path.exists('client_secret.json'):
        print("❌ client_secret.json not found!")
        print("Please download it from Google Cloud Console and place it in the project root.")
        print("See Gmail_api.txt for detailed setup instructions.")
        return
    
    print("=== Gmail Tools Demo ===\n")
    
    # 1. Read messages
    print("1. Reading Gmail Messages")
    read_messages_tool = gmail_toolkit.get_tool("read_messages")
    
    # Read unread messages
    print("   Reading unread messages...")
    result = read_messages_tool(query="is:unread", max_results=5)
    
    if result["success"]:
        print(f"   ✅ Found {result['messages_count']} total unread messages")
        print(f"   Total available: {result['total_available']}")
        print(f"   Showing {len(result['messages'])} messages:\n")
        for msg in result["messages"]:  # Already capped at 10
            print(f"      - From: {msg['from']}")
            print(f"        Subject: {msg['subject']}")
            print(f"        Snippet: {msg['snippet'][:50]}...")
            print()
    else:
        print(f"   ❌ Failed to read messages: {result['error']}")
    
    print("="*50 + "\n")
    
    # 2. Search messages (uses consolidated SearchMessagesTool)
    print("2. Searching Messages")
    search_messages_tool = gmail_toolkit.get_tool("search_messages")
    print("   Searching for messages from Medium...")
    result = search_messages_tool(keyword="from:noreply@medium.com", max_results=3)
    
    if result["success"]:
        print(f"   ✅ Found {result['messages_count']} total matches")
        print(f"   Total available: {result['total_available']}")
        print(f"   Showing {len(result['messages'])} messages:\n")
        for msg in result["messages"]:
            print(f"      - Subject: {msg['subject']}")
            print(f"        Date: {msg['date']}")
            print()
    else:
        print(f"   ❌ Failed to search messages: {result['error']}")
    
    print("="*50 + "\n")
    
    # 3. Read threads (uses refactored ReadThreadsTool with _fetch_threads helper)
    print("3. Reading Gmail Threads")
    read_threads_tool = gmail_toolkit.get_tool("read_threads")
    
    print("   Reading recent threads...")
    result = read_threads_tool(max_results=3)
    
    if result["success"]:
        print(f"   ✅ Found {result['threads_count']} total threads")
        print(f"   Total available: {result['total_available']}")
        print(f"   Showing {len(result['threads'])} threads:\n")
        for thread in result["threads"]:
            print(f"      - Thread ID: {thread['id']}")
            print(f"        Messages in thread: {thread['messages_count']}")
            if thread['messages']:
                first_msg = thread['messages'][0]
                print(f"        Subject: {first_msg['subject']}")
                print(f"        From: {first_msg['from']}")
            print()
    else:
        print(f"   ❌ Failed to read threads: {result['error']}")
    
    print("="*50 + "\n")
    
    # 4. List drafts (fixed API parameter error; capped display to 10)
    print("4. Listing Drafts")
    list_drafts_tool = gmail_toolkit.get_tool("list_drafts")
    
    print("   Listing all drafts...")
    result = list_drafts_tool(max_results=10)
    
    if result["success"]:
        print(f"   ✅ Found {result['drafts_count']} total drafts")
        print(f"    Total available: {result['total_available']}")
        print(f"   Showing {len(result['drafts'])} drafts:\n")
        for draft in result["drafts"]:
            print(f"      - Draft ID: {draft['draft_id']}")
            print(f"        Subject: {draft['subject']}")
            print(f"        To: {draft['to']}")
            print()
    else:
        print(f"   ❌ Failed to list drafts: {result['error']}")
    
    print("="*50 + "\n")
    
    # 5. Create a draft with optional attachments
    print("5. Creating a Draft with Optional Attachments")
    create_draft_tool = gmail_toolkit.get_tool("create_draft")
    
    # Create a test draft (replace with your email)
    test_to = "test@example.com"  # Replace with actual email
    test_subject = "Test Draft from EvoAgentX"
    test_body = "This is a test draft created using EvoAgentX Gmail tools."

    # Optional: attach files when creating the draft
    # CHANGE: Now supports 'attachments' parameter (list of file paths)
    attachment_path = None  # e.g., r"C:\path\to\file.pdf"

    print(f"   Creating draft to: {test_to}")
    if attachment_path:
        try:
            result = create_draft_tool(
                to=test_to,
                subject=test_subject,
                body=test_body,
                attachments=[attachment_path]  # NEW: Attachment support
            )
        except Exception as e:
            print(f"   ⚠️  Error with attachments: {e}")
            result = create_draft_tool(
                to=test_to,
                subject=test_subject,
                body=test_body
            )
    else:
        result = create_draft_tool(
            to=test_to,
            subject=test_subject,
            body=test_body
        )
    
    if result["success"]:
        draft_id = result["draft_id"]
        print(f"   ✅ Draft created successfully!")
        print(f"      Draft ID: {draft_id}")
        print(f"      Message ID: {result['message_id']}")
        
        # Store draft_id for next steps
        created_draft_id = draft_id
    else:
        print(f"   ❌ Failed to create draft: {result['error']}")
        created_draft_id = None
    
    print("="*50 + "\n")
    
    # 6. Search or retrieve a draft (using consolidated SearchDraftTool)
    # CHANGE: RetrieveDraftTool merged into SearchDraftTool (dual capability)
    if created_draft_id:
        print("6. Searching/Retrieving Draft (Consolidated SearchDraftTool)")
        search_draft_tool = gmail_toolkit.get_tool("search_drafts")
        
        print(f"   Retrieving draft ID: {created_draft_id}")
        # SearchDraftTool now handles both: direct fetch by ID and keyword search
        result = search_draft_tool(draft_id=created_draft_id)
        
        if result["success"]:
            print(f"   ✅ Draft retrieved successfully!")
            drafts = result["drafts"]
            if drafts:
                draft = drafts[0]
                print(f"      Subject: {draft['subject']}")
                print(f"      From: {draft['from']}")
                print(f"      To: {draft['to']}")
                print(f"      Body preview: {draft['body'][:100]}...")
        else:
            print(f"   ❌ Failed to retrieve draft: {result['error']}")
    else:
        print("6. Searching/Retrieving Draft")
        print("   ⚠️  Skipped - No draft was created earlier")
    
    print("="*50 + "\n")
    
    # 7. Update the draft (now supports attachments)
    if created_draft_id:
        print("7. Updating Draft with Optional Attachments")
        update_draft_tool = gmail_toolkit.get_tool("update_draft")
        
        updated_subject = "Updated: Test Draft from EvoAgentX"
        updated_body = "This draft has been updated using EvoAgentX Gmail tools."
        
        # CHANGE: UpdateDraftTool now supports adding/updating attachments
        update_attachments = None  # e.g., [r"C:\path\to\new_file.pdf"]
        
        print(f"   Updating draft ID: {created_draft_id}")
        if update_attachments:
            result = update_draft_tool(
                draft_id=created_draft_id,
                subject=updated_subject,
                body=updated_body,
                attachments=update_attachments  # NEW: Attachment support
            )
        else:
            result = update_draft_tool(
                draft_id=created_draft_id,
                subject=updated_subject,
                body=updated_body
            )
        
        if result["success"]:
            print(f"   ✅ Draft updated successfully!")
            print(f"      Updated Draft ID: {result['draft_id']}")
            print(f"      Updated Message ID: {result['message_id']}")
        else:
            print(f"   ❌ Failed to update draft: {result['error']}")
    else:
        print("7. Updating Draft with Optional Attachments")
        print("   ⚠️  Skipped - No draft was created earlier")
    
    print("="*50 + "\n")
    
    # 8. Send draft (independent - creates its own draft and sends it)
    print("8. Send Draft (Create and Send)")
    create_draft_tool = gmail_toolkit.get_tool("create_draft")
    send_draft_tool = gmail_toolkit.get_tool("send_draft")
    
    send_to = "noreply@gmail.com"
    send_subject = "Test Status"
    send_body = "This is message from Evoagent X, Sending Drafting Feature is working."

    # Create a new draft for sending
    print(f"   Creating draft -> To: {send_to}, Subject: {send_subject}")
    create_result = create_draft_tool(
        to=send_to,
        subject=send_subject,
        body=send_body
    )

    if create_result["success"]:
        send_draft_id = create_result["draft_id"]
        print(f"   ✅ Draft created successfully! Draft ID: {send_draft_id}")
        
        # Send the draft
        print(f"   Sending draft ID: {send_draft_id}")
        send_result = send_draft_tool(draft_id=send_draft_id)
        if send_result["success"]:
            print(f"   ✅ Draft sent successfully!")
            print(f"      Message ID: {send_result['message_id']}")
            print(f"      Thread ID: {send_result['thread_id']}")
        else:
            print(f"   ❌ Failed to send draft: {send_result['error']}")
    else:
        print(f"   ❌ Failed to create draft for sending: {create_result['error']}")
    
    print("="*50 + "\n")

    # 9. Delete draft
    print("9. Delete Draft")
    delete_draft_tool = gmail_toolkit.get_tool("delete_draft")
    
    # Try to delete a draft - use the one we created earlier if available
    draft_to_delete_id = None
    if created_draft_id:
        draft_to_delete_id = created_draft_id
        print(f"   Example 1: Deleting draft we created earlier")
        print(f"   Draft ID: {draft_to_delete_id}")
    else:
        # If no draft was created, try to find one from the list
        print("   Example 1: Finding a draft to delete")
        list_drafts_tool = gmail_toolkit.get_tool("list_drafts")
        list_result = list_drafts_tool(max_results=1)
        if list_result.get("success") and list_result.get("drafts"):
            draft_to_delete_id = list_result["drafts"][0]["draft_id"]
            print(f"   Found draft to delete: {draft_to_delete_id}")
        else:
            print("   ⚠️  No drafts available to delete")
    
    if draft_to_delete_id:
        try:
            result = delete_draft_tool(draft_id=draft_to_delete_id)
            if result["success"]:
                print(f"   ✅ Draft deleted successfully!")
                print(f"      Deleted Draft ID: {result['draft_id']}")
                print(f"      Message: {result['message']}")
            else:
                print(f"   ❌ Failed to delete draft: {result['error']}")
        except Exception as e:
            print(f"   ⚠️  Error deleting draft: {e}")
    else:
        print("   ⚠️  Skipping delete - No draft ID available")
    
    print("="*50 + "\n")

    # 10. Trash messages (by query or explicit IDs)
    print("10. Trash Messages (by query or explicit IDs)")
    trash_messages_tool = gmail_toolkit.get_tool("trash_messages")
    
    # Example 1: Trash messages by search query
    print("   Example 1: Trash messages matching a search query")
    print("   (Using a safe query that likely won't match anything)")
    trash_query = 'subject:"SAFE_TO_DELETE_TEST_QUERY" is:read older_than:90d'
    print(f"   Query: {trash_query}")
    try:
        trash_res = trash_messages_tool(query=trash_query, max_results=5)
        if trash_res.get("success"):
            print(f"   ✅ Trash operation completed!")
            print(f"      Trashed IDs: {len(trash_res.get('trashed_ids', []))}")
            print(f"      Failed IDs: {len(trash_res.get('failed_ids', []))}")
            print(f"      Message: {trash_res.get('message', '')}")
        else:
            print(f"   ❌ Trash operation failed: {trash_res.get('error', 'Unknown error')}")
    except Exception as e:
        print(f"   ⚠️  Error during trash operation: {e}")
    
    # Example 2: Trash specific messages by ID (if we have IDs from earlier search)
    print("\n   Example 2: Trash specific messages by ID")
    try:
        search_res = search_messages_tool(keyword='from:noreply@medium.com', max_results=2)
        if search_res.get("success") and search_res.get("messages"):
            msg_ids = [m['id'] for m in search_res['messages'][:1]]  # Take just first one for safety
            print(f"   Found {len(msg_ids)} message(s) to trash")
            if msg_ids:
                print(f"   Trashing message IDs: {msg_ids}")
                trash_res = trash_messages_tool(message_ids=msg_ids)
                if trash_res.get("success"):
                    print(f"   ✅ Trashed {len(trash_res.get('trashed_ids', []))} message(s)")
                else:
                    print(f"   ❌ Trash failed: {trash_res.get('error', 'Unknown error')}")
        else:
            print("   No messages found for this demo trash example")
    except Exception as e:
        print(f"   ⚠️  Error during trash by ID operation: {e}")

    # 11. Modify labels (optional)
    print("="*50 + "\n")
    print("11. Modify Labels on Messages")
    modify_labels_tool = gmail_toolkit.get_tool("modify_labels")

    # ─── Example 1: Add STARRED label by search query ───
    print("\n   📌 Example 1: Search and Star Messages")
    print("   Step 1: Searching for messages from noreply@gmail.com (including attachments)...")
    try:
        search_res = search_messages_tool(keyword='from:noreply@gmail.com', max_results=5, include_attachments=True)
        if search_res.get('success') and search_res.get('messages'):
            ids = [m['id'] for m in search_res['messages']]
            print(f"   ✅ Found {len(ids)} message(s)")
            for i, msg in enumerate(search_res['messages'], 1):
                print(f"      {i}. Subject: {msg['subject']}")
                print(f"         From: {msg['from']}")
                # Display attachment info if present
                attach_count = msg.get('attachments_count', 0)
                if attach_count > 0:
                    print(f"         📎 Attachments: {attach_count}")
                    for att in msg.get('attachments', []):
                        print(f"            - {att['filename']} ({att.get('size', 0)} bytes)")
            
            print(f"\n   Step 2: Adding STARRED label to {len(ids)} message(s)...")
            mod_res = modify_labels_tool(message_ids=ids, add_labels=['STARRED'])
            
            if mod_res.get('success'):
                modified_count = len(mod_res.get('modified', []))
                failed_count = len(mod_res.get('failed', []))
                print(f"   ✅ Label modification completed!")
                print(f"      Modified: {modified_count}")
                print(f"      Failed: {failed_count}")
            else:
                print(f"   ❌ Label modification failed: {mod_res.get('error', 'Unknown error')}")
        else:
            print("   ⚠️  No messages found matching the search query")
    except Exception as e:
        print(f"   ❌ Error in Example 1: {e}")

    # ─── Example 2: Create and apply a custom label ───
    print("\n   📌 Example 2: Apply Custom Label")
    custom_label = "EvoAgentX-Test"
    print(f"   Step 1: Searching for messages matching subject query...")
    try:
        search_res = search_messages_tool(keyword='subject:"Test Draft from EvoAgentX"', max_results=5)
        if search_res.get('success') and search_res.get('messages'):
            ids = [m['id'] for m in search_res['messages']]
            print(f"   ✅ Found {len(ids)} message(s) matching the subject")
            for i, msg in enumerate(search_res['messages'], 1):
                print(f"      {i}. Subject: {msg['subject']}")
            
            print(f"\n   Step 2: Creating and applying custom label '{custom_label}'...")
            mod_res2 = modify_labels_tool(message_ids=ids, add_labels=[custom_label])
            
            if mod_res2.get('success'):
                modified_count = len(mod_res2.get('modified', []))
                failed_count = len(mod_res2.get('failed', []))
                print(f"   ✅ Custom label applied successfully!")
                print(f"      Modified: {modified_count}")
                print(f"      Failed: {failed_count}")
            else:
                print(f"   ❌ Custom label application failed: {mod_res2.get('error', 'Unknown error')}")
        else:
            print(f"   ⚠️  No messages found matching the subject query")
    except Exception as e:
        print(f"   ❌ Error in Example 2: {e}")

    # 12. Download attachments based on user input query
    print("="*50 + "\n")
    print("12. Download Attachments Based on Subject Query (User Input)")
    read_messages_tool = gmail_toolkit.get_tool("read_messages")
    
    print("   This example downloads attachments from emails matching a user query.")
    print("   For demo purposes, we'll search for emails with attachments.\n")
    
    # Simulated user input (in real scenario, use input())
    user_subject_query = input("   Enter a Gmail subject to search for (or press Enter for default 'has:attachment'): ").strip()
    if not user_subject_query:
        user_subject_query = "has:attachment"
    
    # User can specify custom download directory
    custom_download_dir = input("   Enter download directory (or press Enter for default './downloads'): ").strip()
    if not custom_download_dir:
        custom_download_dir = "./downloads"
    
    print(f"\n   Searching for emails with query: {user_subject_query}")
    print(f"   Download directory: {custom_download_dir}\n")
    
    try:
        # Download attachments based on user query
        # Note: include_attachments is auto-enabled when download_attachments=True
        result = read_messages_tool(
            query=user_subject_query,
            include_attachments=True,
            download_attachments=True,
            download_dir=custom_download_dir,
            max_results=5
        )
        
        if result["success"]:
            print(f"   Search completed!")
            print(f"   Found {result['messages_count']} message(s)\n")
            
            total_attachments_downloaded = 0
            
            for idx, msg in enumerate(result["messages"], 1):
                print(f"   Message {idx}:")
                print(f"      From: {msg['from']}")
                print(f"      Subject: {msg['subject']}")
                print(f"      Date: {msg['date']}")
                print(f"      Attachments in message: {msg.get('attachments_count', 0)}")
                
                # Show attachment details before download
                if msg.get('attachments_count', 0) > 0:
                    print(f"      📎 Attachment List:")
                    for att in msg.get('attachments', []):
                        print(f"         • {att['filename']} ({att.get('size', 0):,} bytes)")
                
                # Show downloaded attachment details
                if 'downloaded_attachments' in msg and msg['downloaded_attachments']:
                    print(f"      Downloaded files:")
                    for att in msg['downloaded_attachments']:
                        print(f"         • {att['original_filename']}")
                        print(f"           Saved as: {att['saved_filename']}")
                        print(f"           Size: {att['size']:,} bytes")
                        print(f"           Path: {att['file_path']}")
                        total_attachments_downloaded += 1
                else:
                    if msg.get('attachments_count', 0) > 0:
                        print(f"      ⚠️  Attachments found but download was skipped")
                
                print()
            
            print(f"   📊 Summary:")
            print(f"      Total attachments downloaded: {total_attachments_downloaded}")
            print(f"      Save location: {custom_download_dir}")
            
            if os.path.exists(custom_download_dir):
                files_in_dir = os.listdir(custom_download_dir)
                print(f"      Files in directory: {len(files_in_dir)}")
        else:
            print(f"   ❌ Search failed: {result.get('error', 'Unknown error')}")
    except Exception as e:
        print(f"   ⚠️  Error during attachment download: {e}")

    print("\n=== Demo Complete ===")
    print("Gmail toolkit features demonstrated:")
    print("✅ Read Messages - with query and filtering")
    print("✅ Search Messages - Gmail search syntax support")
    print("✅ Read Threads - conversation view")
    print("✅ List Drafts - view all drafts")
    print("✅ Create Draft - create new email drafts (optional attachment)")
    print("✅ Search/Retrieve Draft - get draft details by ID or keyword")
    print("✅ Update Draft - modify existing drafts (with attachments)")
    print("✅ Send Draft - send drafts as emails")
    print("✅ Delete Draft - remove draft by ID")
    print("✅ Trash Messages - move messages to trash by query or ID")
    print("✅ Modify Labels - add/remove/create labels on messages")
    print("✅ Download Attachments - download files from emails by subject query")

if __name__ == "__main__":
    main()

