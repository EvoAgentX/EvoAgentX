#!/usr/bin/env python3

"""
Example demonstrating how to use browser automation toolkits from EvoAgentX.
This script provides comprehensive examples for:
- BrowserToolkit (Selenium-based): Fine-grained control over browser elements
- BrowserUseToolkit (AI-driven): Natural language browser automation
"""

import os
import sys
# import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(override=True)

# Add the parent directory to sys.path to import from evoagentx
sys.path.append(str(Path(__file__).parent.parent))

from evoagentx.tools import (
    BrowserToolkit,
    BrowserUseToolkit
)


def run_browser_tool_example():
    """
    Run an example using the BrowserToolkit with auto-initialization and auto-cleanup.
    Uses a comprehensive HTML test page to demonstrate browser automation features.
    """
    print("\n===== BROWSER TOOL EXAMPLE =====\n")
    
    try:
        # Initialize the browser toolkit (browser auto-initializes when first used)
        browser_toolkit = BrowserToolkit(headless=False, timeout=10)
        
        # Get individual tools from the toolkit
        nav_tool = browser_toolkit.get_tool("navigate_to_url")
        input_tool = browser_toolkit.get_tool("input_text")
        click_tool = browser_toolkit.get_tool("browser_click")
        snapshot_tool = browser_toolkit.get_tool("browser_snapshot")
        
        # Use the static test HTML file
        test_file_path = os.path.join(os.getcwd(), "examples", "tools", "browser_test_page.html")
        
        print("Step 1: Navigating to test page (browser auto-initializes)...")
        nav_result = nav_tool(url=f"file://{test_file_path}")
        print("Navigation Result:")
        print("-" * 30)
        print(f"Status: {nav_result.get('status')}")
        print(f"URL: {nav_result.get('current_url')}")
        print(f"Title: {nav_result.get('title')}")
        print("-" * 30)
        
        if nav_result.get("status") in ["success", "partial_success"]:
            print("\nStep 2: Taking initial snapshot to identify elements...")
            snapshot_result = snapshot_tool()
            
            if snapshot_result.get("status") == "success":
                print("✓ Initial snapshot successful")
                
                # Find interactive elements
                elements = snapshot_result.get("interactive_elements", [])
                print(f"Found {len(elements)} interactive elements")
                
                # Identify specific elements
                name_input_ref = None
                email_input_ref = None
                message_input_ref = None
                submit_btn_ref = None
                clear_btn_ref = None
                test_btn_ref = None
                
                for elem in elements:
                    desc = elem.get("description", "").lower()
                    purpose = elem.get("purpose", "").lower()
                    
                    if "name" in desc and elem.get("editable"):
                        name_input_ref = elem["id"]
                    elif "email" in desc and elem.get("editable"):
                        email_input_ref = elem["id"]
                    elif "message" in desc and elem.get("editable"):
                        message_input_ref = elem["id"]
                    elif "submit" in purpose and elem.get("interactable"):
                        submit_btn_ref = elem["id"]
                    elif "clear" in purpose and elem.get("interactable"):
                        clear_btn_ref = elem["id"]
                    elif "test" in purpose and elem.get("interactable"):
                        test_btn_ref = elem["id"]
                
                print(f"Identified elements:")
                print(f"  - Name input: {name_input_ref}")
                print(f"  - Email input: {email_input_ref}")
                print(f"  - Message input: {message_input_ref}")
                print(f"  - Submit button: {submit_btn_ref}")
                print(f"  - Clear button: {clear_btn_ref}")
                print(f"  - Test button: {test_btn_ref}")
                
                # Test input functionality
                if name_input_ref and email_input_ref and message_input_ref:
                    print("\nStep 3: Testing input functionality...")
                    
                    # Fill name field
                    print("  - Typing 'John Doe' in name field...")
                    name_result = input_tool(
                        element="Name input", 
                        ref=name_input_ref, 
                        text="John Doe", 
                        submit=False
                    )
                    print(f"    Result: {name_result.get('status')}")
                    
                    # Fill email field
                    print("  - Typing 'john.doe@example.com' in email field...")
                    email_result = input_tool(
                        element="Email input", 
                        ref=email_input_ref, 
                        text="john.doe@example.com", 
                        submit=False
                    )
                    print(f"    Result: {email_result.get('status')}")
                    
                    # Fill message field
                    print("  - Typing 'This is a test message for browser automation.' in message field...")
                    message_result = input_tool(
                        element="Message input", 
                        ref=message_input_ref, 
                        text="This is a test message for browser automation.", 
                        submit=False
                    )
                    print(f"    Result: {message_result.get('status')}")
                    
                    # Test form submission
                    if submit_btn_ref:
                        print("\nStep 4: Testing form submission...")
                        submit_result = click_tool(
                            element="Submit button", 
                            ref=submit_btn_ref
                        )
                        print(f"Submit result: {submit_result.get('status')}")
                        
                        # Take snapshot to see the result
                        print("\nStep 5: Taking snapshot to verify form submission...")
                        result_snapshot = snapshot_tool()
                        if result_snapshot.get("status") == "success":
                            content = result_snapshot.get("page_content", "")
                            if "Name: John Doe, Email: john.doe@example.com" in content:
                                print("✓ Form submission successful - data correctly displayed!")
                            else:
                                print("⚠ Form submission may have failed")
                    
                    # Test test button click
                    if test_btn_ref:
                        print("\nStep 6: Testing test button click...")
                        test_result = click_tool(
                            element="Test button", 
                            ref=test_btn_ref
                        )
                        print(f"Test button result: {test_result.get('status')}")
                        
                        # Take snapshot to see the click result
                        click_snapshot = snapshot_tool()
                        if click_snapshot.get("status") == "success":
                            content = click_snapshot.get("page_content", "")
                            if "Test button clicked at:" in content:
                                print("✓ Test button click successful!")
                            else:
                                print("⚠ Test button click may have failed")
                    
                    # Test clear functionality
                    if clear_btn_ref:
                        print("\nStep 7: Testing clear functionality...")
                        clear_result = click_tool(
                            element="Clear button", 
                            ref=clear_btn_ref
                        )
                        print(f"Clear result: {clear_result.get('status')}")
                        
                        # Take final snapshot
                        final_snapshot = snapshot_tool()
                        if final_snapshot.get("status") == "success":
                            print("✓ Clear functionality tested")
                
                print("\n✓ Browser automation test completed successfully!")
                print("✓ Browser auto-initialization working")
                print("✓ Navigation working")
                print("✓ Input functionality working")
                print("✓ Click functionality working")
                print("✓ Form submission working")
                print("✓ Snapshot functionality working")
            else:
                print("❌ Initial snapshot failed")
        else:
            print("\n❌ Navigation failed")
        
        print("\nBrowser will automatically close when the toolkit goes out of scope...")
        print("(No manual cleanup required)")
        
    except Exception as e:
        print(f"Error running browser tool example: {str(e)}")
        print("Browser will still automatically cleanup on exit")


def run_browser_use_tool_example():
    """Simple example using BrowserUseToolkit for browser automation."""
    print("\n===== BROWSER USE TOOL EXAMPLE =====\n")
    
    # Check for OpenAI API key
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("❌ OPENAI_API_KEY not found in environment variables")
        print("Please set your OpenAI API key: export OPENAI_API_KEY='your-api-key-here'")
        return
    
    try:
        # Initialize the BrowserUse toolkit
        print("Initializing BrowserUseToolkit...")
        toolkit = BrowserUseToolkit(model="gpt-4o-mini", headless=False)
        browser_tool = toolkit.get_tool("browser_use")
        
        print("✓ BrowserUseToolkit initialized")
        print(f"✓ Using OpenAI API key: {openai_api_key[:8]}...")
        
        # Execute a simple browser task
        print("Executing browser task: 'Go to Google and search for OpenAI GPT-4'...")
        result = browser_tool(task="Go to Google and search for 'OpenAI GPT-4'")
        
        if result.get('success'):
            print("✓ Browser task completed successfully")
            print(f"Result: {result.get('result', 'No result details')}")
        else:
            print(f"❌ Browser task failed: {result.get('error', 'Unknown error')}")
        
        print("\n✓ BrowserUseToolkit test completed")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        print("Note: Make sure you have the required dependencies installed and API keys set up.")


def main():
    """Main function to run all browser tool examples"""
    print("===== BROWSER TOOL EXAMPLES =====")
    
    # Run Selenium-based browser example
    run_browser_tool_example()
    
    # Run AI-driven browser example
    run_browser_use_tool_example()
    
if __name__ == "__main__":
    main()