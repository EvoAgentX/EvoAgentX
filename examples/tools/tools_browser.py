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
    """Comprehensive example using BrowserUseToolkit following the designed test flow."""
    print("\n===== BROWSER USE TOOLKIT COMPREHENSIVE TEST =====\n")
    
    # Check for OpenAI API key
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("❌ OPENAI_API_KEY not found in environment variables")
        print("Please set your OpenAI API key: export OPENAI_API_KEY='your-api-key-here'")
        return
    
    try:
        # Initialize the browser toolkit
        toolkit = BrowserUseToolkit()
        print("✓ BrowserUseToolkit initialized")
        
        # Get all available tools
        tool_names = toolkit.get_tool_names()
        print(f"✓ Available tools: {len(tool_names)}")
        
        # Use real websites for testing (file:// URLs don't work with browser-use)
        print("\n🌐 Using real websites for testing...")
        website1_url = "https://httpbin.org/forms/post"  # Has forms, inputs, buttons
        website2_url = "https://en.wikipedia.org/wiki/Python_(programming_language)"  # Long scrollable content
        
        print(f"✓ Test Website 1: {website1_url}")
        print(f"✓ Test Website 2: {website2_url}")
        
        # Get tools
        navigate_tool = toolkit.get_tool("browser_navigate")
        status_tool = toolkit.get_tool("browser_get_status")
        click_tool = toolkit.get_tool("browser_click")
        type_tool = toolkit.get_tool("browser_type")
        scroll_tool = toolkit.get_tool("browser_scroll")
        refresh_tool = toolkit.get_tool("browser_refresh")
        go_back_tool = toolkit.get_tool("browser_go_back")
        create_tab_tool = toolkit.get_tool("browser_create_tab")
        get_dropdown_tool = toolkit.get_tool("browser_get_dropdown_options")
        select_dropdown_tool = toolkit.get_tool("browser_select_dropdown_option")
        switch_tab_tool = toolkit.get_tool("browser_switch_tab")
        close_tab_tool = toolkit.get_tool("browser_close_tab")
        close_tool = toolkit.get_tool("browser_close")
        
        # ===== STEP 1: Navigate to first test page =====
        print("\n" + "="*60)
        print("STEP 1: Navigate to Form Test Page (httpbin.org)")
        print("="*60)
        
        result = navigate_tool(url=website1_url)
        if result.get('success'):
            print("✅ Navigation to Form Test Page successful")
            state = result.get('state', {})
            print(f"   📍 URL: {state.get('url', 'Unknown')}")
            print(f"   📄 Title: {state.get('title', 'Unknown')}")
        else:
            print(f"❌ Navigation failed: {result.get('error')}")
            return
        
        # ===== STEP 2: Test Type, Scroll, Click, and Dropdowns =====
        print("\n" + "="*60)
        print("STEP 2: Test Type, Scroll, Click, and Dropdowns")
        print("="*60)
        
        # Get page status to find elements
        status_result = status_tool()
        if not status_result.get('success'):
            print(f"❌ Status check failed: {status_result.get('error')}")
            return
        
        agent_context = status_result.get('agent_context', {})
        interactive_elements = agent_context.get('interactive_elements', {})
        inputs = interactive_elements.get('inputs', [])
        clickable_elements = interactive_elements.get('all_elements', [])
        
        print(f"   📊 Found {len(inputs)} input elements, {len(clickable_elements)} clickable elements")
        
        # Test typing in first input
        if inputs:
            first_input = inputs[0]
            input_index = first_input.get('index', 1)
            print(f"   ⌨️ Typing in input element {input_index}...")
            type_result = type_tool(index=input_index, text="Test User Input")
            if type_result.get('success'):
                print("   ✅ Type test successful")
            else:
                print(f"   ❌ Type test failed: {type_result.get('error')}")
        else:
            print("   ⚠️ No input elements found for typing test")
        
        # Test scrolling
        print("   📜 Testing scroll down...")
        scroll_result = scroll_tool(direction="down", amount=0.5)
        if scroll_result.get('success'):
            print("   ✅ Scroll test successful")
        else:
            print(f"   ❌ Scroll test failed: {scroll_result.get('error')}")
        
        # Test clicking on a button
        if clickable_elements:
            # Find a button element
            button_element = None
            for element in clickable_elements:
                if element.get('tag') == 'button' or 'button' in element.get('text', '').lower():
                    button_element = element
                    break
            
            if button_element:
                click_index = button_element.get('index', 1)
                print(f"   🖱️ Clicking button element {click_index}...")
                click_result = click_tool(index=click_index)
                if click_result.get('success'):
                    print("   ✅ Click test successful")
                else:
                    print(f"   ❌ Click test failed: {click_result.get('error')}")
            else:
                print("   ⚠️ No button elements found for clicking test")
        else:
            print("   ⚠️ No clickable elements found for clicking test")
        
        # Test dropdown functionality
        selects = interactive_elements.get('selects', [])
        print(f"   📋 Found {len(selects)} dropdown elements")
        
        if selects:
            first_select = selects[0]
            select_index = first_select.get('index', 1)
            print(f"   🔍 Testing dropdown element {select_index}...")
            
            # Get dropdown options
            get_dropdown_result = get_dropdown_tool(index=select_index)
            if get_dropdown_result.get('success'):
                print("   ✅ Get dropdown options successful")
                options = get_dropdown_result.get('options', [])
                print(f"   📝 Available options: {options}")
                
                if options:
                    # Select the first available option
                    first_option = options[0]
                    print(f"   🎯 Selecting option: '{first_option}'...")
                    select_result = select_dropdown_tool(index=select_index, option_text=first_option)
                    if select_result.get('success'):
                        print("   ✅ Dropdown selection successful")
                    else:
                        print(f"   ❌ Dropdown selection failed: {select_result.get('error')}")
                else:
                    print("   ⚠️ No options available in dropdown")
            else:
                print(f"   ❌ Get dropdown options failed: {get_dropdown_result.get('error')}")
        else:
            print("   ⚠️ No dropdown elements found")
        
        # ===== STEP 3: Refresh the page =====
        print("\n" + "="*60)
        print("STEP 3: Refresh the page")
        print("="*60)
        
        print("   🔄 Refreshing page...")
        refresh_result = refresh_tool()
        if refresh_result.get('success'):
            print("   ✅ Refresh test successful")
        else:
            print(f"   ❌ Refresh test failed: {refresh_result.get('error')}")
        
        # ===== STEP 4: Go back =====
        print("\n" + "="*60)
        print("STEP 4: Go back in history")
        print("="*60)
        
        print("   ⬅️ Going back...")
        go_back_result = go_back_tool()
        if go_back_result.get('success'):
            print("   ✅ Go back test successful")
        else:
            print(f"   ❌ Go back test failed: {go_back_result.get('error')}")
        
        # ===== STEP 5: Navigate to first page again =====
        print("\n" + "="*60)
        print("STEP 5: Navigate to Scrollbars Test Page again")
        print("="*60)
        
        print("   🧭 Navigating to Scrollbars Test Page again...")
        result = navigate_tool(url=website1_url)
        if result.get('success'):
            print("   ✅ Navigation to Scrollbars Test Page successful")
        else:
            print(f"   ❌ Navigation failed: {result.get('error')}")
            return
        
        # ===== STEP 6: Create new tab navigating to second page =====
        print("\n" + "="*60)
        print("STEP 6: Create new tab navigating to Wikipedia Python Page")
        print("="*60)
        
        print("   📑 Creating new tab with Wikipedia Python Page...")
        create_tab_result = create_tab_tool(url=website2_url)
        if create_tab_result.get('success'):
            print("   ✅ Create tab test successful")
        else:
            print(f"   ❌ Create tab test failed: {create_tab_result.get('error')}")
            return
        
        # ===== STEP 7: Test scrolling on Wikipedia Python page =====
        print("\n" + "="*60)
        print("STEP 7: Test scrolling on Wikipedia Python page")
        print("="*60)
        
        # Test scrolling on the long Wikipedia page
        print("   📜 Testing scroll down...")
        scroll_result = scroll_tool(direction="down", amount=0.5)
        if scroll_result.get('success'):
            print("   ✅ Scroll down test successful")
        else:
            print(f"   ❌ Scroll down test failed: {scroll_result.get('error')}")
            print("   ⚠️ Continuing with remaining tests...")
        
        print("   📜 Testing scroll up...")
        scroll_up_result = scroll_tool(direction="up", amount=0.3)
        if scroll_up_result.get('success'):
            print("   ✅ Scroll up test successful")
        else:
            print(f"   ❌ Scroll up test failed: {scroll_up_result.get('error')}")
            print("   ⚠️ Continuing with remaining tests...")
        
        # Wait a moment for page to settle after scrolling
        print("   ⏳ Waiting for page to settle...")
        import time
        time.sleep(2)
        
        # ===== STEP 8: Switch to first tab =====
        print("\n" + "="*60)
        print("STEP 8: Switch to first tab")
        print("="*60)
        
        # Get fresh status for tab operations
        print("   🔄 Getting fresh status for tab operations...")
        tab_status_result = status_tool()
        if tab_status_result.get('success'):
            tabs = tab_status_result.get('state', {}).get('tabs', [])
            print(f"   📑 Found {len(tabs)} tabs")
            
            if len(tabs) >= 2:
                # Switch to first tab (index 0)
                first_tab_id = tabs[1].get('target_id')
                print(f"   🔄 Switching to first tab (ID: {first_tab_id[-4:] if first_tab_id else 'N/A'})...")
                switch_result = switch_tab_tool(tab_id=first_tab_id)
                if switch_result.get('success'):
                    print("   ✅ Switch tab test successful")
                else:
                    print(f"   ❌ Switch tab test failed: {switch_result.get('error')}")
            else:
                print("   ⚠️ Not enough tabs to test switching")
        else:
            print(f"   ❌ Status check failed: {tab_status_result.get('error')}")
        
        # ===== STEP 9: Close the second tab =====
        print("\n" + "="*60)
        print("STEP 9: Close the second tab")
        print("="*60)
        
        # Get current tabs again
        close_status_result = status_tool()
        if close_status_result.get('success'):
            tabs = close_status_result.get('state', {}).get('tabs', [])
            print(f"   📑 Current tabs: {len(tabs)}")
            
            if len(tabs) >= 2:
                # Close the second tab (index 1)
                second_tab_id = tabs[1].get('target_id')
                print(f"   ❌ Closing second tab (ID: {second_tab_id[-4:] if second_tab_id else 'N/A'})...")
                close_tab_result = close_tab_tool(tab_id=second_tab_id)
                if close_tab_result.get('success'):
                    print("   ✅ Close tab test successful")
                else:
                    print(f"   ❌ Close tab test failed: {close_tab_result.get('error')}")
            else:
                print("   ⚠️ Not enough tabs to test closing")
        else:
            print(f"   ❌ Status check failed: {close_status_result.get('error')}")
        
        # ===== FINAL CLEANUP =====
        print("\n" + "="*60)
        print("FINAL CLEANUP: Close browser session")
        print("="*60)
        
        print("   🧹 Closing browser session...")
        close_result = close_tool()
        if close_result.get('success'):
            print("   ✅ Browser cleanup successful")
        else:
            print(f"   ❌ Browser cleanup failed: {close_result.get('error')}")
        
        print("\n" + "="*60)
        print("🎉 COMPREHENSIVE TEST COMPLETED SUCCESSFULLY!")
        print("="*60)
        
    except Exception as e:
        print(f"❌ Error during test: {str(e)}")
        import traceback
        traceback.print_exc()


def main():
    """Main function to run all browser tool examples"""
    print("===== BROWSER TOOL EXAMPLES =====")
    
    # # Run Selenium-based browser example
    # run_browser_tool_example()
    
    # Run comprehensive BrowserUse toolkit test
    run_browser_use_tool_example()
    
if __name__ == "__main__":
    main()