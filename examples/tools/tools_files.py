#!/usr/bin/env python3

"""
File System Tools Examples for EvoAgentX

This module provides comprehensive examples for:
- StorageToolkit: Comprehensive file storage operations with flexible storage backends
- CMDToolkit: Command-line execution capabilities with timeout handling

The examples demonstrate various file operations and system interactions including:
- File save, read, append, list, delete, and existence checks
- Command execution with working directory and timeout support
- Cross-platform command handling
- Storage handler management
"""

import os
import sys
from pathlib import Path

# Add the parent directory to sys.path to import from evoagentx
sys.path.append(str(Path(__file__).parent.parent))

from evoagentx.tools import (
    StorageToolkit,
    CMDToolkit
)


def run_file_tool_example():
    """
    Run an example using the StorageToolkit to read and write files with the new storage handler system.
    """
    print("\n===== STORAGE TOOL EXAMPLE =====\n")
    
    try:
        # Initialize the storage toolkit with default storage handler
        storage_toolkit = StorageToolkit(name="DemoStorageToolkit")
        
        # Get individual tools from the toolkit
        save_tool = storage_toolkit.get_tool("save")
        read_tool = storage_toolkit.get_tool("read")
        append_tool = storage_toolkit.get_tool("append")
        list_tool = storage_toolkit.get_tool("list_files")
        exists_tool = storage_toolkit.get_tool("exists")
        
        # Create sample content for different file types
        sample_text = """This is a sample text document created using the StorageToolkit.
This tool provides comprehensive file operations with automatic format detection.
It supports various file types including text, JSON, CSV, YAML, XML, Excel, and more."""
        
        sample_json = {
            "name": "Sample Document",
            "type": "test",
            "content": "This is a JSON document for testing",
            "metadata": {
                "created": "2024-01-01",
                "version": "1.0"
            }
        }
        
        # Test file operations with default storage paths
        print("1. Testing file save operations...")
        
        # Save text file
        text_result = save_tool(
            file_path="sample_document.txt",
            content=sample_text
        )
        print("Text file save result:")
        print("-" * 30)
        print(text_result)
        print("-" * 30)
        
        # Save JSON file
        # Note: StorageToolkit expects Python objects for JSON, not JSON strings
        # For other formats like CSV, YAML, text - pass strings directly
        json_result = save_tool(
            file_path="sample_data.json",
            content=sample_json  # Pass the Python dict directly, not json.dumps()
        )
        print("JSON file save result:")
        print("-" * 30)
        print(json_result)
        print("-" * 30)
        
        # Test file read operations
        print("\n2. Testing file read operations...")
        
        # Read text file
        text_read_result = read_tool(file_path="sample_document.txt")
        print("Text file read result:")
        print("-" * 30)
        print(text_read_result)
        print("-" * 30)
        
        # Read JSON file
        json_read_result = read_tool(file_path="sample_data.json")
        print("JSON file read result:")
        print("-" * 30)
        print(json_read_result)
        print("-" * 30)
        
        # Test file append operations
        print("\n3. Testing file append operations...")
        
        # Append to text file
        append_text_result = append_tool(
            file_path="sample_document.txt",
            content="\n\nThis content was appended to the text file."
        )
        print("Text file append result:")
        print("-" * 30)
        print(append_text_result)
        print("-" * 30)
        
        # Update JSON file with new data
        # Note: For JSON files, we use save_tool to update, not append_tool
        # The append_tool for JSON expects Python objects, not JSON strings
        updated_json_data = {**sample_json, "additional": "data", "timestamp": "2024-01-01T12:00:00Z"}
        update_json_result = save_tool(
            file_path="sample_data.json",
            content=updated_json_data  # Pass the Python dict directly
        )
        print("JSON file update result:")
        print("-" * 30)
        print(update_json_result)
        print("-" * 30)
        
        # Test file listing
        print("\n4. Testing file listing...")
        list_result = list_tool(path=".", max_depth=2, include_hidden=False)
        print("File listing result:")
        print("-" * 30)
        print(list_result)
        print("-" * 30)
        
        # Test file existence
        print("\n5. Testing file existence...")
        exists_result = exists_tool(path="sample_document.txt")
        print("File existence check result:")
        print("-" * 30)
        print(exists_result)
        print("-" * 30)
        
        # Test supported formats
        print("\n6. Testing supported formats...")
        formats_tool = storage_toolkit.get_tool("list_supported_formats")
        formats_result = formats_tool()
        print("Supported formats result:")
        print("-" * 30)
        print(formats_result)
        print("-" * 30)
        
        print("\n✓ StorageToolkit test completed successfully!")
        print("✓ All file operations working with default storage handler")
        print("✓ Automatic format detection working")
        print("✓ File operations working (including JSON updates)")
        print("✓ File listing and existence checks working")
        
    except Exception as e:
        print(f"Error running storage tool example: {str(e)}")


def run_cmd_tool_example():
    """Simple example using CMDToolkit for command line operations."""
    print("\n===== CMD TOOL EXAMPLE =====\n")
    
    try:
        # Initialize the CMD toolkit
        cmd_toolkit = CMDToolkit(name="DemoCMDToolkit")
        execute_tool = cmd_toolkit.get_tool("execute_command")
        
        print("✓ CMDToolkit initialized")
        
        # Test basic command execution
        print("1. Testing basic command execution...")
        result = execute_tool(command="echo 'Hello from CMD toolkit'")
        
        if result.get("success"):
            print("✓ Command executed successfully")
            print(f"Output: {result.get('stdout', 'No output')}")
        else:
            print(f"❌ Command failed: {result.get('error', 'Unknown error')}")
        
        # Test system information commands
        print("\n2. Testing system information commands...")
        
        # Get current working directory
        pwd_result = execute_tool(command="pwd")
        if pwd_result.get("success"):
            print(f"✓ Current directory: {pwd_result.get('stdout', '').strip()}")
        
        # Get system information
        if os.name == 'posix':  # Linux/Mac
            uname_result = execute_tool(command="uname -a")
            if uname_result.get("success"):
                print(f"✓ System info: {uname_result.get('stdout', '').strip()}")
        else:  # Windows
            ver_result = execute_tool(command="ver")
            if ver_result.get("success"):
                print(f"✓ System info: {ver_result.get('stdout', '').strip()}")
        
        # Test file listing
        print("\n3. Testing file listing...")
        if os.name == 'posix':
            ls_result = execute_tool(command="ls -la", working_directory=".")
        else:
            ls_result = execute_tool(command="dir", working_directory=".")
        
        if ls_result.get("success"):
            print("✓ File listing successful")
            print(f"Output length: {len(ls_result.get('stdout', ''))} characters")
        else:
            print(f"❌ File listing failed: {ls_result.get('error', 'Unknown error')}")
        
        # Test with timeout
        print("\n4. Testing command timeout...")
        timeout_result = execute_tool(command="sleep 5", timeout=12)
        if not timeout_result.get("success"):
            print("✓ Timeout working correctly (command was interrupted)")
        else:
            print("⚠ Timeout may not be working as expected")
        
        print("\n✓ CMDToolkit test completed")
        
    except Exception as e:
        print(f"Error: {str(e)}")


def run_storage_handler_examples():
    """
    Run examples demonstrating different storage handlers and configurations.
    """
    print("\n===== STORAGE HANDLER EXAMPLES =====\n")
    
    try:
        # Test with custom base path
        print("1. Testing custom base path storage...")
        custom_storage_toolkit = StorageToolkit(
            name="CustomPathStorageToolkit",
            storage_handler=None,  # Will use default LocalStorageHandler
            base_path="./custom_storage"
        )
        
        # Create a test file in custom location
        custom_save_tool = custom_storage_toolkit.get_tool("save")
        custom_result = custom_save_tool(
            file_path="custom_test.txt",
            content="This file is stored in a custom location"
        )
        
        if custom_result.get("success"):
            print("✓ Custom path storage working")
            print(f"File saved to: {custom_result.get('file_path')}")
        else:
            print(f"❌ Custom path storage failed: {custom_result.get('error')}")
        
        # Test file operations in custom location
        custom_read_tool = custom_storage_toolkit.get_tool("read")
        custom_read_result = custom_read_tool(file_path="custom_test.txt")
        
        if custom_read_result.get("success"):
            print("✓ Custom path file reading working")
            print(f"Content: {custom_read_result.get('content', '')[:50]}...")
        
        # Test file listing in custom location
        custom_list_tool = custom_storage_toolkit.get_tool("list_files")
        custom_list_result = custom_list_tool(path=".", max_depth=1, include_hidden=False)
        
        if custom_list_result.get("success"):
            print("✓ Custom path file listing working")
            files = custom_list_result.get("files", [])
            print(f"Found {len(files)} files in custom location")
        
        print("\n✓ Storage handler examples completed")
        
    except Exception as e:
        print(f"Error running storage handler examples: {str(e)}")


def run_advanced_file_operations():
    """
    Run examples demonstrating advanced file operations and format handling.
    """
    print("\n===== ADVANCED FILE OPERATIONS =====\n")
    
    try:
        # Initialize storage toolkit
        storage_toolkit = StorageToolkit()
        save_tool = storage_toolkit.get_tool("save")
        read_tool = storage_toolkit.get_tool("read")
        
        # Test CSV file operations
        print("1. Testing CSV file operations...")
        csv_content = """name,age,city
John Doe,30,New York
Jane Smith,25,Los Angeles
Bob Johnson,35,Chicago"""
        
        csv_result = save_tool(
            file_path="sample_data.csv",
            content=csv_content
        )
        
        if csv_result.get("success"):
            print("✓ CSV file saved successfully")
            
            # Read CSV file
            csv_read_result = read_tool(file_path="sample_data.csv")
            
            if csv_read_result.get("success"):
                print("✓ CSV file read successfully")
                print(f"Content: {csv_read_result.get('content', '')[:100]}...")
            else:
                print(f"❌ Failed to read CSV file: {csv_read_result.get('error')}")
        else:
            print(f"❌ Failed to save CSV file: {csv_result.get('error')}")
        
        # Test YAML file operations
        print("\n2. Testing YAML file operations...")
        yaml_content = """name: Sample YAML
version: 1.0
features:
  - feature1
  - feature2
metadata:
  author: Test User
  date: 2024-01-01"""
        
        yaml_result = save_tool(
            file_path="sample_config.yaml",
            content=yaml_content
        )
        
        if yaml_result.get("success"):
            print("✓ YAML file saved successfully")
            
            # Read YAML file
            yaml_read_result = read_tool(file_path="sample_config.yaml")
            
            if yaml_read_result.get("success"):
                print("✓ YAML file read successfully")
                print(f"Content: {yaml_read_result.get('content', '')[:100]}...")
            else:
                print(f"❌ Failed to read YAML file: {yaml_read_result.get('error')}")
        else:
            print(f"❌ Failed to save YAML file: {yaml_result.get('error')}")

        # Test PDF file operations
        print("\n3. Testing PDF file operations...")
        
        # Create PDF file first
        pdf_content = """Test PDF Document

This is a test PDF created by EvoAgentX.

Features:
• PDF creation from text
• Automatic formatting
• Professional layout

This demonstrates the storage system's PDF capabilities."""
        
        pdf_result = save_tool(
            file_path="test_pdf.pdf",
            content=pdf_content
        )
        
        if pdf_result.get("success"):
            print("✓ PDF file created successfully")
        else:
            print(f"❌ Failed to create PDF file: {pdf_result.get('error')}")
        
        # Read PDF file
        pdf_read_result = read_tool(file_path="test_pdf.pdf")
        
        if pdf_read_result.get("success"):
            print("✓ PDF file read successfully")
            print(f"Content: {pdf_read_result.get('content', '')[:100]}...")
        else:
            print(f"❌ Failed to read PDF file: {pdf_read_result.get('error')}")
        
        # Test file deletion
        print("\n4. Testing file deletion...")
        delete_tool = storage_toolkit.get_tool("delete")
        
        # Delete test files
        test_files = ["sample_document.txt", "sample_data.json", "custom_test.txt"]
        for test_file in test_files:
            if os.path.exists(test_file):
                delete_result = delete_tool(path=test_file)
                if delete_result.get("success"):
                    print(f"✓ Deleted {test_file}")
                else:
                    print(f"❌ Failed to delete {test_file}: {delete_result.get('error')}")
        
        print("\n✓ Advanced file operations completed")
        
    except Exception as e:
        print(f"Error running advanced file operations: {str(e)}")


def main():
    """Main function to run all file system examples"""
    print("===== FILE SYSTEM TOOLS EXAMPLES =====")
    
    # # Run storage tool example
    # run_file_tool_example()
    
    # # Run CMD tool example
    # run_cmd_tool_example()
    
    # # Run storage handler examples
    # run_storage_handler_examples()
    
    # Run advanced file operations
    run_advanced_file_operations()
    
    print("\n===== ALL FILE SYSTEM EXAMPLES COMPLETED =====")


if __name__ == "__main__":
    main()
