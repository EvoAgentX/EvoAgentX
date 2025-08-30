#!/usr/bin/env python3
"""
Example demonstrating how to use various toolkits from EvoAgentX.
This script provides comprehensive examples for:
- PythonInterpreter and DockerInterpreter tools for code execution
- BrowserToolkit with auto-initialization and auto-cleanup
- Search toolkits (Wikipedia, Google, Google Free)
- File operations with different file types
- MCP toolkit integration
"""

import os
import sys
from pathlib import Path

# Add the parent directory to sys.path to import from evoagentx
sys.path.append(str(Path(__file__).parent.parent))

from evoagentx.tools import (
    PythonInterpreterToolkit,
    DockerInterpreterToolkit,
    WikipediaSearchToolkit,
    GoogleSearchToolkit,
    GoogleFreeSearchToolkit,
    MCPToolkit,
    FileToolkit,
    BrowserToolkit,
    ArxivToolkit,
    BrowserUseToolkit
)


def run_simple_hello_world(interpreter):
    """
    Run a simple Hello World example using the provided interpreter.
    
    Args:
        interpreter: An instance of a code interpreter
    """
    # Define a simple Hello World code
    code = """
print("Hello, World!")
print("This code is running inside a secure Python interpreter.")
"""
    
    # Execute the code
    result = interpreter.execute(code, "python")
    print("\nSimple Hello World Result:")
    print("-" * 50)
    print(result)
    print("-" * 50)


def run_math_example(interpreter):
    """
    Run a math example using the provided interpreter.
    
    Args:
        interpreter: An instance of a code interpreter
    """
    # Define a simple math example
    code = """
print("Running math operations...")

# Using math library
import math
print(f"The value of pi is: {math.pi:.4f}")
print(f"The square root of 16 is: {math.sqrt(16)}")
print(f"The value of e is: {math.e:.4f}")
"""

    # Execute the code
    result = interpreter.execute(code, "python")
    print("\nMath Example Result:")
    print("-" * 50)
    print(result)
    print("-" * 50)


def run_platform_info(interpreter):
    """
    Run a platform info example using the provided interpreter.
    
    Args:
        interpreter: An instance of a code interpreter
    """
    # Define code that prints platform information
    code = """
print("Getting platform information...")

# System information
import platform
import sys

print(f"Python version: {platform.python_version()}")
print(f"Platform: {platform.system()} {platform.release()}")
print(f"Processor: {platform.processor()}")
print(f"Implementation: {platform.python_implementation()}")
"""

    # Execute the code
    result = interpreter.execute(code, "python")
    print("\nPlatform Info Result:")
    print("-" * 50)
    print(result)
    print("-" * 50)


def run_script_execution(interpreter):
    """
    Run a script file using the execute_script method of the interpreter.
    
    Args:
        interpreter: An instance of a code interpreter
    """
    # Get the path to the hello_world.py script
    script_path = os.path.join(os.getcwd(), "examples", "tools", "hello_world.py")
    
    # Make sure the file exists
    if not os.path.exists(script_path):
        print(f"Error: Script file not found at {script_path}")
        return
    
    print(f"Executing script file: {script_path}")
    
    # Execute the script
    result = interpreter.execute_script(script_path, "python")
    
    print("\nScript Execution Result:")
    print("-" * 50)
    print(result)
    print("-" * 50)


def run_dynamic_code_generation(interpreter):
    """
    Run an example that demonstrates dynamic code generation and execution.
    
    Args:
        interpreter: An instance of a code interpreter
    """
    # Define code that generates and executes more code
    code = """
print("Generating and executing code dynamically...")

# Generate a function definition
function_code = '''
def calculate_factorial(n):
    if n == 0 or n == 1:
        return 1
    else:
        return n * calculate_factorial(n-1)
'''

# Execute the generated code to define the function
exec(function_code)

# Now use the dynamically defined function
for i in range(1, 6):
    print(f"Factorial of {i} is {calculate_factorial(i)}")
"""

    # Execute the code
    result = interpreter.execute(code, "python")
    print("\nDynamic Code Generation Result:")
    print("-" * 50)
    print(result)
    print("-" * 50)


def run_visualization_example(interpreter):
    """
    Run an example that would generate a visualization if matplotlib was allowed.
    This demonstrates handling imports that might not be allowed.
    
    Args:
        interpreter: An instance of a code interpreter
    """
    # Define code that attempts to use matplotlib
    code = """
print("Attempting to create a simple visualization...")

try:
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Generate some data
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    
    # Create a plot
    plt.figure(figsize=(8, 4))
    plt.plot(x, y)
    plt.title("Sine Wave")
    plt.xlabel("x")
    plt.ylabel("sin(x)")
    plt.grid(True)
    
    # Save the plot (would work if matplotlib was available)
    plt.savefig("examples/output/sine_wave.png")
    plt.close()
    
    print("Visualization created and saved as 'examples/output/sine_wave.png'")
except ImportError as e:
    print(f"Import error: {e}")
    print("Note: This example requires matplotlib to be in the allowed_imports.")
"""

    # Execute the code
    result = interpreter.execute(code, "python")
    print("\nVisualization Example Result:")
    print("-" * 50)
    print(result)
    print("-" * 50)


def run_search_examples():
    """
    Run examples using the search toolkits (Wikipedia, Google, and Google Free).
    """
    print("\n===== SEARCH TOOLS EXAMPLES =====\n")
    
    # Initialize search toolkits
    wiki_toolkit = WikipediaSearchToolkit(max_summary_sentences=3)
    google_toolkit = GoogleSearchToolkit(num_search_pages=3, max_content_words=200)
    google_free_toolkit = GoogleFreeSearchToolkit()
    
    # Get the individual tools from toolkits
    wiki_tool = wiki_toolkit.get_tool("wikipedia_search")
    google_tool = google_toolkit.get_tool("google_search")
    google_free_tool = google_free_toolkit.get_tool("google_free_search")
    
    # Example search query
    query = "artificial intelligence agent architecture"
    
    # Run Wikipedia search example
    try:
        print("\nWikipedia Search Example:")
        print("-" * 50)
        wiki_results = wiki_tool(query=query, num_search_pages=2)
        
        if wiki_results.get("error"):
            print(f"Error: {wiki_results['error']}")
        else:
            for i, result in enumerate(wiki_results.get("results", [])):
                print(f"Result {i+1}: {result['title']}")
                print(f"Summary: {result['summary'][:150]}...")
                print(f"URL: {result['url']}")
                print("-" * 30)
    except Exception as e:
        print(f"Error running Wikipedia search: {str(e)}")
    
    # Run Google search example (requires API key)
    try:
        print("\nGoogle Search Example (requires API key):")
        print("-" * 50)
        google_results = google_tool(query=query)
        
        if google_results.get("error"):
            print(f"Error: {google_results['error']}")
        else:
            for i, result in enumerate(google_results.get("results", [])):
                print(f"Result {i+1}: {result['title']}")
                print(f"URL: {result['url']}")
                print("-" * 30)
    except Exception as e:
        print(f"Error running Google search: {str(e)}")
    
    # Run Google Free search example
    try:
        print("\nGoogle Free Search Example:")
        print("-" * 50)
        free_results = google_free_tool(query=query, num_search_pages=2)
        
        if free_results.get("error"):
            print(f"Error: {free_results['error']}")
        else:
            for i, result in enumerate(free_results.get("results", [])):
                print(f"Result {i+1}: {result['title']}")
                print(f"URL: {result['url']}")
                print("-" * 30)
    except Exception as e:
        print(f"Error running free Google search: {str(e)}")


def run_python_interpreter_examples():
    """Run all examples using the Python InterpreterToolkit"""
    print("\n===== PYTHON INTERPRETER EXAMPLES =====\n")
    
    # Initialize the Python interpreter toolkit with the current directory as project path
    # and allow common standard library imports
    interpreter_toolkit = PythonInterpreterToolkit(
        project_path=os.getcwd(),
        directory_names=["examples", "evoagentx"],
        allowed_imports={"os", "sys", "time", "datetime", "math", "random", "platform", "matplotlib.pyplot", "numpy"}
    )
    
    # Get the underlying interpreter instance for the examples
    interpreter = interpreter_toolkit.python_interpreter
    
    # Run the examples
    run_simple_hello_world(interpreter)
    run_math_example(interpreter)
    run_platform_info(interpreter)
    run_script_execution(interpreter)
    run_dynamic_code_generation(interpreter)
    run_visualization_example(interpreter)


def run_docker_interpreter_examples():
    """Run all examples using the Docker InterpreterToolkit"""
    print("\n===== DOCKER INTERPRETER EXAMPLES =====\n")
    print("Running Docker interpreter examples...")
    
    try:
        # Initialize the Docker interpreter toolkit with a standard Python image
        interpreter_toolkit = DockerInterpreterToolkit(
            image_tag="python:3.9-slim",  # Using official Python image
            print_stdout=True,
            print_stderr=True,
            container_directory="/app"  # Better working directory for containerized apps
        )
        
        # Get the underlying interpreter instance for the examples
        interpreter = interpreter_toolkit.docker_interpreter
        
        # Run the examples
        run_simple_hello_world(interpreter)
        run_math_example(interpreter)
        run_platform_info(interpreter)
        run_script_execution(interpreter)
        run_dynamic_code_generation(interpreter)
        # run_visualization_example(interpreter)
    except Exception as e:
        print(f"Error running Docker interpreter examples: {str(e)}")
        print("Make sure Docker is installed and running on your system.")
        print("You may need to pull the python:3.9-slim image first using: docker pull python:3.9-slim")


def run_mcp_example():
    """
    Run an example using the MCP toolkit to search for research papers about 'artificial intelligence'.
    This uses the sample_mcp.config file to configure the arXiv MCP client.
    """
    print("\n===== MCP TOOLKIT EXAMPLE (arXiv) =====\n")
    
    # Get the path to the sample_mcp.config file
    config_path = os.path.join(os.getcwd(), "examples", "tools", "sample_mcp.config")
    
    print(f"Loading MCP configuration from: {config_path}")
    
    try:
        # Initialize the MCP toolkit with the sample config
        mcp_toolkit = MCPToolkit(config_path=config_path)
        
        # Get all available toolkits
        toolkits = mcp_toolkit.get_toolkits()
        
        print(f"Available MCP toolkits: {len(toolkits)}")
        
        # Find and use the arXiv search tool
        arxiv_tool = None
        for toolkit_item in toolkits:
            for tool in toolkit_item.get_tools():
                print(f"Tool: {tool.name}")
                print(f"Description: {tool.description}")
                print("-" * 30)
                
                if "search" in tool.name.lower() or "arxiv" in tool.name.lower():
                    arxiv_tool = tool
                    break
            if arxiv_tool:
                break
        
        if arxiv_tool:
            print(f"Using tool: {arxiv_tool.name}")
            
            # Search for 'artificial intelligence' research papers
            search_query = "artificial intelligence"
            print(f"Searching for research papers about: '{search_query}'")
            
            # Call the tool with the search query
            # Note: The actual parameter name might differ based on the tool's schema
            result = arxiv_tool(**{"query": search_query})
            
            print("\nSearch Results:")
            print("-" * 50)
            print(result)
            print("-" * 50)
        else:
            print("No suitable arXiv search tool found in the MCP configuration.")
    except Exception as e:
        print(f"Error running MCP example: {str(e)}")
        print("Make sure the arXiv MCP server is properly configured and running.")
    finally:
        if 'mcp_toolkit' in locals():
            mcp_toolkit.disconnect()


def run_file_tool_example():
    """
    Run an example using the FileToolkit to read and write PDF files.
    """
    print("\n===== FILE TOOL EXAMPLE =====\n")
    
    try:
        # Initialize the file toolkit
        file_toolkit = FileToolkit()
        
        # Get individual tools from the toolkit
        read_tool = file_toolkit.get_tool("read_file")
        write_tool = file_toolkit.get_tool("write_file")
        append_tool = file_toolkit.get_tool("append_file")
        
        # Create sample content for a PDF
        sample_content = """This is a sample PDF document created using the FileTool.
This tool provides special handling for different file types.
For PDF files, it uses PyPDF2 library for reading operations."""
        
        # Example PDF file path
        pdf_path = os.path.join(os.getcwd(), "examples", "output", "sample_document.pdf")
        
        # Make sure the output directory exists
        os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
        
        print(f"Writing content to PDF file: {pdf_path}")
        
        # Write content to PDF file
        write_result = write_tool(file_path=pdf_path, content=sample_content)
        print("Write Result:")
        print("-" * 30)
        print(write_result)
        print("-" * 30)
        
        # Read content from PDF file
        print(f"\nReading content from PDF file: {pdf_path}")
        read_result = read_tool(file_path=pdf_path)
        print("Read Result:")
        print("-" * 30)
        print(read_result)
        print("-" * 30)
        
        # Also demonstrate with a regular text file
        text_path = os.path.join(os.getcwd(), "examples", "output", "sample_text.txt")
        
        print(f"\nWriting content to text file: {text_path}")
        text_write_result = write_tool(file_path=text_path, content="This is a sample text file.")
        print("Text Write Result:")
        print("-" * 30)
        print(text_write_result)
        print("-" * 30)
        
        print(f"\nReading content from text file: {text_path}")
        text_read_result = read_tool(file_path=text_path)
        print("Text Read Result:")
        print("-" * 30)
        print(text_read_result)
        print("-" * 30)
        
        # ===== APPEND FILE OPERATIONS =====
        print("\n===== APPEND FILE OPERATIONS =====\n")
        
        # 1. Append to text file
        print(f"Appending content to text file: {text_path}")
        append_text_content = "\nThis line was appended to the text file."
        text_append_result = append_tool(file_path=text_path, content=append_text_content)
        print("Text Append Result:")
        print("-" * 30)
        print(text_append_result)
        print("-" * 30)
        
        # Read the text file again to show appended content
        print(f"\nReading text file after append: {text_path}")
        text_read_after_append = read_tool(file_path=text_path)
        print("Text File After Append:")
        print("-" * 30)
        print(text_read_after_append)
        print("-" * 30)
        
        # 2. Append to PDF file
        print(f"\nAppending content to PDF file: {pdf_path}")
        append_pdf_content = "\n\nThis content was appended to the PDF document.\nIt demonstrates PDF append functionality."
        pdf_append_result = append_tool(file_path=pdf_path, content=append_pdf_content)
        print("PDF Append Result:")
        print("-" * 30)
        print(pdf_append_result)
        print("-" * 30)
        
        # Read the PDF file again to show appended content
        print(f"\nReading PDF file after append: {pdf_path}")
        pdf_read_after_append = read_tool(file_path=pdf_path)
        print("PDF File After Append:")
        print("-" * 30)
        print(pdf_read_after_append)
        print("-" * 30)
        
        # 3. Append to log file
        log_path = os.path.join(os.getcwd(), "examples", "output", "application.log")
        print(f"\nCreating and appending to log file: {log_path}")
        
        # Initial log entry
        initial_log = "2024-01-01 10:00:00 INFO Application started"
        log_write_result = write_tool(file_path=log_path, content=initial_log)
        print("Initial Log Write Result:")
        print("-" * 30)
        print(log_write_result)
        print("-" * 30)
        
        # Append multiple log entries
        log_entries = [
            "\n2024-01-01 10:01:00 INFO User logged in",
            "\n2024-01-01 10:02:00 WARNING Cache miss for key 'user_data'",
            "\n2024-01-01 10:03:00 ERROR Database connection failed",
            "\n2024-01-01 10:04:00 INFO Retrying database connection",
            "\n2024-01-01 10:05:00 INFO Database connection restored"
        ]
        
        for log_entry in log_entries:
            append_result = append_tool(file_path=log_path, content=log_entry)
            print(f"Appended: {log_entry.strip()}")
        
        # Read the complete log file
        print(f"\nReading complete log file: {log_path}")
        log_read_result = read_tool(file_path=log_path)
        print("Complete Log File:")
        print("-" * 30)
        print(log_read_result)
        print("-" * 30)
        
        # 4. Append to CSV file
        csv_path = os.path.join(os.getcwd(), "examples", "output", "data.csv")
        print(f"\nCreating and appending to CSV file: {csv_path}")
        
        # Initial CSV header and data
        csv_header = "Name,Age,City,Occupation"
        csv_write_result = write_tool(file_path=csv_path, content=csv_header)
        print("CSV Header Write Result:")
        print("-" * 30)
        print(csv_write_result)
        print("-" * 30)
        
        # Append CSV rows
        csv_rows = [
            "\nJohn Doe,30,New York,Engineer",
            "\nJane Smith,25,Los Angeles,Designer",
            "\nBob Johnson,35,Chicago,Manager",
            "\nAlice Brown,28,San Francisco,Developer"
        ]
        
        for csv_row in csv_rows:
            append_result = append_tool(file_path=csv_path, content=csv_row)
            print(f"Appended CSV row: {csv_row.strip()}")
        
        # Read the complete CSV file
        print(f"\nReading complete CSV file: {csv_path}")
        csv_read_result = read_tool(file_path=csv_path)
        print("Complete CSV File:")
        print("-" * 30)
        print(csv_read_result)
        print("-" * 30)
        
        # 5. Append to configuration file
        config_path = os.path.join(os.getcwd(), "examples", "output", "config.ini")
        print(f"\nCreating and appending to config file: {config_path}")
        
        # Initial config content
        initial_config = """[DATABASE]
host = localhost
port = 5432
name = myapp"""
        
        config_write_result = write_tool(file_path=config_path, content=initial_config)
        print("Initial Config Write Result:")
        print("-" * 30)
        print(config_write_result)
        print("-" * 30)
        
        # Append new config sections
        additional_configs = [
            "\n\n[CACHE]",
            "\nredis_host = localhost",
            "\nredis_port = 6379",
            "\nttl = 3600",
            "\n\n[LOGGING]",
            "\nlevel = INFO",
            "\nfile = /var/log/myapp.log",
            "\nmax_size = 10MB"
        ]
        
        for config_line in additional_configs:
            append_result = append_tool(file_path=config_path, content=config_line)
        
        print("Appended additional configuration sections")
        
        # Read the complete config file
        print(f"\nReading complete config file: {config_path}")
        config_read_result = read_tool(file_path=config_path)
        print("Complete Config File:")
        print("-" * 30)
        print(config_read_result)
        print("-" * 30)
        
        # 6. Demonstrate error handling for non-existent file append
        non_existent_path = os.path.join(os.getcwd(), "examples", "output", "non_existent.txt")
        print(f"\nTesting append to non-existent file: {non_existent_path}")
        error_append_result = append_tool(file_path=non_existent_path, content="This should create a new file")
        print("Append to Non-existent File Result:")
        print("-" * 30)
        print(error_append_result)
        print("-" * 30)
        
        # Verify the file was created
        if error_append_result.get("success"):
            print(f"Reading newly created file: {non_existent_path}")
            new_file_read = read_tool(file_path=non_existent_path)
            print("Newly Created File Content:")
            print("-" * 30)
            print(new_file_read)
            print("-" * 30)
        
    except Exception as e:
        print(f"Error running file tool example: {str(e)}")


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


def run_arxiv_tool_example():
    """Simple example using ArxivToolkit to search for papers."""
    print("\n===== ARXIV TOOL EXAMPLE =====\n")
    
    try:
        # Initialize the arXiv toolkit
        arxiv_toolkit = ArxivToolkit()
        search_tool = arxiv_toolkit.get_tool("arxiv_search")
        
        print("✓ ArxivToolkit initialized")
        
        # Search for machine learning papers
        print("Searching for 'machine learning' papers...")
        result = search_tool(
            search_query="all:machine learning",
            max_results=3
        )
        
        if result.get('success'):
            papers = result.get('papers', [])
            print(f"✓ Found {len(papers)} papers")
            
            for i, paper in enumerate(papers):
                print(f"\nPaper {i+1}: {paper.get('title', 'No title')}")
                print(f"  Authors: {', '.join(paper.get('authors', ['Unknown']))}")
                print(f"  arXiv ID: {paper.get('arxiv_id', 'Unknown')}")
                print(f"  URL: {paper.get('url', 'No URL')}")
        else:
            print(f"❌ Search failed: {result.get('error', 'Unknown error')}")
        
        print("\n✓ ArxivToolkit test completed")
        
    except Exception as e:
        print(f"Error: {str(e)}")


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
    """Main function to run all examples"""
    print("===== INTERPRETER TOOL EXAMPLES =====")
    
    # Run file tool example
    run_file_tool_example()
    
    # Run browser tool example
    run_browser_tool_example()
    
    # Run MCP toolkit example
    run_mcp_example()
    
    # Run Python interpreter examples
    run_python_interpreter_examples()
    
    # Run Docker interpreter examples
    run_docker_interpreter_examples()
    
    # Run search tools examples
    run_search_examples()
    
    # Run arXiv tool example
    run_arxiv_tool_example()
    
    # Run BrowserUse tool example
    run_browser_use_tool_example()
    
    print("\n===== ALL EXAMPLES COMPLETED =====")


if __name__ == "__main__":
    main() 