#!/usr/bin/env python3
"""
Example demonstrating how to use the PythonInterpreter and DockerInterpreter tools.
This script provides examples for running simple Python code and script files.
"""

import os
import sys
from pathlib import Path

# Add the parent directory to sys.path to import from evoagentx
sys.path.append(str(Path(__file__).parent.parent))

from evoagentx.tools import (
    PythonInterpreterToolKit,
    DockerInterpreterToolKit,
    WikipediaSearchToolKit,
    GoogleSearchToolKit,
    GoogleFreeSearchToolKit,
    MCPToolkit,
    FileToolKit,
    BrowserToolKit
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
    script_path = os.path.join(os.getcwd(), "examples", "output", "tools", "hello_world.py")
    
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
    plt.savefig("sine_wave.png")
    plt.close()
    
    print("Visualization created and saved as 'sine_wave.png'")
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
    wiki_toolkit = WikipediaSearchToolKit(max_summary_sentences=3)
    google_toolkit = GoogleSearchToolKit(num_search_pages=3, max_content_words=200)
    google_free_toolkit = GoogleFreeSearchToolKit()
    
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
    """Run all examples using the Python Interpreter ToolKit"""
    print("\n===== PYTHON INTERPRETER EXAMPLES =====\n")
    
    # Initialize the Python interpreter toolkit with the current directory as project path
    # and allow common standard library imports
    interpreter_toolkit = PythonInterpreterToolKit(
        project_path=os.getcwd(),
        directory_names=["examples", "evoagentx"],
        allowed_imports={"os", "sys", "time", "datetime", "math", "random", "platform"}
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
    """Run all examples using the Docker Interpreter ToolKit"""
    print("\n===== DOCKER INTERPRETER EXAMPLES =====\n")
    print("Running Docker interpreter examples...")
    
    try:
        # Initialize the Docker interpreter toolkit with a standard Python image
        interpreter_toolkit = DockerInterpreterToolKit(
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
        run_visualization_example(interpreter)
    except Exception as e:
        print(f"Error running Docker interpreter examples: {str(e)}")
        print("Make sure Docker is installed and running on your system.")
        print("You may need to pull the python:3.9-slim image first using: docker pull python:3.9-slim")


def run_mcp_example():
    """
    Run an example using the MCP toolkit to search for job information about 'data scientist'.
    This uses the sample_mcp.config file to configure the MCP client.
    """
    print("\n===== MCP TOOLKIT EXAMPLE =====\n")
    
    # Get the path to the sample_mcp.config file
    config_path = os.path.join(os.getcwd(), "examples", "sample_mcp.config")
    
    print(f"Loading MCP configuration from: {config_path}")
    
    try:
        # Initialize the MCP toolkit with the sample config
        toolkit = MCPToolkit(config_path=config_path)
        
        # Get all available toolkits
        toolkits = toolkit.get_tools()
        
        print(f"Available MCP toolkits: {len(toolkits)}")
        
        # Find and use the hirebase search tool
        hirebase_tool = None
        for toolkit_item in toolkits:
            for tool in toolkit_item.tools:
                print(f"Tool: {tool.name}")
                print(f"Description: {tool.description}")
                print("-" * 30)
                
                if "hire" in tool.name.lower() or "search" in tool.name.lower():
                    hirebase_tool = tool
                    break
            if hirebase_tool:
                break
        
        if hirebase_tool:
            print(f"Using tool: {hirebase_tool.name}")
            
            # Search for 'data scientist' job information
            search_query = "data scientist"
            print(f"Searching for job information about: '{search_query}'")
            
            # Call the tool with the search query
            # Note: The actual parameter name might differ based on the tool's schema
            result = hirebase_tool(**{"query": search_query})
            
            print("\nSearch Results:")
            print("-" * 50)
            print(result)
            print("-" * 50)
        else:
            print("No suitable hiring or search tool found in the MCP configuration.")
    except Exception as e:
        print(f"Error running MCP example: {str(e)}")
        print("Make sure the hirebase MCP server is properly configured with a valid API key.")
    finally:
        if 'toolkit' in locals():
            toolkit.disconnect()


def run_file_tool_example():
    """
    Run an example using the FileToolKit to read and write PDF files.
    """
    print("\n===== FILE TOOL EXAMPLE =====\n")
    
    try:
        # Initialize the file toolkit
        file_toolkit = FileToolKit()
        
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
    Run an example using the BrowserToolKit to initialize browser, 
    go to Google, search for "test", and then close the browser.
    """
    print("\n===== BROWSER TOOL EXAMPLE =====\n")
    
    try:
        # Initialize the browser toolkit (with visible browser window if headless is False)
        browser_toolkit = BrowserToolKit(headless=True, timeout=10)
        
        # Get individual tools from the toolkit
        init_tool = browser_toolkit.get_tool("initialize_browser")
        nav_tool = browser_toolkit.get_tool("navigate_to_url")
        input_tool = browser_toolkit.get_tool("input_text")
        click_tool = browser_toolkit.get_tool("browser_click")
        close_tool = browser_toolkit.get_tool("close_browser")
        
        print("Step 1: Initializing browser...")
        init_result = init_tool()
        print("Browser Initialization Result:")
        print("-" * 30)
        print(init_result)
        print("-" * 30)
        
        if init_result.get("status") == "success":
            print("\nStep 2: Navigating to Google...")
            nav_result = nav_tool(url="https://www.google.com")
            print("Navigation Result:")
            print("-" * 30)
            print(f"Status: {nav_result.get('status')}")
            print(f"URL: {nav_result.get('current_url')}")
            print(f"Title: {nav_result.get('title')}")
            
            # Show available interactive elements
            if nav_result.get("snapshot") and nav_result["snapshot"].get("interactive_elements"):
                elements = nav_result["snapshot"]["interactive_elements"]
                print(f"Found {len(elements)} interactive elements:")
                for elem in elements[:5]:  # Show first 5 elements
                    print(f"  - {elem['id']}: {elem.get('description', 'No description')}")
            print("-" * 30)
            
            if nav_result.get("status") == "success":
                # Find the search input box and search button
                elements = nav_result.get("snapshot", {}).get("interactive_elements", [])
                search_input_ref = None
                search_button_ref = None
                
                for elem in elements:
                    desc = elem.get("description", "").lower()
                    if "search" in desc and ("input" in desc or "textbox" in desc):
                        search_input_ref = elem["id"]
                    elif "search" in desc and ("button" in desc or "submit" in desc):
                        search_button_ref = elem["id"]
                
                if search_input_ref:
                    print(f"\nStep 3: Typing 'test' in search box (element {search_input_ref})...")
                    input_result = input_tool(
                        element="Search box", 
                        ref=search_input_ref, 
                        text="test", 
                        submit=False
                    )
                    print("Input Result:")
                    print("-" * 30)
                    print(input_result)
                    print("-" * 30)
                    
                    if search_button_ref:
                        print(f"\nStep 4: Clicking search button (element {search_button_ref})...")
                        click_result = click_tool(
                            element="Search button", 
                            ref=search_button_ref
                        )
                        print("Click Result:")
                        print("-" * 30)
                        print(click_result)
                        print("-" * 30)
                    else:
                        print("\nStep 4: Search button not found, submitting with Enter key...")
                        submit_result = input_tool(
                            element="Search box", 
                            ref=search_input_ref, 
                            text="", 
                            submit=True
                        )
                        print("Submit Result:")
                        print("-" * 30)
                        print(submit_result)
                        print("-" * 30)
        
        print("Closing browser...")
        close_result = close_tool()
        print("Browser Close Result:")
        print("-" * 30)
        print(close_result)
        print("-" * 30)
        
    except Exception as e:
        print(f"Error running browser tool example: {str(e)}")
        # Make sure to close browser even if there's an error
        try:
            if 'browser_toolkit' in locals():
                close_tool = browser_toolkit.get_tool("close_browser")
                close_tool()
        except Exception as e:
            print(f"Error closing browser: {str(e)}")


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
    
    print("\n===== ALL EXAMPLES COMPLETED =====")


if __name__ == "__main__":
    main() 