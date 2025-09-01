#!/usr/bin/env python3

"""
Code Interpreter Examples for EvoAgentX

This module provides comprehensive examples for:
- PythonInterpreterToolkit: Execute Python code in a secure environment
- DockerInterpreterToolkit: Execute code within isolated Docker containers

The examples demonstrate various use cases including:
- Basic code execution (Hello World, Math operations)
- Platform information and system details
- Script file execution
- Dynamic code generation
- Visualization attempts (with import handling)
"""

import os
import sys
from pathlib import Path

# Add the parent directory to sys.path to import from evoagentx
sys.path.append(str(Path(__file__).parent.parent))

from evoagentx.tools import (
    PythonInterpreterToolkit,
    DockerInterpreterToolkit
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
        # run_visualization_example(interpreter)  # Commented out for Docker examples
    except Exception as e:
        print(f"Error running Docker interpreter examples: {str(e)}")
        print("Make sure Docker is installed and running on your system.")
        print("You may need to pull the python:3.9-slim image first using: docker pull python:3.9-slim")


def main():
    """Main function to run interpreter examples"""
    print("===== CODE INTERPRETER EXAMPLES =====")
    
    # Run Python interpreter examples
    run_python_interpreter_examples()
    
    # Run Docker interpreter examples
    run_docker_interpreter_examples()
    
    print("\n===== ALL INTERPRETER EXAMPLES COMPLETED =====")


if __name__ == "__main__":
    main()
