#!/usr/bin/env python3
"""
Hello World script for testing the execute_script method of interpreters.
This script contains examples of using various standard library imports.
"""

import math
import sys
import os
import datetime
import random

# Print hello world
print("Hello, World! This is a script file being executed.")

# Show math operations
print("\n--- Math Operations ---")
print(f"The value of pi is: {math.pi:.4f}")
print(f"The square root of 16 is: {math.sqrt(16)}")
print(f"The value of e is: {math.e:.4f}")

# Show system information
print("\n--- System Information ---")
print(f"Python version: {sys.version}")
print(f"Platform: {sys.platform}")
print(f"Current directory: {os.getcwd()}")

# Show date and time
print("\n--- Date and Time ---")
now = datetime.datetime.now()
print(f"Current date and time: {now}")
print(f"Current year: {now.year}")
print(f"Current month: {now.month}")
print(f"Current day: {now.day}")

# Generate random numbers
print("\n--- Random Numbers ---")
print(f"Random integer between 1 and 10: {random.randint(1, 10)}")
print(f"Random float between 0 and 1: {random.random()}")
print(f"Random choice from a list: {random.choice(['apple', 'banana', 'cherry', 'date'])}")

# Print ASCII art for EvoAgentX
print("\n--- ASCII Art ---")
print(" ______              _                     _   __   __ ")
print("|  ____|            /\\                    | | |  \\ /  |")
print("| |__   __   ___   /  \\    __ _  ___ _ __| |_|   |   |")
print("|  __|  \\ \\ / / | / /\\ \\  / _` |/ _ \\ '_ \\_   _|      |")
print("| |____  \\ V /| |/ ____ \\| (_| |  __/ | | || ||  |\\/|  |")
print("|______|  \\_/ |_/_/    \\_\\__, |\\___|_| |_||_||__|  |__|")
print("                          __/ |                        ")
print("                         |___/                         ")

print("\nScript execution completed successfully!")
