# EvoAgentX Tools Tutorial

This document provides a comprehensive guide to using the various tools available in the EvoAgentX library. These tools include code interpreters and search utilities designed to enhance AI agent capabilities.

## Table of Contents

- [Overview](#overview)
- [Configuration](#configuration)
  - [Environment Variables](#environment-variables)
- [Code Interpreters](#code-interpreters)
  - [Python Interpreter](#python-interpreter)
  - [Docker Interpreter](#docker-interpreter)
- [Search Tools](#search-tools)
  - [Wikipedia Search](#wikipedia-search)
  - [Google Search](#google-search)
  - [Free Google Search](#free-google-search)

## Overview

EvoAgentX provides a set of tools that can be integrated into AI agents to extend their capabilities. These tools fall into two main categories:

1. **Code Interpreters**: Safely execute code in various languages
2. **Search Tools**: Retrieve information from external sources like Wikipedia and Google

All tools inherit from the base `Tool` class, providing a consistent interface for integration.

## Configuration

### Environment Variables

EvoAgentX uses environment variables for configuration, particularly for sensitive information like API keys. You can set these variables in a `.env` file in your project root:

```
# Search API Keys
GOOGLE_API_KEY=your_google_api_key_here
GOOGLE_SEARCH_ENGINE_ID=your_search_engine_id_here
```

You can also set these variables in your shell environment:

```bash
# Linux/Mac
export GOOGLE_API_KEY=your_google_api_key_here
export GOOGLE_SEARCH_ENGINE_ID=your_search_engine_id_here

# Windows (Command Prompt)
set GOOGLE_API_KEY=your_google_api_key_here
set GOOGLE_SEARCH_ENGINE_ID=your_search_engine_id_here

# Windows (PowerShell)
$env:GOOGLE_API_KEY="your_google_api_key_here"
$env:GOOGLE_SEARCH_ENGINE_ID="your_search_engine_id_here"
```

## Code Interpreters

### Python Interpreter

The Python Interpreter allows for secure execution of Python code within a controlled environment. It performs static analysis to detect unauthorized imports and other security risks before execution.

#### Features

- Static analysis using AST to detect unauthorized imports
- Controlled execution environment
- Support for project-specific module loading
- Restriction of unsafe system-wide libraries

#### Example Usage

```python
from evoagentx.tools import InterpreterPython

# Initialize the interpreter with project path and allowed imports
interpreter = InterpreterPython(
    project_path="/path/to/your/project",
    allowed_imports={"math", "numpy", "os", "sys"}
)

# Execute Python code
code = """
import math
from math import sqrt

def calculate_area(radius):
    return math.pi * radius ** 2

print(calculate_area(5))
print(sqrt(16))
"""

result = interpreter.execute(code)
print(result)
```

### Docker Interpreter

The Docker Interpreter provides an isolated environment for executing Python code inside Docker containers. This offers an additional layer of security by fully isolating the execution environment from the host system.

#### Features

- Secure code execution in isolated Docker containers
- Support for mounting local directories
- Capture of standard output and error messages
- Optional user confirmation before execution

#### Example Usage

```python
from evoagentx.tools import DockerInterpreter

# Initialize the Docker interpreter
interpreter = DockerInterpreter(
    require_confirm=False,
    print_stdout=True,
    host_directory="/path/to/your/project",
    container_directory="/home/app/"
)

# Execute Python code in a Docker container
code = """
import os
print('Hello from Docker!')
print(f'Current directory: {os.getcwd()}')
"""

result = interpreter.execute(code, "python")
print(result)
```

## Search Tools

### Wikipedia Search

The Wikipedia Search tool enables querying Wikipedia to find relevant articles and extract key information.

#### Features

- Retrieve article titles, summaries, and content
- Control over number of search results
- Content length customization
- Filtering of ambiguous or non-existent pages

#### Example Usage

```python
from evoagentx.tools import SearchWiki

# Initialize the Wikipedia search tool
wiki_search = SearchWiki(
    num_search_pages=5,
    max_content_words=500
)

# Search for information
results = wiki_search.search("Python programming language", max_sentences=15)

# Process the results
if "results" in results:
    for article in results["results"]:
        print(f"Title: {article['title']}")
        print(f"Summary: {article['summary'][:150]}...")
        print(f"URL: {article['url']}")
        print("-" * 40)
else:
    print(f"Error: {results.get('error')}")
```

### Google Search

The Google Search tool utilizes the Google Custom Search API to perform structured search queries, requiring an API key for access.

#### Features

- Secure and authenticated search queries
- Structured retrieval of search results
- Content extraction and summarization
- Reliable and unrestricted access (compared to free alternatives)

#### Example Usage

```python
import os
from evoagentx.tools import SearchGoogle

# Initialize the Google search tool
google_search = SearchGoogle(
    num_search_pages=5,
    max_content_words=500
)

# Use environment variables for API credentials (securely)
search_params = {
    "api_key": os.environ.get("GOOGLE_API_KEY"),
    "search_engine_id": os.environ.get("GOOGLE_SEARCH_ENGINE_ID")
}

# Search for information
results = google_search.search("artificial intelligence trends", search_params)

# Process the results
if "results" in results and results["error"] is None:
    for item in results["results"]:
        print(f"Title: {item['title']}")
        print(f"Content: {item['content'][:150]}...")
        print(f"URL: {item['url']}")
        print("-" * 40)
else:
    print(f"Error: {results.get('error')}")
```

For improved security, you can also create a dedicated config class:

```python
import os
from pydantic import BaseSettings

class SearchConfig(BaseSettings):
    GOOGLE_API_KEY: str
    GOOGLE_SEARCH_ENGINE_ID: str
    
    class Config:
        env_file = ".env"

# Use the config
config = SearchConfig()
search_params = {
    "api_key": config.GOOGLE_API_KEY,
    "search_engine_id": config.GOOGLE_SEARCH_ENGINE_ID
}
```

### Free Google Search

The Free Google Search tool provides a lightweight alternative for Google searches without requiring an API key, though it comes with certain limitations.

#### Features

- No API key required
- Retrieval of search results
- Content extraction from linked pages
- Subject to Google's public rate limits

#### Example Usage

```python
from evoagentx.tools import SearchGoogleFree

# Initialize the Free Google search tool
free_search = SearchGoogleFree(
    num_search_pages=5,
    max_content_words=500
)

# Search for information
results = free_search.search("machine learning tutorials")

# Process the results - Note: This tool returns a list directly when successful
if isinstance(results, list):
    for item in results:
        print(f"Title: {item['title']}")
        print(f"Content: {item['content'][:150]}...")
        print(f"URL: {item['url']}")
        print("-" * 40)
elif isinstance(results, dict) and "error" in results:
    print(f"Error: {results['error']}")
```

## Custom Tool Development

If you need to create your own custom tools, you can extend the base Tool class:

```python
from evoagentx.tools import Tool

class MyCustomTool(Tool):
    def get_tool_info(self):
        return {
            "description": "Description of your custom tool",
            "inputs": {
                "parameter1": {
                    "type": "str",
                    "description": "Description of parameter1",
                    "required": True
                }
            },
            "outputs": {
                "result": {
                    "type": "str",
                    "description": "Description of the output"
                }
            }
        }
    
    def my_function(self, parameter1):
        # Implement your tool's functionality
        return f"Result: {parameter1}"
```
