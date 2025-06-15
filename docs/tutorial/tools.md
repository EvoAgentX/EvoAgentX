# Working with Tools in EvoAgentX

This tutorial walks you through using EvoAgentX's powerful tool ecosystem. Tools allow agents to interact with the external world, perform computations, and access information. We'll cover:

1. **Understanding the Tool Architecture**: Learn about the base Tool class and its functionality
2. **Code Interpreters**: Execute Python code safely using Python and Docker interpreters
3. **Search Tools**: Access information from the web using Wikipedia and Google search tools
4. **File Operations**: Handle file reading and writing with special support for different file formats
5. **Browser Automation**: Control web browsers to interact with websites and web applications
6. **MCP Tools**: Connect to external services using the Model Context Protocol

By the end of this tutorial, you'll understand how to leverage these tools in your own agents and workflows.

---

## 1. Understanding the Tool Architecture

At the core of EvoAgentX's tool ecosystem is the `Tool` base class, which provides a standardized interface for all tools. 

```python
from evoagentx.tools.tool import Tool
```

The `Tool` class implements three key methods:

- `get_tool_schemas()`: Returns OpenAI-compatible function schemas for the tool
- `get_tools()`: Returns a list of callable functions that the tool provides
- `get_tool_descriptions()`: Returns descriptions of the tool's functionality

All tools in EvoAgentX extend this base class, ensuring a consistent interface for agents to use them.

### Key Concepts

- **Tool Integration**: Tools seamlessly integrate with agents via function calling protocols
- **Schemas**: Each tool provides schemas that describe its functionality, parameters, and outputs
- **Modularity**: Tools can be easily added to any agent that supports function calling

---

## 2. Code Interpreters

EvoAgentX provides two main code interpreter tools:

1. **PythonInterpreter**: Executes Python code in a controlled environment
2. **DockerInterpreter**: Executes code within isolated Docker containers

### 2.1 PythonInterpreter

**The PythonInterpreter provides a secure environment for executing Python code with fine-grained control over imports, directory access, and execution context. It uses a sandboxing approach to restrict potentially harmful operations.**

#### 2.1.1 Setup

```python
from evoagentx.tools.interpreter_python import PythonInterpreter

# Initialize with specific allowed imports and directory access
interpreter = PythonInterpreter(
    project_path=".",  # Default is current directory
    directory_names=["examples", "evoagentx"],
    allowed_imports={"os", "sys", "math", "random", "datetime"}
)
```

#### 2.1.2 Available Methods

The `PythonInterpreter` provides the following callable methods:

##### Method 1: execute(code, language)

**Description**: Executes Python code directly in a secure environment.

**Usage Example**:
```python
# Execute a simple code snippet
result = interpreter.execute("""
print("Hello, World!")
import math
print(f"The value of pi is: {math.pi:.4f}")
""", "python")

print(result)
```

**Return Type**: `str`

**Sample Return**:
```
Hello, World!
The value of pi is: 3.1416
```

---

##### Method 2: execute_script(file_path, language)

**Description**: Executes a Python script file in a secure environment.

**Usage Example**:
```python
# Execute a Python script file
script_path = "examples/hello_world.py"
script_result = interpreter.execute_script(script_path, "python")
print(script_result)
```

**Return Type**: `str`

**Sample Return**:
```
Running hello_world.py...
Hello from the script file!
Script execution completed.
```

#### 2.1.3 Setup Hints

- **Project Path**: The `project_path` parameter should point to the root directory of your project to ensure proper file access. Default is the current directory (".").

- **Directory Names**: The `directory_names` list specifies which directories within your project can be imported from. This is important for security to prevent unauthorized access. Default is an empty list `[]`.

- **Allowed Imports**: The `allowed_imports` set restricts which Python modules can be imported in executed code. Default is an empty list `[]`.
  - **Important**: If `allowed_imports` is set to an empty list, no import restrictions are applied.
  - When specified, add only the modules you consider safe:

```python
# Example with restricted imports
interpreter = PythonInterpreter(
    project_path=os.getcwd(),
    directory_names=["examples", "evoagentx", "tests"],
    allowed_imports={
        "os", "sys", "time", "datetime", "math", "random", 
        "json", "csv", "re", "collections", "itertools"
    }
)

# Example with no import restrictions
interpreter = PythonInterpreter(
    project_path=os.getcwd(),
    directory_names=["examples", "evoagentx"],
    allowed_imports=[]  # Allows any module to be imported
)
```

---

### 2.2 DockerInterpreter

**The DockerInterpreter executes code in isolated Docker containers, providing maximum security and environment isolation. It allows safe execution of potentially risky code with custom environments, dependencies, and complete resource isolation. Docker must be installed and running on your machine to use this tool.**

#### 2.2.1 Setup

```python
from evoagentx.tools.interpreter_docker import DockerInterpreter

# Initialize with a specific Docker image
interpreter = DockerInterpreter(
    image_tag="fundingsocietiesdocker/python3.9-slim",
    print_stdout=True,
    print_stderr=True,
    container_directory="/app"
)
```

#### 2.2.2 Available Methods

The `DockerInterpreter` provides the following callable methods:

##### Method 1: execute(code, language)

**Description**: Executes code inside a Docker container.

**Usage Example**:
```python
# Execute Python code in a Docker container
result = interpreter.execute("""
import platform
print(f"Python version: {platform.python_version()}")
print(f"Platform: {platform.system()} {platform.release()}")
""", "python")

print(result)
```

**Return Type**: `str`

**Sample Return**:
```
Python version: 3.9.16
Platform: Linux 5.15.0-1031-azure
```

---

##### Method 2: execute_script(file_path, language)

**Description**: Executes a script file inside a Docker container.

**Usage Example**:
```python
# Execute a Python script file in Docker
script_path = "examples/docker_test.py"
script_result = interpreter.execute_script(script_path, "python")
print(script_result)
```

**Return Type**: `str`

**Sample Return**:
```
Running container with script: /app/script_12345.py
Hello from the Docker container!
Container execution completed.
```

#### 2.2.3 Setup Hints

- **Docker Requirements**: Ensure Docker is installed and running on your system before using this interpreter.

- **Image Management**: You need to provide **either** an `image_tag` **or** a `dockerfile_path`, not both:
  - **Option 1: Using an existing image**
    ```python
    interpreter = DockerInterpreter(
        image_tag="python:3.9-slim",  # Uses an existing Docker Hub image
        container_directory="/app"
    )
    ```
  
  - **Option 2: Building from a Dockerfile**
    ```python
    interpreter = DockerInterpreter(
        dockerfile_path="path/to/Dockerfile",  # Builds a custom image
        image_tag="my-custom-image-name",      # Name for the built image
        container_directory="/app"
    )
    ```

- **File Access**:
  - To make local files available in the container, use the `host_directory` parameter:
  ```python
  interpreter = DockerInterpreter(
      image_tag="python:3.9-slim",
      host_directory="/path/to/local/files",
      container_directory="/app/data"
  )
  ```
  - This mounts the local directory to the specified container directory, making all files accessible.

- **Container Lifecycle**:
  - The Docker container is created when you initialize the interpreter and removed when the interpreter is destroyed.
  - For long-running sessions, you can set `print_stdout` and `print_stderr` to see real-time output.

- **Troubleshooting**:
  - If you encounter permission issues, ensure your user has Docker privileges.
  - For network-related errors, check if your Docker daemon has proper network access.

---

## 3. Search Tools

EvoAgentX provides several search tools to retrieve information from various sources:

1. **SearchWiki**: Search Wikipedia for information
2. **SearchGoogle**: Search Google using the official API
3. **SearchGoogleFree**: Search Google without requiring an API key

### 3.1 SearchWiki

**The SearchWiki tool retrieves information from Wikipedia articles, providing summaries, full content, and metadata. It offers a straightforward way to incorporate encyclopedic knowledge into your agents without complex API setups.**

#### 3.1.1 Setup

```python
from evoagentx.tools.search_wiki import SearchWiki

# Initialize with custom parameters
wiki_search = SearchWiki(max_sentences=3)
```

#### 3.1.2 Available Methods

The `SearchWiki` provides the following callable method:

##### Method: search(query)

**Description**: Searches Wikipedia for articles matching the query.

**Usage Example**:
```python
# Search Wikipedia for information
results = wiki_search.search(
    query="artificial intelligence agent architecture"
)

# Process the results
for i, result in enumerate(results.get("results", [])):
    print(f"Result {i+1}: {result['title']}")
    print(f"Summary: {result['summary']}")
    print(f"URL: {result['url']}")
```

**Return Type**: `dict`

**Sample Return**:
```python
{
    "results": [
        {
            "title": "Artificial intelligence",
            "summary": "Artificial intelligence (AI) is the intelligence of machines or software, as opposed to the intelligence of humans or animals. AI applications include advanced web search engines, recommendation systems, voice assistants...",
            "url": "https://en.wikipedia.org/wiki/Artificial_intelligence"
        },
        {
            "title": "Intelligent agent",
            "summary": "In artificial intelligence, an intelligent agent (IA) is anything which can perceive its environment, process those perceptions, and respond in pursuit of its own goals...",
            "url": "https://en.wikipedia.org/wiki/Intelligent_agent"
        }
    ]
}
```

---

### 3.2 SearchGoogle

**The SearchGoogle tool enables web searches through Google's official Custom Search API, providing high-quality search results with content extraction. It requires API credentials but offers more reliable and comprehensive search capabilities.**

#### 3.2.1 Setup

```python
from evoagentx.tools.search_google import SearchGoogle

# Initialize with custom parameters
google_search = SearchGoogle(
    num_search_pages=3,
    max_content_words=200
)
```

#### 3.2.2 Available Methods

The `SearchGoogle` provides the following callable method:

##### Method: search(query)

**Description**: Searches Google for content matching the query.

**Usage Example**:
```python
# Search Google for information
results = google_search.search(
    query="evolutionary algorithms for neural networks"
)

# Process the results
for i, result in enumerate(results.get("results", [])):
    print(f"Result {i+1}: {result['title']}")
    print(f"URL: {result['url']}")
    print(f"Content: {result['content'][:150]}...")
```

**Return Type**: `dict`

**Sample Return**:
```python
{
    "results": [
        {
            "title": "Evolutionary Algorithms for Neural Networks - A Systematic Review",
            "url": "https://example.com/paper1",
            "content": "This paper provides a comprehensive review of evolutionary algorithms applied to neural network optimization. Key approaches include genetic algorithms, particle swarm optimization, and differential evolution..."
        },
        {
            "title": "Applying Genetic Algorithms to Neural Network Training",
            "url": "https://example.com/article2",
            "content": "Genetic algorithms offer a powerful approach to optimizing neural network architectures and weights. This article explores how evolutionary computation can overcome limitations of gradient-based methods..."
        }
    ]
}
```

#### 3.2.3 Setup Hints

- **API Requirements**: This tool requires Google Custom Search API credentials. Set them in your environment:
  ```python
  # In your .env file or environment variables
  GOOGLE_API_KEY=your_google_api_key_here
  GOOGLE_SEARCH_ENGINE_ID=your_search_engine_id_here
  ```

- **Obtaining Credentials**:
  1. Create a project in [Google Cloud Console](https://console.cloud.google.com/)
  2. Enable the Custom Search API
  3. Create API credentials
  4. Set up a Custom Search Engine at [https://cse.google.com/cse/](https://cse.google.com/cse/)

---

### 3.3 SearchGoogleFree

**The SearchGoogleFree tool provides web search capability without requiring any API keys or authentication. It offers a simpler alternative to the official Google API with basic search results suitable for most general queries.**

#### 3.3.1 Setup

```python
from evoagentx.tools.search_google_f import SearchGoogleFree

# Initialize the free Google search
google_free = SearchGoogleFree(
    num_search_pages=3,
    max_content_words=500
)
```

#### 3.3.2 Available Methods

The `SearchGoogleFree` provides the following callable method:

##### Method: search(query)

**Description**: Searches Google for content matching the query without requiring an API key.

**Usage Example**:
```python
# Search Google without an API key
results = google_free.search(
    query="reinforcement learning algorithms"
)

# Process the results
for i, result in enumerate(results.get("results", [])):
    print(f"Result {i+1}: {result['title']}")
    print(f"URL: {result['url']}")
```

**Return Type**: `dict`

**Sample Return**:
```python
{
    "results": [
        {
            "title": "Introduction to Reinforcement Learning Algorithms",
            "url": "https://example.com/intro-rl",
            "snippet": "A comprehensive overview of reinforcement learning algorithms including Q-learning, SARSA, and policy gradient methods."
        },
        {
            "title": "Top 10 Reinforcement Learning Algorithms for Beginners",
            "url": "https://example.com/top-rl",
            "snippet": "Learn about the most commonly used reinforcement learning algorithms with practical examples and implementation tips."
        }
    ]
}
```

---

## 4. File Operations

EvoAgentX provides tools for handling file operations, including reading and writing files with special support for different file formats like PDFs.

### 4.1 FileTool

**The FileTool provides file handling capabilities with special support for different file formats. It offers standard file operations for text files and specialized handlers for formats like PDF, which use PyPDF2 for reading and basic PDF creation functionality.**

#### 4.1.1 Setup

```python
from evoagentx.tools.file_tool import FileTool

# Initialize the file tool
file_tool = FileTool()

# Get all available tools/methods
available_tools = file_tool.get_tools()
print(f"Available methods: {[tool.__name__ for tool in available_tools]}")
# Output: ['read_file', 'write_file', 'append_file']
```

#### 4.1.2 Available Methods

The `FileTool` provides exactly **3 callable methods** accessible via `get_tools()`:

##### Method 1: read_file(file_path)

**Description**: Reads content from a file with special handling for different file types.

**Usage Example**:
```python
# Read a text file
text_result = file_tool.read_file("examples/sample.txt")
print(text_result)

# Read a PDF file (automatically detected by extension)
pdf_result = file_tool.read_file("examples/document.pdf")
print(pdf_result)
```

**Return Type**: `dict`

**Sample Return**:
```python
# For text files
{
    "success": True,
    "content": "This is the content of the text file.",
    "file_path": "examples/sample.txt",
    "file_type": ".txt"
}

# For PDF files
{
    "success": True,
    "content": "Extracted text from the PDF document...",
    "file_path": "examples/document.pdf",
    "file_type": "pdf",
    "pages": 5
}
```

---

##### Method 2: write_file(file_path, content, mode)

**Description**: Writes content to a file with special handling for different file types.

**Usage Example**:
```python
# Write to a text file
text_result = file_tool.write_file(
    "examples/output.txt", 
    "This is new content for the file."
)

# Write to a PDF file (creates a basic PDF)
pdf_result = file_tool.write_file(
    "examples/new_document.pdf", 
    "This content will be in a PDF."
)
```

**Return Type**: `dict`

**Sample Return**:
```python
{
    "success": True,
    "message": "Content written to examples/output.txt",
    "file_path": "examples/output.txt"
}
```

---

##### Method 3: append_file(file_path, content)

**Description**: Appends content to a file with special handling for different file types.

**Usage Example**:
```python
# Append to a text file
result = file_tool.append_file(
    "examples/log.txt", 
    "\nNew log entry: Operation completed."
)
print(result)
```

**Return Type**: `dict`

**Sample Return**:
```python
{
    "success": True,
    "message": "Content appended to examples/log.txt",
    "file_path": "examples/log.txt"
}
```

#### 4.1.3 Setup Hints

- **File Type Detection**: The tool automatically detects file types based on file extensions and applies appropriate handlers.

- **PDF Support**: 
  - Reading PDFs uses PyPDF2 to extract text content from all pages
  - Writing PDFs creates basic PDF documents (for advanced PDF creation with text formatting, consider using reportlab)
  - PDF appending is currently limited and may require additional libraries

- **Error Handling**: All methods return a dictionary with `success` field indicating whether the operation succeeded, and an `error` field if it failed.

- **File Paths**: Use absolute or relative paths. The tool will create directories if they don't exist when writing files.

---

## 5. Browser Automation

EvoAgentX provides powerful browser automation tools for controlling web browsers to interact with websites and web applications.

### 5.1 BrowserTool

**The BrowserTool provides comprehensive browser automation capabilities using Selenium WebDriver. It allows agents to navigate websites, interact with elements, fill forms, and extract information from web pages with full visual browser control or headless operation.**

#### 5.1.1 Setup and Initialization

```python
from evoagentx.tools.browser_tool import BrowserTool

# Initialize with visible browser window
browser_tool = BrowserTool(
    browser_type="chrome",
    headless=False,
    timeout=10
)

# Initialize with headless browser (no visible window)
browser_tool = BrowserTool(
    browser_type="chrome",
    headless=True,
    timeout=10
)

# Get all available tools/methods
available_tools = browser_tool.get_tools()
print(f"Available methods: {[tool.__name__ for tool in available_tools]}")
# Output: ['initialize_browser', 'navigate_to_url', 'input_text', 'browser_click', 'browser_snapshot', 'browser_console_messages', 'close_browser']

# IMPORTANT: Always initialize the browser before using other methods
result = browser_tool.initialize_browser()
if result["status"] == "success":
    print("Browser is ready for use!")
```

#### 5.1.2 Available Methods

The `BrowserTool` provides exactly **7 callable methods** accessible via `get_tools()`:

##### Method 1: initialize_browser()

**Description**: Starts or restarts a browser session. **MUST be called before any other browser operations.**

**Usage Example**:
```python
# Initialize the browser
result = browser_tool.initialize_browser()
print(result)
```

**Return Type**: `dict`

**Sample Return**:
```python
{
    "status": "success",
    "message": "Browser initialized successfully"
}
```

---

##### Method 2: navigate_to_url(url)

**Description**: Navigates to a specific URL and captures a snapshot of interactive elements.

**Usage Example**:
```python
# Navigate to a website
result = browser_tool.navigate_to_url("https://www.google.com")
print(f"Title: {result['title']}")
print(f"Found {len(result['snapshot']['interactive_elements'])} interactive elements")

# Show available elements
for elem in result['snapshot']['interactive_elements'][:3]:
    print(f"Element {elem['id']}: {elem['description']}")
```

**Return Type**: `dict`

**Sample Return**:
```python
{
    "status": "success",
    "current_url": "https://www.google.com",
    "title": "Google",
    "snapshot": {
        "interactive_elements": [
            {"id": "e0", "description": "Search textbox", "element_type": "input"},
            {"id": "e1", "description": "Google Search button", "element_type": "button"},
            {"id": "e2", "description": "I'm Feeling Lucky button", "element_type": "button"}
        ]
    }
}
```

---

##### Method 3: input_text(element, ref, text, submit)

**Description**: Types text into an input field using element references from snapshots.

**Usage Example**:
```python
# Type text in a search box (element e0 from snapshot)
result = browser_tool.input_text(
    element="Search box",
    ref="e0",
    text="artificial intelligence",
    submit=False
)

# Type and submit with Enter key
result = browser_tool.input_text(
    element="Search box",
    ref="e0", 
    text="machine learning",
    submit=True
)
```

**Return Type**: `dict`

**Sample Return**:
```python
{
    "status": "success",
    "message": "Text input successful",
    "element_ref": "e0"
}
```

---

##### Method 4: browser_click(element, ref)

**Description**: Clicks on buttons, links, or other clickable elements using references from snapshots.

**Usage Example**:
```python
# Click a button (element e1 from snapshot)
result = browser_tool.browser_click(
    element="Search button",
    ref="e1"
)
print(result)
```

**Return Type**: `dict`

**Sample Return**:
```python
{
    "status": "success", 
    "message": "Element clicked successfully",
    "element_ref": "e1"
}
```

---

##### Method 5: browser_snapshot()

**Description**: Captures a fresh snapshot of the current page with all interactive elements.

**Usage Example**:
```python
# Take a new snapshot after page changes
result = browser_tool.browser_snapshot()
print(f"Found {len(result['interactive_elements'])} elements")
```

**Return Type**: `dict`

**Sample Return**:
```python
{
    "title": "Google Search Results", 
    "current_url": "https://www.google.com/search?q=ai",
    "interactive_elements": [
        {"id": "e0", "description": "Search results link", "element_type": "link"},
        {"id": "e1", "description": "Next page button", "element_type": "button"}
    ]
}
```

---

##### Method 6: browser_console_messages()

**Description**: Retrieves JavaScript console messages (logs, warnings, errors) from the browser for debugging.

**Usage Example**:
```python
# Get console messages for debugging
result = browser_tool.browser_console_messages()
print("Console messages:")
for msg in result.get("messages", []):
    print(f"[{msg['level']}] {msg['message']}")
```

**Return Type**: `dict`

**Sample Return**:
```python
{
    "status": "success",
    "messages": [
        {"level": "INFO", "message": "Page loaded successfully", "timestamp": "2024-01-01T12:00:00"},
        {"level": "WARNING", "message": "Deprecated API usage detected", "timestamp": "2024-01-01T12:00:01"}
    ]
}
```

---

##### Method 7: close_browser()

**Description**: Closes the browser and ends the session. **MUST be called when done to free resources.**

**Usage Example**:
```python
# Close the browser
result = browser_tool.close_browser()
print(result)
```

**Return Type**: `dict`

**Sample Return**:
```python
{
    "status": "success",
    "message": "Browser closed successfully"
}
```

#### 5.1.3 Complete Workflow Example

Here's a complete example showing proper initialization and cleanup:

```python
from evoagentx.tools.browser_tool import BrowserTool

# Step 1: Initialize the tool
browser_tool = BrowserTool(headless=False, timeout=10)

try:
    # Step 2: Initialize browser session
    init_result = browser_tool.initialize_browser()
    if init_result["status"] != "success":
        raise Exception(f"Failed to initialize browser: {init_result}")
    
    # Step 3: Navigate to website
    nav_result = browser_tool.navigate_to_url("https://www.google.com")
    elements = nav_result["snapshot"]["interactive_elements"]
    
    # Step 4: Find search elements
    search_input = next((e for e in elements if "search" in e["description"].lower() and "input" in e["description"].lower()), None)
    search_button = next((e for e in elements if "search" in e["description"].lower() and "button" in e["description"].lower()), None)
    
    if search_input and search_button:
        # Step 5: Perform search
        browser_tool.input_text(element="Search box", ref=search_input["id"], text="EvoAgentX")
        browser_tool.browser_click(element="Search button", ref=search_button["id"])
        
        # Step 6: Get console messages for debugging
        console_result = browser_tool.browser_console_messages()
        print(f"Console messages: {len(console_result.get('messages', []))}")
    
except Exception as e:
    print(f"Browser operation failed: {e}")
    
finally:
    # Step 7: ALWAYS close the browser to free resources
    close_result = browser_tool.close_browser()
    print(f"Browser cleanup: {close_result['message']}")
```

#### 5.1.4 Setup Hints

- **Browser Requirements**: 
  - Chrome is the default and most stable option
  - Ensure you have Chrome or ChromeDriver installed for Selenium
  - For other browsers, install the appropriate WebDriver

- **Initialization and Cleanup**:
  - **ALWAYS** call `initialize_browser()` first - no other methods will work without it
  - **ALWAYS** call `close_browser()` when done to free system resources
  - Use try-finally blocks to ensure cleanup happens even if errors occur
  - The browser tool maintains internal state, so proper initialization/cleanup is critical

- **Headless vs Visual Mode**:
  - Set `headless=False` to see the browser window (useful for debugging and demonstrations)
  - Set `headless=True` for production or automated workflows
  - Visual mode helps understand what the automation is doing

- **Element References and Snapshots**:
  - All interactions use element IDs like "e0", "e1", "e2" from snapshots
  - Element IDs are refreshed after navigation or page changes
  - Always use the most recent snapshot's element references
  - The `navigate_to_url()` method automatically captures a snapshot
  - Use `browser_snapshot()` to refresh element references after dynamic content changes

- **Method Execution Order**:
  ```python
  # Required workflow pattern
  browser_tool.initialize_browser()          # 1. Start browser (required first)
  nav_result = browser_tool.navigate_to_url(url)  # 2. Go to page, get elements
  browser_tool.input_text(ref="e0", text="query")  # 3. Use element refs from snapshot
  browser_tool.browser_click(ref="e1")       # 4. Click using element refs
  browser_tool.close_browser()               # 5. Clean up (required last)
  ```

- **Error Handling Best Practices**:
  ```python
  browser_tool = BrowserTool(headless=False)
  try:
      # Always check initialization result
      init_result = browser_tool.initialize_browser()
      if init_result["status"] != "success":
          raise Exception("Browser initialization failed")
      
      # Your browser operations here
      nav_result = browser_tool.navigate_to_url("https://example.com")
      # ... more operations
      
  except Exception as e:
      print(f"Browser operation failed: {e}")
  finally:
      # CRITICAL: Always close browser to free resources
      browser_tool.close_browser()
  ```

- **Timeouts and Performance**:
  - The `timeout` parameter controls how long to wait for elements to load
  - Increase timeout for slow websites or complex pages
  - Use `browser_console_messages()` to debug JavaScript errors or performance issues

---

## 6. MCP Tools

**The Model Context Protocol (MCP) toolkit provides a standardized way to connect to external services through the MCP protocol. It enables agents to access specialized tools like job search services, data processing utilities, and other MCP-compatible APIs without requiring direct integration of each service.**

### 6.1 MCPToolkit

#### 6.1.1 Setup

```python
from evoagentx.tools.mcp import MCPToolkit

# Initialize with a configuration file
mcp_toolkit = MCPToolkit(config_path="examples/sample_mcp.config")

# Or initialize with a configuration dictionary
config = {
    "mcpServers": {
        "hirebase": {
            "command": "uvx",
            "args": ["hirebase-mcp"],
            "env": {"HIREBASE_API_KEY": "your_api_key_here"}
        }
    }
}
mcp_toolkit = MCPToolkit(config=config)
```

#### 6.1.2 Available Methods

The `MCPToolkit` provides the following callable methods:

##### Method 1: get_tools()

**Description**: Returns a list of all available tools from connected MCP servers.

**Usage Example**:
```python
# Get all available MCP tools
tools = mcp_toolkit.get_tools()

# Display available tools
for i, tool in enumerate(tools):
    print(f"Tool {i+1}: {tool.name}")
    print(f"Description: {tool.descriptions[0]}")
```

**Return Type**: `List[Tool]`

**Sample Return**:
```
[MCPTool(name="HirebaseSearch", descriptions=["Search for job information by providing keywords"]), 
 MCPTool(name="HirebaseAnalyze", descriptions=["Analyze job market trends for given skills"])]
```

---

##### Method 2: disconnect()

**Description**: Disconnects from all MCP servers and cleans up resources.

**Usage Example**:
```python
# When done with the MCP toolkit
mcp_toolkit.disconnect()
```

**Return Type**: `None`

#### 6.1.3 Using MCP Tools

Once you have obtained the tools from the MCPToolkit, you can use them like any other EvoAgentX tool:

```python
# Get all tools from the toolkit
tools = mcp_toolkit.get_tools()

# Find a specific tool
hirebase_tool = None
for tool in tools:
    if "hire" in tool.name.lower() or "search" in tool.name.lower():
        hirebase_tool = tool
        break

if hirebase_tool:
    # Use the tool to search for information
    search_query = "data scientist"
    result = hirebase_tool.tools[0](**{"query": search_query})
    
    print(f"Search results for '{search_query}':")
    print(result)
```

#### 6.1.4 Setup Hints

- **Configuration File**: The configuration file should follow the MCP protocol's server configuration format:
  ```json
  {
      "mcpServers": {
          "serverName": {
              "command": "executable_command",
              "args": ["command_arguments"],
              "env": {"ENV_VAR_NAME": "value"}
          }
      }
  }
  ```

- **Server Types**:
  - **Command-based servers**: Use the `command` field to specify an executable
  - **URL-based servers**: Use the `url` field to specify a server endpoint

- **Connection Management**:
  - Always call `disconnect()` when you're done using the MCPToolkit to free resources
  - Use a try-finally block for automatic cleanup:
    ```python
    try:
        toolkit = MCPToolkit(config_path="config.json")
        tools = toolkit.get_tools()
        # Use tools here
    finally:
        toolkit.disconnect()
    ```

- **Error Handling**:
  - The MCPToolkit will log warning messages if it can't connect to servers
  - It's good practice to implement error handling around tool calls:
    ```python
    try:
        result = tool.tools[0](**{"query": "example query"})
    except Exception as e:
        print(f"Error calling MCP tool: {str(e)}")
    ```

- **Environment Variables**:
  - API keys and other sensitive information can be provided via environment variables in the config
  - You can also set them in your environment before running your application

---

## Summary

In this tutorial, we've explored the tool ecosystem in EvoAgentX:

1. **Tool Architecture**: Understood the base Tool class and its standardized interface
2. **Code Interpreters**: Learned how to execute Python code securely using both Python and Docker interpreters
3. **Search Tools**: Discovered how to access web information using Wikipedia and Google search tools
4. **File Operations**: Learned how to handle file operations with special support for different file formats
5. **Browser Automation**: Learned how to control web browsers to interact with websites and web applications
6. **MCP Tools**: Learned how to connect to external services using the Model Context Protocol

Tools in EvoAgentX extend your agents' capabilities by providing access to external resources and computation. By combining these tools with agents and workflows, you can build powerful AI systems that can retrieve information, perform calculations, and interact with the world.

For more advanced usage and customization options, refer to the [API documentation](../api/tools.md) and explore the examples in the repository. 