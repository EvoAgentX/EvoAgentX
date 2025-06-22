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

### 4.1 File Tools

**EvoAgentX provides comprehensive file handling capabilities through both individual file tools and a unified FileToolKit. The tools support standard file operations for text files and specialized handlers for formats like PDF using PyPDF2.**

#### 4.1.1 FileToolKit Usage (Recommended)

The `FileToolKit` provides a convenient way to access all file-related tools:

```python
from evoagentx.tools.file_tool import FileToolKit

# Initialize the file toolkit
file_toolkit = FileToolKit()

# Get all available tools/methods
available_tools = file_toolkit.get_tools()
print(f"Available methods: {[tool.name for tool in available_tools]}")
# Output: ['read_file', 'write_file', 'append_file']

# Get individual tools from the toolkit
read_tool = file_toolkit.get_tool("read_file")
write_tool = file_toolkit.get_tool("write_file")
append_tool = file_toolkit.get_tool("append_file")
```

#### 4.1.2 Available Methods

The `FileToolKit` provides exactly **3 callable methods** accessible via `get_tool()`:

##### Method 1: read_file(file_path)

**Description**: Read content from a file with special handling for different file types like PDFs.

**Usage Example**:
```python
# Read a text file
read_tool = file_toolkit.get_tool("read_file")
text_result = read_tool(file_path="examples/sample.txt")
print(text_result)

# Read a PDF file (automatically detected by extension)
pdf_result = read_tool(file_path="examples/document.pdf")
print(pdf_result)
```

**Parameters**:
- `file_path` (str): Path to the file to read

**Return Type**: `Dict[str, Any]`

**Sample Return**:
```python
{
    "success": True,
    "content": "File content here...",
    "file_path": "examples/sample.txt",
    "file_type": "text"
}
```

---

##### Method 2: write_file(file_path, content)

**Description**: Write content to a file with special handling for different file types like PDFs.

**Usage Example**:
```python
# Write to a text file
write_tool = file_toolkit.get_tool("write_file")
text_result = write_tool(
    file_path="examples/output.txt", 
    content="This is new content for the file."
)

# Write to a PDF file (creates a basic PDF)
pdf_result = write_tool(
    file_path="examples/new_document.pdf", 
    content="This content will be in a PDF."
)
```

**Parameters**:
- `file_path` (str): Path to the file to write
- `content` (str): Content to write to the file

**Return Type**: `Dict[str, Any]`

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

**Description**: Append content to a file with special handling for different file types like PDFs.

**Usage Example**:
```python
# Append to a text file
append_tool = file_toolkit.get_tool("append_file")
result = append_tool(
    file_path="examples/log.txt", 
    content="\nNew log entry: Operation completed."
)
print(result)
```

**Parameters**:
- `file_path` (str): Path to the file to append to
- `content` (str): Content to append to the file

**Return Type**: `Dict[str, Any]`

**Sample Return**:
```python
{
    "success": True,
    "message": "Content appended to examples/log.txt",
    "file_path": "examples/log.txt"
}
```

#### 4.1.3 Direct Tool Usage (Alternative)

You can also import and use individual file tools directly:

```python
from evoagentx.tools.file_tool import ReadFileTool, WriteFileTool, AppendFileTool, FileToolBase

# Create shared file base for special format handling
file_base = FileToolBase()

# Create individual tools
read_tool = ReadFileTool(file_base=file_base)
write_tool = WriteFileTool(file_base=file_base)
append_tool = AppendFileTool(file_base=file_base)

# Use the tools directly
result = read_tool(file_path="example.txt")
```

---

## 5. Browser Tools

EvoAgentX provides comprehensive browser automation capabilities through the `BrowserToolKit` class and individual browser tool classes. These tools allow agents to control web browsers, navigate pages, interact with elements, and extract information.

## Setup

### Using BrowserToolKit (Recommended)

```python
from evoagentx.tools import BrowserToolKit

# Initialize the browser toolkit
toolkit = BrowserToolKit(
    browser_type="chrome",  # Options: "chrome", "firefox", "safari", "edge"  
    headless=False,         # Set to True for background operation
    timeout=10              # Default timeout in seconds
)

# Get specific tools
initialize_tool = toolkit.get_tool("initialize_browser")
navigate_tool = toolkit.get_tool("navigate_to_url")
input_tool = toolkit.get_tool("input_text")
click_tool = toolkit.get_tool("browser_click")
snapshot_tool = toolkit.get_tool("browser_snapshot")
console_tool = toolkit.get_tool("browser_console_messages")
close_tool = toolkit.get_tool("close_browser")
```

### Using Individual Browser Tools (Alternative)

```python
from evoagentx.tools import BrowserTool

# Initialize the browser tool directly
browser = BrowserTool(
    browser_type="chrome",
    headless=False,
    timeout=10
)
```

## Available Methods

### 1. initialize_browser()

Start or restart a browser session. Must be called before any other browser operations.

**Parameters:**
- None required

**Sample Return:**
```python
{
    "status": "success",
    "message": "Browser chrome initialized successfully"
}
```

**Usage:**
```python
# Using ToolKit
result = toolkit.get_tool("initialize_browser")()

# Using BrowserTool directly  
result = browser.initialize_browser()
```

### 2. navigate_to_url(url, timeout=None)

Navigate to a URL and automatically capture a snapshot of all page elements for interaction.

**Parameters:**
- `url` (str, required): Complete URL with protocol (e.g., "https://example.com")
- `timeout` (int, optional): Custom timeout in seconds

**Sample Return:**
```python
{
    "status": "success", 
    "title": "Example Domain",
    "url": "https://example.com",
    "accessibility_tree": {...},  # Full page structure
    "page_content": "Example Domain\n\nThis domain is for use in illustrative examples...",
    "interactive_elements": [
        {
            "id": "e0",
            "description": "More information.../link", 
            "purpose": "link",
            "label": "More information...",
            "category": "navigation",
            "isPrimary": False,
            "visible": True,
            "interactable": True
        }
    ]
}
```

**Usage:**
```python
# Using ToolKit
result = toolkit.get_tool("navigate_to_url")(url="https://example.com")

# Using BrowserTool directly
result = browser.navigate_to_url("https://example.com")
```

### 3. input_text(element, ref, text, submit=False, slowly=True)

Type text into form fields, search boxes, or other input elements using element references from snapshots.

**Parameters:**
- `element` (str, required): Human-readable description (e.g., "Search field", "Username input")
- `ref` (str, required): Element ID from snapshot (e.g., "e0", "e1", "e2") 
- `text` (str, required): Text to input
- `submit` (bool, optional): Press Enter after typing (default: False)
- `slowly` (bool, optional): Type character by character to trigger JS events (default: True)

**Sample Return:**
```python
{
    "status": "success",
    "message": "Successfully input text into Search field and submitted",
    "element": "Search field", 
    "text": "python tutorial"
}
```

**Usage:**
```python
# Using ToolKit
result = toolkit.get_tool("input_text")(
    element="Search field",
    ref="e1", 
    text="python tutorial",
    submit=True
)

# Using BrowserTool directly
result = browser.input_text(
    element="Search field",
    ref="e1",
    text="python tutorial", 
    submit=True
)
```

### 4. browser_click(element, ref)

Click on buttons, links, or other clickable elements using element references from snapshots.

**Parameters:**
- `element` (str, required): Human-readable description (e.g., "Login button", "Next page link")
- `ref` (str, required): Element ID from snapshot (e.g., "e0", "e1", "e2")

**Sample Return:**
```python
{
    "status": "success",
    "message": "Successfully clicked Login button",
    "element": "Login button",
    "new_url": "https://example.com/dashboard"  # If navigation occurred
}
```

**Usage:**
```python
# Using ToolKit
result = toolkit.get_tool("browser_click")(
    element="Login button",
    ref="e3"
)

# Using BrowserTool directly  
result = browser.browser_click(element="Login button", ref="e3")
```

### 5. browser_snapshot()

Capture a fresh snapshot of the current page state, including all interactive elements. Use this after page changes not caused by navigation or clicking.

**Parameters:**
- None required

**Sample Return:**
```python
{
    "status": "success",
    "title": "Search Results - Example",
    "url": "https://example.com/search?q=python",
    "accessibility_tree": {...},  # Complete page structure
    "page_content": "Search Results\n\nResult 1: Python Tutorial...",
    "interactive_elements": [
        {
            "id": "e0",
            "description": "search/search box",
            "purpose": "search box", 
            "label": "Search",
            "category": "search",
            "isPrimary": True,
            "visible": True,
            "editable": True
        },
        {
            "id": "e1", 
            "description": "Search/submit button",
            "purpose": "submit button",
            "label": "Search",
            "category": "action",
            "isPrimary": True,
            "visible": True,
            "interactable": True
        }
    ]
}
```

**Usage:**
```python
# Using ToolKit
result = toolkit.get_tool("browser_snapshot")()

# Using BrowserTool directly
result = browser.browser_snapshot()
```

### 6. browser_console_messages()

Retrieve JavaScript console messages (logs, warnings, errors) for debugging web applications.

**Parameters:**
- None required

**Sample Return:**
```python
{
    "status": "success",
    "console_messages": [
        {
            "level": "INFO",
            "message": "Page loaded successfully",
            "timestamp": "2024-01-15T10:30:45.123Z"
        },
        {
            "level": "WARNING", 
            "message": "Deprecated API usage detected",
            "timestamp": "2024-01-15T10:30:46.456Z"
        },
        {
            "level": "ERROR",
            "message": "Failed to load resource: net::ERR_BLOCKED_BY_CLIENT", 
            "timestamp": "2024-01-15T10:30:47.789Z"
        }
    ]
}
```

**Usage:**
```python
# Using ToolKit
result = toolkit.get_tool("browser_console_messages")()

# Using BrowserTool directly
result = browser.browser_console_messages()
```

### 7. close_browser()

Close the browser session and free system resources. Always call this when finished.

**Parameters:**
- None required

**Sample Return:**
```python
{
    "status": "success",
    "message": "Browser session closed successfully"
}
```

**Usage:**
```python
# Using ToolKit
result = toolkit.get_tool("close_browser")()

# Using BrowserTool directly
result = browser.close_browser()
```

## Element Reference System

The browser tools use a unique element reference system:

1. **Element IDs**: After taking a snapshot, interactive elements are assigned unique IDs like `e0`, `e1`, `e2`, etc.
2. **Element References**: These IDs map internally to specific selectors (CSS, XPath, ID, etc.)
3. **Interactive Elements**: Only elements that can be clicked, typed into, or otherwise interacted with are included
4. **Element Properties**: Each element includes description, purpose, label, category, and visibility information

## Best Practices

### Setup and Initialization
- Always call `initialize_browser()` first
- Use `headless=True` for server environments or background automation
- Set appropriate `timeout` values for slow-loading pages

### Element Interaction
- Always take a snapshot with `navigate_to_url()` or `browser_snapshot()` before interacting with elements
- Use the exact element IDs (`e0`, `e1`, etc.) returned from snapshots
- Provide descriptive `element` parameters to make interactions clear
- Use `submit=True` in `input_text()` for form submissions

### Error Handling and Debugging
- Check return status before proceeding with next operations
- Use `browser_console_messages()` to debug JavaScript errors
- Take fresh snapshots after page state changes
- Handle timeout errors gracefully

### Resource Management
- Always call `close_browser()` when finished
- Only keep one browser session active per tool instance
- Consider using context managers for automatic cleanup

## Complete Example

```python
from evoagentx.tools import BrowserToolKit

# Initialize browser toolkit
toolkit = BrowserToolKit(browser_type="chrome", headless=False)

try:
    # Start browser
    result = toolkit.get_tool("initialize_browser")()
    print(f"Browser init: {result['status']}")
    
    # Navigate to page and get snapshot
    result = toolkit.get_tool("navigate_to_url")(url="https://example.com")
    print(f"Navigation: {result['status']}")
    print(f"Found {len(result['interactive_elements'])} interactive elements")
    
    # Find and interact with elements
    for element in result['interactive_elements']:
        if 'search' in element['purpose'].lower():
            # Input text into search field
            search_result = toolkit.get_tool("input_text")(
                element="Search field",
                ref=element['id'],
                text="python tutorial",
                submit=True
            )
            print(f"Search: {search_result['status']}")
            break
    
    # Take a fresh snapshot after search
    snapshot = toolkit.get_tool("browser_snapshot")()
    print(f"New snapshot: {len(snapshot['interactive_elements'])} elements")
    
    # Check console for any errors
    console = toolkit.get_tool("browser_console_messages")()
    if console['console_messages']:
        print(f"Console messages: {len(console['console_messages'])}")
        
finally:
    # Always close browser
    toolkit.get_tool("close_browser")()
    print("Browser closed")
```

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


