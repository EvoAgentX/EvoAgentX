# Working with Tools in EvoAgentX

This tutorial walks you through using EvoAgentX's powerful tool ecosystem. Tools allow agents to interact with the external world, perform computations, and access information. We'll cover:

1. **Understanding the Tool Architecture**: Learn about the base Tool class and Toolkit system
2. **Code Interpreters**: Execute Python code safely using Python and Docker interpreters
3. **Search Tools**: Access information from the web using Wikipedia and Google search tools
4. **File Operations**: Handle file reading and writing with special support for different file formats
5. **Browser Automation**: Control web browsers using both traditional Selenium-based automation and AI-driven natural language automation
6. **MCP Tools**: Connect to external services using the Model Context Protocol

By the end of this tutorial, you'll understand how to leverage these tools in your own agents and workflows.

---

## 1. Understanding the Tool Architecture

At the core of EvoAgentX's tool ecosystem are the `Tool` base class and the `Toolkit` system, which provide a standardized interface for all tools. 

```python
from evoagentx.tools import FileToolkit, PythonInterpreterToolkit, BrowserToolkit, BrowserUseToolkit
```

The `Tool` class implements a standardized interface with:

- `name`: The tool's unique identifier
- `description`: What the tool does
- `inputs`: Schema defining the tool's parameters
- `required`: List of required parameters
- `__call__()`: The method that executes the tool's functionality

The `Toolkit` system groups related tools together, providing:

- `get_tool(tool_name)`: Returns a specific tool by name
- `get_tools()`: Returns all available tools in the toolkit
- `get_tool_schemas()`: Returns OpenAI-compatible schemas for all tools

### Key Concepts

- **Toolkit Integration**: Tools are organized into toolkits for related functionality
- **Tool Access**: Individual tools are accessed via `toolkit.get_tool(tool_name)`
- **Schemas**: Each tool provides schemas that describe its functionality, parameters, and outputs
- **Modularity**: Toolkits can be easily added to any agent that supports function calling

---

## 2. Code Interpreters

EvoAgentX provides two main code interpreter toolkits:

1. **PythonInterpreterToolkit**: Executes Python code in a controlled environment
2. **DockerInterpreterToolkit**: Executes code within isolated Docker containers

### 2.1 PythonInterpreterToolkit

**The PythonInterpreterToolkit provides a secure environment for executing Python code with fine-grained control over imports, directory access, and execution context. It uses a sandboxing approach to restrict potentially harmful operations.**

#### 2.1.1 Setup

```python
from evoagentx.tools import PythonInterpreterToolkit

# Initialize with specific allowed imports and directory access
toolkit = PythonInterpreterToolkit(
    project_path=".",  # Default is current directory
    directory_names=["examples", "evoagentx"],
    allowed_imports={"os", "sys", "math", "random", "datetime"}
)
```

#### 2.1.2 Available Methods

The `PythonInterpreterToolkit` provides the following tools:

##### Tool 1: python_execute

**Description**: Executes Python code directly in a secure environment.

**Usage Example**:
```python
# Get the execute tool
execute_tool = toolkit.get_tool("python_execute")

# Execute a simple code snippet
result = execute_tool(code="""
print("Hello, World!")
import math
print(f"The value of pi is: {math.pi:.4f}")
""", language="python")

print(result)
```

**Return Type**: `str`

**Sample Return**:
```
Hello, World!
The value of pi is: 3.1416
```

---

##### Tool 2: python_execute_script

**Description**: Executes a Python script file in a secure environment.

**Usage Example**:
```python
# Get the execute script tool
execute_script_tool = toolkit.get_tool("python_execute_script")

# Execute a Python script file
script_result = execute_script_tool(file_path="examples/hello_world.py", language="python")
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
toolkit = PythonInterpreterToolkit(
    project_path=os.getcwd(),
    directory_names=["examples", "evoagentx", "tests"],
    allowed_imports={
        "os", "sys", "time", "datetime", "math", "random", 
        "json", "csv", "re", "collections", "itertools"
    }
)

# Example with no import restrictions
toolkit = PythonInterpreterToolkit(
    project_path=os.getcwd(),
    directory_names=["examples", "evoagentx"],
    allowed_imports=set()  # Allows any module to be imported
)
```

---

### 2.2 DockerInterpreterToolkit

**The DockerInterpreterToolkit executes code in isolated Docker containers, providing maximum security and environment isolation. It allows safe execution of potentially risky code with custom environments, dependencies, and complete resource isolation. Docker must be installed and running on your machine to use this toolkit.**

#### 2.2.1 Setup

```python
from evoagentx.tools import DockerInterpreterToolkit

# Initialize with a specific Docker image
toolkit = DockerInterpreterToolkit(
    image_tag="fundingsocietiesdocker/python3.9-slim",
    print_stdout=True,
    print_stderr=True,
    container_directory="/app"
)
```

#### 2.2.2 Available Methods

The `DockerInterpreterToolkit` provides the following tools:

##### Tool 1: docker_execute

**Description**: Executes code inside a Docker container.

**Usage Example**:
```python
# Get the execute tool
execute_tool = toolkit.get_tool("docker_execute")

# Execute Python code in a Docker container
result = execute_tool(code="""
import platform
print(f"Python version: {platform.python_version()}")
print(f"Platform: {platform.system()} {platform.release()}")
""", language="python")

print(result)
```

**Return Type**: `str`

**Sample Return**:
```
Python version: 3.9.16
Platform: Linux 5.15.0-1031-azure
```

---

##### Tool 2: docker_execute_script

**Description**: Executes a script file inside a Docker container.

**Usage Example**:
```python
# Get the execute script tool
execute_script_tool = toolkit.get_tool("docker_execute_script")

# Execute a Python script file in Docker
script_result = execute_script_tool(file_path="examples/docker_test.py", language="python")
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

- **Docker Requirements**: Ensure Docker is installed and running on your system before using this toolkit.

- **Image Management**: You need to provide **either** an `image_tag` **or** a `dockerfile_path`, not both:
  - **Option 1: Using an existing image**
    ```python
    toolkit = DockerInterpreterToolkit(
        image_tag="python:3.9-slim",  # Uses an existing Docker Hub image
        container_directory="/app"
    )
    ```
  
  - **Option 2: Building from a Dockerfile**
    ```python
    toolkit = DockerInterpreterToolkit(
        dockerfile_path="path/to/Dockerfile",  # Builds a custom image
        image_tag="my-custom-image-name",      # Name for the built image
        container_directory="/app"
    )
    ```

- **File Access**:
  - To make local files available in the container, use the `host_directory` parameter:
  ```python
  toolkit = DockerInterpreterToolkit(
      image_tag="python:3.9-slim",
      host_directory="/path/to/local/files",
      container_directory="/app/data"
  )
  ```
  - This mounts the local directory to the specified container directory, making all files accessible.

- **Container Lifecycle**:
  - The Docker container is created when you initialize the toolkit and removed when the toolkit is destroyed.
  - For long-running sessions, you can set `print_stdout` and `print_stderr` to see real-time output.

- **Troubleshooting**:
  - If you encounter permission issues, ensure your user has Docker privileges.
  - For network-related errors, check if your Docker daemon has proper network access.

---

## 3. Search Tools

EvoAgentX provides several search toolkits to retrieve information from various sources:

1. **WikipediaSearchToolkit**: Search Wikipedia for information
2. **GoogleSearchToolkit**: Search Google using the official API
3. **GoogleFreeSearchToolkit**: Search Google without requiring an API key

### 3.1 WikipediaSearchToolkit

**The WikipediaSearchToolkit retrieves information from Wikipedia articles, providing summaries, full content, and metadata. It offers a straightforward way to incorporate encyclopedic knowledge into your agents without complex API setups.**

#### 3.1.1 Setup

```python
from evoagentx.tools import WikipediaSearchToolkit

# Initialize with custom parameters
toolkit = WikipediaSearchToolkit(max_summary_sentences=3)
```

#### 3.1.2 Available Methods

The `WikipediaSearchToolkit` provides the following callable tool:

##### Tool: wikipedia_search

**Description**: Searches Wikipedia for articles matching the query.

**Usage Example**:
```python
# Get the search tool
search_tool = toolkit.get_tool("wikipedia_search")

# Search Wikipedia for information
results = search_tool(
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

### 3.2 GoogleSearchToolkit

**The GoogleSearchToolkit enables web searches through Google's official Custom Search API, providing high-quality search results with content extraction. It requires API credentials but offers more reliable and comprehensive search capabilities.**

#### 3.2.1 Setup

```python
from evoagentx.tools import GoogleSearchToolkit

# Initialize with custom parameters
toolkit = GoogleSearchToolkit(
    num_search_pages=3,
    max_content_words=200
)
```

#### 3.2.2 Available Methods

The `GoogleSearchToolkit` provides the following callable tool:

##### Tool: google_search

**Description**: Searches Google for content matching the query.

**Usage Example**:
```python
# Get the search tool
search_tool = toolkit.get_tool("google_search")

# Search Google for information
results = search_tool(
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

- **API Requirements**: This toolkit requires Google Custom Search API credentials. Set them in your environment:
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

### 3.3 GoogleFreeSearchToolkit

**The GoogleFreeSearchToolkit provides web search capability without requiring any API keys or authentication. It offers a simpler alternative to the official Google API with basic search results suitable for most general queries.**

#### 3.3.1 Setup

```python
from evoagentx.tools import GoogleFreeSearchToolkit

# Initialize the free Google search toolkit
toolkit = GoogleFreeSearchToolkit(
    num_search_pages=3,
    max_content_words=500
)
```

#### 3.3.2 Available Methods

The `GoogleFreeSearchToolkit` provides the following callable tool:

##### Tool: google_free_search

**Description**: Searches Google for content matching the query without requiring an API key.

**Usage Example**:
```python
# Get the search tool
search_tool = toolkit.get_tool("google_free_search")

# Search Google without an API key
results = search_tool(
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
            "content": "A comprehensive overview of reinforcement learning algorithms including Q-learning, SARSA, and policy gradient methods."
        },
        {
            "title": "Top 10 Reinforcement Learning Algorithms for Beginners",
            "url": "https://example.com/top-rl",
            "content": "Learn about the most commonly used reinforcement learning algorithms with practical examples and implementation tips."
        }
    ]
}
```

---

## 4. File Operations

EvoAgentX provides comprehensive file handling capabilities through the FileToolkit, including reading and writing files with special support for different file formats like PDFs.

### 4.1 FileToolkit

**EvoAgentX provides comprehensive file handling capabilities through the FileToolkit. The toolkit supports standard file operations for text files and specialized handlers for formats like PDF using PyPDF2.**

#### 4.1.1 FileToolkit Usage

The `FileToolkit` provides a convenient way to access all file-related tools:

```python
from evoagentx.tools import FileToolkit

# Initialize the file toolkit
toolkit = FileToolkit()

# Get all available tools
available_tools = toolkit.get_tools()
print(f"Available tools: {[tool.name for tool in available_tools]}")
# Output: ['read_file', 'write_file', 'append_file']

# Get individual tools from the toolkit
read_tool = toolkit.get_tool("read_file")
write_tool = toolkit.get_tool("write_file")
append_tool = toolkit.get_tool("append_file")
```

#### 4.1.2 Available Methods

The `FileToolkit` provides exactly **3 callable tools**:

##### Tool 1: read_file

**Description**: Read content from a file with special handling for different file types like PDFs.

**Usage Example**:
```python
# Read a text file
read_tool = toolkit.get_tool("read_file")
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

##### Tool 2: write_file

**Description**: Write content to a file with special handling for different file types like PDFs.

**Usage Example**:
```python
# Write to a text file
write_tool = toolkit.get_tool("write_file")
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

##### Tool 3: append_file

**Description**: Append content to a file with special handling for different file types like PDFs.

**Usage Example**:
```python
# Append to a text file
append_tool = toolkit.get_tool("append_file")
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

---

## 5. Browser Tools

EvoAgentX provides comprehensive browser automation capabilities through two different toolkits:

1. **BrowserToolkit** (Selenium-based): Provides fine-grained control over browser elements with detailed snapshots and element references
2. **BrowserUseToolkit** (Browser-Use based): Offers natural language browser automation using AI-driven interactions

## Setup

### Option 1: BrowserToolkit (Selenium-based)

Best for: Fine-grained control, detailed element inspection, complex automation workflows

```python
from evoagentx.tools import BrowserToolkit

# Initialize the browser toolkit
toolkit = BrowserToolkit(
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

### Option 2: BrowserUseToolkit (Browser-Use based)

Best for: Natural language interactions, AI-driven automation, simple task descriptions

```python
from evoagentx.tools import BrowserUseToolkit

# Initialize the browser-use toolkit
toolkit = BrowserUseToolkit(
    model="gpt-4o-mini",          # LLM model for browser control
    api_key="your-api-key",       # OpenAI API key (or use environment variable)
    browser_type="chromium",      # Options: "chromium", "firefox", "webkit"
    headless=False                # Set to True for background operation
)

# Get the browser automation tool
browser_tool = toolkit.get_tool("browser_use")
```

## Available Methods

### BrowserToolkit (Selenium-based) Methods

### 1. initialize_browser

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
# Get and use the tool
initialize_tool = toolkit.get_tool("initialize_browser")
result = initialize_tool()
```

### 2. navigate_to_url

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
# Get and use the tool
navigate_tool = toolkit.get_tool("navigate_to_url")
result = navigate_tool(url="https://example.com")
```

### 3. input_text

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
# Get and use the tool
input_tool = toolkit.get_tool("input_text")
result = input_tool(
    element="Search field",
    ref="e1", 
    text="python tutorial",
    submit=True
)
```

### 4. browser_click

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
# Get and use the tool
click_tool = toolkit.get_tool("browser_click")
result = click_tool(
    element="Login button",
    ref="e3"
)
```

### 5. browser_snapshot

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
# Get and use the tool
snapshot_tool = toolkit.get_tool("browser_snapshot")
result = snapshot_tool()
```

### 6. browser_console_messages

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
# Get and use the tool
console_tool = toolkit.get_tool("browser_console_messages")
result = console_tool()
```

### 7. close_browser

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
# Get and use the tool
close_tool = toolkit.get_tool("close_browser")
result = close_tool()
```

---

### BrowserUseToolkit (Browser-Use based) Methods

### browser_use

Execute browser automation tasks using natural language descriptions. This single tool handles all browser interactions through AI-driven automation.

**Parameters:**
- `task` (str, required): Natural language description of the task to perform

**Sample Return:**
```python
{
    "success": True,
    "result": "Successfully navigated to Google and searched for 'OpenAI GPT-4'. Found 10 search results on the page."
}
```

**Usage:**
```python
# Get and use the tool
browser_tool = toolkit.get_tool("browser_use")

# Navigate and search
result = browser_tool(task="Go to Google and search for 'OpenAI GPT-4'")
print(f"Task result: {result}")

# Fill out a form
result = browser_tool(task="Fill out the contact form with name 'John Doe', email 'john@example.com', and message 'Hello world'")
print(f"Form result: {result}")

# Click on specific elements
result = browser_tool(task="Click the 'Sign Up' button and then fill out the registration form")
print(f"Registration result: {result}")
```

**Natural Language Task Examples:**
- "Go to https://example.com and click the login button"
- "Search for 'machine learning' on the current page"
- "Fill out the form with my name and email address"
- "Click the first result in the search results"
- "Navigate to the pricing page and take a screenshot"
- "Find the download button and click it"
- "Scroll down to the bottom of the page and click 'Load More'"

## Element Reference System

The browser tools use a unique element reference system:

1. **Element IDs**: After taking a snapshot, interactive elements are assigned unique IDs like `e0`, `e1`, `e2`, etc.
2. **Element References**: These IDs map internally to specific selectors (CSS, XPath, ID, etc.)
3. **Interactive Elements**: Only elements that can be clicked, typed into, or otherwise interacted with are included
4. **Element Properties**: Each element includes description, purpose, label, category, and visibility information

## Best Practices

### BrowserToolkit (Selenium-based) Best Practices

#### Setup and Initialization
- Always call `initialize_browser()` first
- Use `headless=True` for server environments or background automation
- Set appropriate `timeout` values for slow-loading pages

#### Element Interaction
- Always take a snapshot with `navigate_to_url()` or `browser_snapshot()` before interacting with elements
- Use the exact element IDs (`e0`, `e1`, etc.) returned from snapshots
- Provide descriptive `element` parameters to make interactions clear
- Use `submit=True` in `input_text()` for form submissions

#### Error Handling and Debugging
- Check return status before proceeding with next operations
- Use `browser_console_messages()` to debug JavaScript errors
- Take fresh snapshots after page state changes
- Handle timeout errors gracefully

#### Resource Management
- Always call `close_browser()` when finished
- Only keep one browser session active per toolkit instance
- Consider using context managers for automatic cleanup

### BrowserUseToolkit (Browser-Use based) Best Practices

#### Setup and Initialization
- Ensure you have a valid OpenAI API key set in your environment
- Install the browser-use package: `pip install browser-use`
- Use `headless=True` for server environments or background automation
- Choose the appropriate LLM model for your use case (gpt-4o-mini is cost-effective)

#### Task Description
- Write clear, specific task descriptions in natural language
- Include complete context (e.g., "Go to https://example.com and...")
- Break complex tasks into smaller, sequential steps
- Be specific about what you want to achieve

#### Error Handling
- Check the `success` field in the response before proceeding
- Handle cases where the AI may not complete the task successfully
- Provide fallback logic for critical automation flows

#### Resource Management
- The browser session is managed automatically by the Browser-Use library
- No manual cleanup is required
- Use appropriate model settings to manage API costs

## Complete Examples

### BrowserToolkit (Selenium-based) Example

```python
from evoagentx.tools import BrowserToolkit

# Initialize browser toolkit
toolkit = BrowserToolkit(browser_type="chrome", headless=False)

try:
    # Start browser
    initialize_tool = toolkit.get_tool("initialize_browser")
    result = initialize_tool()
    print(f"Browser init: {result['status']}")
    
    # Navigate to page and get snapshot
    navigate_tool = toolkit.get_tool("navigate_to_url")
    result = navigate_tool(url="https://example.com")
    print(f"Navigation: {result['status']}")
    print(f"Found {len(result['interactive_elements'])} interactive elements")
    
    # Find and interact with elements
    input_tool = toolkit.get_tool("input_text")
    for element in result['interactive_elements']:
        if 'search' in element['purpose'].lower():
            # Input text into search field
            search_result = input_tool(
                element="Search field",
                ref=element['id'],
                text="python tutorial",
                submit=True
            )
            print(f"Search: {search_result['status']}")
            break
    
    # Take a fresh snapshot after search
    snapshot_tool = toolkit.get_tool("browser_snapshot")
    snapshot = snapshot_tool()
    print(f"New snapshot: {len(snapshot['interactive_elements'])} elements")
    
    # Check console for any errors
    console_tool = toolkit.get_tool("browser_console_messages")
    console = console_tool()
    if console['console_messages']:
        print(f"Console messages: {len(console['console_messages'])}")
        
finally:
    # Always close browser
    close_tool = toolkit.get_tool("close_browser")
    close_tool()
    print("Browser closed")
```

### BrowserUseToolkit (Browser-Use based) Example

```python
from evoagentx.tools import BrowserUseToolkit
import os

# Initialize browser-use toolkit
toolkit = BrowserUseToolkit(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY"),  # Set your OpenAI API key
    browser_type="chromium",
    headless=False
)

# Get the browser automation tool
browser_tool = toolkit.get_tool("browser_use")

# Example 1: Simple navigation and search
print("=== Example 1: Search task ===")
result = browser_tool(task="Go to https://google.com and search for 'Python programming tutorial'")
print(f"Search result: {result}")

# Example 2: Form filling
print("\n=== Example 2: Form filling ===")
result = browser_tool(task="Go to https://httpbin.org/forms/post and fill out the form with name 'John Doe' and email 'john@example.com', then submit it")
print(f"Form result: {result}")

# Example 3: Complex navigation
print("\n=== Example 3: Complex navigation ===")
result = browser_tool(task="Go to https://news.ycombinator.com, find the first article, and click on it")
print(f"Navigation result: {result}")

# Example 4: Information extraction
print("\n=== Example 4: Information extraction ===")
result = browser_tool(task="Go to https://example.com and tell me what the main heading says")
print(f"Extraction result: {result}")

print("\nAll browser automation tasks completed!")
```

### Choosing Between the Toolkits

**Use BrowserToolkit when:**
- You need precise control over individual elements
- You want to inspect detailed page structure
- You're building complex automation workflows
- You need to debug specific browser interactions
- You want to minimize API costs (no LLM calls for basic actions)
- You're working with simple, single-page / few-page interactions

**Use BrowserUseToolkit when:**
- You prefer natural language task descriptions
- You want AI-driven decision making in browser interactions
- You're building conversational agents that need to browse the web
- You want to quickly prototype browser automation tasks
- You're comfortable with LLM API costs for enhanced capabilities
- You're working with complex multi-page workflows

## Important Limitations and Requirements

### BrowserToolkit Limitations

**⚠️ Human Verification Issues:**
- **CAPTCHA and Security Checks**: The Selenium-based BrowserToolkit may struggle with human verification systems, CAPTCHAs, and other anti-bot measures

**⚠️ Complex Multi-Page Tasks:**
- **Limited Context**: The toolkit works best with single-page / few-page interactions and may struggle with complex workflows that span multiple pages
- **State Management**: Maintaining application state across page navigations can be challenging
- **Dynamic Content**: Heavily JavaScript-dependent sites with dynamic content loading may cause issues

### BrowserUseToolkit Limitations

**⚠️ Model Performance Dependency:**
- **Weaker Models**: The BrowserUseToolkit may perform poorly with less powerful models like `gpt-4o-mini`
- **Cost Consideration**: More powerful models increase API costs

### Browser Driver Requirements

**🔧 Browser Driver Setup:**

Both toolkits require browser drivers to be installed:

**For BrowserToolkit (Selenium):**
- **Recommended**: Google Chrome with ChromeDriver

**For BrowserUseToolkit (Browser-Use):**
- **Required**: Playwright browser installation
- **Installation**: visit https://github.com/browser-use/browser-use
- **Recommended**: Use Chromium for best compatibility

### Performance Recommendations

**For BrowserToolkit:**
- Use for simple, predictable automation tasks
- Implement proper error handling for network issues
- Consider using headless mode for server deployments
- Test thoroughly with target websites before production use

**For BrowserUseToolkit:**
- Use powerful models (gpt-4o or better) for reliable performance
- Break complex tasks into smaller, sequential steps
- Implement fallback mechanisms for critical workflows
- Monitor API usage and costs carefully
- Test with representative tasks to validate model performance

---

## 6. MCP Tools

**The Model Context Protocol (MCP) Toolkit provides a standardized way to connect to external services through the MCP protocol. It enables agents to access specialized tools like job search services, data processing utilities, and other MCP-compatible APIs without requiring direct integration of each service.**

### 6.1 MCPToolkit

#### 6.1.1 Setup

```python
from evoagentx.tools import MCPToolkit

# Initialize with a configuration file
toolkit = MCPToolkit(config_path="examples/sample_mcp.config")

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
toolkit = MCPToolkit(config=config)
```

#### 6.1.2 Available Methods

The `MCPToolkit` provides the following callable methods:

##### Method 1: get_tools()

**Description**: Returns a list of all available tools from connected MCP servers.

**Usage Example**:
```python
# Get all available MCP tools
tools = toolkit.get_toolkits()

# Display available tools
for i, tool in enumerate(tools):
    print(f"Tool {i+1}: {tool.name}")
    print(f"Description: {tool.description}")
```

**Return Type**: `List[Tool]`

**Sample Return**:
```
[MCPTool(name="HirebaseSearch", description="Search for job information by providing keywords"), 
 MCPTool(name="HirebaseAnalyze", description="Analyze job market trends for given skills")]
```

---

##### Method 2: disconnect()

**Description**: Disconnects from all MCP servers and cleans up resources.

**Usage Example**:
```python
# When done with the MCP toolkit
toolkit.disconnect()
```

**Return Type**: `None`

#### 6.1.3 Using MCP Tools

Once you have obtained the tools from the MCPToolkit, you can use them like any other EvoAgentX tool:

```python
# Get all tools from the toolkit
tools = toolkit.get_toolkits()

# Find a specific tool
hirebase_tool = None
for tool in tools:
    if "hire" in tool.name.lower() or "search" in tool.name.lower():
        hirebase_tool = tool
        break

if hirebase_tool:
    # Use the tool to search for information
    search_query = "data scientist"
    result = hirebase_tool(query=search_query)
    
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
        tools = toolkit.get_toolkits()
        # Use tools here
    finally:
        toolkit.disconnect()
    ```

- **Error Handling**:
  - The MCPToolkit will log warning messages if it can't connect to servers
  - It's good practice to implement error handling around tool calls:
    ```python
    try:
        result = tool(query="example query")
    except Exception as e:
        print(f"Error calling MCP tool: {str(e)}")
    ```

- **Environment Variables**:
  - API keys and other sensitive information can be provided via environment variables in the config
  - You can also set them in your environment before running your application

---

## Summary

In this tutorial, we've explored the tool ecosystem in EvoAgentX:

1. **Tool Architecture**: Understood the base Tool class and Toolkit system providing standardized interfaces
2. **Code Interpreters**: Learned how to execute Python code securely using both Python and Docker interpreter toolkits
3. **Search Tools**: Discovered how to access web information using Wikipedia and Google search toolkits
4. **File Operations**: Learned how to handle file operations with special support for different file formats
5. **Browser Automation**: Learned how to control web browsers using both Selenium-based fine-grained control and AI-driven natural language automation
6. **MCP Tools**: Learned how to connect to external services using the Model Context Protocol

Tools in EvoAgentX extend your agents' capabilities by providing access to external resources and computation. By combining these toolkits with agents and workflows, you can build powerful AI systems that can retrieve information, perform calculations, and interact with the world.

For more advanced usage and customization options, refer to the [API documentation](../api/tools.md) and explore the examples in the repository.


