# Working with Tools in EvoAgentX

This tutorial walks you through using EvoAgentX's powerful tool ecosystem. Tools allow agents to interact with the external world, perform computations, and access information. We'll cover:

1. **Understanding the Tool Architecture**: Learn about the base Tool class and Toolkit system
2. **Code Interpreters**: Execute Python code safely using Python and Docker interpreters
3. **Search Tools**: Access information from the web using Wikipedia and Google search tools
4. **File Operations**: Handle file reading and writing with special support for different file formats
5. **Browser Automation**: Control web browsers using both traditional Selenium-based automation and AI-driven natural language automation
6. **MCP Tools**: Connect to external services using the Model Context Protocol
7. **Image Handling Tools**: Comprehensive capabilities for image analysis, generation, and manipulation using various AI services and APIs

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

## 3. Search and Request Tools

EvoAgentX provides comprehensive search and request toolkits to retrieve information from various sources and perform HTTP operations:

1. **WikipediaSearchToolkit**: Search Wikipedia for information
2. **GoogleSearchToolkit**: Search Google using the official API
3. **GoogleFreeSearchToolkit**: Search Google without requiring an API key
4. **DDGSSearchToolkit**: Search using DuckDuckGo
5. **SerpAPIToolkit**: Multi-engine search (Google, Bing, Baidu, Yahoo, DuckDuckGo)
6. **SerperAPIToolkit**: Google search via SerperAPI
7. **RequestToolkit**: Perform HTTP operations (GET, POST, PUT, DELETE)
8. **ArxivToolkit**: Search for research papers
9. **RSSToolkit**: Fetch and validate RSS feeds

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
            "content": "Full article content here...",
            "url": "https://en.wikipedia.org/wiki/Artificial_intelligence"
        },
        {
            "title": "Intelligent agent",
            "summary": "In artificial intelligence, an intelligent agent (IA) is anything which can perceive its environment, process those perceptions, and respond in pursuit of its own goals...",
            "content": "Full article content here...",
            "url": "https://en.wikipedia.org/wiki/Intelligent_agent"
        }
    ],
    "error": None
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
    ],
    "error": None
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
    ],
    "error": None
}
```

---

### 3.4 DDGSSearchToolkit

**The DDGSSearchToolkit provides web search capabilities using DuckDuckGo, offering privacy-focused search results without requiring API keys. It supports multiple backends and provides comprehensive search results with content extraction.**

#### 3.4.1 Setup

```python
from evoagentx.tools import DDGSSearchToolkit

# Initialize with custom parameters
toolkit = DDGSSearchToolkit(
    num_search_pages=3,
    max_content_words=300,
    backend="auto",  # Options: "auto", "duckduckgo", "google", "bing"
    region="us-en"   # Language and region settings
)
```

#### 3.4.2 Available Methods

The `DDGSSearchToolkit` provides the following callable tool:

##### Tool: ddgs_search

**Description**: Searches the web using DuckDuckGo with optional backend selection.

**Usage Example**:
```python
# Get the search tool
search_tool = toolkit.get_tool("ddgs_search")

# Search using DuckDuckGo
results = search_tool(
    query="machine learning applications",
    num_search_pages=2,
    backend="duckduckgo"
)

# Process the results
for i, result in enumerate(results.get("results", [])):
    print(f"Result {i+1}: {result['title']}")
    print(f"URL: {result['url']}")
    print(f"Content: {result['content'][:100]}...")
```

**Return Type**: `dict`

**Sample Return**:
```python
{
    "results": [
        {
            "title": "Machine Learning Applications in Healthcare",
            "content": "Machine learning is revolutionizing healthcare through predictive analytics, medical imaging analysis, and personalized treatment plans...",
            "url": "https://example.com/ml-healthcare"
        },
        {
            "title": "Top 10 Machine Learning Applications in 2024",
            "content": "From autonomous vehicles to recommendation systems, machine learning is transforming industries across the board...",
            "url": "https://example.com/top-ml-apps"
        }
    ],
    "error": None
}
```

#### 3.4.3 Setup Hints

- **Backend Options**: The toolkit supports multiple search backends:
  - `"auto"`: Automatically selects the best available backend
  - `"duckduckgo"`: Uses DuckDuckGo's search engine
  - `"google"`: Uses Google search (may require additional setup)
  - `"bing"`: Uses Bing search
  - `"brave"`: Uses Brave search
  - `"yahoo"`: Uses Yahoo search

- **Region Settings**: Set the `region` parameter to match your target audience:
  - `"us-en"`: English (United States)
  - `"uk-en"`: English (United Kingdom)
  - `"de-de"`: German (Germany)
  - And many more language-region combinations

---

### 3.5 SerpAPIToolkit

**The SerpAPIToolkit provides access to multiple search engines through SerpAPI, including Google, Bing, Baidu, Yahoo, and DuckDuckGo. It offers comprehensive search results with content scraping capabilities and supports various search parameters.**

#### 3.5.1 Setup

```python
from evoagentx.tools import SerpAPIToolkit

# Initialize with custom parameters
toolkit = SerpAPIToolkit(
    num_search_pages=3,
    max_content_words=300,
    enable_content_scraping=True  # Enable content extraction from search results
)
```

#### 3.5.2 Available Methods

The `SerpAPIToolkit` provides the following callable tool:

##### Tool: serpapi_search

**Description**: Searches multiple engines using SerpAPI with comprehensive result processing.

**Usage Example**:
```python
# Get the search tool
search_tool = toolkit.get_tool("serpapi_search")

# Search using Google engine
results = search_tool(
    query="artificial intelligence trends 2024",
    num_search_pages=3,
    max_content_words=300,
    engine="google",        # Options: "google", "bing", "baidu", "yahoo", "duckduckgo"
    location="United States",
    language="en"
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
            "title": "AI Trends 2024: What's Next in Artificial Intelligence",
            "content": "The artificial intelligence landscape in 2024 is marked by significant advances in generative AI, multimodal models, and AI governance frameworks...",
            "url": "https://example.com/ai-trends-2024",
            "type": "organic",
            "priority": 2,
            "position": 1,
            "site_content": "Full scraped content from the webpage..."
        },
        {
            "title": "Knowledge: Artificial Intelligence",
            "content": "**Artificial Intelligence**\n\nAI is the simulation of human intelligence in machines...",
            "url": "https://example.com/ai-knowledge",
            "type": "knowledge_graph",
            "priority": 1
        }
    ],
    "raw_data": {
        "news_results": [...],
        "related_questions": [...]
    },
    "search_metadata": {
        "query": "artificial intelligence trends 2024",
        "location": "United States",
        "total_results": "1,234,567",
        "search_time": "0.45"
    },
    "error": None
}
```

#### 3.5.3 Setup Hints

- **API Requirements**: This toolkit requires a SerpAPI key. Set it in your environment:
  ```python
  # In your .env file or environment variables
  SERPAPI_KEY=your_serpapi_key_here
  ```

- **Engine Selection**: Choose the search engine that best fits your needs:
  - `"google"`: Most comprehensive results, good for general queries
  - `"bing"`: Good for news and current events
  - `"baidu"`: Excellent for Chinese language content
  - `"yahoo"`: Good for news and finance
  - `"duckduckgo"`: Privacy-focused, no tracking
  - `"brave"`: Privacy-focused search engine

- **Content Scraping**: Enable `enable_content_scraping=True` to extract full content from search results, providing richer information for analysis.

---

### 3.6 SerperAPIToolkit

**The SerperAPIToolkit provides Google search capabilities through SerperAPI, offering high-quality search results with content extraction. It's an alternative to the official Google API with simplified setup and comprehensive search capabilities.**

#### 3.6.1 Setup

```python
from evoagentx.tools import SerperAPIToolkit

# Initialize with custom parameters
toolkit = SerperAPIToolkit(
    num_search_pages=3,
    max_content_words=300,
    enable_content_scraping=True  # Enable content extraction
)
```

#### 3.6.2 Available Methods

The `SerperAPIToolkit` provides the following callable tool:

##### Tool: serperapi_search

**Description**: Searches Google using SerperAPI with content extraction capabilities.

**Usage Example**:
```python
# Get the search tool
search_tool = toolkit.get_tool("serperapi_search")

# Search Google with content extraction
results = search_tool(
    query="deep learning frameworks comparison",
    num_search_pages=3,
    max_content_words=300,
    location="United States",
    language="en"
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
            "title": "Deep Learning Framework Comparison: TensorFlow vs PyTorch",
            "content": "A comprehensive comparison of the two most popular deep learning frameworks, covering performance, ease of use, and community support...",
            "url": "https://example.com/dl-framework-comparison",
            "type": "organic",
            "priority": 2,
            "position": 1,
            "site_content": "Full scraped content from the webpage..."
        },
        {
            "title": "Knowledge: Deep Learning Frameworks",
            "content": "**Deep Learning Frameworks**\n\nSoftware libraries that provide tools for building and training neural networks...",
            "url": "https://example.com/dl-knowledge",
            "type": "knowledge_graph",
            "priority": 1
        }
    ],
    "raw_data": {
        "relatedSearches": [...]
    },
    "search_metadata": {
        "query": "deep learning frameworks comparison",
        "engine": "google",
        "type": "search",
        "credits": 100
    },
    "error": None
}
```

#### 3.6.3 Setup Hints

- **API Requirements**: This toolkit requires a SerperAPI key. Set it in your environment:
  ```python
  # In your .env file or environment variables
  SERPERAPI_KEY=your_serperapi_key_here
  ```

- **Content Extraction**: Enable `enable_content_scraping=True` to get full content from search results, providing richer information for analysis and processing.

- **Location and Language**: Use the `location` and `language` parameters to get region-specific and language-specific results.

---

### 3.7 RequestToolkit

**The RequestToolkit provides comprehensive HTTP operations for making web requests, including GET, POST, PUT, and DELETE operations. It's essential for building agents that need to interact with web APIs and services.**

#### 3.7.1 Setup

```python
from evoagentx.tools import RequestToolkit

# Initialize the request toolkit
toolkit = RequestToolkit(name="DemoRequestToolkit")
```

#### 3.7.2 Available Methods

The `RequestToolkit` provides the following callable tool:

##### Tool: http_request

**Description**: Performs HTTP requests with support for all major HTTP methods and data types.

**Usage Example**:
```python
# Get the HTTP request tool
http_tool = toolkit.get_tool("http_request")

# GET request with query parameters
get_result = http_tool(
    url="https://httpbin.org/get",
    method="GET",
    params={"test": "param", "example": "value"}
)

# POST request with JSON data
post_result = http_tool(
    url="https://httpbin.org/post",
    method="POST",
    json_data={"name": "Test User", "email": "test@example.com"},
    headers={"Content-Type": "application/json"}
)

# PUT request with form data
put_result = http_tool(
    url="https://httpbin.org/put",
    method="PUT",
    data={"update": "new value", "timestamp": "2024-01-01"}
)

# DELETE request
delete_result = http_tool(
    url="https://httpbin.org/delete",
    method="DELETE"
)
```

**Parameters**:
- `url` (str, required): The target URL for the request
- `method` (str, required): HTTP method (GET, POST, PUT, DELETE, etc.)
- `params` (dict, optional): Query parameters for GET requests
- `data` (dict, optional): Form data for POST/PUT requests
- `json_data` (dict, optional): JSON data for POST/PUT requests
- `headers` (dict, optional): Custom HTTP headers
- `return_raw` (bool, optional): If true, return raw response content; if false, return processed content (default: false)
- `save_file_path` (str, optional): Optional file path to save the response content

**Return Type**: `dict`

**Sample Return**:
```python
{
    "success": True,
    "status_code": 200,
    "content": "Response content here...",
    "url": "https://httpbin.org/get",
    "method": "GET",
    "content_type": "application/json",
    "content_length": 1234,
    "headers": {"Content-Type": "application/json"}
}
```

#### 3.7.3 Setup Hints

- **HTTP Methods**: The toolkit supports all standard HTTP methods:
  - `GET`: Retrieve data (use `params` for query parameters)
  - `POST`: Submit data (use `data` for form data or `json_data` for JSON)
  - `PUT`: Update data (use `data` for form data or `json_data` for JSON)
  - `DELETE`: Remove data

- **Data Types**: Choose the appropriate data parameter:
  - `params`: For query parameters in GET requests
  - `data`: For form-encoded data
  - `json_data`: For JSON payloads

- **Error Handling**: Always check the `success` field in the response before processing the content.

---

### 3.8 ArxivToolkit

**The ArxivToolkit provides access to arXiv, the preprint repository for physics, mathematics, computer science, and other scientific disciplines. It enables agents to search for and retrieve research papers and academic content.**

#### 3.8.1 Setup

```python
from evoagentx.tools import ArxivToolkit

# Initialize the arXiv toolkit
toolkit = ArxivToolkit()
```

#### 3.8.2 Available Methods

The `ArxivToolkit` provides the following callable tool:

##### Tool: arxiv_search

**Description**: Searches arXiv for research papers matching the query.

**Usage Example**:
```python
# Get the search tool
search_tool = toolkit.get_tool("arxiv_search")

# Search for research papers
results = search_tool(
    search_query="all:machine learning",
    max_results=5
)

# Process the results
if results.get('success'):
    papers = results.get('papers', [])
    for i, paper in enumerate(papers):
        print(f"Paper {i+1}: {paper.get('title', 'No title')}")
        print(f"  Authors: {', '.join(paper.get('authors', ['Unknown']))}")
        print(f"  arXiv ID: {paper.get('arxiv_id', 'Unknown')}")
        print(f"  URL: {paper.get('url', 'No URL')}")
```

**Parameters**:
- `search_query` (str, required): Search query in arXiv format
- `max_results` (int, optional): Maximum number of results to return

**Return Type**: `dict`

**Sample Return**:
```python
{
    "success": True,
    "papers": [
        {
            "title": "Deep Learning for Natural Language Processing",
            "authors": ["Smith, J.", "Johnson, A."],
            "arxiv_id": "2401.00123",
            "url": "https://arxiv.org/abs/2401.00123",
            "summary": "This paper presents a comprehensive survey of deep learning approaches...",
            "published_date": "2024-01-01T00:00:00Z",
            "categories": ["cs.AI", "cs.CL"],
            "primary_category": "cs.AI",
            "links": {
                "html": "https://arxiv.org/abs/2401.00123",
                "pdf": "https://arxiv.org/pdf/2401.00123"
            }
        }
    ]
}
```

#### 3.8.3 Setup Hints

- **Query Format**: Use arXiv's search syntax for best results:
  - `all:machine learning`: Search all fields for "machine learning"
  - `ti:neural networks`: Search title for "neural networks"
  - `au:Smith`: Search author for "Smith"
  - `cat:cs.AI`: Search computer science AI category

- **Categories**: arXiv uses category codes for different fields:
  - `cs.AI`: Artificial Intelligence
  - `cs.LG`: Machine Learning
  - `cs.CL`: Computation and Language
  - `cs.CV`: Computer Vision and Pattern Recognition

---

### 3.9 RSSToolkit

**The RSSToolkit provides functionality to fetch, validate, and process RSS feeds from various sources. It enables agents to monitor news sources, blogs, and other regularly updated content.**

#### 3.9.1 Setup

```python
from evoagentx.tools import RSSToolkit

# Initialize the RSS toolkit
toolkit = RSSToolkit(name="DemoRSSToolkit")
```

#### 3.9.2 Available Methods

The `RSSToolkit` provides the following callable tools:

##### Tool 1: rss_fetch

**Description**: Fetches RSS feeds and returns the latest entries.

**Usage Example**:
```python
# Get the fetch tool
fetch_tool = toolkit.get_tool("rss_fetch")

# Fetch RSS feed
results = fetch_tool(
    feed_url="https://feeds.bbci.co.uk/news/rss.xml",
    max_entries=5
)

# Process the results
if results.get('success'):
    entries = results.get('entries', [])
    print(f"Fetched {len(entries)} entries from '{results.get('title')}'")
    
    for entry in entries:
        print(f"Title: {entry.get('title', 'No title')}")
        print(f"Published: {entry.get('published', 'Unknown')}")
        print(f"Link: {entry.get('link', 'No link')}")
```

**Parameters**:
- `feed_url` (str, required): URL of the RSS feed to fetch
- `max_entries` (int, optional): Maximum number of entries to return (default: 10)
- `fetch_webpage_content` (bool, optional): Whether to fetch and extract content from article webpages (default: true)

**Return Type**: `dict`

**Sample Return**:
```python
{
    "success": True,
    "title": "BBC News",
    "entries": [
        {
            "title": "Breaking News: AI Breakthrough",
            "published": "2024-01-01T10:00:00Z",
            "link": "https://bbc.com/news/ai-breakthrough",
            "author": "BBC News",
            "summary": "Scientists announce major breakthrough in artificial intelligence...",
            "description": "Detailed description of the AI breakthrough...",
            "tags": ["AI", "Technology", "Science"],
            "categories": ["Technology"],
            "webpage_content": "Full webpage content if fetched...",
            "webpage_content_fetched": true
        }
    ]
}
```

---

##### Tool 2: rss_validate

**Description**: Validates RSS feeds to check if they are accessible and properly formatted.

**Usage Example**:
```python
# Get the validate tool
validate_tool = toolkit.get_tool("rss_validate")

# Validate RSS feed
result = validate_tool(url="https://feeds.bbci.co.uk/news/rss.xml")

# Check validation result
if result.get('success') and result.get('is_valid'):
    print(f"✓ Valid {result.get('feed_type')} feed: {result.get('title', 'Unknown')}")
else:
    print(f"❌ Invalid feed: {result.get('error', 'Unknown error')}")
```

**Parameters**:
- `url` (str, required): URL of the RSS feed to validate

**Return Type**: `dict`

**Sample Return**:
```python
{
    "success": True,
    "is_valid": True,
    "feed_type": "RSS",
    "title": "BBC News",
    "description": "Latest news from BBC"
}
```

#### 3.9.3 Setup Hints

- **Feed Sources**: Popular RSS feeds include:
  - News: BBC, CNN, Reuters
  - Tech: TechCrunch, Ars Technica
  - Science: Nature, Science Daily
  - Blogs: Personal and professional blogs

- **Validation**: Always validate RSS feeds before processing to ensure they are accessible and properly formatted.

- **Rate Limiting**: Be respectful of feed sources and implement appropriate delays between requests.

---

## Summary of Search and Request Tools

The search and request tools in EvoAgentX provide comprehensive access to information from various sources:

| Toolkit | Purpose | API Key Required | Best For |
|---------|---------|------------------|----------|
| **WikipediaSearchToolkit** | Encyclopedic knowledge | ❌ | General information, definitions |
| **GoogleSearchToolkit** | Web search (official API) | ✅ | High-quality, reliable results |
| **GoogleFreeSearchToolkit** | Web search (no API) | ❌ | Simple queries, no setup |
| **DDGSSearchToolkit** | Privacy-focused search | ❌ | Privacy-conscious applications |
| **SerpAPIToolkit** | Multi-engine search | ✅ | Comprehensive, multi-source results |
| **SerperAPIToolkit** | Google search alternative | ✅ | Google results with content extraction |
| **RequestToolkit** | HTTP operations | ❌ | API interactions, web scraping |
| **ArxivToolkit** | Research papers | ❌ | Academic research, scientific content |
| **RSSToolkit** | News and updates | ❌ | Real-time information, monitoring |

Choose the appropriate toolkit based on your specific needs, API key availability, and the type of information you need to retrieve.

## 4. FileSystem Tools

FileSystem tools provide capabilities for file operations, storage management, and command-line execution. These tools are essential for managing data persistence, file manipulation, and system interactions.

### 4.1 StorageToolkit

**The StorageToolkit provides comprehensive file storage operations including saving, loading, appending, and managing various file formats with flexible storage backends.**

#### 4.1.1 Setup

```python
from evoagentx.tools import StorageToolkit
from evoagentx.tools.storage_file import LocalStorageHandler

# Initialize with local storage
storage_handler = LocalStorageHandler(base_path="./data")
toolkit = StorageToolkit(storage_handler=storage_handler)

# Or use default storage
toolkit = StorageToolkit()  # Uses LocalStorageHandler with current directory
```

#### 4.1.2 Available Methods

```python
# Get available tools
tools = toolkit.get_tools()
print(f"Available tools: {[tool.name for tool in tools]}")

# Available tools:
# - save: Save content to files
# - read: Read content from files
# - append: Append content to existing files
# - list_files: List files in storage directory
# - delete: Delete files
# - exists: Check if file exists
# - list_supported_formats: List supported file formats
```

#### 4.1.3 Usage Example

```python
# Save text content
save_result = toolkit.save(
    content="Hello, this is a test file!",
    file_path="test.txt"
)

# Save JSON content
import json
json_data = {"name": "test", "value": 123}
save_result = toolkit.save(
    content=json.dumps(json_data),
    file_path="data.json"
)

# Load content
read_result = toolkit.read(file_path="test.txt")
print(f"Loaded content: {read_result}")

# Append content
append_result = toolkit.append(
    content="\nThis is appended content.",
    file_path="test.txt"
)

# List files
list_result = toolkit.list_files(path=".", max_depth=2, include_hidden=False)
print(f"Files in directory: {list_result}")

# Check if file exists
exists_result = toolkit.exists(path="test.txt")
print(f"File exists: {exists_result}")

# Delete file
delete_result = toolkit.delete(file_path="test.txt")

# List supported formats
formats_result = toolkit.list_supported_formats()
print(f"Supported formats: {formats_result}")
```

#### 4.1.4 Parameters

**save:**
- `file_path` (str): Path where to save the file
- `content` (str): Content to save
- `encoding` (str, optional): File encoding (default: "utf-8")
- `indent` (int, optional): Indentation for JSON files
- `sheet_name` (str, optional): Sheet name for Excel files
- `root_tag` (str, optional): Root tag for XML files

**read:**
- `file_path` (str): Path of the file to read
- `encoding` (str, optional): File encoding (default: "utf-8")
- `sheet_name` (str, optional): Sheet name for Excel files
- `head` (int, optional): Number of characters to return (default: 0 means return everything)

**append:**
- `file_path` (str): Path of the file to append to
- `content` (str): Content to append
- `encoding` (str, optional): File encoding (default: "utf-8")

**list_files:**
- `path` (str, optional): Directory to list (default: current directory)
- `max_depth` (int, optional): Maximum depth for recursive listing (default: 3)
- `include_hidden` (bool, optional): Whether to include hidden files (default: False)

**exists:**
- `path` (str): Path of the file to check

**delete:**
- `file_path` (str): Path of the file to delete

**list_supported_formats:**
- No parameters required

#### 4.1.5 Return Type

All tools return `dict` with success/error information.

#### 4.1.6 Sample Return

```python
# Success response for save
{
    "success": True,
    "message": "File 'test.txt' created successfully",
    "file_path": "./data/test.txt",
    "full_path": "/absolute/path/to/data/test.txt",
    "size": 45
}

# Success response for read
{
    "success": True,
    "message": "File 'test.txt' read successfully",
    "file_path": "./data/test.txt",
    "full_path": "/absolute/path/to/data/test.txt",
    "content": "Hello, this is a test file!",
    "size": 45
}

# Error response
{
    "success": False,
    "message": "Error creating file: Permission denied",
    "file_path": "./data/test.txt"
}
```

#### 4.1.7 Setup Hints

- **Storage Backends**: The toolkit supports different storage handlers:
  - `LocalStorageHandler`: Local file system storage
  - `FileStorageHandler`: Abstract base class for custom implementations
  - Custom handlers can be implemented for cloud storage, databases, etc.

- **Base Path**: Set a base path for organized file storage:
  ```python
  storage_handler = LocalStorageHandler(base_path="./project_data")
  ```

- **File Formats**: Supports any text-based format (txt, json, csv, yaml, etc.)

---

### 4.2 CMDToolkit

**The CMDToolkit provides command-line execution capabilities, allowing you to run system commands, scripts, and shell operations with proper timeout handling and result processing.**

#### 4.2.1 Setup

```python
from evoagentx.tools import CMDToolkit

# Initialize with default settings
toolkit = CMDToolkit()

# Or customize settings
toolkit = CMDToolkit(
    timeout=30,  # Command timeout in seconds
    working_directory="./scripts"  # Default working directory
)
```

#### 4.2.2 Available Methods

```python
# Get available tools
tools = toolkit.get_tools()
print(f"Available tools: {[tool.name for tool in tools]}")

# Available tools:
# - execute_command: Execute command-line commands
```

#### 4.2.3 Usage Example

```python
# Execute a simple command
result = toolkit.execute_command(command="echo 'Hello, World!'")
print(f"Command output: {result}")

# Execute with working directory
result = toolkit.execute_command(
    command="pwd",
    working_directory="/tmp"
)

# Execute with timeout
result = toolkit.execute_command(
    command="sleep 10",
    timeout=5  # Will timeout after 5 seconds
)

# Execute complex command
result = toolkit.execute_command(
    command="ls -la | grep '\.py$'",
    working_directory="./src"
)

# Cross-platform commands
import platform
if platform.system() == "Windows":
    result = toolkit.execute_command(command="dir")
else:
    result = toolkit.execute_command(command="ls -la")
```

#### 4.2.4 Parameters

**execute_command:**
- `command` (str): The command to execute
- `working_directory` (str, optional): Working directory for the command
- `timeout` (int, optional): Timeout in seconds (overrides toolkit default)

#### 4.2.5 Return Type

Returns `dict` with command execution results.

#### 4.2.6 Sample Return

```python
# Success response
{
    "success": True,
    "command": "echo 'Hello, World!'",
    "stdout": "Hello, World!\n",
    "stderr": "",
    "return_code": 0,
    "system": "linux",
    "shell": "bash",
    "storage_handler": "LocalStorageHandler",
    "storage_base_path": "./workplace/cmd"
}

# Error response
{
    "success": False,
    "error": "Command timed out after 5 seconds",
    "command": "sleep 10",
    "stdout": "",
    "stderr": "",
    "return_code": None
}

# Command failure
{
    "success": False,
    "error": "Permission denied by user",
    "command": "rm -rf /",
    "stdout": "",
    "stderr": "",
    "return_code": None
}
```

#### 4.2.7 Setup Hints

- **Timeout Handling**: Always set appropriate timeouts for long-running commands
- **Working Directory**: Use working directory to execute commands in specific locations
- **Cross-Platform**: Commands should work on both Windows and Unix-like systems
- **Security**: Be careful with user input in commands to prevent command injection
- **Error Handling**: Check both `success` and `return_code` for proper error handling

---

### 4.3 Storage Handler Introduction

**Storage handlers provide the underlying storage abstraction for the StorageToolkit, allowing you to implement custom storage backends for different environments and requirements.**

#### 4.3.1 Available Storage Handlers

**LocalStorageHandler:**
```python
from evoagentx.tools.storage_file import LocalStorageHandler

# Basic local storage
handler = LocalStorageHandler()

# With custom base path
handler = LocalStorageHandler(base_path="./data")

# With custom encoding
handler = LocalStorageHandler(encoding="utf-8")
```

**FileStorageHandler (Abstract Base):**
```python
from evoagentx.tools.storage_handler import FileStorageHandler

class CustomStorageHandler(FileStorageHandler):
    def __init__(self, bucket_name: str, credentials: dict):
        self.bucket_name = bucket_name
        self.credentials = credentials
    
    def create_file(self, content: str, file_path: str, encoding: str = "utf-8") -> dict:
        # Custom save implementation
        pass
    
    def read_file(self, file_path: str, encoding: str = "utf-8") -> dict:
        # Custom load implementation
        pass
    
    def update_file(self, content: str, file_path: str, encoding: str = "utf-8") -> dict:
        # Custom update implementation
        pass
```

#### 4.3.2 Storage Handler Methods

All storage handlers implement these core methods:

- **`create_file(content, file_path, encoding)`**: Create/save content to file
- **`read_file(file_path, encoding)`**: Read content from file  
- **`update_file(content, file_path, encoding)`**: Update content in file
- **`delete_file(file_path)`**: Delete file
- **`list_files(path, max_depth, include_hidden)`**: List files in directory
- **`exists(path)`**: Check if file exists

#### 4.3.3 Custom Storage Implementation

```python
class CloudStorageHandler(FileStorageHandler):
    def __init__(self, bucket_name: str, credentials: dict):
        self.bucket_name = bucket_name
        self.credentials = credentials
    
    def create_file(self, content: str, file_path: str, encoding: str = "utf-8") -> dict:
        try:
            # Upload to cloud storage
            # ... cloud-specific implementation
            return {
                "success": True,
                "message": "File uploaded to cloud storage",
                "file_path": file_path,
                "file_size": len(content.encode(encoding))
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "file_path": file_path
            }
```

#### 4.3.4 Setup Hints

- **Base Path**: Always set a meaningful base path for organized storage
- **Encoding**: Use UTF-8 for international character support
- **Error Handling**: Implement proper error handling in custom handlers
- **Permissions**: Ensure proper file permissions for read/write operations
- **Backup**: Consider implementing backup strategies for critical data

---

### 4.4 FileSystem Tools Summary

| Tool | Purpose | Key Features | Use Cases |
|------|---------|--------------|-----------|
| **StorageToolkit** | File operations | Save, load, append, list, delete | Data persistence, file management |
| **CMDToolkit** | Command execution | Shell commands, timeout handling | System administration, automation |
| **Storage Handler** | Storage abstraction | Custom backends, cloud storage | Flexible storage solutions |

**Common Use Cases:**
- **Data Persistence**: Save and load application data, configurations, logs
- **File Management**: Organize, backup, and manage project files
- **System Automation**: Execute scripts, manage services, monitor systems
- **Cross-Platform**: Work consistently across different operating systems
- **Custom Storage**: Implement cloud storage, database storage, or other backends

**Best Practices:**
- Always handle errors gracefully
- Use appropriate timeouts for long-running operations
- Implement proper file path validation
- Consider security implications of command execution
- Use meaningful base paths for organized storage

## 5. Database Tools

Database tools provide comprehensive database management capabilities including relational databases (PostgreSQL), document databases (MongoDB), and vector databases (FAISS). These tools enable agents to perform complex data operations, semantic search, and data persistence with automatic storage management.

### 5.1 MongoDBToolkit

**The MongoDBToolkit provides comprehensive document database operations for MongoDB, including querying, inserting, updating, and deleting documents with support for complex queries, aggregation pipelines, and metadata filtering.**

#### 5.1.1 Setup

```python
from evoagentx.tools import MongoDBToolkit

# Initialize with default storage
toolkit = MongoDBToolkit(
    name="DemoMongoDBToolkit",
    database_name="demo_db",
    auto_save=True
)

# Or with custom configuration
toolkit = MongoDBToolkit(
    name="CustomMongoDBToolkit",
    database_name="my_database",
    auto_save=False,
    host="localhost",
    port=27017
)
```

#### 5.1.2 Available Methods

```python
# Get available tools
tools = toolkit.get_tools()
print(f"Available tools: {[tool.name for tool in tools]}")

# Available tools:
# - mongodb_execute_query: Execute MongoDB queries and aggregation pipelines
# - mongodb_find: Find documents with filtering, projection, and sorting
# - mongodb_update: Update documents in collections
# - mongodb_delete: Delete documents with filters
# - mongodb_info: Get database and collection information
```

#### 5.1.3 Usage Example

```python
# Execute complex queries
execute_tool = toolkit.get_tool("mongodb_execute_query")
find_tool = toolkit.get_tool("mongodb_find")
delete_tool = toolkit.get_tool("mongodb_delete")

# Insert products data
products = [
    {"id": "P001", "name": "Laptop", "category": "Electronics", "price": 999.99, "stock": 50},
    {"id": "P002", "name": "Mouse", "category": "Electronics", "price": 29.99, "stock": 100},
    {"id": "P003", "name": "Desk Chair", "category": "Furniture", "price": 199.99, "stock": 25}
]

# Insert using execute tool
result = execute_tool(
    query=json.dumps(products),
    query_type="insert",
    collection_name="products"
)

# Find electronics products
find_result = find_tool(
    collection_name="products",
    filter='{"category": "Electronics"}',
    sort='{"price": -1}'
)

# Delete furniture products
delete_result = delete_tool(
    collection_name="products",
    filter='{"category": "Furniture"}',
    multi=True
)
```

#### 5.1.4 Parameters

**mongodb_execute_query:**
- `query` (str): MongoDB query (JSON string for find, array for aggregation)
- `query_type` (str, optional): Type of query (select, aggregate) - auto-detected if not provided
- `collection_name` (str): Collection name (required for all operations)

**mongodb_find:**
- `collection_name` (str): Collection name to query
- `filter` (str, optional): MongoDB filter query (JSON string, e.g., '{"age": {"$gt": 18}}')
- `projection` (str, optional): Fields to include/exclude (JSON string, e.g., '{"name": 1, "_id": 0}')
- `sort` (str, optional): Sort criteria (JSON string, e.g., '{"age": -1}')
- `limit` (int, optional): Maximum number of documents to return
- `skip` (int, optional): Number of documents to skip

**mongodb_delete:**
- `collection_name` (str): Collection name to delete from
- `filter` (str): MongoDB filter query (JSON string)
- `multi` (bool): Whether to delete multiple documents (required parameter)

#### 5.1.5 Return Type

All tools return `dict` with success/error information and data. The actual return structure follows a standardized format:

```python
{
    "success": True,                    # Boolean indicating success/failure
    "data": {...},                      # Actual result data
    "query_type": "select",             # Type of query executed
    "execution_time": 0.001,            # Time taken to execute
    "row_count": 2,                     # Number of rows/documents affected
    "metadata": {...}                   # Additional metadata (collection names, filters, etc.)
}
```

#### 5.1.6 Sample Return

```python
# Success response for find
{
    "success": True,
    "data": [
        {
            "id": "P001",
            "name": "Laptop",
            "category": "Electronics",
            "price": 999.99,
            "stock": 50
        },
        {
            "id": "P002",
            "name": "Mouse",
            "category": "Electronics",
            "price": 29.99,
            "stock": 100
        }
    ],
    "query_type": "select",
    "execution_time": 0.001,
    "row_count": 2,
    "metadata": {
        "collection_name": "products",
        "filter_applied": {"category": "Electronics"}
    }
}

# Success response for delete
{
    "success": True,
    "data": {
        "deleted_count": 1
    },
    "query_type": "delete",
    "execution_time": 0.002,
    "row_count": 1,
    "metadata": {
        "collection_name": "products"
    }
}
```

#### 5.1.7 Setup Hints

- **Auto-save**: Enable `auto_save=True` for automatic database persistence
- **Query Format**: Use JSON strings for filters, projections, and sort criteria
- **Collection Names**: Always specify the collection name for operations
- **Error Handling**: Check the `success` field before processing results

---

### 5.2 PostgreSQLToolkit

**The PostgreSQLToolkit provides comprehensive relational database operations for PostgreSQL, including SQL execution, table creation, data querying, updating, and deletion with automatic query type detection and result processing.**

#### 5.2.1 Setup

```python
from evoagentx.tools import PostgreSQLToolkit

# Initialize with default storage
toolkit = PostgreSQLToolkit(
    name="DemoPostgreSQLToolkit",
    database_name="demo_db",
    auto_save=True
)

# Or with custom configuration
toolkit = PostgreSQLToolkit(
    name="CustomPostgreSQLToolkit",
    database_name="my_database",
    host="localhost",
    port=5432,
    user="myuser",
    password="mypassword"
)
```

#### 5.2.2 Available Methods

```python
# Get available tools
tools = toolkit.get_tools()
print(f"Available tools: {[tool.name for tool in tools]}")

# Available tools:
# - postgresql_execute: Execute arbitrary SQL queries
# - postgresql_find: Find (SELECT) rows from tables
# - postgresql_update: Update rows in tables
# - postgresql_create: Create tables and other objects
# - postgresql_delete: Delete rows from tables
# - postgresql_info: Get database and table information
```

#### 5.2.3 Usage Example

```python
# Get tools
execute_tool = toolkit.get_tool("postgresql_execute")
find_tool = toolkit.get_tool("postgresql_find")
create_tool = toolkit.get_tool("postgresql_create")
delete_tool = toolkit.get_tool("postgresql_delete")

# Create users table
create_sql = """
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    age INTEGER,
    department VARCHAR(50)
);
"""

result = create_tool(create_sql)

# Insert users
insert_sql = """
INSERT INTO users (name, email, age, department) VALUES
('Alice Johnson', 'alice@example.com', 28, 'Engineering'),
('Bob Smith', 'bob@example.com', 32, 'Marketing'),
('Carol Davis', 'carol@example.com', 25, 'Engineering')
ON CONFLICT (email) DO NOTHING;
"""

result = execute_tool(insert_sql)

# Query engineers
find_result = find_tool(
    "users",
    where="department = 'Engineering'",
    columns="name, age",
    sort="age ASC"
)

# Delete marketing users
delete_result = delete_tool(
    "users",
    "department = 'Marketing'"
)
```

#### 5.2.4 Parameters

**postgresql_execute:**
- `query` (str): SQL query to execute (can be SELECT, INSERT, UPDATE, DELETE, etc.)
- `query_type` (str, optional): Type of query (select, insert, update, delete, create, drop, alter, index) - auto-detected if not provided

**postgresql_find:**
- `table_name` (str): Table name to query
- `where` (str, optional): WHERE clause (e.g., 'age > 18')
- `columns` (str, optional): Comma-separated columns to select (default: '*')
- `limit` (int, optional): Maximum number of rows to return
- `offset` (int, optional): Number of rows to skip
- `sort` (str, optional): ORDER BY clause (e.g., 'age ASC')

**postgresql_update:**
- `table_name` (str): Table name to update
- `set` (str): SET clause (e.g., 'status = \'active\'')
- `where` (str, optional): WHERE clause

**postgresql_create:**
- `query` (str): CREATE statement (e.g., CREATE TABLE ...)

**postgresql_delete:**
- `table_name` (str): Table name to delete from
- `where` (str): WHERE clause for deletion

#### 5.2.5 Return Type

All tools return `dict` with success/error information and data. The actual return structure follows a standardized format:

```python
{
    "success": True,                    # Boolean indicating success/failure
    "data": {...},                      # Actual result data
    "query_type": "select",             # Type of query executed
    "execution_time": 0.001,            # Time taken to execute
    "row_count": 2,                     # Number of rows affected
    "metadata": {...}                   # Additional metadata
}
```

#### 5.2.6 Sample Return

```python
# Success response for find
{
    "success": True,
    "data": [
        {
            "name": "Alice Johnson",
            "age": 28
        },
        {
            "name": "Carol Davis",
            "age": 25
        }
    ],
    "query_type": "select",
    "execution_time": 0.003,
    "row_count": 2,
    "metadata": {}
}

# Success response for delete
{
    "success": True,
    "data": {
        "rowcount": 1
    },
    "query_type": "delete",
    "execution_time": 0.001,
    "row_count": 1,
    "metadata": {}
}
```

#### 5.2.7 Setup Hints

- **Auto-save**: Enable `auto_save=True` for automatic database persistence
- **Query Types**: The toolkit automatically detects query types, but you can specify them explicitly
- **SQL Injection**: Always use parameterized queries for user input
- **Transactions**: Use explicit transaction management for complex operations

---

### 5.3 FAISSToolkit

**The FAISSToolkit provides comprehensive vector database operations using FAISS, enabling semantic search, document insertion with automatic chunking and embedding, and advanced metadata filtering for building intelligent search applications.**

#### 5.3.1 Setup

```python
from evoagentx.tools import FaissToolkit
from evoagentx.rag.rag_config import RAGConfig, EmbeddingConfig, ChunkerConfig
from evoagentx.storages.storages_config import StoreConfig, DBConfig, VectorStoreConfig

# Basic setup with default configuration
toolkit = FaissToolkit(
    name="ExampleFaissToolkit",
    default_corpus_id="example_corpus"
)

# Advanced setup with custom configuration
storage_config = StoreConfig(
    dbConfig=DBConfig(
        db_name="sqlite",
        path="./example_faiss.db"
    ),
    vectorConfig=VectorStoreConfig(
        vector_name="faiss",
        dimensions=1536,  # For OpenAI embeddings
        index_type="flat_l2"
    )
)

rag_config = RAGConfig(
    embedding=EmbeddingConfig(
        provider="openai",
        model_name="text-embedding-ada-002"
    ),
    chunker=ChunkerConfig(
        chunk_size=500,
        chunk_overlap=50
    )
)

toolkit = FaissToolkit(
    name="CustomFaissToolkit",
    storage_config=storage_config,
    rag_config=rag_config,
    default_corpus_id="custom_corpus"
)
```

#### 5.3.2 Available Methods

```python
# Get available tools
tools = toolkit.get_tools()
print(f"Available tools: {[tool.name for tool in tools]}")

# Available tools:
# - faiss_query: Query the vector database with semantic search
# - faiss_insert: Insert documents with automatic chunking and embedding
# - faiss_delete: Delete documents by ID or metadata filters
# - faiss_list: List all corpora and their configurations
# - faiss_stats: Get database and corpus statistics
```

#### 5.3.3 Usage Example

```python
# Get tools
insert_tool = toolkit.get_tool("faiss_insert")
query_tool = toolkit.get_tool("faiss_query")
stats_tool = toolkit.get_tool("faiss_stats")
delete_tool = toolkit.get_tool("faiss_delete")

# Insert AI knowledge documents
documents = [
    "Artificial Intelligence (AI) is a branch of computer science that aims to create intelligent machines capable of performing tasks that typically require human intelligence.",
    "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed.",
    "Deep learning is a specialized form of machine learning that uses neural networks with multiple layers to analyze and learn from data."
]

# Insert with metadata
result = insert_tool(
    documents=documents,
    metadata={
        "source": "AI_knowledge_base",
        "topic": "artificial_intelligence",
        "language": "en"
    }
)

# Perform semantic search
search_result = query_tool(
    query="How do machines learn?",
    top_k=3,
    similarity_threshold=0.1
)

# Get database statistics
stats_result = stats_tool()

# Delete documents by metadata filter
delete_result = delete_tool(
    metadata_filters={"source": "AI_knowledge_base"}
)
```

#### 5.3.4 Parameters

**faiss_query:**
- `query` (str): The search query text to find semantically similar content
- `corpus_id` (str, optional): Optional corpus ID to search in (uses default if not provided)
- `top_k` (int, optional): Number of top results to return (default: 5)
- `similarity_threshold` (float, optional): Minimum similarity threshold for results (default: 0.0)
- `metadata_filters` (dict, optional): Optional metadata filters to apply to search results

**faiss_insert:**
- `documents` (list): Array of documents to insert (can be strings, file paths, or structured objects)
- `corpus_id` (str, optional): Optional corpus ID to insert into (uses default if not provided)
- `metadata` (dict, optional): Optional metadata to add to all documents
- `batch_size` (int, optional): Batch size for processing documents (default: 100)

**faiss_delete:**
- `doc_ids` (list, optional): List of document IDs to delete
- `metadata_filters` (dict, optional): Metadata filters for deletion
- `corpus_id` (str, optional): Corpus ID to delete from (uses default if not provided)

**faiss_stats:**
- `corpus_id` (str, optional): Optional corpus ID to get statistics for (returns global stats if not provided)

#### 5.3.5 Return Type

All tools return `dict` with success/error information and data. The actual return structure follows a standardized format:

```python
{
    "success": True,                    # Boolean indicating success/failure
    "data": {...},                      # Actual result data
    "query_type": "select",             # Type of query executed
    "execution_time": 0.001,            # Time taken to execute
    "row_count": 2,                     # Number of documents/vectors affected
    "metadata": {...}                   # Additional metadata
}
```

#### 5.3.6 Sample Return

```python
# Success response for insert
{
    "success": True,
    "data": {
        "documents_inserted": 3,
        "chunks_created": 6,
        "corpus_id": "example_corpus"
    }
}

# Success response for query
{
    "success": True,
    "data": {
        "total_results": 3,
        "results": [
            {
                "score": 0.8923,
                "content": "Machine learning is a subset of artificial intelligence...",
                "doc_id": "doc_001",
                "metadata": {"source": "AI_knowledge_base", "topic": "artificial_intelligence"}
            }
        ]
    }
}

# Success response for stats
{
    "success": True,
    "data": {
        "total_corpora": 1,
        "corpora": ["example_corpus"],
        "embedding_model": "text-embedding-ada-002",
        "vector_store_type": "faiss"
    }
}
```

#### 5.3.7 Setup Hints

- **API Key**: Set `OPENAI_API_KEY` environment variable for OpenAI embeddings
- **File Processing**: The toolkit automatically detects file paths and reads content
- **Chunking**: Configure chunk size and overlap for optimal document processing
- **Metadata**: Use metadata filters for efficient document organization and retrieval
- **Corpora**: Use different corpus IDs to organize documents by topic or source

---

### 5.4 Database Tools Summary

| Toolkit | Purpose | Key Features | Use Cases |
|---------|---------|--------------|-----------|
| **MongoDBToolkit** | Document database | JSON queries, aggregation, flexible schema | Content management, user data, logs |
| **PostgreSQLToolkit** | Relational database | SQL operations, ACID compliance, complex queries | Business data, analytics, structured information |
| **FAISSToolkit** | Vector database | Semantic search, embeddings, metadata filtering | AI applications, content search, similarity matching |

**Common Use Cases:**
- **Data Persistence**: Store and retrieve application data with automatic persistence
- **Content Management**: Manage documents, user data, and metadata
- **Semantic Search**: Build intelligent search applications with vector similarity
- **Analytics**: Perform complex queries and data analysis
- **Real-time Applications**: Handle concurrent database operations

**Best Practices:**
- Always check `success` field before processing results
- Use appropriate metadata for efficient document organization
- Implement proper error handling for database operations
- Consider transaction management for complex operations
- Use connection pooling for high-traffic applications

## 6. Image Handling Tools

Image handling tools provide comprehensive capabilities for image analysis, generation, and manipulation using various AI services and APIs. These tools enable agents to work with visual content, generate images from text descriptions, and analyze image content.

### 6.1 ImageAnalysisToolkit

**The ImageAnalysisToolkit provides AI-powered image analysis capabilities using OpenAI's GPT-4 Vision model through OpenRouter API. It can analyze images, extract information, and provide detailed descriptions of visual content.**

#### 6.1.1 Setup

```python
from evoagentx.tools import ImageAnalysisToolkit

# Initialize with OpenRouter API (requires OPENROUTER_API_KEY)
toolkit = ImageAnalysisToolkit(
    name="DemoImageAnalysisToolkit",
    api_key="your-openrouter-api-key",  # Or set OPENROUTER_API_KEY environment variable
    model="gpt-4o"  # Default model for image analysis
)
```

#### 6.1.2 Available Methods

```python
# Get available tools
tools = toolkit.get_tools()
print(f"Available tools: {[tool.name for tool in tools]}")

# Available tools:
# - image_analysis: Analyze images and extract information
```

#### 6.1.3 Usage Example

```python
# Get the image analysis tool
analysis_tool = toolkit.get_tool("image_analysis")

# Analyze an image file
result = analysis_tool(
    image_path="path/to/image.jpg",
    prompt="Describe what you see in this image in detail"
)

# Analyze an image URL
result = analysis_tool(
    image_url="https://example.com/image.jpg",
    prompt="What objects and activities can you identify in this image?"
)

# Analyze with custom model
result = analysis_tool(
    image_path="screenshot.png",
    prompt="Analyze this screenshot and identify the main UI elements",
    model="gpt-4o-mini"
)
```

#### 6.1.4 Parameters

**image_analysis:**
- `image_path` (str, optional): Local file path to the image
- `image_url` (str, optional): URL of the image to analyze
- `prompt` (str, required): Text prompt describing what to analyze
- `model` (str, optional): AI model to use for analysis (default: "gpt-4o")

**Note**: Provide either `image_path` or `image_url`, not both.

#### 6.1.5 Return Type

Returns `dict` with analysis results and metadata.

#### 6.1.6 Sample Return

```python
# Success response
{
    "content": "This image shows a modern office workspace with a laptop computer, a coffee mug, and several documents arranged on a wooden desk. The lighting appears to be natural daylight coming from a window on the left side. The workspace looks organized and professional, with a minimalist aesthetic.",
    "usage": {
        "prompt_tokens": 15,
        "completion_tokens": 45,
        "total_tokens": 60
    }
}

# Error response
{
    "error": "Image file not found at specified path"
}
```

#### 6.1.7 Setup Hints

- **API Key**: Set `OPENROUTER_API_KEY` environment variable or pass directly to toolkit
- **Image Formats**: Supports common formats: JPEG, PNG, GIF, WebP
- **File Size**: Images should be under 20MB for optimal processing
- **Models**: Uses GPT-4 Vision models for best image understanding

---

### 6.2 OpenAIImageGenerationToolkit

**The OpenAIImageGenerationToolkit provides access to OpenAI's DALL-E image generation capabilities, allowing you to create high-quality images from text descriptions with various customization options.**

#### 6.2.1 Setup

```python
from evoagentx.tools import OpenAIImageGenerationToolkit

# Initialize with OpenAI API (requires OPENAI_API_KEY)
toolkit = OpenAIImageGenerationToolkit(
    name="DemoOpenAIImageToolkit",
    api_key="your-openai-api-key",  # Or set OPENAI_API_KEY environment variable
    organization_id="your-organization-id",  # Or set OPENAI_ORGANIZATION_ID environment variable
    model="gpt-4o"  # Default model for image generation
)
```

#### 6.2.2 Available Methods

```python
# Get available tools
tools = toolkit.get_tools()
print(f"Available tools: {[tool.name for tool in tools]}")

# Available tools:
# - image_generation: Generate images from text descriptions
```

#### 6.2.3 Usage Example

```python
# Get the image generation tool
gen_tool = toolkit.get_tool("image_generation")

# Generate a simple image
result = gen_tool(
    prompt="A serene mountain landscape at sunset with a lake in the foreground",
    size="1024x1024",
    quality="standard",
    style="vivid"
)

# Generate with specific parameters
result = gen_tool(
    prompt="A futuristic city skyline with flying cars and neon lights",
    size="1792x1024",
    quality="hd",
    style="natural"
)

# Generate with custom settings
result = gen_tool(
    prompt="A cute robot playing with a cat in a garden",
    size="1024x1024",
    quality="standard"
)

# Generate with custom settings
result = gen_tool(
    prompt="A cute robot playing with a cat in a garden",
    size="1024x1024",
    quality="standard"
)
```

#### 6.2.4 Parameters

**image_generation:**
- `prompt` (str, required): Detailed description of the image to generate
- `size` (str, optional): Image dimensions (default: "1024x1024")
  - Options: "1024x1024", "1792x1024", "1024x1792"
- `quality` (str, optional): Image quality (default: "standard")
  - Options: "standard", "hd"
- `style` (str, optional): Artistic style (default: "vivid")
  - Options: "vivid", "natural"


#### 6.2.5 Return Type

Returns `dict` with generation results including file path and storage handler information.

#### 6.2.6 Sample Return

```python
# Success response
{
    "file_path": "generated_image.png",
    "storage_handler": "LocalStorageHandler"
}

# Error response
{
    "error": "Your request was rejected as a result of our safety system. Your prompt may contain text that is not allowed by our safety system."
}
```

#### 6.2.7 Setup Hints

- **API Key**: Set `OPENAI_API_KEY` environment variable or pass directly to toolkit
- **Organization ID**: Set `OPENAI_ORGANIZATION_ID` environment variable or pass directly to toolkit
- **Prompt Quality**: Detailed, descriptive prompts produce better results
- **Safety Filters**: Content must comply with OpenAI's safety guidelines
- **Rate Limits**: Be mindful of API rate limits and costs

---

### 6.3 FluxImageGenerationToolkit

**The FluxImageGenerationToolkit provides access to Flux Kontext Max image generation capabilities, offering high-quality image creation with advanced customization options and various artistic styles.**

#### 6.3.1 Setup

```python
from evoagentx.tools import FluxImageGenerationToolkit

# Initialize with BFL API (requires BFL_API_KEY)
toolkit = FluxImageGenerationToolkit(
    name="DemoFluxImageToolkit",
    api_key="your-bfl-api-key",  # Or set BFL_API_KEY environment variable
    model="kontext-max"  # Default model for image generation
)
```

#### 6.3.2 Available Methods

```python
# Get available tools
tools = toolkit.get_tools()
print(f"Available tools: {[tool.name for tool in tools]}")

# Available tools:
# - flux_image_generation: Generate images using Flux Kontext Max
```

#### 6.3.3 Usage Example

```python
# Get the image generation tool
gen_tool = toolkit.get_tool("flux_image_generation")

# Generate a basic image
result = gen_tool(
    prompt="A magical forest with glowing mushrooms and fairy lights",
    aspect_ratio="1:1"
)

# Generate with advanced parameters
result = gen_tool(
    prompt="A steampunk airship flying over Victorian London",
    aspect_ratio="16:9",
    seed=42,
    output_format="png",
    prompt_upsampling=True,
    safety_tolerance=2
)

# Generate with specific style
result = gen_tool(
    prompt="A cyberpunk street scene with neon lights and rain",
    aspect_ratio="2:1",
    seed=12345,
    output_format="webp"
)
```

#### 6.3.4 Parameters

**flux_image_generation:**
- `prompt` (str, required): Detailed description of the image to generate
- `aspect_ratio` (str, optional): Image aspect ratio (default: "1:1")
  - Options: "1:1", "16:9", "9:16", "2:1", "1:2", "4:3", "3:4"
- `seed` (int, optional): Random seed for reproducible results
- `output_format` (str, optional): Output image format (default: "png")
  - Options: "png", "webp", "jpeg"
- `prompt_upsampling` (bool, optional): Enable prompt upsampling for better quality
- `safety_tolerance` (int, optional): Safety filter tolerance (default: 2)
  - Options: 1 (low), 2 (medium), 3 (high)

#### 6.3.5 Return Type

Returns `dict` with generation results including file path and storage handler information.

#### 6.3.6 Sample Return

```python
# Success response
{
    "file_path": "./flux_generated_images/flux_42.jpeg",
    "storage_handler": "LocalStorageHandler"
}

# Error response
{
    "success": False,
    "error": "Invalid API key provided",
    "error_code": "AUTH_ERROR"
}
```

#### 6.3.7 Setup Hints

- **API Key**: Set `BFL_API_KEY` environment variable or pass directly to toolkit
- **Aspect Ratios**: Choose appropriate ratios for your use case
- **Seeds**: Use consistent seeds for reproducible results
- **Safety**: Adjust safety tolerance based on your content requirements

---

### 6.4 Image Handling Tools Summary

| Toolkit | Purpose | Key Features | Use Cases |
|---------|---------|--------------|-----------|
| **ImageAnalysisToolkit** | Image understanding | Visual analysis, content extraction | Content moderation, image description, visual QA |
| **OpenAIImageGenerationToolkit** | AI image creation | DALL-E 3, high quality, safety filters | Creative content, marketing materials, concept art |
| **FluxImageGenerationToolkit** | Advanced image generation | Kontext Max, multiple formats, style control | Professional graphics, artistic content, design work |

**Common Use Cases:**
- **Content Creation**: Generate images for websites, presentations, and marketing
- **Visual Analysis**: Analyze user-uploaded content, screenshots, and photos
- **Creative Projects**: Create artwork, illustrations, and concept designs
- **Documentation**: Generate visual aids and explanatory images
- **Research**: Create visualizations and experimental imagery

**Best Practices:**
- Always check API key requirements and set appropriate environment variables
- Use detailed, descriptive prompts for better generation results
- Be mindful of content safety guidelines and API rate limits
- Consider image formats and aspect ratios for your specific use case
- Test with different models and parameters to find optimal settings

**API Key Requirements:**
- **ImageAnalysisToolkit**: `OPENROUTER_API_KEY` (for GPT-4 Vision access)
- **OpenAIImageGenerationToolkit**: `OPENAI_API_KEY` (for DALL-E access)
- **FluxImageGenerationToolkit**: `BFL_API_KEY` (for Flux Kontext Max access)

---

## 7. Browser Tools

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

1. **Element IDs**: After taking a snapshot, interactive elements are assigned unique IDs like `e0`, `e1`, `