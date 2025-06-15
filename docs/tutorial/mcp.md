# Working with MCP Tools in EvoAgentX

This tutorial walks you through using the Model Context Protocol (MCP) integration in EvoAgentX. MCP allows agents to interact with various tools and services through a standardized protocol. The implementation uses FastMCP 2.0 for enhanced performance and reliability. We'll cover:

1. **Understanding MCP**: Learn about the Model Context Protocol and its integration in EvoAgentX
2. **Setting Up MCP Tools**: Configure and initialize MCP tools using different methods
3. **Using MCP Tools**: Access and use MCP tools within your agent workflows

By the end of this tutorial, you'll understand how to leverage MCP tools in your own EvoAgentX applications.

---

## 1. Understanding MCP

The Model Context Protocol (MCP) is a standardized way for language models to communicate with external tools and services. EvoAgentX provides integration with MCP through the `MCPTool` and `MCPToolkit` classes, powered by FastMCP 2.0.

```python
from evoagentx.tools.mcp import MCPToolkit, MCPTool
```

MCP integration in EvoAgentX allows you to:

- Connect to MCP-compatible servers (both stdio and HTTP-based)
- Discover available tools from these servers
- Use these tools through a consistent interface
- Benefit from FastMCP 2.0's enhanced performance and reliability

### Key Concepts

- **MCP Servers**: External processes or services that provide tools through the MCP protocol
- **Tool Discovery**: Automatically discover and expose tools provided by MCP servers
- **Standardized Interface**: All MCP tools follow the same interface pattern, making them easy to use

---

## 2. Setting Up MCP Tools

EvoAgentX provides two main ways to set up MCP tools:

1. **From a Configuration File**: Load server configurations from a JSON file
2. **From a Configuration Dictionary**: Configure servers directly in your code

### 2.1 From a Configuration File

You can set up MCP tools by providing a configuration file path:

```python
from evoagentx.tools.mcp import MCPToolkit

# Initialize with a configuration file
mcp_toolkit = MCPToolkit(config_path="examples/mcp.config")

# Get all available tools
tools = mcp_toolkit.get_tools()
```

#### 2.1.1 Configuration File Format

The configuration file should be a JSON file that follows FastMCP 2.0's configuration format. Here's an example:

```json
{
    "mcpServers": {
        "github": {
            "command": "npx",
            "args": [
                "-y",
                "@modelcontextprotocol/server-github"
            ],
            "env": {
                "GITHUB_PERSONAL_ACCESS_TOKEN": "your_github_token_here"
            }
        },
        "pdf-reader-mcp": {
            "command": "node",
            "args": ["/path/to/pdf-reader-mcp/dist/index.js"],
            "name": "PDF Reader (Local Build)"
        },
        "hirebase": {
            "command": "uvx",
            "args": [
                "hirebase-mcp" 
            ],
            "env": {
                "HIREBASE_API_KEY": "your_api_key_here" 
            }
        }
    }
}
```

Each server configuration can include:

- `command`: The command to start the MCP server (required)
- `args`: Command-line arguments for the server (optional)
- `env`: Environment variables to set when running the server (optional)
- `timeout`: Connection timeout in seconds (optional, defaults to 120.0)

### 2.2 From a Configuration Dictionary

Alternatively, you can configure MCP tools directly in your code:

```python
import os
from evoagentx.tools.mcp import MCPToolkit

# Configuration dictionary
config = {
    "mcpServers": {
        "github": {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-github"],
            "env": {
                "GITHUB_PERSONAL_ACCESS_TOKEN": os.environ.get("GITHUB_TOKEN")
            }
        },
        "pythonTools": {
            "command": "python",
            "args": ["-m", "mcp_python_tools"],
            "env": {
                "PYTHONPATH": "/path/to/python"
            }
        }
    }
}

# Initialize with the configuration dictionary
mcp_toolkit = MCPToolkit(config=config)

# Get all available tools
tools = mcp_toolkit.get_tools()
```

---

## 3. Using MCP Tools

Once your MCP toolkit is initialized, you can access and use the available tools.

### 3.1 Getting Available Tools

The `get_tools()` method returns a list of all available tools from all connected MCP servers:

```python
# Initialize the MCP toolkit
mcp_toolkit = MCPToolkit(config_path="examples/mcp.config")

# Get all available tools
tools = mcp_toolkit.get_tools()
```

### 3.2 Accessing Tool Information

Each tool provides consistent access to its functionality:

```python
# Loop through available tools
for tool in tools:
    # Get tool name
    name = tool.name
    print(f"Tool: {name}")
    
    # Get tool descriptions
    descriptions = tool.get_tool_descriptions()
    print(f"Description: {descriptions[0]}")
    
    # Get tool schemas
    schemas = tool.get_tool_schemas()
    print(f"Schema: {schemas[0]}")
    
    # Get callable functions
    callables = tool.get_tools()
    print(f"Available functions: {len(callables)}")
```

**Sample Output:**

```
Tool: GitHubSearchRepository
Description: Search for repositories on GitHub
Schema: {
  "type": "function",
  "function": {
    "name": "GitHubSearchRepository",
    "description": "Search for repositories on GitHub",
    "parameters": {
      "type": "object",
      "properties": {
        "query": {
          "type": "string",
          "description": "The search query"
        },
        "sort": {
          "type": "string",
          "enum": ["stars", "forks", "updated"],
          "description": "How to sort the results"
        },
        "per_page": {
          "type": "integer",
          "description": "Number of results per page"
        }
      },
      "required": ["query"]
    }
  }
}
Available functions: 1
...
```

### 3.3 Using MCP Tools in Practice

Here's a complete example of initializing and using MCP tools:

```python
from evoagentx.tools.mcp import MCPToolkit

# Initialize the MCP toolkit
mcp_toolkit = MCPToolkit(config_path="examples/mcp.config")

# Get all available tools
tools = mcp_toolkit.get_tools()

# Find a specific tool by name
calculator_tool = next((tool for tool in tools if tool.name == "Calculator"), None)

if calculator_tool:
    # Get the callable function
    calculate = calculator_tool.get_tools()[0]
    
    # Call the function with appropriate parameters
    result = calculate(expression="2 + 2 * 3")
    print(f"Calculation result: {result}")

# Clean up when done
mcp_toolkit.disconnect()
```

### 3.4 Handling MCP Server Lifecycle

The MCP toolkit handles server connections automatically, but you should properly disconnect when you're done:

```python
# Initialize the toolkit
mcp_toolkit = MCPToolkit(config_path="examples/mcp.config")

try:
    # Use MCP tools
    tools = mcp_toolkit.get_tools()
    # ... work with tools ...
finally:
    # Disconnect from servers
    mcp_toolkit.disconnect()
```

You can also use the toolkit as a context manager:

```python
from evoagentx.tools.mcp import MCPClient

# Setup server configurations
server_configs = {
    "command": "npx",
    "args": ["-y", "@modelcontextprotocol/server-github"],
    "env": {"GITHUB_PERSONAL_ACCESS_TOKEN": "your_token_here"}
}

with MCPClient(server_configs) as mcp_tools:
    # Use MCP tools here
    # Automatic disconnection happens when exiting the context
    for tool in mcp_tools:
        print(f"Available tool: {tool.name}")
```

### 3.5 Using MCP in Practical Applications

Once you've initialized your MCP toolkit and obtained the tools, you can use them directly:

```python
# Initialize MCP toolkit
mcp_toolkit = MCPToolkit(config_path="mcp_config.json")
mcp_tools = mcp_toolkit.get_tools()

try:
    # Find a specific tool by name
    github_search = next((tool for tool in mcp_tools if "GitHubSearchRepository" in tool.name), None)
    
    if github_search:
        # Get the callable function
        search_function = github_search.get_tools()[0]
        
        # Call the function with appropriate parameters
        result = search_function(query="large language models", sort="stars")
        print(f"Search results: {result}")
finally:
    # Always disconnect when done
    mcp_toolkit.disconnect()
```

---

## 4. Using MCP with Agents

MCP tools integrate seamlessly with EvoAgentX agents through tool calling actions. Here's how to use MCP tools with agents.

### 4.1 Creating a Customized Agent with Tool Calling Action

The most flexible way to use MCP tools is to create a customized agent with tool mapping:

```python
import os 
from dotenv import load_dotenv
from evoagentx.models import OpenAILLMConfig
from evoagentx.agents import CustomizeAgent
from evoagentx.prompts import StringTemplate 
from evoagentx.tools.mcp import MCPToolkit

# Load environment variables and setup OpenAI config
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai_config = OpenAILLMConfig(model="gpt-4o-mini", openai_key=OPENAI_API_KEY, stream=True, output_response=True)

# Initialize MCP toolkit and get tools
mcp_toolkit = MCPToolkit(config_path="examples/mcp.config")
tools = mcp_toolkit.get_tools()

# Create tool mapping
tools_mapping = {}
tools_schemas = [(tool.get_tool_schemas(), tool) for tool in tools]
tools_schemas = [(j, k) for i, k in tools_schemas for j in i]
for tool_schema, tool in tools_schemas:
    tool_name = tool_schema["function"]["name"]
    tools_mapping[tool_name] = tool

# Create a customized agent with the tool mapping
code_writer = CustomizeAgent(
    name="CodeWriter",
    description="Writes Python code based on requirements",
    prompt_template=StringTemplate(
        instruction="Summarize all your tools."
    ), 
    llm_config=openai_config,
    inputs=[
        {"name": "instruction", "type": "string", "description": "The goal you need to achieve"}
    ],
    outputs=[
        {"name": "result", "type": "string", "description": "The tools you have"}
    ],
    tool_names=[tool.name for tool in tools],
    tool_dict={tool.name: tool for tool in tools}
)

# Execute the agent
message = code_writer(
    inputs={"instruction": "Summarize all your tools."}
)
print(f"Response from {code_writer.name}:")
print(message.content.result)
```

This example demonstrates:
1. Setting up the MCP toolkit and getting available tools
2. Creating a mapping between tool names and tool objects
3. Configuring a customized agent with the tool mapping
4. Executing the agent with specific inputs

The agent can now use any of the MCP tools available through the tool mapping.

### 4.2 Integration with Workflow and Agent Generation

MCP tools can also be integrated into workflow and agent generation systems. Here's a simplified example:

```python
from evoagentx.tools.mcp import MCPToolkit
from evoagentx.agents.agent_manager import AgentManager
from evoagentx.workflow.workflow_generator import WorkFlowGenerator
from evoagentx.models.openai_model import OpenAILLM

# Initialize MCP toolkit and add to agent manager
mcp_toolkit = MCPToolkit(config_path="mcp_config.json")
mcp_tools = mcp_toolkit.get_tools()

agent_manager = AgentManager(tools=mcp_tools)
workflow_generator = WorkFlowGenerator(llm=llm, tools=mcp_tools)

# Generate a workflow that can use MCP tools
workflow = workflow_generator.generate_workflow(
    goal="Analyze GitHub repositories and create a report"
)

# Add generated agents to the agent manager
agent_manager.add_agents_from_workflow(workflow_graph=workflow, llm_config=llm_config)
```

## 5. Troubleshooting

Here are some common issues and their solutions:

- **Connection Timeout**: If you encounter timeout errors connecting to MCP servers, increase the timeout value in your configuration (default is 120.0 seconds).

- **Missing Tools**: If expected tools aren't available, check if the MCP server is running correctly and whether it's properly configured to expose those tools.

- **Tool Execution Errors**: FastMCP 2.0 provides specific error types:
  - `ClientError`: Indicates client-side issues (connection, configuration, etc.)
  - `McpError`: Indicates server-side MCP protocol errors
  Check the error message for details about what went wrong.

- **Server Not Starting**: If an MCP server fails to start, verify the command path and environment variables in your configuration.

For more information about MCP and FastMCP, visit:
- [Model Context Protocol](https://github.com/modelcontextprotocol/protocol)
- [FastMCP Documentation](https://gofastmcp.com/clients/client)
