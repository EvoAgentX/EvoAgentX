# Working with MCP Tools in EvoAgentX

This tutorial walks you through using the Model Context Protocol (MCP) integration in EvoAgentX. MCP allows agents to interact with various tools and services through a standardized protocol. The implementation uses FastMCP 2.0 for enhanced performance and reliability. We'll cover:

1. **Understanding MCP**: Learn about the Model Context Protocol and its integration in EvoAgentX
2. **Setting Up MCP Tools**: Configure and initialize MCP tools using different methods
3. **Using MCP Tools**: Access and use MCP tools within your agent workflows

By the end of this tutorial, you'll understand how to leverage MCP tools in your own EvoAgentX applications.

---

## 1. Understanding MCP

The Model Context Protocol (MCP) is a standardized way for language models to communicate with external tools and services. EvoAgentX provides integration with MCP through the `MCPToolkit` class, powered by FastMCP 2.0.

```python
from evoagentx.tools import MCPToolkit
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
from evoagentx.tools import MCPToolkit

# Initialize with a configuration file
toolkit = MCPToolkit(config_path="examples/mcp.config")

# Get all available tools
tools = toolkit.get_toolkits()
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
from evoagentx.tools import MCPToolkit

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
toolkit = MCPToolkit(config=config)

# Get all available tools
tools = toolkit.get_toolkits()
```

---

## 3. Using MCP Tools

Once your MCP toolkit is initialized, you can access and use the available tools.

### 3.1 Getting Available Tools

The `get_tools()` method returns a list of all available tools from all connected MCP servers:

```python
# Initialize the MCP toolkit
toolkit = MCPToolkit(config_path="examples/mcp.config")

# Get all available tools
tools = toolkit.get_toolkits()
```

### 3.2 Accessing Tool Information

Each tool provides consistent access to its functionality:

```python
# Loop through available tools
for tool in tools:
    # Get tool name
    name = tool.name
    print(f"Tool: {name}")
    
    # Get tool description
    description = tool.description
    print(f"Description: {description}")
    
    # Get tool inputs schema
    inputs = tool.inputs
    print(f"Inputs: {inputs}")
    
    # Get required parameters
    required = tool.required
    print(f"Required: {required}")
```

**Sample Output:**

```
Tool: HirebaseSearch
Description: Search for job information by providing keywords and filters
Inputs: {
    "query": {
        "anyOf": [{"type": "string"}, {"type": "null"}],
        "default": None,
        "title": "Query"
    },
    "and_keywords": {
        "anyOf": [{"items": {"type": "string"}, "type": "array"}, {"type": "null"}],
        "default": None,
        "title": "And Keywords"
    },
    "or_keywords": {
        "anyOf": [{"items": {"type": "string"}, "type": "array"}, {"type": "null"}],
        "default": None,
        "title": "Or Keywords"
    },
    "country": {
        "anyOf": [{"items": {"type": "string"}, "type": "array"}, {"type": "null"}],
        "default": None,
        "title": "Country"
    },
    "city": {
        "anyOf": [{"items": {"type": "string"}, "type": "array"}, {"type": "null"}],
        "default": None,
        "title": "City"
    },
    "company": {
        "anyOf": [{"items": {"type": "string"}, "type": "array"}, {"type": "null"}],
        "default": None,
        "title": "Company"
    },
    "salary_from": {
        "anyOf": [{"type": "number"}, {"type": "null"}],
        "default": None,
        "title": "Salary From"
    },
    "salary_to": {
        "anyOf": [{"type": "number"}, {"type": "null"}],
        "default": None,
        "title": "Salary To"
    },
    "limit": {
        "anyOf": [{"type": "integer"}, {"type": "null"}],
        "default": 10,
        "title": "Limit"
    }
}
Required: []
...
```

### 3.3 Using MCP Tools in Practice

Here's a complete example of initializing and using MCP tools:

```python
from evoagentx.tools import MCPToolkit

# Initialize the MCP toolkit
toolkit = MCPToolkit(config_path="examples/mcp.config")

# Get all available tools
tools = toolkit.get_toolkits()

# Find a specific tool by name
hirebase_tool = next((tool for tool in tools if "hirebase" in tool.name.lower() or "search" in tool.name.lower()), None)

if hirebase_tool:
    # Call the tool with appropriate parameters (showing complex input schema)
    result = hirebase_tool(
        query="data scientist",
        and_keywords=["python", "machine learning"],
        country=["United States", "Canada"],
        salary_from=80000,
        limit=20
    )
    print(f"Job search results: {result}")

# Clean up when done
toolkit.disconnect()
```

### 3.4 Handling MCP Server Lifecycle

The MCP toolkit handles server connections automatically, but you should properly disconnect when you're done:

```python
# Initialize the toolkit
toolkit = MCPToolkit(config_path="examples/mcp.config")

try:
    # Use MCP tools
    tools = toolkit.get_toolkits()
    # ... work with tools ...
finally:
    # Disconnect from servers
    toolkit.disconnect()
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
toolkit = MCPToolkit(config_path="mcp_config.json")
mcp_tools = toolkit.get_toolkits()

try:
    # Find a specific tool by name
    job_search = next((tool for tool in mcp_tools if "hirebase" in tool.name.lower() or "search" in tool.name.lower()), None)
    
    if job_search:
        # Call the tool with appropriate parameters (demonstrating optional arrays and filters)
        result = job_search(
            query="software engineer",
            and_keywords=["react", "javascript"],
            or_keywords=["node.js", "typescript"],
            city=["San Francisco", "New York"],
            salary_from=100000,
            salary_to=150000,
            limit=15
        )
        print(f"Job search results: {result}")
finally:
    # Always disconnect when done
    toolkit.disconnect()
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
from evoagentx.tools import MCPToolkit

# Load environment variables and setup OpenAI config
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai_config = OpenAILLMConfig(model="gpt-4o-mini", openai_key=OPENAI_API_KEY, stream=True, output_response=True)

# Initialize MCP toolkit and get tools
toolkit = MCPToolkit(config_path="examples/mcp.config")
tools = toolkit.get_toolkits()

# Create a customized agent with the tools
mcp_agent = CustomizeAgent(
    name="MCPAgent",
    description="A MCP agent that can use the tools provided by the MCP server",
    prompt_template=StringTemplate(
        instruction="Do some operations based on the user's instruction."
    ), 
    llm_config=openai_config,
    inputs=[
        {"name": "instruction", "type": "string", "description": "The goal you need to achieve"}
    ],
    outputs=[
        {"name": "result", "type": "string", "description": "The result of the tool call"}
    ],
    tools=tools  # Pass tools directly, not as tool_names and tool_dict
)

# Optional: Save and load the agent configuration
mcp_agent.save_module("examples/mcp_agent.json")
mcp_agent.load_module("examples/mcp_agent.json", llm_config=openai_config, tools=tools)

# Execute the agent with a realistic task
message = mcp_agent(
    inputs={"instruction": "Search for data scientist jobs in San Francisco with Python skills"}
)

print(f"Response from {mcp_agent.name}:")
print(message.content.result)

# Clean up when done
toolkit.disconnect()
```

This example demonstrates:
1. Setting up the MCP toolkit and getting available tools
2. Configuring a customized agent with the tools
3. Executing the agent with specific inputs

The agent can now use any of the MCP tools available through the toolkit.

### 4.2 Integration with Workflow and Agent Generation

MCP tools can also be integrated into workflow and agent generation systems. Here's a simplified example:

```python
from evoagentx.tools import MCPToolkit
from evoagentx.agents.agent_manager import AgentManager
from evoagentx.workflow.workflow_generator import WorkFlowGenerator
from evoagentx.models.openai_model import OpenAILLM

# Initialize MCP toolkit and add to agent manager
toolkit = MCPToolkit(config_path="mcp_config.json")
mcp_tools = toolkit.get_toolkits()

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
