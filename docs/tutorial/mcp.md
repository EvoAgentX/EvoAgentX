# Working with MCP Tools in EvoAgentX

This tutorial walks you through using the Model Context Protocol (MCP) integration in EvoAgentX. MCP allows agents to interact with various tools and services through a standardized protocol. We'll cover:

1. **Understanding MCP**: Learn about the Model Context Protocol and its integration in EvoAgentX
2. **Setting Up MCP Tools**: Configure and initialize MCP tools using different methods
3. **Using MCP Tools**: Access and use MCP tools within your agent workflows

By the end of this tutorial, you'll understand how to leverage MCP tools in your own EvoAgentX applications.

---

## 1. Understanding MCP

The Model Context Protocol (MCP) is a standardized way for language models to communicate with external tools and services. EvoAgentX provides integration with MCP through the `MCPTool` and `MCPToolkit` classes.

```python
from evoagentx.tools.mcp import MCPToolkit, MCPTool
```

MCP integration in EvoAgentX allows you to:

- Connect to MCP-compatible servers (both stdio and SSE-based)
- Discover available tools from these servers
- Use these tools through a consistent interface

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

The configuration file should be a JSON file with the following structure:

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
- `name`: A human-readable name for the server (optional)
- `timeout`: Connection timeout in seconds (optional)

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

The most flexible way to use MCP tools is to create a customized agent with a tool calling action:

```python
import os
from evoagentx.tools.mcp import MCPToolkit
from evoagentx.actions.tool_calling import ToolCalling
from evoagentx.agents.customize_agent import CustomizeAgent
from evoagentx.models.model_configs import OpenAILLMConfig
from evoagentx.models.openai_model import OpenAILLM
from evoagentx.core.message import MessageType

# Initialize MCP toolkit and LLM
mcp_toolkit = MCPToolkit(config_path="mcp_config.json")
mcp_tools = mcp_toolkit.get_tools()

llm_config = OpenAILLMConfig(model="gpt-4o-mini", openai_key=os.environ.get("OPENAI_API_KEY"))
llm = OpenAILLM(config=llm_config)

# Create a tool calling action with MCP tools
tool_action = ToolCalling(max_tool_try=3)
tool_action.add_tools(mcp_tools)

# Create a customized agent with the tool calling action
agent = CustomizeAgent(
    name="GitHubToolAgent",
    description="An agent that can search GitHub repositories",
    prompt="Please help me find information from GitHub using the available tools.",
    llm_config=llm_config,
    actions=[tool_action]
)

try:
    # Execute the agent with a query
    response = agent.execute(
        action_name=tool_action.name,
        action_input_data={"query": "Find repositories about LLMs on GitHub"},
        return_msg_type=MessageType.RESPONSE
    )
    print(response.content)
finally:
    mcp_toolkit.disconnect()
```

### 4.2 Creating a CusToolCaller for Practical Use

For a more direct approach, you can use `CusToolCaller` which is specifically designed for tool usage:

```python
import os
from evoagentx.tools.mcp import MCPToolkit
from evoagentx.agents.cus_tool_caller import CusToolCaller
from evoagentx.models.model_configs import OpenAILLMConfig
from evoagentx.models.openai_model import OpenAILLM

# Initialize MCP toolkit and LLM
mcp_toolkit = MCPToolkit(config_path="mcp_config.json")
mcp_tools = mcp_toolkit.get_tools()

llm_config = OpenAILLMConfig(model="gpt-4o-mini", openai_key=os.environ.get("OPENAI_API_KEY"))
llm = OpenAILLM(config=llm_config)

# Create the CusToolCaller with GitHub-related tools
github_tools = [tool for tool in mcp_tools if "GitHub" in tool.name]
tool_caller = CusToolCaller(
    name="GitHubAnalyzer",
    description="An agent that analyzes GitHub repositories",
    prompt="You are an expert in analyzing GitHub repositories.",
    llm_config=llm_config,
    tools=github_tools
)

try:
    # Execute the tool caller with a query
    response = tool_caller.execute(
        action_name=tool_caller.tool_calling_action_name,
        action_input_data={"query": "Find the top 3 LLM frameworks"}
    )
    print(response.content)
finally:
    mcp_toolkit.disconnect()
```

### 4.3 Integration with Workflow and Agent Generation

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

- **Connection Timeout**: If you encounter timeout errors connecting to MCP servers, increase the timeout value in your configuration.

- **Missing Tools**: If expected tools aren't available, check if the MCP server is running correctly and whether it's properly configured to expose those tools.

- **Tool Execution Errors**: If a tool call fails, check the arguments you're passing and ensure they match the expected schema for that tool.

- **Server Not Starting**: If an MCP server fails to start, verify the command path and environment variables in your configuration.

For more information about MCP itself, visit:
- [Model Context Protocol](https://github.com/modelcontextprotocol/protocol)
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)
