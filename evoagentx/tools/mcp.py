"""
This tool is inspired / modified from MCP Python SDK and mcpadapt projects, now enhanced with FastMCP 2.0. 
You may find more information about by visiting the following links:
- https://github.com/modelcontextprotocol/python-sdk
- https://github.com/grll/mcpadapt
- https://gofastmcp.com/clients/client

FastMCP 2.0 Integration Notes:
- Replaced official MCP SDK with FastMCP for better performance and reliability
- Maintains the same synchronous API with internal async handling via threading
- Auto-infers transport types (stdio/HTTP) based on configuration
- Enhanced error handling with FastMCP's exception hierarchy
- Backwards compatible with existing MCP server configurations
"""
import threading
import asyncio
from functools import partial
from typing import Optional, Any, List, Dict, Callable, Union
from evoagentx.tools.tool import Tool,Toolkit
from evoagentx.core.logging import logger
from contextlib import AsyncExitStack, asynccontextmanager

# FastMCP 2.0 imports - replacing official MCP SDK
from fastmcp import Client
from fastmcp.exceptions import ClientError, McpError
import json

class MCPTool(Tool):
    name: str = "MCPTool"
    description: str = "MCP tool wrapper"
    inputs: Dict[str, Dict[str, Any]] = {}
    required: Optional[List[str]] = None
    function: Callable = None
    
    def __init__(self, name: str, description: str, inputs: Dict[str, Dict[str, str]], required: Optional[List[str]] = None, function: Callable = None):
        super().__init__(name=name, description=description, inputs=inputs, required=required)
        self.function = function
    
    @property
    def __name__(self):
        return self.name
    
    def __call__(self, **kwargs):
        if not self.function:
            raise ValueError("Function not set for MCPTool")
        return self.function(**kwargs)
    
    @classmethod
    def validate_attributes(cls):
        required_attributes = {
            "name": str,
            "description": str,
            "inputs": dict
        }
        
        for attr, attr_type in required_attributes.items():
            if not hasattr(cls, attr):
                raise ValueError(f"Attribute {attr} is required")
            if not isinstance(getattr(cls, attr), attr_type):
                raise ValueError(f"Attribute {attr} must be of type {attr_type}")

        if cls.required:
            for required_input in cls.required:
                if required_input not in cls.inputs:
                    raise ValueError(f"Required input '{required_input}' is not found in inputs")


class MCPClient:
    def __init__(
        self, 
        server_configs: Union[Dict[str, Any], List[Dict[str, Any]]],
        connect_timeout: float = 120.0,
    ):
        if isinstance(server_configs, dict):
            self.server_configs = [server_configs]
        else:
            self.server_configs = server_configs
        
        self.event_loop = asyncio.new_event_loop()
        self.sessions: list[Client] = []
        self.mcp_tools: list[list[Any]] = []
        
        self.task = None
        self.thread_running = threading.Event()
        self.working_thread = threading.Thread(target=self._run_event, daemon=True)
        self.connect_timeout = connect_timeout
        
        self.tools = None
        self.tool_schemas = None
        self.tool_descriptions = None
    
    def _disconnect(self):
        if self.task and not self.task.done():
            self.event_loop.call_soon_threadsafe(self.task.cancel)
        self.working_thread.join()
        self.event_loop.close()
    
    def _connect(self):
        self.working_thread.start()
        if not self.thread_running.wait(timeout=self.connect_timeout):
            raise TimeoutError(
                f"Couldn't connect to the MCP server after {self.connect_timeout} seconds"
            )
    
    def __enter__(self):
        self._connect()
        return self.get_tools()
    
    def __del__(self):
        self._disconnect()
    
    def __exit__(self, exc_type, exc_value, traceback):
        self._disconnect()
    
    def _run_event(self):
        """Runs the event loop in a separate thread (for synchronous usage)."""
        print("Running event loop")
        asyncio.set_event_loop(self.event_loop)

        async def setup():
            try:
                async with AsyncExitStack() as stack:
                    connections = [
                        await stack.enter_async_context(self._start_server(config))
                        for config in self.server_configs
                    ]
                    self.sessions, self.mcp_tools = [list(c) for c in zip(*connections)]
                    self.thread_running.set()
                    
                    done_future = self.event_loop.create_future()
                    await done_future
            except Exception as e:
                logger.error(f"Error in MCP event loop: {str(e)}")
                self.thread_running.set()
                raise

        self.task = self.event_loop.create_task(setup())
        
        try:
            self.event_loop.run_until_complete(self.task)
        except asyncio.CancelledError:
            logger.info("MCP client event loop was cancelled")
        except Exception as e:
            logger.error(f"Error in MCP event loop: {str(e)}")

    @asynccontextmanager
    async def _start_server(self, config: Dict[str, Any]):
        client = Client(config)
        async with client:
            tools = await client.list_tools()
            yield client, tools

    def create_tool(
        self,
        session: Client,
        mcp_tools: List[Any],  # List of FastMCP tool objects for a single server
        config: Dict[str, Any],
    ) ->Toolkit:
        # Define the sync call function once
        def _sync_call_tool(name: str, **kwargs) -> Any:
            try:
                if "arguments" in kwargs and len(kwargs) == 1:
                    arguments = kwargs["arguments"]
                else:
                    arguments = kwargs
                
                logger.info(f"Calling MCP tool: {name} with arguments: {arguments}")
                future = asyncio.run_coroutine_threadsafe(
                    session.call_tool(name, arguments), self.event_loop
                )
                result = future.result(timeout=60)
                logger.info(f"MCP tool {name} call completed successfully")
                return result
            except (TimeoutError, ClientError, McpError) as e:
                logger.error(f"Error calling MCP tool {name}: {str(e)}")
                raise
            except Exception as e:
                logger.error(f"Unexpected error calling MCP tool {name}: {str(e)}")
                raise

        # Collect all schemas and descriptions
        all_tools = []

        for mcp_tool in mcp_tools:
            input_schema = getattr(mcp_tool, 'inputSchema', {})
            if not input_schema and hasattr(mcp_tool, 'input_schema'):
                input_schema = mcp_tool.input_schema
            
            properties = input_schema.get("properties", {})
            required = input_schema.get("required", [])
            
            # Convert MCP properties to Tool.inputs format
            # Tool.inputs expects Dict[str, Dict[str, str]] format
            
            inputs = properties
            
            # Create the partial function and add __name__ attribute
            partial_func = partial(_sync_call_tool, mcp_tool.name)
            partial_func.__name__ = mcp_tool.name
            
            tool = MCPTool(
                name=mcp_tool.name,
                description=getattr(mcp_tool, 'description', None) or "",
                inputs=inputs,
                required=required,
                function=partial_func
            )
            all_tools.append(tool)
        
        tool_collection =Toolkit(name=next(iter(config.get("mcpServers").keys())), tools=all_tools)
        return tool_collection

    
    def get_tools(self) -> List[Toolkit]:
        """Return a list ofToolkits, one per server."""
        if not self.sessions:
            raise RuntimeError("Session not initialized")

        return [
            self.create_tool(session, tools, config)
            for session, tools, config in zip(self.sessions, self.mcp_tools, self.server_configs)
        ]
    
class MCPToolkit:
    def __init__(self, servers: Optional[list[MCPClient]] = None, config_path: Optional[str] = None, config: Optional[dict[str, Any]] = None):
        parameters = []
        if config_path:
            parameters += self._from_config_file(config_path)
        if config:
            parameters += self._from_config(config)
        
        self.servers = [MCPClient(parameters)]
        if servers:
            self.servers += servers
        for server in self.servers:
            try:
                server._connect()
                logger.info("Successfully connected to MCP servers")
            except TimeoutError as e:
                logger.warning(f"Timeout connecting to MCP servers: {str(e)}. Some tools may not be available.")
            except Exception as e:
                logger.error(f"Error connecting to MCP servers: {str(e)}")
    
    def _from_config_file(self, config_path: str):
        try:
            with open(config_path, "r") as f:
                server_configs = json.load(f)
            return self._from_config(server_configs)
        except FileNotFoundError:
            logger.error(f"Config file not found: {config_path}")
            return []
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in config file: {config_path}")
            return []
    
    def _from_config(self, server_configs: dict[str, Any]):
        if not isinstance(server_configs, dict):
            logger.error("Server configuration must be a dictionary")
            return []
            
        # If there's no mcpServers key, treat it as a single server config
        if "mcpServers" not in server_configs:
            raise ValueError("Server configuration must contain 'mcpServers' key")
            
        # Separate each server into its own config with the same format
        server_list = []
        for server_name, server_config in server_configs["mcpServers"].items():
            individual_config = {
                "mcpServers": {
                    server_name: server_config
                }
            }
            server_list.append(individual_config)
            
        return server_list
    
    def disconnect(self):
        for server in self.servers:
            server._disconnect()
        
    def get_tools(self):
        """Return a flattened list of all tools across all servers"""
        all_tools = []
        for server in self.servers:
            try:
                tools = server.get_tools()
                all_tools.extend(tools)
                logger.info(f"Added {len(tools)} tools from MCP server")
            except Exception as e:
                logger.error(f"Error getting tools from MCP server: {str(e)}")
        return all_tools