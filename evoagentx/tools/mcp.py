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
from pydantic import Field
from functools import partial
from typing import Optional, Any, List, Dict, Callable, Union
from evoagentx.tools.tool import Tool
from evoagentx.core.logging import logger
from contextlib import AsyncExitStack, asynccontextmanager

# FastMCP 2.0 imports - replacing official MCP SDK
from fastmcp import Client
from fastmcp.exceptions import ClientError, McpError
import json

class MCPTool(Tool):
    descriptions: List[str] = Field(default_factory=list, description="A list of descriptions for the tool")
    schemas: List[dict[str, Any]] = Field(default_factory=list, description="A list of schemas for the tool")
    tools: List[Callable] = Field(default_factory=list, description="A list of tools for the tool")

    def __init__(
        self,
        name: str = "MCPTool",
        descriptions: List[str] = ["Default MCP Tool description"],
        schemas: List[dict[str, Any]] = [],
        tools: List[Callable] = [],
    ):
        super().__init__(
            name=name,
            descriptions=descriptions,
            schemas=schemas,
            tools=tools
        )

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return self.schemas
    
    def get_tool_descriptions(self) -> List[str]:
        return self.descriptions
    
    def get_tools(self) -> List[Callable]:
        return self.tools

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
        function: Callable[[dict | None], Any],
        mcp_tool: Any,  # FastMCP tool object
    ) -> MCPTool:
        input_schema = getattr(mcp_tool, 'inputSchema', {})
        if not input_schema and hasattr(mcp_tool, 'input_schema'):
            input_schema = mcp_tool.input_schema
            
        properties = input_schema.get("properties", {})
        required = input_schema.get("required", [])

        parameters = {
            "type": "object",
            "properties": properties,
            "required": required,
        }

        schema = {
            "type": "function",
            "function": {
                "name": mcp_tool.name,
                "description": getattr(mcp_tool, 'description', None) or "No description provided.",
                "parameters": parameters,
            },
        }
        
        tool = MCPTool(
            name=mcp_tool.name,
            descriptions=[getattr(mcp_tool, 'description', None) or ""],
            schemas=[schema],
            tools=[function],
        )

        return tool
    
    def get_tools(self) -> List[Tool]:
        if not self.sessions:
            raise RuntimeError("Session not initialized")

        def _sync_call_tool(
            session, name: str, **kwargs
        ) -> Any:
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

        return [
            self.create_tool(partial(_sync_call_tool, session, tool.name), tool)
            for session, tools in zip(self.sessions, self.mcp_tools)
            for tool in tools
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
            
        # Pass the config directly to FastMCP
        return [server_configs]
    
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