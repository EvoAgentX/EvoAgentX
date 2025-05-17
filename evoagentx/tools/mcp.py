"""
This tool is inspired / modified from MCP Python SDK and mcpadapt projects. You may find more information about by visiting the following links:
- https://github.com/modelcontextprotocol/python-sdk
- https://github.com/grll/mcpadapt

"""
import threading
import asyncio
from functools import partial
from typing import Optional, Any, List, Callable
from evoagentx.tools.tool import Tool
from evoagentx.core.logging import logger
from contextlib import AsyncExitStack, asynccontextmanager

import mcp
import os
import json
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.sse import sse_client




class MCPTool(Tool):
    def __init__(
        self,
        name: str = "MCP Tool",
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
    


class MCPClient:
    
    def __init__(self, 
                 server_configs: StdioServerParameters | dict[str, Any] | list[StdioServerParameters | dict[str, Any]],
                 connect_timeout: float = 120.0,
                 ):
        
        if isinstance(server_configs, list):
            self.server_configs = server_configs
        else:
            self.server_configs = [server_configs]
        
        self.event_loop = asyncio.new_event_loop()
        
        self.sessions:list[mcp.ClientSession] = []
        self.mcp_tools:list[list[mcp.types.Tool]] = []
        
        self.task = None
        self.thread_running = threading.Event()
        ## Testing
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
        # check connection to mcp server is ready
        if not self.thread_running.wait(timeout=self.connect_timeout):
            raise TimeoutError(
                f"Couldn't connect to the MCP server after {self.connect_timeout} seconds"
            )
    
    def __enter__(self):
        self.working_thread.start()
        # check connection to mcp server is ready
        if not self.thread_running.wait(timeout=self.connect_timeout):
            raise TimeoutError(
                f"Couldn't connect to the MCP server after {self.connect_timeout} seconds"
            )
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
                        await stack.enter_async_context(self._start_server(params))
                        for params in self.server_configs
                    ]
                    self.sessions, self.mcp_tools = [list(c) for c in zip(*connections)]
                    self.thread_running.set()  # Signal initialization is complete
                    
                    # Use asyncio.Future instead of Event for better cleanup
                    done_future = self.event_loop.create_future()
                    await done_future
            except Exception as e:
                logger.error(f"Error in MCP event loop: {str(e)}")
                self.thread_running.set()  # Still set the event so we don't hang
                raise

        # Create and run the task directly in the event loop
        self.task = self.event_loop.create_task(setup())
        
        try:
            self.event_loop.run_until_complete(self.task)
        except asyncio.CancelledError:
            logger.info("MCP client event loop was cancelled")
        except Exception as e:
            logger.error(f"Error in MCP event loop: {str(e)}")

    @asynccontextmanager
    async def _start_server(self, server_config: StdioServerParameters | dict[str, Any]):
        if isinstance(server_config, StdioServerParameters):
            client = stdio_client(server_config)
        elif isinstance(server_config, dict):
            client = sse_client(**server_config)
        else:
            raise ValueError("Invalid server config type: {}".format(type(server_config)))
        
        async with client as (read, write):
            async with ClientSession(read, write) as session:
                # Initialize the connection and get the tools from the mcp server
                await session.initialize()
                tools = await session.list_tools()
                yield session, tools.tools

    def create_tool(
        self,
        function: Callable[[dict | None], mcp.types.CallToolResult],
        mcp_tool: mcp.types.Tool,
    ) -> MCPTool:

        # make sure jsonref are resolved
        input_schema = mcp_tool.inputSchema
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
                "description": mcp_tool.description or "No description provided.",
                "parameters": parameters,
            },
        }
        
        tool = MCPTool(
            mcp_tool.name,
            [mcp_tool.description or ""],
            [schema],
            [function],
        )

        return tool
    
    def get_tools(self) -> List[Tool]:
        if not self.sessions:
            raise RuntimeError("Session not initialized")

        def _sync_call_tool(
            session, name: str, **kwargs
        ) -> mcp.types.CallToolResult:
            try:
                # Handle both direct parameters and arguments-wrapped parameters
                if "arguments" in kwargs and len(kwargs) == 1:
                    # Original format: {"arguments": {...}}
                    arguments = kwargs["arguments"]
                else:
                    # New format: direct parameters
                    arguments = kwargs
                
                logger.info(f"Calling MCP tool: {name} with arguments: {arguments}")
                future = asyncio.run_coroutine_threadsafe(
                    session.call_tool(name, arguments), self.event_loop
                )
                result = future.result(timeout=60)  # Increased timeout from 15 to 60 seconds
                logger.info(f"MCP tool {name} call completed successfully")
                return result
            except TimeoutError:
                logger.error(f"Timeout calling MCP tool: {name}")
                raise
            except Exception as e:
                logger.error(f"Error calling MCP tool {name}: {str(e)}")
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
        parameters = []
        if server_configs.get("mcpServers", None):
            server_configs = server_configs.get("mcpServers")
        for name, cfg in server_configs.items():
            if not isinstance(cfg, dict):
                logger.warning(f"Configuration for server '{name}' must be a dictionary")
                continue

            if "command" not in cfg and "url" not in cfg:
                logger.warning(f"Missing required 'command' or 'url' field for server '{name}'")
                continue
                
            command_or_url = cfg.get("command") or cfg.get("url")
            # Get timeout from config or use default
            timeout = cfg.get("timeout", None)
            parameters.append(StdioServerParameters(
                command=command_or_url, 
                args=cfg.get("args", []), 
                env={**os.environ, **cfg.get("env", {})}, 
                timeout=timeout, 
                headers=cfg.get("headers", None)
            ))
            logger.info(f"Configured MCP server: {name} with command: {command_or_url}")
            
        return parameters
    
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
