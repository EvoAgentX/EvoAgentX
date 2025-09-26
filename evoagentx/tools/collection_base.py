from typing import Callable, Dict, Any, List, Optional, Union
from pydantic import Field

from .tool import Tool, Toolkit
from ..core.logging import logger
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError


class ToolCollection(Tool):
    """
    Base class for tool collections that orchestrate multiple tools.
    A ToolCollection is a Tool that manages and executes multiple tools in a specific order.
    """
    
    # Tool interface requirements
    name: str = "ToolCollection"
    description: str = "A collection that orchestrates multiple tools"
    inputs: Dict[str, Dict[str, str]] = {
        "query": {
            "type": "string", 
            "description": "The input query to process through the tool collection"
        }
    }
    required: Optional[List[str]] = ["query"]
    
    # Collection-specific attributes
    execution_order: List[str] = Field(default_factory=list, description="Order to execute tools")
    default_order: Optional[List[str]] = Field(default_factory=list, description="Default order to execute tools")
    argument_mapping_function: Optional[Dict[str, Callable]] = Field(default_factory=dict, description="Argument mapping functions for each tool")
    output_mapping_function: Optional[Dict[str, Callable]] = Field(default_factory=dict, description="Output mapping functions for each tool")

    def __init__(
        self,
        name: Optional[str] = None,
        description: Optional[str] = None,
        kits: Optional[List[Union[Tool, Toolkit]]] = None,
        execution_order: Optional[List[str]] = None,
        argument_mapping_function: Optional[Dict[str, Callable]] = None,
        output_mapping_function: Optional[Dict[str, Callable]] = None,
        per_tool_timeout: Optional[float] = None,
        ):
        
        # Call parent constructor first
        super().__init__()
        
        # Initialize Tool attributes
        if name is not None:
            self.name = name
        if description is not None:
            self.description = description
        
        # Process tools from kits
        tools = []
        kits = kits or []
        for kit in kits:
            if isinstance(kit, Tool):
                tools.append(kit)
            elif isinstance(kit, Toolkit):
                tools += kit.get_tools()
            else:
                try:
                    cur_name = kit.name
                except:
                    cur_name = str(kit)
                logger.warning(f"{cur_name} not a valid Tool/Toolkit, skip adding to the collection")
        
        # Initialize collection attributes
        self.tools = tools
        self.argument_mapping_function = argument_mapping_function or {}
        self.output_mapping_function = output_mapping_function or {}
        self.per_tool_timeout = per_tool_timeout
        self._init_module(execution_order)
    
    def _init_module(self, preferred_order: Optional[List[str]] = None) -> None:
        """Initialize the tool mappings and execution order."""
        tool_mappings = {}
        order_list = []
        
        if not preferred_order:
            preferred_order = self.default_order or []
        
        # Create tool name to tool mapping
        available_tools = {tool.name: tool for tool in self.tools}
        
        for name in preferred_order:
            cur_tool = available_tools.get(name)
            if not cur_tool:
                logger.warning(f"Tool {name} not found in available tools...")
                continue
            if not self.argument_mapping_function.get(name, None):
                logger.warning(f"Argument mapping function for {name} cannot be found...")
            tool_mappings[name] = cur_tool
            order_list.append(name)
        
        self.execution_order = order_list
        self.tool_mappings = tool_mappings
    
    def _get_next_execute(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> Optional[str]:
        """Get the next tool to execute based on current outputs."""
        for tool_name in self.execution_order:
            if tool_name not in outputs:
                return tool_name
        return None


    def _run_pipeline(self, inputs: Dict[str, Any], outputs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute the collection pipeline using mapping functions and execution order."""
        outputs = outputs or {}
        while True:
            next_tool_name = self._get_next_execute(inputs, outputs)
            if not next_tool_name:
                break
            tool = self.tool_mappings.get(next_tool_name)
            if not tool:
                break
            try:
                arg_map_fn = self.argument_mapping_function.get(next_tool_name, lambda i, o: i)
                mapped_args = arg_map_fn(inputs, outputs)

                # Enforce timeout via ThreadPoolExecutor if configured
                resolved_timeout = self.per_tool_timeout
                if resolved_timeout is not None and resolved_timeout > 0:
                    executor = ThreadPoolExecutor(max_workers=1)
                    future = executor.submit(tool, **mapped_args)
                    try:
                        result = future.result(timeout=resolved_timeout)
                    except FuturesTimeoutError:
                        out_map_fn = self.output_mapping_function.get(next_tool_name)
                        err = {"error": f"timeout after {resolved_timeout}s"}
                        try:
                            outputs[next_tool_name] = out_map_fn(err) if out_map_fn else err
                        except Exception:
                            outputs[next_tool_name] = err
                        # Ensure we don't block on shutdown; proceed to next tool immediately
                        executor.shutdown(wait=False, cancel_futures=True)
                        continue
                    except Exception as inner_e:
                        # Handle tool execution exceptions and normalize
                        out_map_fn = self.output_mapping_function.get(next_tool_name)
                        error_obj = {"error": str(inner_e)}
                        try:
                            outputs[next_tool_name] = out_map_fn(error_obj) if out_map_fn else error_obj
                        except Exception:
                            outputs[next_tool_name] = error_obj
                        executor.shutdown(wait=False, cancel_futures=True)
                        continue
                    else:
                        # Successful result, convert and shutdown executor
                        out_map_fn = self.output_mapping_function.get(next_tool_name, lambda r: r)
                        outputs[next_tool_name] = out_map_fn(result)
                        executor.shutdown(wait=False, cancel_futures=True)
                else:
                    result = tool(**mapped_args)

                    out_map_fn = self.output_mapping_function.get(next_tool_name, lambda r: r)
                    outputs[next_tool_name] = out_map_fn(result)
            except Exception as e:
                logger.exception(f"Error executing tool {next_tool_name}: {e}")
                # Normalize errors through output mapping if possible
                out_map_fn = self.output_mapping_function.get(next_tool_name)
                error_obj = {"error": str(e)}
                try:
                    outputs[next_tool_name] = out_map_fn(error_obj) if out_map_fn else error_obj
                except Exception:
                    outputs[next_tool_name] = error_obj
        return outputs
    
    def __call__(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        Execute the tool collection with the given query.
        
        Args:
            query: The input query to process through the tool collection
            
        Returns:
            Dict containing the results from all executed tools
        """
        inputs = {"query": query, **kwargs}
        return self._run_pipeline(inputs)
