import json
import os
from typing import Any, Dict, List, Optional

from ..core.logging import logger
from ..models.model_configs import LLMConfig
from ..agents import CustomizeAgent

from .tool import Tool, Toolkit
from .storage_file import StorageToolkit
from .search_serpapi import SerpAPIToolkit
from .interpreter_docker import DockerInterpreterToolkit
from .interpreter_python import PythonInterpreterToolkit


class GeneratedCodeTool(Tool):
    """
    A dynamically created code-based tool backed by the interpreter toolkit.

    The tool wraps LLM-generated source code (e.g. Python) into a callable Tool.
    When invoked, it executes the stored code through a base code executor
    (Docker or Python interpreter) and passes a generic ``payload`` object
    into the code as the ``payload`` variable.

    The generated code should assign its final result to a variable named
    ``result``. This tool will then serialize ``result`` to JSON and return
    it to the caller. If serialization fails, the raw textual output from the
    code executor is returned instead.
    """

    name: str = "generated_code_tool"
    description: str = (
        "A dynamic tool that executes LLM-generated source code using a shared "
        "code execution backend."
    )
    inputs: Dict[str, Dict[str, str]] = {
        "payload": {
            "type": "object",
            "description": (
                "Optional payload object made available to the generated code "
                "as the variable `payload`."
            ),
        }
    }
    required: Optional[List[str]] = []

    def __init__(
        self,
        code_executor: Optional[Tool],
        tool_name: str,
        description: str,
        code: str,
        language: str = "python",
    ):
        super().__init__()
        self.name = tool_name
        self.description = description or f"Generated code tool '{tool_name}'"
        self._code_executor = code_executor
        self._source_code = code
        self._language = language or "python"

    def __call__(self, payload: dict = None) -> Dict[str, Any]:
        if self._code_executor is None:
            return {
                "success": False,
                "error": "Code executor is not configured for this generated tool.",
            }

        payload = payload or {}
        try:
            payload_json = json.dumps(payload, ensure_ascii=False)
        except TypeError:
            # Fallback for non-JSON-serializable payloads
            payload_json = json.dumps(
                {"raw_payload_repr": repr(payload)}, ensure_ascii=False
            )

        # Inject payload and user code into a small wrapper that expects the
        # user to set `result` and prints it as JSON.
        wrapper_code = (
            "import json\n\n"
            f"payload = json.loads({json.dumps(payload_json)})\n\n"
            "result = None\n\n"
            f"{self._source_code}\n\n"
            "print(json.dumps(result, ensure_ascii=False))\n"
        )

        try:
            output = self._code_executor(code=wrapper_code, language=self._language)
        except Exception as exc:
            logger.error(
                "Error executing generated tool '%s' via code executor: %s",
                self.name,
                exc,
            )
            return {"success": False, "error": str(exc)}

        if output is None:
            return {
                "success": False,
                "error": "Code executor returned no output for generated tool.",
            }

        # Try to parse the output as JSON; if that fails, return raw text.
        try:
            parsed = json.loads(output)
            return {
                "success": True,
                "result": parsed,
                "raw_output": output,
            }
        except Exception:
            return {
                "success": True,
                "result": output,
            }


class AlitaDynamicToolkit(Toolkit):
    """
    Toolkit that manages dynamically created code-based tools.

    It exposes meta-tools that allow the agent to create, list, delete,
    and reload generated tools. Tool definitions (name, description, code,
    language) can optionally be persisted to disk and reloaded in later runs.
    """

    def __init__(
        self,
        name: str = "AlitaDynamicToolkit",
        base_toolkits: Optional[List[Toolkit]] = None,
        dynamic_tools_path: str = "./workplace/alita/dynamic_tools.json",
        persist_new_tools: bool = True,
        load_existing_tools: bool = True,
    ):
        # First create meta-tools and initialize the underlying Toolkit/BaseModel
        # so that Pydantic's internal state is ready before we assign any
        # additional attributes on this instance.
        create_tool = CreateGeneratedTool(dynamic_toolkit=self)
        call_tool = CallGeneratedTool(dynamic_toolkit=self)
        list_tool = ListGeneratedTools(dynamic_toolkit=self)
        remove_tool = RemoveGeneratedTool(dynamic_toolkit=self)
        reload_tool = ReloadDynamicTools(dynamic_toolkit=self)

        tools: List[Tool] = [create_tool, call_tool, list_tool, remove_tool, reload_tool]

        # dynamic_tools_path and persist_new_tools are stored as extra fields on
        # the Pydantic BaseModel (BaseModule) and later used by helper methods.
        super().__init__(
            name=name,
            tools=tools,
            dynamic_tools_path=dynamic_tools_path,
            persist_new_tools=persist_new_tools,
        )

        base_toolkits = base_toolkits or []
        self.base_tool_registry: Dict[str, Tool] = {}
        for toolkit in base_toolkits:
            try:
                for tool in toolkit.get_tools():
                    if tool.name in self.base_tool_registry:
                        logger.warning(
                            "Duplicate base tool name '%s' encountered when "
                            "initializing AlitaDynamicToolkit; keeping the "
                            "first instance.",
                            tool.name,
                        )
                        continue
                    self.base_tool_registry[tool.name] = tool
            except Exception as exc:
                logger.error(
                    "Failed to read tools from base toolkit '%s': %s",
                    getattr(toolkit, "name", "unknown"),
                    exc,
                )

        # Resolve a code execution tool from the base toolkits, preferring
        # Docker if available and falling back to the Python interpreter.
        self._code_executor: Optional[Tool] = None
        for preferred_name in ("docker_execute", "python_execute"):
            candidate = self.base_tool_registry.get(preferred_name)
            if candidate is not None:
                self._code_executor = candidate
                break

        if self._code_executor is None:
            logger.warning(
                "AlitaDynamicToolkit initialized without a code execution tool; "
                "generated tools will not be executable."
            )

        self._dynamic_tools: Dict[str, GeneratedCodeTool] = {}

        # Optionally load any previously saved generated tools from disk and
        # register them into the toolkit.
        if load_existing_tools:
            loaded = self._load_from_disk()
            for tool in loaded:
                self.register_dynamic_tool(tool)

    # ----------------- Dynamic tool management helpers -----------------

    def register_dynamic_tool(self, tool: GeneratedCodeTool) -> None:
        """Register a new generated tool in memory and expose it via this toolkit."""
        existing = self._dynamic_tools.get(tool.name)
        if existing is not None:
            logger.info("Overwriting existing dynamic tool '%s'.", tool.name)
            # Remove old instance from the tools list if it exists
            self.tools = [t for t in self.tools if t is not existing]

        self._dynamic_tools[tool.name] = tool
        if tool not in self.tools:
            self.tools.append(tool)

    def remove_dynamic_tool(self, tool_name: str) -> bool:
        """Remove a generated tool by name."""
        tool = self._dynamic_tools.pop(tool_name, None)
        if tool is None:
            return False
        self.tools = [t for t in self.tools if t is not tool]
        return True

    def _ensure_directory(self) -> None:
        directory = os.path.dirname(self.dynamic_tools_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

    def save_dynamic_tools(self) -> None:
        """Persist current dynamic tool definitions to disk."""
        if not self.persist_new_tools:
            return

        self._ensure_directory()
        payload: List[Dict[str, Any]] = []
        for tool in self._dynamic_tools.values():
            payload.append(
                {
                    "name": tool.name,
                    "description": tool.description,
                    "code": tool._source_code,
                    "language": tool._language,
                }
            )

        try:
            with open(self.dynamic_tools_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
        except Exception as exc:
            logger.error("Failed to save dynamic tools to '%s': %s", self.dynamic_tools_path, exc)

    def _load_from_disk(self) -> List[GeneratedCodeTool]:
        """Load dynamic tool definitions from disk."""
        if not os.path.exists(self.dynamic_tools_path):
            return []

        try:
            with open(self.dynamic_tools_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except Exception as exc:
            logger.error("Failed to load dynamic tools from '%s': %s", self.dynamic_tools_path, exc)
            return []

        loaded: List[GeneratedCodeTool] = []
        if not isinstance(raw, list):
            logger.error("Dynamic tools file '%s' must contain a list.", self.dynamic_tools_path)
            return []

        for item in raw:
            try:
                name = item.get("name")
                desc = item.get("description", "")
                if not name:
                    logger.warning("Skipping dynamic tool without a name in '%s'.", self.dynamic_tools_path)
                    continue
                code = item.get("code", "")
                language = item.get("language", "python")
                if not code:
                    logger.warning(
                        "Skipping dynamic tool '%s' in '%s' because it has no code.",
                        name,
                        self.dynamic_tools_path,
                    )
                    continue
                tool = GeneratedCodeTool(
                    code_executor=self._code_executor,
                    tool_name=name,
                    description=desc,
                    code=code,
                    language=language,
                )
                self._dynamic_tools[name] = tool
                loaded.append(tool)
            except Exception as exc:
                logger.error("Failed to reconstruct dynamic tool from '%s': %s", item, exc)

        return loaded

    def reload_from_disk(self) -> Dict[str, Any]:
        """Reload dynamic tools from disk and replace in-memory definitions."""
        loaded = self._load_from_disk()
        # Remove all previously registered generated tools
        current_names = set(self._dynamic_tools.keys())
        for name in current_names:
            self.remove_dynamic_tool(name)

        for tool in loaded:
            self.register_dynamic_tool(tool)

        return {
            "success": True,
            "count": len(loaded),
            "tool_names": [t.name for t in loaded],
        }


class CreateGeneratedTool(Tool):
    """
    Tool that creates a new code-based tool and registers it into
    :class:`AlitaDynamicToolkit`.
    """

    name: str = "create_generated_tool"
    description: str = (
        "Create a new generated code tool using the underlying code execution "
        "backend. The provided code will be executed with a `payload` variable "
        "available and must assign the final result to a variable named "
        "`result`."
    )
    inputs: Dict[str, Dict[str, str]] = {
        "tool_name": {
            "type": "string",
            "description": "Unique name of the new generated tool.",
        },
        "description": {
            "type": "string",
            "description": "Human-readable description of the new tool.",
        },
        "code": {
            "type": "string",
            "description": (
                "Source code for the generated tool. The code will be executed "
                "with a `payload` variable in scope and must assign its final "
                "output to `result`."
            ),
        },
        "language": {
            "type": "string",
            "description": "Programming language of the code (default: python).",
        },
    }
    required: Optional[List[str]] = ["tool_name", "code"]

    def __init__(self, dynamic_toolkit: "AlitaDynamicToolkit"):
        super().__init__()
        self.dynamic_toolkit = dynamic_toolkit

    def __call__(
        self,
        tool_name: str,
        description: str = "",
        code: str = "",
        language: str = "python",
    ) -> Dict[str, Any]:
        if not code or not code.strip():
            return {"success": False, "error": "`code` must be a non-empty string."}

        generated_tool = GeneratedCodeTool(
            code_executor=self.dynamic_toolkit._code_executor,
            tool_name=tool_name,
            description=description or f"Generated code tool '{tool_name}'",
            code=code,
            language=language,
        )
        self.dynamic_toolkit.register_dynamic_tool(generated_tool)
        self.dynamic_toolkit.save_dynamic_tools()

        return {
            "success": True,
            "tool_name": tool_name,
            "description": generated_tool.description,
            "language": language,
        }


class CallGeneratedTool(Tool):
    """
    Tool that calls an existing generated tool by name.

    This is a stable entry point that the agent can always see, even though
    the underlying generated tools are created at runtime.
    """

    name: str = "call_generated_tool"
    description: str = (
        "Call a previously created generated tool by name with a payload "
        "object. The payload will be passed to the generated tool as its "
        "`payload` argument."
    )
    inputs: Dict[str, Dict[str, str]] = {
        "tool_name": {
            "type": "string",
            "description": "Name of the generated tool to call.",
        },
        "payload": {
            "type": "object",
            "description": "Arbitrary payload object passed to the generated tool.",
        },
    }
    required: Optional[List[str]] = ["tool_name"]

    def __init__(self, dynamic_toolkit: "AlitaDynamicToolkit"):
        super().__init__()
        self.dynamic_toolkit = dynamic_toolkit

    def __call__(self, tool_name: str, payload: dict = None) -> Dict[str, Any]:
        tool = self.dynamic_toolkit._dynamic_tools.get(tool_name)
        if tool is None:
            # Try to reload from disk in case the tool was created in a
            # previous run or another process.
            self.dynamic_toolkit.reload_from_disk()
            tool = self.dynamic_toolkit._dynamic_tools.get(tool_name)
            if tool is None:
                return {
                    "success": False,
                    "error": f"Generated tool '{tool_name}' not found.",
                }

        try:
            return tool(payload=payload or {})
        except Exception as exc:
            logger.error(
                "Error while calling generated tool '%s': %s",
                tool_name,
                exc,
            )
            return {"success": False, "error": str(exc)}


class ListGeneratedTools(Tool):
    """
    Tool that lists all currently registered generated tools.
    """

    name: str = "list_generated_tools"
    description: str = "List all dynamic generated tools managed by AlitaDynamicToolkit."
    inputs: Dict[str, Dict[str, str]] = {}
    required: Optional[List[str]] = []

    def __init__(self, dynamic_toolkit: "AlitaDynamicToolkit"):
        super().__init__()
        self.dynamic_toolkit = dynamic_toolkit

    def __call__(self) -> Dict[str, Any]:
        tools_info = [
            {"name": name, "description": tool.description}
            for name, tool in self.dynamic_toolkit._dynamic_tools.items()
        ]
        return {"success": True, "tools": tools_info}


class RemoveGeneratedTool(Tool):
    """
    Tool that removes a previously created generated tool.
    """

    name: str = "remove_generated_tool"
    description: str = "Remove a dynamic generated tool by name."
    inputs: Dict[str, Dict[str, str]] = {
        "tool_name": {
            "type": "string",
            "description": "Name of the generated tool to remove.",
        }
    }
    required: Optional[List[str]] = ["tool_name"]

    def __init__(self, dynamic_toolkit: "AlitaDynamicToolkit"):
        super().__init__()
        self.dynamic_toolkit = dynamic_toolkit

    def __call__(self, tool_name: str) -> Dict[str, Any]:
        success = self.dynamic_toolkit.remove_dynamic_tool(tool_name)
        if not success:
            return {
                "success": False,
                "error": f"Generated tool '{tool_name}' not found.",
            }
        self.dynamic_toolkit.save_dynamic_tools()
        return {"success": True, "tool_name": tool_name}


class ReloadDynamicTools(Tool):
    """
    Tool that reloads generated tool definitions from disk at runtime.
    """

    name: str = "reload_generated_tools"
    description: str = (
        "Reload dynamic generated tools from the configured persistence path."
    )
    inputs: Dict[str, Dict[str, str]] = {}
    required: Optional[List[str]] = []

    def __init__(self, dynamic_toolkit: "AlitaDynamicToolkit"):
        super().__init__()
        self.dynamic_toolkit = dynamic_toolkit

    def __call__(self) -> Dict[str, Any]:
        return self.dynamic_toolkit.reload_from_disk()


def create_alita_agent(
    llm_config: LLMConfig,
    persist_dynamic_tools: bool = True,
    load_existing_dynamic_tools: bool = True,
    dynamic_tools_path: str = "./workplace/alita/dynamic_tools.json",
    use_docker: bool = True,
    serpapi_api_key: Optional[str] = None,
) -> CustomizeAgent:
    """
    Build the Alita agent with search, storage, code execution, and dynamic tools.

    The agent is implemented as a :class:`CustomizeAgent` with a single action
    that can:

    - Call SerpAPI search via :class:`SerpAPIToolkit`
    - Read and write files via :class:`StorageToolkit`
    - Execute code via :class:`DockerInterpreterToolkit` (preferred) or
      :class:`PythonInterpreterToolkit` as a fallback
    - Create, list, remove, and reload code-based generated tools via
      :class:`AlitaDynamicToolkit`

    Args:
        llm_config: Language model configuration used by the agent.
        persist_dynamic_tools: If True, newly created generated tools are
            persisted to disk.
        load_existing_dynamic_tools: If True, previously persisted generated
            tools are loaded at initialization.
        dynamic_tools_path: JSON file path used to store generated tool
            definitions (name, description, code, language).
        use_docker: Prefer Docker-based code execution when True; falls back to
            the Python interpreter if Docker initialization fails.
        serpapi_api_key: Optional explicit SerpAPI API key. If not provided,
            the SerpAPIToolkit will fall back to the SERPAPI_KEY environment
            variable.
    """

    # Search toolkit (SerpAPI)
    serp_toolkit = SerpAPIToolkit(
        api_key=serpapi_api_key,
    )

    # Storage toolkit, scoped to a dedicated base path
    storage_toolkit = StorageToolkit(
        name="AlitaStorageToolkit",
        base_path="./workplace/alita_storage",
    )

    # Code execution toolkit: prefer Docker for stronger isolation.
    code_toolkit: Toolkit
    if use_docker:
        try:
            code_toolkit = DockerInterpreterToolkit(
                image_tag="python:3.9-slim",
                print_stdout=True,
                print_stderr=True,
                container_directory="/app",
            )
        except Exception as exc:
            logger.warning(
                "Failed to initialize DockerInterpreterToolkit (%s); "
                "falling back to PythonInterpreterToolkit.",
                exc,
            )
            code_toolkit = PythonInterpreterToolkit(
                project_path=".",
                directory_names=["examples", "evoagentx"],
                allowed_imports=set(),
            )
    else:
        code_toolkit = PythonInterpreterToolkit(
            project_path=".",
            directory_names=["examples", "evoagentx"],
            allowed_imports=set(),
        )

    base_toolkits: List[Toolkit] = [serp_toolkit, storage_toolkit, code_toolkit]

    dynamic_toolkit = AlitaDynamicToolkit(
        base_toolkits=base_toolkits,
        dynamic_tools_path=dynamic_tools_path,
        persist_new_tools=persist_dynamic_tools,
        load_existing_tools=load_existing_dynamic_tools,
    )

    prompt = (
        "You are Alita, an autonomous AI agent. "
        "You can search the web, read and write files, execute code in a "
        "safe environment, and create reusable code-based tools that "
        "encapsulate custom logic. Carefully decide when to call tools and "
        "provide the best possible final answer to the user.\n\n"
        "You may create new generated tools with 'create_generated_tool' by "
        "supplying source code that uses the `payload` variable and assigns "
        "its final output to `result`. To run a generated tool, always call "
        "the stable tool 'call_generated_tool' with the appropriate "
        "`tool_name` and `payload` instead of calling the generated tool name "
        "directly. Generated tools can be listed, removed, and reloaded using "
        "the management tools.\n\n"
        "User instruction: {instruction}"
    )

    agent = CustomizeAgent(
        name="AlitaAgent",
        description=(
            "An autonomous agent that can search the web, work with files, "
            "execute code safely, and dynamically create code-based tools."
        ),
        prompt=prompt,
        llm_config=llm_config,
        inputs=[
            {
                "name": "instruction",
                "type": "string",
                "description": "High-level task description for Alita.",
            }
        ],
        outputs=[
            {
                "name": "result",
                "type": "string",
                "description": "The final result or summary produced by Alita.",
            }
        ],
        parse_mode="str",
        tools=[serp_toolkit, storage_toolkit, code_toolkit, dynamic_toolkit],
        max_tool_calls=10,
    )

    return agent


def alita_agent(*args, **kwargs) -> CustomizeAgent:
    """
    Convenience wrapper for :func:`create_alita_agent`.

    This lets callers simply do ``alita_agent(llm_config=...)`` to obtain
    a fully initialized agent instance.
    """

    return create_alita_agent(*args, **kwargs)
