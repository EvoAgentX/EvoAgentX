# Alita Agent (Search + Storage + Code + Dynamic Tools)

This page introduces the **Alita agent**: a single agent that can search the web, read and write files, execute code in an isolated environment, and **create new tools at runtime**.

The implementation lives in:

- `evoagentx/tools/alita_agent.py`
- Example usage: `examples/alita_agent_example.py`

## What toolkits Alita uses

Alita is built on top of existing EvoAgentX toolkits:

- **Web search** – `SerpAPIToolkit`
  - Uses [SerpAPI](https://serpapi.com/) to search the web.
  - Reads the API key from `SERPAPI_KEY` (or the explicit argument).
- **File storage** – `StorageToolkit`
  - Configured with `base_path="./workplace/alita_storage"`.
  - All read/write operations for Alita go under this directory.
- **Code execution** – `DockerInterpreterToolkit` (preferred) or `PythonInterpreterToolkit` (fallback)
  - Docker version:
    - Runs code in a container (image `python:3.9-slim` by default).
    - Uses its own workspace under `./workplace/docker`.
  - Python version:
    - Executes code locally in a controlled environment.
    - Uses `./workplace/python` as its workspace.
- **Dynamic tools** – `AlitaDynamicToolkit`
  - Manages code-based generated tools.
  - Uses the code execution toolkit internally to run generated code.

## How to configure and create the agent

You can create Alita with the helper function `create_alita_agent`:

```python
from evoagentx.models import OpenRouterConfig  # or another LLM config
from evoagentx.tools.alita_agent import create_alita_agent

llm_config = OpenRouterConfig(
    model="openai/gpt-5-mini",
    openrouter_key="YOUR_OPENROUTER_API_KEY",
    stream=True,
    output_response=True,
)

agent = create_alita_agent(
    llm_config=llm_config,
    persist_dynamic_tools=True,
    load_existing_dynamic_tools=True,
    dynamic_tools_path="./workplace/alita/dynamic_tools.json",
    use_docker=True,
    serpapi_api_key=None,  # or pass your SERPAPI_KEY explicitly
)

message = agent(inputs={"instruction": "Your high-level task description"})
print(message.content.result)
```

Key configuration options:

- `persist_dynamic_tools: bool`
  - If `True`, every newly created generated tool is saved to a JSON file.
- `load_existing_dynamic_tools: bool`
  - If `True`, previously saved generated tools are loaded at startup.
- `dynamic_tools_path: str`
  - Path to the JSON file storing generated tool definitions  
    (default: `./workplace/alita/dynamic_tools.json`).
- `use_docker: bool`
  - If `True`, prefer Docker-based code execution.
  - If Docker initialization fails, Alita automatically falls back to the Python interpreter.

## How dynamic tools are created and used

Alita does **not** directly register each new tool as a separate function name in the agent.  
Instead, it exposes a small set of stable meta-tools that the LLM can always see:

- `create_generated_tool`
  - Inputs: `tool_name`, `description`, `code`, `language`
  - Creates a new `GeneratedCodeTool` and stores it in an internal registry.
  - The generated code receives a variable named `payload` and must assign the final result to `result`.
- `call_generated_tool`
  - Inputs: `tool_name`, `payload`
  - Looks up the generated tool by name and executes it with the given `payload`.
- `list_generated_tools`, `remove_generated_tool`, `reload_generated_tools`
  - Manage the internal registry and persisted tools.

Conceptually:

1. The LLM first uses `create_generated_tool` to define a new tool:
   - It writes the function body as Python code.
   - It chooses a `tool_name` and describes what the tool expects in `payload`.
2. Later, when it wants to use that tool, it calls `call_generated_tool`:
   - It passes the same `tool_name`.
   - It builds a `payload` object that matches what the generated code expects.

The generated tool itself is always called with a single `payload` argument.  
The internal structure of `payload` (e.g. fields like `text`, `path`, etc.) is decided by the LLM when it writes the code and description.

## Example: creating and using a generated tool

The file `examples/alita_agent_example.py` contains two examples:

1. **Example 1** – Use Alita to:
   - Search the EvoAgentX GitHub repository via SerpAPI.
   - Summarize what the project is.
   - Save the summary to `project_summary.txt` in `./workplace/alita_storage`.
2. **Example 2** – Let the agent itself:
   - Use `create_generated_tool` to build a `payload_text_analyzer` tool that:
     - Reads `payload["text"]`.
     - Returns a JSON object with keys `original`, `upper`, and `length`.
   - Use `call_generated_tool` to run this tool on a sample text.
   - Explain in natural language what the tool did and show the JSON result.

You can run the example with `uv`:

```bash
uv run examples/alita_agent_example.py
```

Make sure to set:

- `OPENROUTER_API_KEY` (or another LLM provider key, depending on your config)
- `SERPAPI_KEY` (if you want web search)

This should give you a concrete demonstration of:

- How Alita uses search, storage, and code execution tools.
- How it creates a new generated tool at runtime.
- How it calls that generated tool via `call_generated_tool` to solve a task. 

