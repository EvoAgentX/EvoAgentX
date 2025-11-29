# Alita 智能体（搜索 + 存储 + 代码执行 + 动态工具）

本页面简要介绍 **Alita 智能体**：一个集成了网页搜索、文件读写、代码执行，并且可以在运行过程中 **动态创建工具** 的单智能体。

核心实现位置：

- `evoagentx/tools/alita_agent.py`
- 使用示例：`examples/alita_agent_example.py`

## 使用了哪些工具

Alita 基于 EvoAgentX 已有的工具体系组合而成：

- **网页搜索** – `SerpAPIToolkit`
  - 通过 [SerpAPI](https://serpapi.com/) 进行搜索。
  - 优先使用传入的 `serpapi_api_key`，否则从环境变量 `SERPAPI_KEY` 读取。
- **文件存储** – `StorageToolkit`
  - 配置 `base_path="./workplace/alita_storage"`。
  - Alita 的读写文件都被限制在这个目录下。
- **代码执行** – `DockerInterpreterToolkit`（优先）和 `PythonInterpreterToolkit`（兜底）
  - Docker 版本：
    - 默认镜像 `python:3.9-slim`。
    - 使用 `./workplace/docker` 作为容器相关的工作区。
  - Python 版本：
    - 在本地受控环境中执行 Python 代码。
    - 使用 `./workplace/python` 作为工作区。
- **动态工具** – `AlitaDynamicToolkit`
  - 负责管理“由 LLM 生成代码的工具”。
  - 内部依赖上面的代码执行工具来运行生成的代码。

## 如何配置并创建 Alita

推荐使用工具函数 `create_alita_agent` 创建智能体：

```python
from evoagentx.models import OpenRouterConfig  # 也可以使用其他 LLM 配置
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
    serpapi_api_key=None,  # 或显式传入 SERPAPI_KEY
)

message = agent(inputs={"instruction": "你的任务描述"})
print(message.content.result)
```

关键参数说明：

- `persist_dynamic_tools: bool`
  - 是否在创建新工具后，将工具定义持久化到一个 JSON 文件。
- `load_existing_dynamic_tools: bool`
  - 初始化时是否从 JSON 文件中加载之前创建过的工具。
- `dynamic_tools_path: str`
  - 动态工具定义的 JSON 文件路径（默认 `./workplace/alita/dynamic_tools.json`）。
- `use_docker: bool`
  - 是否优先使用 Docker 环境执行代码。
  - 如果 Docker 初始化失败，会自动回退到本地 Python 解释器。

## 动态工具的原理：创建与调用

Alita 不会在 Agent 层面为每个新工具单独注册一个函数名，而是通过一组“元工具（meta tools）”来管理运行时生成的工具。LLM 能看到并调用的固定工具包括：

- `create_generated_tool`
  - 输入：`tool_name`, `description`, `code`, `language`
  - 作用：创建一个新的 `GeneratedCodeTool`，并存入内部的 `_dynamic_tools` 字典。
  - LLM 提供的 `code` 会在执行时被注入一个变量 `payload`，并要求最终把结果赋值给 `result`。
- `call_generated_tool`
  - 输入：`tool_name`, `payload`
  - 作用：按名称找到对应的生成工具，并将 `payload` 作为其输入执行。
- `list_generated_tools` / `remove_generated_tool` / `reload_generated_tools`
  - 用于查看、删除和从磁盘重载已生成的工具。

运行时的核心过程可以概括为：

1. LLM 首先通过 `create_generated_tool` 创建一个工具，例如：
   - `tool_name="payload_text_analyzer"`
   - `code` 内部用 `payload["text"]` 做处理，并将结果写入 `result`。
2. 当需要使用这个工具时，LLM 再调用 `call_generated_tool`：
   - 传入同样的 `tool_name`。
   - 构造一个符合自己代码逻辑的 `payload` 对象，例如 `{"text": "Hello from Alita"}`。

内部的 `GeneratedCodeTool` 始终以单个参数 `payload` 作为输入。  
`payload` 的内部结构（字段名、类型等）完全由 LLM 在编写 `code` 和 `description` 时约定，而不是框架强制规定。

## 示例：创建并使用一个生成工具

`examples/alita_agent_example.py` 提供了两个示例：

1. **示例 1**：使用 Alita
   - 通过 SerpAPI 搜索 EvoAgentX 的 GitHub 仓库信息；
   - 总结项目特点；
   - 将总结内容写入 `./workplace/alita_storage/project_summary.txt`。
2. **示例 2**：让 Agent 自己创建并使用工具
   - 使用 `create_generated_tool` 创建一个名为 `payload_text_analyzer` 的工具：
     - 从 `payload["text"]` 读取文本；
     - 返回包含 `original`, `upper`, `length` 三个字段的 JSON 结果。
   - 使用 `call_generated_tool`，给定 `tool_name="payload_text_analyzer"` 和一个 `payload` 文本进行实际调用；
   - 在最终回答中解释工具做了什么，并展示实际的 JSON 输出。

你可以通过 `uv` 运行示例：

```bash
uv run examples/alita_agent_example.py
```

运行前请确保设置好：

- `OPENROUTER_API_KEY`（或你使用的其他 LLM 提供商的 Key）
- `SERPAPI_KEY`（如需启用网页搜索）

这个示例将展示：

- Alita 如何综合使用搜索、文件存储和代码执行工具；
- 它如何在运行中创建一个新的代码型工具；
- 它如何通过 `call_generated_tool` 调用这个工具来完成具体任务。 

