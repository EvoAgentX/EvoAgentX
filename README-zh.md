<!-- Add logo here -->
<div align="center">
  <a href="https://github.com/EvoAgentX/EvoAgentX">
    <img src="./assets/EAXLoGo.svg" alt="EvoAgentX" width="50%">
  </a>
</div>

<h2 align="center">
    构建自进化的 AI 智能体生态系统
</h2>

<div align="center">

[![文档](https://img.shields.io/badge/-文档-0A66C2?logo=readthedocs&logoColor=white&color=7289DA&labelColor=grey)](https://EvoAgentX.github.io/EvoAgentX/)
[![Discord](https://img.shields.io/badge/Chat-Discord-5865F2?&logo=discord&logoColor=white)](https://discord.gg/EvoAgentX)
[![Twitter](https://img.shields.io/badge/Follow-@EvoAgentX-e3dee5?&logo=x&logoColor=white)](https://x.com/EvoAgentX)
[![Wechat](https://img.shields.io/badge/微信-EvoAgentX-brightgreen?logo=wechat&logoColor=white)]()
[![GitHub star chart](https://img.shields.io/github/stars/EvoAgentX/EvoAgentX?style=social)](https://star-history.com/#EvoAgentX/EvoAgentX)
[![GitHub fork](https://img.shields.io/github/forks/EvoAgentX/EvoAgentX?style=social)](https://github.com/EvoAgentX/EvoAgentX/fork)
[![License](https://img.shields.io/badge/License-MIT-blue.svg?)](https://github.com/EvoAgentX/EvoAgentX/blob/main/LICENSE)
<!-- [![EvoAgentX 首页](https://img.shields.io/badge/EvoAgentX-Homepage-blue?logo=homebridge)](https://EvoAgentX.github.io/EvoAgentX/) -->
<!-- [![hf_space](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-EvoAgentX-ffc107?color=ffc107&logoColor=white)](https://huggingface.co/EvoAgentX) -->
</div>

<div align="center">

<h3 align="center">

<a href="./README.md">English</a> | <a href="./README-zh.md" style="text-decoration: underline;">简体中文</a>

</h3>

</div>

<h4 align="center">
  <i>面向 Agent 工作流评估与演化的自动化框架</i>
</h4>

<p align="center">
  <img src="./assets/framework_zh.png">
</p>


## 🔥 最新动态
- **[2025年5月]** 🎉 **EvoAgentX** 正式发布！

## ⚡开始使用
- [安装指南](#安装指南)
- [配置指南](#配置指南)
- [示例：自动工作流生成](#示例自动工作流生成)
- [演示视频](#演示视频)
- [教程与用例](#教程与用例)

## 安装指南

我们推荐使用 `pip` 安装 EvoAgentX：

```bash
pip install evoagentx
```

若需本地开发或更详细的安装步骤（例如使用 conda），请参阅：[EvoAgentX 安装指南]((./docs/installation.md))。

<details>
<summary>本地开发示例（可选）：</summary>

```bash
git clone https://github.com/EvoAgentX/EvoAgentX.git
cd EvoAgentX

# 创建 Conda 虚拟环境
conda create -n evoagentx python=3.10

# 激活环境
conda activate evoagentx

# 安装依赖
pip install -r requirements.txt
# 或者开发者模式安装
pip install -e .
```
</details>


## 配置指南
要使用 EvoAgentX 中的语言大模型（如 OpenAI），需要设置 API 密钥。

#### 方式一：通过环境变量设置 API 密钥

- Linux/macOS: 
```bash
export OPENAI_API_KEY=<你的 OpenAI API 密钥>
```

- Windows 命令提示符：
```cmd 
set OPENAI_API_KEY=<你的 OpenAI API 密钥>
```

-  Windows PowerShell:
```powershell
$env:OPENAI_API_KEY="<你的 OpenAI API 密钥>"  # 注意引号不可省略
```

然后你可以在 Python 中这样获取密钥：
```python
import os
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
```

#### 方式二：使用 .env 文件

- 在项目根目录创建 .env 文件：
```bash
OPENAI_API_KEY=<你的 OpenAI API 密钥>
```

然后在 Python 中加载：
```python
from dotenv import load_dotenv 
import os 

load_dotenv()  # 加载 .env 文件中的环境变量
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
```
> 🔐 提示：请将 .env 添加到 .gitignore，以避免泄露敏感信息。


### 配置并使用语言模型（LLM）

一旦设置好 API 密钥，可以初始化语言模型如下：

```python
from evoagentx.models import OpenAILLMConfig, OpenAILLM

# 加载 API 密钥
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 定义 LLM 配置
openai_config = OpenAILLMConfig(
    model="gpt-4o-mini",       # 指定模型名称
    openai_key=OPENAI_API_KEY, # 设置密钥
    stream=True,               # 开启流式响应
    output_response=True       # 控制台输出响应内容
)

# 初始化语言模型
llm = OpenAILLM(config=openai_config)

# 生成测试响应
response = llm.generate(prompt="什么是 Agentic Workflow？")
print(response)
```
> 📖 更多模型类型和参数说明请见：[LLM 模块指南](./docs/modules/llm.md)。


## 示例：自动工作流生成

配置好 API 密钥和语言模型后，你可以使用 EvoAgentX 自动生成和执行多智能体工作流。

🧩 核心步骤：
1. 定义自然语言目标
2. 用 WorkFlowGenerator 自动生成工作流
3. 使用 AgentManager 实例化智能体
4. 调用 WorkFlow 执行整个流程

💡 用例：

```python
from evoagentx.workflow import WorkFlowGenerator, WorkFlowGraph, WorkFlow
from evoagentx.agents import AgentManager

goal = "生成可在浏览器中玩的 Tetris（俄罗斯方块）HTML 游戏代码"
workflow_graph = WorkFlowGenerator(llm=llm).generate_workflow(goal)

agent_manager = AgentManager()
agent_manager.add_agents_from_workflow(workflow_graph, llm_config=openai_config)

workflow = WorkFlow(graph=workflow_graph, agent_manager=agent_manager, llm=llm)
output = workflow.execute()
print(output)
```

你还可以：
- 📊 可视化工作流：`workflow_graph.display()`
- 💾 保存 / 加载工作流：`save_module()` / `from_file()`

> 📂 查看完整示例请访问 [`workflow_demo.py`](./examples/workflow_demo.py)。

## 演示视频
🎥 演示视频即将上线，敬请期待！

> 在此之前，你可以先阅读 [EvoAgentX 快速入门指南](./docs/quickstart.md)，按照步骤上手使用 EvoAgentX。

## 教程与用例

探索如何有效地使用 EvoAgentX:

| Cookbook | Description |
|:---|:---|
| **[构建你的第一个智能体](./docs/tutorial/first_agent.md)** | 逐步创建第一个智能体的完全指南. |
| **[手动构建工作流](./docs/tutorial/first_workflow.md)** | 学习如何设计和实现写作智能体工作流. |
| **[基准和评估教程](./docs/tutorial/benchmark_and_evaluation.md)** | 智能体性能评估和准则测试指南. |
| **[SEW优化器教程](./docs/tutorial/sew_optimizer.md)** | 学习优化技术，提升智能体工作效率. |

🛠️ 按照教程构建和优化你的 EvoAgentX 工作流。

💡 通过这些实际案例，发掘 EvoAgentX 在你的项目中的潜力！

## 🙋 支持

### 加入社区

📢 参与并跟随  **EvoAgentX** 的发展历程！  
🚩 加入我们的社区，获取最新动态，分享你的想法，并与全球的AI爱好者合作。

- [Discord](https://discord.com/invite/EvoAgentX) — 实时聊天，讨论和协作。
- [X (formerly Twitter)](https://x.com/EvoAgentX) — 获取新闻、更新和洞察。
- [WeChat]() — 与中国社区连接。

### 联系信息

如果你有任何问题或反馈，请随时联系我们。我们非常欢迎您的建议！

- **邮箱:** evoagentx.ai@gmail.com

我们将在2-3个工作日内回复所有问题。

## 🙌 为EvoAgentX做贡献
感谢以下优秀的贡献者

<a href="https://github.com/EvoAgentX/EvoAgentX/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=EvoAgentX/EvoAgentX" />
</a>

我们感谢你对我们开源项目的兴趣。我们提供了 [贡献指南文档](https://github.com/clayxai/EvoAgentX/blob/main/CONTRIBUTING.md) 其中列出了为EvoAgentX做贡献的步骤。请参考此指南，确保顺利合作并取得成功。 🤝🚀

[![Star History Chart](https://api.star-history.com/svg?repos=EvoAgentX/EvoAgentX&type=Date)](https://www.star-history.com/#EvoAgentX/EvoAgentX&Date)


## 📄 许可证
本仓库中的源代码根据 [MIT 许可证](./LICENSE) 提供。
