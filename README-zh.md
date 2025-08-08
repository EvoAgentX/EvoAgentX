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

[![EvoAgentX 主页](https://img.shields.io/badge/EvoAgentX-主页-blue?logo=homebridge)](https://evoagentx.org/)
[![文档](https://img.shields.io/badge/-文档-0A66C2?logo=readthedocs&logoColor=white&color=7289DA&labelColor=grey)](https://EvoAgentX.github.io/EvoAgentX/)
[![Discord](https://img.shields.io/badge/Chat-Discord-5865F2?&logo=discord&logoColor=white)](https://discord.gg/XWBZUJFwKe)
[![Twitter](https://img.shields.io/badge/Follow-@EvoAgentX-e3dee5?&logo=x&logoColor=white)](https://x.com/EvoAgentX)
[![Wechat](https://img.shields.io/badge/微信-EvoAgentX-brightgreen?logo=wechat&logoColor=white)](./assets/wechat_info.md)
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
  <img src="./assets/framework_zh.jpg">
</p>


## 🔥 最新消息
- **[2025年5月]** 🎉 **EvoAgentX** 正式发布！

## ⚡ 快速开始
- [🔥 最新消息](#-最新消息)
- [⚡ 快速开始](#-快速开始)
- [安装](#安装)
- [LLM配置](#llm配置)
  - [API密钥配置](#api密钥配置)
  - [配置并使用LLM](#配置并使用llm)
- [自动工作流生成](#自动工作流生成)
- [演示视频](#演示视频)
  - [✨ 最终结果](#-最终结果)
- [进化算法](#进化算法)
  - [📊 结果](#-结果)
- [应用案例](#应用案例)
- [教程与使用案例](#教程与使用案例)
- [🎯 路线图](#-路线图)
- [🙋 支持](#-支持)
  - [加入社区](#加入社区)
  - [联系信息](#联系信息)
  - [观看往期社区会议](#观看往期社区会议)
- [🙌 为EvoAgentX做贡献](#-为evoagentx做贡献)
- [📚 致谢](#-致谢)
- [📄 许可证](#-许可证)

## 安装

我们建议使用 `pip` 安装 EvoAgentX：

```bash
pip install git+https://github.com/EvoAgentX/EvoAgentX.git
```

对于本地开发或详细设置（例如，使用conda），请参考 [EvoAgentX 安装指南](./docs/zh/installation.md)。

<details>
<summary>示例（可选，用于本地开发）：</summary>

```bash
git clone https://github.com/EvoAgentX/EvoAgentX.git
cd EvoAgentX
# 创建新的conda环境
conda create -n evoagentx python=3.10

# 激活环境
conda activate evoagentx

# 安装包
pip install -r requirements.txt
# 或者以开发模式安装
pip install -e .
```
</details>

## LLM配置

### API密钥配置 

要在EvoAgentX中使用LLM（如OpenAI），您必须设置API密钥。

<details>
<summary>选项1：通过环境变量设置API密钥</summary> 

- Linux/macOS: 
```bash
export OPENAI_API_KEY=<your-openai-api-key>
```

- Windows命令提示符: 
```cmd 
set OPENAI_API_KEY=<your-openai-api-key>
```

- Windows PowerShell:
```powershell
$env:OPENAI_API_KEY="<your-openai-api-key>" #必要的
```

设置后，您可以在Python代码中访问密钥：
```python
import os
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
```
</details>

<details>
<summary>选项2：使用.env文件</summary> 

- 在项目根目录创建.env文件并添加以下内容：
```bash
OPENAI_API_KEY=<your-openai-api-key>
```

然后在Python中加载：
```python
from dotenv import load_dotenv 
import os 

load_dotenv() # 从.env文件加载环境变量
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
```
</details>
<!-- > 🔐 提示：不要忘记将`.env`添加到`.gitignore`以避免提交敏感信息。 -->

### 配置并使用LLM
设置好API密钥后，使用以下方式初始化LLM：

```python
from evoagentx.models import OpenAILLMConfig, OpenAILLM

# 从环境中加载API密钥
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 定义LLM配置
openai_config = OpenAILLMConfig(
    model="gpt-4o-mini",       # 指定模型名称
    openai_key=OPENAI_API_KEY, # 直接传递密钥
    stream=True,               # 启用流式响应
    output_response=True       # 将响应打印到标准输出
)

# 初始化语言模型
llm = OpenAILLM(config=openai_config)

# 从LLM生成响应
response = llm.generate(prompt="什么是智能体工作流？")
```
> 📖 有关支持的模型和配置选项的更多详情：[LLM模块指南](./docs/zh/modules/llm.md)。


## 自动工作流生成
配置好API密钥和语言模型后，您可以在EvoAgentX中自动生成和执行多智能体工作流。

🧩 核心步骤：
1. 定义自然语言目标
2. 使用`WorkFlowGenerator`生成工作流
3. 使用`AgentManager`实例化智能体
4. 通过`WorkFlow`执行工作流

💡 最小示例：
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

您还可以：
- 📊 可视化工作流：`workflow_graph.display()`
- 💾 保存/加载工作流：`save_module()` / `from_file()`

> 📂 完整的工作示例，请查看[`workflow_demo.py`](https://github.com/EvoAgentX/EvoAgentX/blob/main/examples/workflow_demo.py)


## 演示视频


[![在YouTube上观看](https://img.shields.io/badge/-在YouTube上观看-red?logo=youtube&labelColor=grey)](https://www.youtube.com/watch?v=Wu0ZydYDqgg)
[![在bilibili上观看](https://img.shields.io/badge/-在bilibili上观看-00A1D6?logo=bilibili&labelColor=white)](https://www.bilibili.com/video/BV1mEJizyE7H/?vd_source=02f8f3a7c8865b3af6378d9680393f5a)

<div align="center">
  <video src="https://github.com/user-attachments/assets/906a6086-e98d-4df3-84b0-808020ddd520.mp4" autoplay loop muted playsinline width="600">
    您的浏览器不支持视频标签。
  </video>
</div>

在此演示中，我们通过两个示例展示了EvoAgentX的工作流生成和执行能力：

- 应用1：简历的智能职位推荐
- 应用2：A股股票的视觉分析


### ✨ 最终结果

<table>
  <tr>
    <td align="center">
      <img src="./assets/demo_result_1.png" width="400"><br>
      <strong>应用1：</strong><br>职位推荐
    </td>
    <td align="center">
      <img src="./assets/demo_result_2.jpeg" width="400"><br>
      <strong>应用2：</strong><br>股票视觉分析
    </td>
  </tr>
</table>

## 进化算法

我们将一些现有的智能体/工作流进化算法集成到EvoAgentX中，包括[TextGrad](https://www.nature.com/articles/s41586-025-08661-4)、[MIPRO](https://arxiv.org/abs/2406.11695)和[AFlow](https://arxiv.org/abs/2410.10762)。

为了评估性能，我们使用它们在三个不同任务上优化相同的智能体系统：多跳问答(HotPotQA)、代码生成(MBPP)和推理(MATH)。我们随机抽取50个样本进行验证，另外100个样本用于测试。

> 提示：我们已将这些基准测试和评估代码集成到EvoAgentX中。有关更多详情，请参考[基准测试和评估教程](https://github.com/EvoAgentX/EvoAgentX/blob/main/docs/zh/tutorial/benchmark_and_evaluation.md)。

### 📊 结果 

| 方法     | HotPotQA<br>(F1%) | MBPP<br>(Pass@1 %) | MATH<br>(解题率 %) |
|----------|--------------------|---------------------|--------------------------|
| 原始     | 63.58              | 69.00               | 66.00                    |
| TextGrad | 71.02              | 71.00               | 76.00                    |
| AFlow    | 65.09              | 79.00               | 71.00                    |
| MIPRO    | 69.16              | 68.00               | 72.30       

更多详情请参考`examples/optimization`文件夹。

## 应用案例 

我们使用我们的框架在[GAIA](https://huggingface.co/spaces/gaia-benchmark/leaderboard)基准测试中优化现有的多智能体系统。我们选择了来自GAIA排行榜的两个具有代表性的开源且可运行的多智能体框架：[Open Deep Research](https://github.com/huggingface/smolagents/tree/main/examples/open_deep_research)和[OWL](https://github.com/camel-ai/owl)。

我们应用EvoAgentX来优化它们的提示词。下图显示了优化后的智能体在GAIA基准验证集上的性能。

<table>
  <tr>
    <td align="center" width="50%">
      <img src="./assets/open_deep_research_optimization_report.png" alt="Open Deep Research优化" width="100%"><br>
      <strong>Open Deep Research</strong>
    </td>
    <td align="center" width="50%">
      <img src="./assets/owl_optimization_result.png" alt="OWL优化" width="100%"><br>
      <strong>OWL智能体</strong>
    </td>
  </tr>
</table>

> 完整优化报告：[Open Deep Research](https://github.com/eax6/smolagents)和[OWL](https://github.com/TedSIWEILIU/owl)。  

## 教程与使用案例

> 💡 **EvoAgentX新手？** 从[快速入门指南](./docs/zh/quickstart.md)开始，逐步了解基本使用方法。


通过以下资源了解如何有效使用EvoAgentX：

| 指南 | Colab 笔记本 | 描述 |
|:---|:---|:---|
| **[构建您的第一个智能体](./docs/zh/tutorial/first_agent.md)** | **[构建您的第一个智能体](./docs/ColabNotebook/tutorial_notebooks_zh/first_agent.ipynb)** | 快速创建和管理具有多动作能力的智能体。 |
| **[构建您的第一个工作流](./docs/zh/tutorial/first_workflow.md)** | **[构建您的第一个工作流](./docs/ColabNotebook/tutorial_notebooks_zh/first_workflow.ipynb)** | 学习如何使用多个智能体构建协作工作流。 |
| **[使用工具](./docs/zh/tutorial/tools.md)** | **[使用工具](./docs/ColabNotebook/tutorial_notebooks_zh/tools.ipynb)** | 掌握 EvoAgentX 强大的工具生态系统，实现智能体交互。 |
| **[自动工作流生成](./docs/zh/quickstart.md#automatic-workflow-generation-and-execution)** | **[自动工作流生成](./docs/ColabNotebook/tutorial_notebooks_zh/quickstart.ipynb)** | 从自然语言目标自动生成工作流。 |
| **[基准测试与评估教程](./docs/zh/tutorial/benchmark_and_evaluation.md)** | **[基准测试与评估教程](./docs/ColabNotebook/tutorial_notebooks_zh/benchmark_and_evaluation.ipynb)** | 使用基准数据集评估智能体性能。 |
| **[TextGrad优化器教程](./docs/zh/tutorial/textgrad_optimizer.md)** | **[TextGrad优化器教程](./docs/ColabNotebook/tutorial_notebooks_zh/textgrad_optimizer.ipynb)** | 使用TextGrad自动优化多智能体工作流中的提示词。 |
| **[AFlow优化器教程](./docs/zh/tutorial/aflow_optimizer.md)** | **[AFlow优化器教程](./docs/ColabNotebook/tutorial_notebooks_zh/aflow_optimizer.ipynb)** | 使用AFlow自动优化多智能体工作流的提示词和结构。 |
<!-- | **[SEW优化器教程](./docs/zh/tutorial/sew_optimizer.md)** | 创建SEW（自进化工作流）来增强智能体系统。 | -->

🛠️ 通过这些教程构建和优化您的EvoAgentX工作流。

🚀 我们正在积极扩展我们的用例库和优化策略。**更多内容即将推出—敬请期待！**

## 🎯 路线图
- [ ] **模块化进化算法**：将优化算法抽象为即插即用的模块，便于集成到自定义工作流中。 
- [ ] **开发任务模板和智能体模块**：构建典型任务的可重用模板和标准化智能体组件，以简化应用开发。
- [ ] **集成自进化智能体算法**：整合更多最新和先进的智能体自进化技术，跨多个维度，包括提示词调整、工作流结构和记忆模块。 
- [ ] **启用可视化工作流编辑界面**：提供工作流结构显示和编辑的可视化界面，以提高可用性和调试能力。 



## 🙋 支持

### 加入社区

📢 保持联系，成为**EvoAgentX**旅程的一部分！  
🚩 加入我们的社区获取最新更新，分享您的想法，并与全球AI爱好者合作。

- [Discord](https://discord.gg/XWBZUJFwKe) — 实时聊天、讨论和协作。
- [X（前Twitter）](https://x.com/EvoAgentX) — 关注我们获取新闻、更新和洞见。
- [微信](https://github.com/EvoAgentX/EvoAgentX/blob/main/assets/wechat_info.md) — 连接我们的中文社区。

### 将会议添加到您的日历

📅 点击下面的链接将EvoAgentX周会（周日，16:30-17:30 GMT+8）添加到您的日历：

👉 [添加到您的Google日历](https://calendar.google.com/calendar/u/0/r/eventedit?text=EvoAgentX+周会（腾讯会议）&dates=20250629T083000Z/20250629T093000Z&details=会议链接：https://meeting.tencent.com/dm/5UuNxo7Detz0&location=Online&recur=RRULE:FREQ=WEEKLY;BYDAY=SU;UNTIL=20270523T093000Z&ctz=Asia/Shanghai)

👉 [添加到您的腾讯会议](https://meeting.tencent.com/dm/5UuNxo7Detz0)

👉 [下载EvoAgentX_Weekly_Meeting.ics文件](./EvoAgentX_Weekly_Meeting.ics)

### 联系信息

如果您对本项目有任何问题或反馈，请随时联系我们。我们非常感谢您的建议！

- **电子邮件：** evoagentx.ai@gmail.com

我们将在2-3个工作日内回复所有问题。

### 观看往期社区会议
- [Bilibili](https://space.bilibili.com/3493105294641286/favlist?fid=3584589186&ftype=create&spm_id_from=333.788.0.0)
- [Youtube](https://studio.youtube.com/playlist/PL_kuPS05qA1hyU6cLX--bJ93Km2-md8AA/edit)
## 🙌 为EvoAgentX做贡献
感谢这些优秀的贡献者

<a href="https://github.com/EvoAgentX/EvoAgentX/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=EvoAgentX/EvoAgentX" />
</a>

我们感谢您对我们开源计划的贡献兴趣。我们提供了[贡献指南](https://github.com/EvoAgentX/EvoAgentX/blob/main/CONTRIBUTING.md)文档，其中概述了为EvoAgentX做贡献的步骤。请参考此指南以确保顺利协作和成功贡献。 🤝🚀

[![Star历史图表](https://api.star-history.com/svg?repos=EvoAgentX/EvoAgentX&type=Date)](https://www.star-history.com/#EvoAgentX/EvoAgentX&Date)


## 📚 致谢 
本项目基于几个杰出的开源项目：[AFlow](https://github.com/FoundationAgents/MetaGPT/tree/main/metagpt/ext/aflow)、[TextGrad](https://github.com/zou-group/textgrad)、[DSPy](https://github.com/stanfordnlp/dspy)、[LiveCodeBench](https://github.com/LiveCodeBench/LiveCodeBench)等。我们衷心感谢这些框架的开发者和维护者对开源社区的宝贵贡献。

## 📄 许可证

本仓库中的源代码根据[MIT许可证](./LICENSE)提供。
