<!-- Add logo here -->
<div align="center">
  <a href="https://github.com/EvoAgentX/EvoAgentX">
    <img src="./assets/EAXLoGo.svg" alt="EvoAgentX" width="50%">
  </a>
</div>

<h2 align="center">
    Building a Self-Evolving Ecosystem of AI Agents
</h2>

<div align="center">

[![Docs](https://img.shields.io/badge/-Documentation-0A66C2?logo=readthedocs&logoColor=white&color=7289DA&labelColor=grey)](https://EvoAgentX.github.io/EvoAgentX/)
[![Discord](https://img.shields.io/badge/Chat-Discord-5865F2?&logo=discord&logoColor=white)](https://discord.gg/EvoAgentX)
[![Twitter](https://img.shields.io/badge/Follow-@EvoAgentX-e3dee5?&logo=x&logoColor=white)](https://x.com/EvoAgentX)
[![Wechat](https://img.shields.io/badge/WeChat-EvoAgentX-brightgreen?logo=wechat&logoColor=white)]()
[![GitHub star chart](https://img.shields.io/github/stars/EvoAgentX/EvoAgentX?style=social)](https://star-history.com/#EvoAgentX/EvoAgentX)
[![GitHub fork](https://img.shields.io/github/forks/EvoAgentX/EvoAgentX?style=social)](https://github.com/EvoAgentX/EvoAgentX/fork)
[![License](https://img.shields.io/badge/License-MIT-blue.svg?)](https://github.com/EvoAgentX/EvoAgentX/blob/main/LICENSE)
<!-- [![EvoAgentX Homepage](https://img.shields.io/badge/EvoAgentX-Homepage-blue?logo=homebridge)](https://EvoAgentX.github.io/EvoAgentX/) -->
<!-- [![hf_space](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-EvoAgentX-ffc107?color=ffc107&logoColor=white)](https://huggingface.co/EvoAgentX) -->
</div>

<div align="center">

<h3 align="center">

<a href="./README.md" style="text-decoration: underline;">English</a> | <a href="./README-zh.md">简体中文</a>

</h3>

</div>

<h4 align="center">
  <i>An automated framework for evaluating and evolving agentic workflows.</i>
</h4>

<p align="center">
  <img src="./assets/framework_en.png">
</p>


## 🔥 Latest News
- **[May 2025]** 🎉 **EvoAgentX** has been officially released!

## ⚡Get Started
- [Installation](#installation)
- [Configuration](#configuration)
- [Examples: Automatic WorkFlow Generation](#examples-automatic-workflow-generation)
- [Demo Video](#demo-video)
- [Tutorial and Use Cases](#tutorial-and-use-cases)

## Installation

We recommend installing EvoAgentX using `pip`:

```bash
pip install evoagentx
```

For local development or detailed setup (e.g., using conda), refer to the [Installation Guide for EvoAgentX](./docs/installation.md).

<details>
<summary>Example (optional, for local development):</summary>

```bash
git clone https://github.com/EvoAgentX/EvoAgentX.git
cd EvoAgentX
# Create a new conda environment
conda create -n evoagentx python=3.10

# Activate the environment
conda activate evoagentx

# Install the package
pip install -r requirements.txt
# OR install in development mode
pip install -e .
```
</details>

## Configuration

To use LLMs with EvoAgentX (e.g., OpenAI), you must set up your API key.

#### Option 1: Set API Key via Environment Variable

- Linux/macOS: 
```bash
export OPENAI_API_KEY=<your-openai-api-key>
```

- Windows Command Prompt: 
```cmd 
set OPENAI_API_KEY=<your-openai-api-key>
```

-  Windows PowerShell:
```powershell
$env:OPENAI_API_KEY="<your-openai-api-key>" # " is required 
```

Once set, you can access the key in your Python code with:
```python
import os
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
```

#### Option 2: Use .env File

- Create a .env file in your project root:
```bash
OPENAI_API_KEY=<your-openai-api-key>
```

Then load it in Python:
```python
from dotenv import load_dotenv 
import os 

load_dotenv() # Loads environment variables from .env file
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
```

> 🔐 Tip: Don’t forget to add `.env` to your `.gitignore` to avoid committing secrets.

### Configure and Use the LLM
Once the API key is set, initialise the LLM with:

```python
from evoagentx.models import OpenAILLMConfig, OpenAILLM

# Load the API key from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Define LLM configuration
openai_config = OpenAILLMConfig(
    model="gpt-4o-mini",       # Specify the model name
    openai_key=OPENAI_API_KEY, # Pass the key directly
    stream=True,               # Enable streaming response
    output_response=True       # Print response to stdout
)

# Initialize the language model
llm = OpenAILLM(config=openai_config)

# Generate a response from the LLM
response = llm.generate(prompt="What is Agentic Workflow?")
```
> 📖 More details on supported models and config options: [LLM module guide](./docs/modules/llm.md).


## Examples: Automatic WorkFlow Generation 
Once your API key and language model are configured, you can automatically generate and execute multi-agent workflows in EvoAgentX.

🧩 Core Steps:
1. Define a natural language goal
2. Generate the workflow with WorkFlowGenerator
3. Instantiate agents using AgentManager
4. Execute the workflow via WorkFlow

💡 Minimal Example:
```python
from evoagentx.workflow import WorkFlowGenerator, WorkFlowGraph, WorkFlow
from evoagentx.agents import AgentManager

goal = "Generate html code for the Tetris game"
workflow_graph = WorkFlowGenerator(llm=llm).generate_workflow(goal)

agent_manager = AgentManager()
agent_manager.add_agents_from_workflow(workflow_graph, llm_config=openai_config)

workflow = WorkFlow(graph=workflow_graph, agent_manager=agent_manager, llm=llm)
output = workflow.execute()
print(output)
```

You can also:
- 📊 Visualise the workflow: `workflow_graph.display()`
- 💾 Save/load workflows: `save_module()` / `from_file()`

> 📂 For a complete working example, check out the [`workflow_demo.py`](./examples/workflow_demo.py)


## Demo Video
🎥 Demo video coming soon – stay tuned!

> In the meantime, check out the [EvoAgentX Quickstart Guide](./docs/quickstart.md) for a step-by-step guide to get started with EvoAgentX.

## Tutorial and Use Cases

Explore how to effectively use EvoAgentX with the following resources:

| Cookbook | Description |
|:---|:---|
| **[Build Your First Agent](./docs/tutorial/first_agent.md)** | A comprehensive guide to creating your first agent step-by-step. |
| **[Building Workflows Manually](./docs/tutorial/first_workflow.md)** | Learn how to design and implement collaborative agent workflows. |
| **[Benchmark and Evaluation Tutorial](./docs/tutorial/benchmark_and_evaluation.md)** | Guidelines for evaluating and benchmarking agent performance. |
| **[SEW Optimizer Tutorial](./docs/tutorial/sew_optimizer.md)** | Learn optimization techniques for enhancing agent workflows. |

🛠️ Follow the tutorials to build and optimize your EvoAgentX workflows.

💡 Discover real-world applications and unleash the potential of EvoAgentX in your projects!

## 🙋 Support

### Join the Community

📢 Stay connected and be part of the **EvoAgentX** journey!  
🚩 Join our community to get the latest updates, share your ideas, and collaborate with AI enthusiasts worldwide.

- [Discord](https://discord.com/invite/EvoAgentX) — Chat, discuss, and collaborate in real-time.
- [X (formerly Twitter)](https://x.com/EvoAgentX) — Follow us for news, updates, and insights.
- [WeChat]() — Connect with our Chinese community.

### Contact Information

If you have any questions or feedback about this project, please feel free to contact us. We highly appreciate your suggestions!

- **Email:** evoagentx.ai@gmail.com

We will respond to all questions within 2-3 business days.

## 🙌 Contributing to EvoAgentX
Thanks go to these awesome contributors

<a href="https://github.com/EvoAgentX/EvoAgentX/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=EvoAgentX/EvoAgentX" />
</a>

We appreciate your interest in contributing to our open-source initiative. We provide a document of [contributing guidelines](https://github.com/clayxai/EvoAgentX/blob/main/CONTRIBUTING.md) which outlines the steps for contributing to EvoAgentX. Please refer to this guide to ensure smooth collaboration and successful contributions. 🤝🚀

[![Star History Chart](https://api.star-history.com/svg?repos=EvoAgentX/EvoAgentX&type=Date)](https://www.star-history.com/#EvoAgentX/EvoAgentX&Date)


## 📄 License

Source code in this repository is made available under the [MIT License](./LICENSE).
