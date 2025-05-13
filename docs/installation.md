# Installation Guide for EvoAgentX

This guide will walk you through the process of installing EvoAgentX on your system, setting up the required dependencies, and configuring the framework for your projects.

## Prerequisites

Before installing EvoAgentX, make sure you have the following prerequisites:

- Python 3.10 or higher
- pip (Python package installer)
- Git (for cloning the repository)
- Conda (recommended for environment management, but optional)

## Installation Methods

There are several ways to install EvoAgentX. Choose the method that best suits your needs.

### Method 1: Using pip (Recommended)

The simplest way to install EvoAgentX is using pip:

```bash
pip install evoagentx
```

### Method 2: From Source (For Development)

If you want to contribute to EvoAgentX or need the latest development version, you can install it directly from the source:

```bash
# Clone the repository
git clone https://github.com/EvoAgentX/EvoAgentX/

# Navigate to the project directory
cd EvoAgentX

# Install the package in development mode
pip install -e .
```

### Method 3: Using Conda Environment (Recommended for Isolation)

If you prefer to use Conda for managing your Python environments, follow these steps:

```bash hl_lines="4-5"
# Create a new conda environment
conda create -n evoagentx python=3.10

# Activate the environment
conda activate evoagentx

# Install the package
pip install -r requirements.txt
# OR install in development mode
pip install -e .
```

<!-- ## Configuration Setup

### API Keys Configuration

EvoAgentX requires API keys for certain functionalities, especially when using language models like OpenAI's GPT models. Here's how to set them up:

1. Create a `.env` file in your project root directory based on the `.env.example` template:

```bash
cp .env.example .env
```

2. Open the `.env` file and add your API keys:

```
# OpenAI API Key
OPENAI_API_KEY=your_openai_api_key_here

# Other API Keys as needed
# ANTHROPIC_API_KEY=your_anthropic_api_key_here
# GOOGLE_API_KEY=your_google_api_key_here
```

Alternatively, you can set these environment variables directly in your shell:

```bash
export OPENAI_API_KEY=your_openai_api_key_here
```

### Configuring Model Settings

You can configure model settings programmatically:

```python
from evoagentx.models import OpenAILLMConfig, OpenAILLM

# Configure your language model
openai_config = OpenAILLMConfig(
    model="gpt-4o",  # or any other supported model
    openai_key="your_api_key",  # optional if set in .env
    stream=True,
    temperature=0.7
)
model = OpenAILLM(config=openai_config)
``` -->

## Verifying Installation

To verify that EvoAgentX has been installed correctly, run the following Python code:

```python
import evoagentx

# Print the version
print(evoagentx.__version__)
```

You should see the current version of EvoAgentX printed to the console.

<!-- ## Running a Simple Test

Here's a simple test to ensure everything is working correctly:

```python
from evoagentx.models import OpenAILLMConfig, OpenAILLM
from evoagentx.agents import AgentManager

# Configure your language model
openai_config = OpenAILLMConfig(
    model="gpt-4o-mini",
    stream=True
)
model = OpenAILLM(config=openai_config)

# Create an agent manager
agent_manager = AgentManager()

# If you see no errors, your installation is working correctly
print("EvoAgentX installation successful!")
```

## Troubleshooting

### Common Issues

#### Missing Dependencies

If you encounter errors about missing dependencies, try reinstalling with all optional dependencies:

```bash
pip install evoagentx[all]
```

#### API Key Issues

If you're experiencing authentication errors:

1. Double-check that your API keys are correctly set in the `.env` file or as environment variables
2. Ensure that your API keys have the necessary permissions and are valid
3. Check if your API key has sufficient credits or quota remaining

#### Version Conflicts

If you encounter version conflicts with other packages, consider using a virtual environment:

```bash
# Create a new virtual environment
python -m venv evoagentx_env

# Activate the environment
# On Windows
evoagentx_env\Scripts\activate
# On macOS/Linux
source evoagentx_env/bin/activate

# Install EvoAgentX
pip install evoagentx
```

### Getting Help

If you continue to experience issues:

1. Check the [GitHub Issues](https://github.com/EvoAgentX/EvoAgentX/issues) page for similar problems
2. Join our [Discord community](https://discord.gg/q5hBjHVz) for real-time help
3. Email us at evoagentx.ai@gmail.com -->

<!-- ## Next Steps

Now that you have EvoAgentX installed, check out the following resources:

- [Basic Usage Guide](../tutorials/basic_usage.md) - Learn how to create your first workflow
- [API Reference](../api.md) - Explore the full API documentation
- [Examples](../examples/index.md) - See practical examples of EvoAgentX in action -->
