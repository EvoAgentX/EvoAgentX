# Installation Guide for EvoAgentX

This guide will walk you through the process of installing EvoAgentX on your system, setting up the required dependencies, and configuring the framework for your projects.

## Prerequisites

Before installing EvoAgentX, make sure you have the following prerequisites:

- Python 3.11 or higher
- pip (Python package installer)
- Git (for cloning the repository)
- Conda (recommended for environment management, but optional)

## Installation Methods

There are several ways to install EvoAgentX. Choose the method that best suits your needs.

### Method 1: Using pip (Recommended)

The simplest way to install EvoAgentX is using pip:

```bash
pip install git+https://github.com/EvoAgentX/EvoAgentX.git
```

Optional feature sets:

```bash
pip install "evoagentx[rag]"
pip install "evoagentx[tools]"
pip install "evoagentx[multimodal]"
pip install "evoagentx[optimizers]"
pip install "evoagentx[benchmarks]"
pip install "evoagentx[viz]"
pip install "evoagentx[all]"
```

### Method 2: From Source (For Development)

If you want to contribute to EvoAgentX or need the latest development version, you can install it directly from the source:

```bash
# Clone the repository
git clone https://github.com/EvoAgentX/EvoAgentX.git

# Navigate to the project directory
cd EvoAgentX

# Install the package in development mode
pip install -e .
```

### Method 3: Using Conda Environment (Recommended for Isolation)

If you prefer to use Conda for managing your Python environments, follow these steps:

```bash hl_lines="4-5"
# Create a new conda environment
conda create -n evoagentx python=3.11

# Activate the environment
conda activate evoagentx

# Install the package dependencies
pip install -e ".[dev]"
# Or install with optional features
pip install -e ".[dev,rag,tools]"
# OR install in development mode
pip install -e .
```

## Verifying Installation

To verify that EvoAgentX has been installed correctly, run the following Python code:

```python
import evoagentx

# Print the version
print(evoagentx.__version__)
```

You should see the current version of EvoAgentX printed to the console.
