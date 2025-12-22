# EvoAgentX 安装指南

本指南将引导您完成在系统上安装 EvoAgentX、设置所需依赖项以及为您的项目配置框架的全过程。

## 前置条件

在安装 EvoAgentX 之前，请确保您具备以下前置条件：

- Python 3.10 或更高版本  
- pip（Python 包安装器）  
- Git（用于克隆仓库）  
- Conda（推荐用于环境管理，但可选）  

## 安装方法

安装 EvoAgentX 有多种方式。请选择最符合您需求的方法。

### 方法 1：使用 pip（推荐）

安装 EvoAgentX 最简单的方法是使用 pip：

```bash
pip install git+https://github.com/EvoAgentX/EvoAgentX.git
```

### 方法 2：从源代码安装（开发者专用）

如果您希望为 EvoAgentX 做贡献，或需要获取最新的开发版本，可以直接从源代码安装：

```bash
# 克隆仓库
git clone https://github.com/EvoAgentX/EvoAgentX/

# 进入项目目录
cd EvoAgentX

# 以开发模式安装包
pip install -e .
```

### 方法 3：使用 Conda 环境（推荐用于隔离）

如果您偏好使用 Conda 来管理 Python 环境，请按以下步骤操作：

```bash hl_lines="4-5"
# 创建新的conda环境
conda create -n evoagentx python=3.10

# 激活环境
conda activate evoagentx

# 安装包
pip install -r requirements.txt
# 或者以开发模式安装
pip install -e .
```

## 验证安装

要验证 EvoAgentX 是否已正确安装，请运行以下 Python 代码：

```python
import evoagentx

# 打印版本
print(evoagentx.__version__)
```

您应该能在控制台看到 EvoAgentX 当前的版本号。
