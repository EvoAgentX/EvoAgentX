# Workflow Evaluation Pipeline 使用指南

## 快速开始

### 1. 基础使用 (推荐)

#### Windows系统：
```cmd
# 运行所有三层评估 (默认配置)
python evaluation_pipeline\run_evaluation_cli.py

# 或者使用批处理文件
evaluation_pipeline\run_evaluation.bat

# 查看帮助信息
python evaluation_pipeline\run_evaluation_cli.py --help
```

#### Linux/macOS系统：
```bash
# 运行所有三层评估 (默认配置)
./evaluation_pipeline/run_evaluation.sh

# 或者使用Python脚本 (跨平台)
python3 evaluation_pipeline/run_evaluation_cli.py

# 查看帮助信息
./evaluation_pipeline/run_evaluation.sh --help
```

### 2. 自定义配置运行

#### Windows系统：
```cmd
# 只运行第一层和第二层
python evaluation_pipeline\run_evaluation_cli.py -l 1,2

# 使用8个进程，批次大小为20
python evaluation_pipeline\run_evaluation_cli.py -p 8 -b 20

# 指定自定义数据目录
python evaluation_pipeline\run_evaluation_cli.py -d path\to\your\test\data

# 清理之前的结果后重新运行
python evaluation_pipeline\run_evaluation_cli.py --clean

# 查看当前状态
python evaluation_pipeline\run_evaluation_cli.py --status

# 创建示例配置文件
python evaluation_pipeline\run_evaluation_cli.py --create-config
```

#### Linux/macOS系统：
```bash
# 只运行第一层和第二层
./evaluation_pipeline/run_evaluation.sh -l 1,2
# 或者: python3 evaluation_pipeline/run_evaluation_cli.py -l 1,2

# 使用8个进程，批次大小为20
./evaluation_pipeline/run_evaluation.sh -p 8 -b 20

# 指定自定义数据目录
./evaluation_pipeline/run_evaluation.sh -d /path/to/your/test/data

# 清理之前的结果后重新运行
./evaluation_pipeline/run_evaluation.sh --clean
```

### 3. 使用Python直接运行

```bash
# 运行完整评估
python3 evaluation_pipeline/run_evaluation.py

# 运行特定层次
python3 evaluation_pipeline/run_evaluation.py --layers 1,3

# 自定义配置
python3 evaluation_pipeline/run_evaluation.py --max-processes 6 --batch-size 15
```

## 配置选项

### 命令行参数

| 参数 | 描述 | 默认值 |
|------|------|--------|
| `-l, --layers` | 要运行的层次 (1,2,3) | "1,2,3" |
| `-p, --processes` | 最大进程数 | 4 |
| `-b, --batch-size` | 批处理大小 | 10 |
| `-d, --data-dir` | 测试数据目录 | "evaluation_pipeline/workflow_generation_eval_data" |
| `-c, --config` | 配置文件路径 | None |
| `--clean` | 清理之前的结果 | False |
| `--dry-run` | 显示将要执行的命令但不实际运行 | False |

### 环境变量配置

可以通过环境变量配置默认值：

```bash
export EVAL_MAX_PROCESSES=8
export EVAL_MAX_THREADS=4
export EVAL_BATCH_SIZE=20
export EVAL_MAX_RETRIES=5
export EVAL_LAYERS=1,2,3

# 然后运行评估
./evaluation_pipeline/run_evaluation.sh
```

## 数据准备

### 测试数据格式

在 `evaluation_pipeline/workflow_generation_eval_data/` 目录中放置JSON格式的测试数据文件：

```json
{
    "workflow_name": "content_analysis",
    "workflow_id": "a1b2c3d4-e5f6-4789-9012-34567890abcd",
    "workflow_requirement": "Analyze RSS news content to extract key topics, sentiment, and auto-categorize articles",
    "workflow_inputs": [
        {
            "name": "article_content",
            "type": "string",
            "description": "Full article text content",
            "required": true
        }
    ],
    "workflow_outputs": [
        {
            "name": "categories",
            "type": "array",
            "description": "Extracted categories with confidence scores"
        }
    ]
}
```

## 运行示例

### 场景1：开发阶段快速测试

```bash
# 只运行结构评估，快速验证workflow生成
./evaluation_pipeline/run_evaluation.sh -l 1 -p 2 -b 5
```

### 场景2：完整性能评估

```bash
# 运行所有层次，使用更多进程加速
./evaluation_pipeline/run_evaluation.sh -p 8 -b 20
```

### 场景3：特定层次调试

```bash
# 清理之前结果，只运行输出质量评估
./evaluation_pipeline/run_evaluation.sh --clean -l 3
```

### 场景4：大规模测试

```bash
# 使用最大资源进行大规模测试
./evaluation_pipeline/run_evaluation.sh -p 12 -b 50 -d /path/to/large/dataset
```

## 结果查看

### 输出文件位置

- **结果目录**: `evaluation_pipeline/results/`
- **检查点目录**: `evaluation_pipeline/checkpoints/`

### 结果文件

| 文件名 | 描述 |
|--------|------|
| `layer_1_structure_evaluation.json` | 第一层结构评估结果 |
| `layer_2_execution_evaluation.json` | 第二层执行评估结果 |
| `layer_3_output_evaluation.json` | 第三层输出质量评估结果 |
| `comprehensive_evaluation_report_*.json` | 综合评估报告 |

### 实时监控

运行过程中，脚本会实时显示：
- 当前执行的层次和进度
- 每个批次的完成状态
- 成功率和错误统计
- 执行时间信息

## 断点续传

系统自动支持断点续传：

1. **自动保存检查点**: 每完成几个批次会自动保存进度
2. **自动恢复**: 重新运行时会自动从上次中断的地方继续
3. **手动清理**: 使用 `--clean` 参数可以清理所有之前的进度

## 错误处理

### 常见问题

1. **内存不足**：减少 `--processes` 或 `--batch-size` 参数
2. **网络超时**：系统会自动重试LLM API调用
3. **数据格式错误**：检查测试数据JSON格式是否正确

### 日志查看

所有错误信息会显示在控制台输出中，包括：
- 详细的错误堆栈信息
- 失败的测试案例ID
- 重试次数和结果

## 高级使用

### 自定义配置文件

创建配置文件 `custom_config.json`：

```json
{
    "max_processes": 8,
    "batch_size": 25,
    "max_retries": 5,
    "retry_delay": 3.0,
    "layers_to_run": [1, 2, 3],
    "evaluation_llm_model": "gpt-4",
    "workflow_execution_timeout": 600.0
}
```

然后使用：

```bash
./evaluation_pipeline/run_evaluation.sh -c custom_config.json
```

### 编程接口使用

```python
from evaluation_pipeline import EvaluationConfig, WorkflowEvaluationPipeline

# 创建配置
config = EvaluationConfig(
    max_processes=8,
    batch_size=20,
    layers_to_run=[1, 2, 3]
)

# 运行评估
pipeline = WorkflowEvaluationPipeline(config)
results = pipeline.run_complete_evaluation()

# 查看结果
print(f"总体成功率: {results['overall_summary']['average_quality_score']}")
```

## 性能优化建议

1. **进程数设置**: 通常设置为CPU核心数的75-100%
2. **批次大小**: 根据内存大小调整，通常10-50之间
3. **数据预处理**: 确保测试数据文件不要过大，建议单个文件<1MB
4. **网络稳定**: 确保网络连接稳定，特别是LLM API调用时

## 注意事项

- 首次运行会自动创建必要的目录结构
- 大量测试数据可能需要较长时间，建议先用小批量测试
- 确保有足够的磁盘空间存储结果和检查点文件
- LLM API调用需要相应的API密钥配置
