# 动作图

## 简介

`ActionGraph` 类是 EvoAgentX 框架中的一个基础组件，用于在单个任务中创建和执行操作（动作）序列。它提供了一种结构化的方式来定义、管理和执行一系列需要按特定顺序执行的操作，以完成任务。

动作图表示一组按预定义顺序执行的运算符（动作），用于处理输入并产生输出。与在更高层次管理多个任务及其依赖关系的 `WorkFlowGraph` 不同，`ActionGraph` 专注于单个任务内的详细执行步骤。

## 架构

### 动作图架构

`ActionGraph` 由几个关键组件组成：

1. **运算符**：
   
    每个运算符代表可以作为任务一部分执行的特定操作或动作，具有以下属性：

    - `name`：运算符的唯一标识符
    - `description`：运算符功能的详细描述
    - `llm`：用于执行运算符的 LLM
    - `outputs_format`：运算符输出的结构化格式
    - `interface`：调用运算符的接口
    - `prompt`：执行此运算符时用于指导 LLM 的模板

2. **LLM**：
   
    ActionGraph 使用语言学习模型（LLM）来执行运算符。它接收 `llm_config` 作为输入并创建 LLM 实例，该实例将被传递给运算符执行。LLM 提供了执行每个动作所需的推理和生成能力。

3. **执行流程**：
   
    ActionGraph 定义了特定的执行顺序：

    - 动作按预定顺序执行（在 `execute` 或 `async_execute` 方法中使用代码指定）
    - 每个动作可以使用之前动作的结果
    - 在所有动作执行完成后产生最终输出

### 与工作流图的比较

虽然 `ActionGraph` 和 `WorkFlowGraph` 都管理执行流程，但它们在抽象层次上有所不同：

| 特性 | 动作图 | 工作流图 |
|---------|-------------|---------------|
| 范围 | 单个任务执行 | 多任务工作流编排 |
| 组件 | 运算符（动作） | 节点（任务）和边（依赖关系） |
| 重点 | 任务内的详细步骤 | 不同任务之间的关系 |
| 灵活性 | 固定执行顺序 | 基于依赖关系的动态执行 |
| 主要用途 | 定义可重用的任务执行模式 | 编排复杂的多步骤工作流 |
| 粒度 | 细粒度操作 | 粗粒度任务 |

## 使用方法

### 基本动作图创建

```python
from evoagentx.workflow import ActionGraph
from evoagentx.workflow.operators import Custom
from evoagentx.models import OpenAILLMConfig 

# 创建 LLM 配置
llm_config = OpenAILLMConfig(model="gpt-4o-mini", openai_key="xxx")

# 创建自定义 ActionGraph
class MyActionGraph(ActionGraph):
    def __init__(self, llm_config, **kwargs):

        name = kwargs.pop("name") if "name" in kwargs else "Custom Action Graph"
        description = kwargs.pop("description") if "description" in kwargs else "A custom action graph for text processing"
        # 基于 `llm_config` 创建 LLM 实例 `self._llm` 并将其传递给运算符
        super().__init__(name=name, description=description, llm_config=llm_config, **kwargs)
        # 定义运算符
        self.extract_entities = Custom(self._llm) # , prompt="Extract key entities from the following text: {input}")
        self.analyze_sentiment = Custom(self._llm) # , prompt="Analyze the sentiment of the following text: {input}")
        self.summarize = Custom(self._llm) # , prompt="Summarize the following text in one paragraph: {input}")

    def execute(self, text: str) -> dict:
        # 按顺序执行运算符（指定运算符的执行顺序）
        entities = self.extract_entities(input=text, instruction="Extract key entities from the provided text")["response"]
        sentiment = self.analyze_sentiment(input=text, instruction="Analyze the sentiment of the provided text")["response"]
        summary = self.summarize(input=text, instruction="Summarize the provided text in one paragraph")["response"]

        # 返回组合结果
        return {
            "entities": entities,
            "sentiment": sentiment,
            "summary": summary
        }

# 创建动作图
action_graph = MyActionGraph(llm_config=llm_config)

# 执行动作图
result = action_graph.execute(text="This is a test text")
print(result)
```

### 在工作流图中使用动作图

您可以直接使用 `ActionGraph` 或在 `WorkFlowGraph` 中将其作为节点使用。

```python
from evoagentx.workflow.workflow_graph import WorkFlowNode, WorkFlowGraph
from evoagentx.workflow.action_graph import QAActionGraph
from evoagentx.core.base_config import Parameter
from evoagentx.models import OpenAILLMConfig, OpenAILLM
from evoagentx.workflow import WorkFlow

# 创建 LLM 配置
llm_config = OpenAILLMConfig(model="gpt-4o-mini", openai_key="xxx", stream=True, output_response=True)
llm = OpenAILLM(llm_config)

# 创建动作图
qa_graph = QAActionGraph(llm_config=llm_config)

# 创建使用动作图的工作流节点
qa_node = WorkFlowNode(
    name="QATask",
    description="Answer questions using a QA system",
    # 输入名称应与动作图的 `execute` 方法中的参数匹配
    inputs=[Parameter(name="problem", type="string", description="The problem to answer")],
    outputs=[Parameter(name="answer", type="string", description="The answer to the problem")],
    action_graph=qa_graph  # 使用 action_graph 而不是 agents
)

# 创建工作流图
workflow_graph = WorkFlowGraph(goal="Answer a question", nodes=[qa_node])

# 定义工作流
workflow = WorkFlow(graph=workflow_graph, llm=llm)

# 执行工作流
result = workflow.execute(inputs={"problem": "What is the capital of France?"})
print(result)
```

!!! warning 
    在 `WorkFlowNode` 中使用 `ActionGraph` 时，`WorkFlowNode` 的 `inputs` 参数应与 `ActionGraph` 的 `execute` 方法中所需的参数匹配。`execute` 方法应返回一个**字典**或 `LLMOutputParser` 实例，其键与 `WorkFlowNode` 中 `outputs` 的名称匹配。

### 保存和加载动作图

```python
# 保存动作图
action_graph.save_module("examples/output/my_action_graph.json")

# 加载动作图
from evoagentx.workflow.action_graph import ActionGraph
loaded_graph = ActionGraph.from_file("examples/output/my_action_graph.json", llm_config=llm_config)
```

`ActionGraph` 类提供了一种强大的方式来定义单个任务内的复杂操作序列，补充了 EvoAgentX 框架中 `WorkFlowGraph` 的高级编排功能。
