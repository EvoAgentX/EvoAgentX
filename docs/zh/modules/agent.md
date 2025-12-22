# Agent

## 简介

`Agent` 类是 EvoAgentX 框架中创建智能 AI 代理的基础构建块。它提供了一种结构化的方式来组合语言模型、动作和内存管理。

## 架构

Agent 由几个关键组件组成：

1. **大语言模型 (LLM)**：

    LLM 通过 `llm` 或 `llm_config` 参数指定，作为代理的基础构建块。它负责解释上下文、生成响应和做出高级决策。LLM 将被传递给动作以执行特定任务。

2. **动作 (Actions)**：

    动作是代理的基本操作单元。每个动作封装了一个特定任务，是实际调用 LLM 进行推理、生成或决策的地方。虽然代理提供整体编排，但 LLM 通过动作执行其核心功能。每个动作都被设计为只做一件事——比如检索知识、总结输入或调用 API——并且可以包含以下组件：

    - **prompt**：用于指导 LLM 执行此特定任务的提示模板。
    - **inputs_format**：传入动作的输入的预期结构和键。
    - **outputs_format**：用于解释和解析 LLM 输出的格式。
    - **tools**：可以在动作中集成和使用的可选工具。

3. **内存组件**：

    内存允许代理在交互过程中保留和回忆相关信息，增强上下文感知能力。EvoAgentX 框架中有两种类型的内存：

    - **短期内存**：维护当前任务的中间对话或上下文。
    - **长期内存**（可选）：存储可以跨越会话或任务的持久知识。这使代理能够从过去的经验中学习、维护用户偏好或随时间构建知识库。

## 使用方法

### 基本 Agent 创建

要创建代理，您需要定义代理将执行的动作。每个动作都被定义为一个继承自 `Action` 类的类。动作类应该定义以下组件：`name`、`description`、`prompt`、`inputs_format` 和 `outputs_format`，并实现 `execute` 方法（如果您想异步使用代理，还需要实现 `async_exectue`）。

```python
from evoagentx.agents import Agent
from evoagentx.models import OpenAILLMConfig
from evoagentx.actions import Action, ActionInput, ActionOutput

# 定义一个使用 LLM 回答问题的简单动作

class AnswerQuestionInput(ActionInput):
    question: str

class AnswerQuestionOutput(ActionOutput):
    answer: str

class AnswerQuestionAction(Action):

    def __init__(
        self, 
        name = "answer_question",
        description = "Answers a factual question using the LLM",   
        prompt = "Answer the following question as accurately as possible:\n\n{question}",
        inputs_format = AnswerQuestionInput,
        outputs_format = AnswerQuestionOutput,
        **kwargs
    ):
        super().__init__(
            name=name, 
            description=description, 
            prompt=prompt, 
            inputs_format=inputs_format, 
            outputs_format=outputs_format, 
            **kwargs
        )
    
    def execute(self, llm, inputs, sys_msg = None, return_prompt = False, **kwargs) -> AnswerQuestionOutput:
        question = inputs.get("question")
        prompt = self.prompt.format(question=question)
        response = llm.generate(
            prompt=prompt, 
            system_message=sys_msg,
            parser=self.outputs_format, 
            parse_mode="str"
        )

        if return_prompt:
            return response, prompt
        return response 

    async def async_execute(self, llm, inputs, sys_msg = None, return_prompt = False, **kwargs) -> AnswerQuestionOutput:
        question = inputs.get("question")
        prompt = self.prompt.format(question=question)
        response = await llm.async_generate(
            prompt=prompt, 
            system_message=sys_msg,
            parser=self.outputs_format, 
            parse_mode="str"
        )   
        if return_prompt:
            return response, prompt
        return response 

# 配置 LLM
llm_config = OpenAILLMConfig(model="gpt-4o-mini", openai_key="your-api-key")

# 创建代理
agent = Agent(
    name="AssistantAgent",
    description="Answers a factual question using the LLM",
    llm_config=llm_config,
    system_prompt="You are a helpful assistant.",
    actions=[AnswerQuestionAction()]
)
```

### 执行动作

您可以直接像函数一样调用 `Agent` 实例。这将内部使用指定的 `action_name` 和 `action_input_data` 调用匹配动作的 `execute()` 方法。

```python
# 使用输入数据执行动作
message = agent(
    action_name="answer_question",
    action_input_data={"question": "What is the capital of France?"}
)

# 访问输出
result = message.content.answer 
```

### 异步执行

您也可以在异步上下文中调用 `Agent` 实例。如果动作定义了 `async_execute` 方法，当您 `await` 代理时，它将自动使用。

```python
# 异步执行动作
import asyncio 

async def main():
    message = await agent(
        action_name="answer_question",
        action_input_data={"question": "What is the capital of France?"}
    )
    return message.content.answer 

result = asyncio.run(main())
print(result)
```

## 内存管理

代理维护短期内存以跟踪对话上下文：

```python
# 访问代理的内存
messages = agent.short_term_memory.get(n=5)  # 获取最后 5 条消息

# 清除内存
agent.clear_short_term_memory()
```

## 代理配置文件

您可以获取代理及其功能的人类可读描述：

```python
# 获取所有动作的描述
profile = agent.get_agent_profile()
print(profile)

# 获取特定动作的描述
profile = agent.get_agent_profile(action_names=["answer_question"])
print(profile)
```

## 提示管理

访问和修改代理使用的提示：

```python
# 获取所有提示
prompts = agent.get_prompts()
# prompts 是一个字典，结构如下：
# {'answer_question': {'system_prompt': 'You are a helpful assistant.', 'prompt': 'Answer the following question as accurately as possible:\n\n{question}'}}

# 设置特定提示
agent.set_prompt(
    action_name="answer_question",
    prompt="Please provide a clear and concise answer to the following query:\n\n{question}",
    system_prompt="You are a helpful assistant." # 可选，如果未提供，系统提示将保持不变
)

# 更新所有提示
prompts_dict = {
    "answer_question": {
        "system_prompt": "You are an expert in providing concise, accurate information.",
        "prompt": "Please answer this question with precision and clarity:\n\n{question}"
    }
}
agent.set_prompts(prompts_dict)
```

## 保存和加载代理

代理可以被持久化和重新加载：

```python
# 保存代理
agent.save_module("./agents/my_agent.json")

# 加载代理（需要再次提供 llm_config）
loaded_agent = Agent.from_file(
    "./agents/my_agent.json", 
    llm_config=llm_config
)
```

## 上下文提取

代理包含内置的上下文提取机制，可以自动从对话历史中派生动动作的适当输入：

```python
# 在没有显式输入数据的情况下执行时，上下文会自动提取
response = agent.execute(
    action_name="action_name",
    msgs=conversation_history
)

# 手动获取动作输入
action = agent.get_action("action_name")
inputs = agent.get_action_inputs(action)
```
