# 构建你的第一个代理

在 EvoAgentX 中，代理是设计用来自主完成特定任务的智能组件。本教程将引导你了解在 EvoAgentX 中创建和使用代理的基本概念：

1. **使用 CustomizeAgent 创建简单代理**：学习如何使用自定义提示创建基本代理
2. **使用多个动作**：创建可以执行多个任务的更复杂的代理
3. **保存和加载代理**：学习如何保存和加载你的代理

通过本教程，你将能够创建简单和复杂的代理，了解它们如何处理输入和输出，以及如何在项目中保存和重用它们。

## 1. 使用 CustomizeAgent 创建简单代理

创建代理最简单的方法是使用 `CustomizeAgent`，它允许你快速定义一个具有特定提示的代理。

首先，让我们导入必要的组件并设置 LLM：

```python
import os 
from dotenv import load_dotenv
from evoagentx.models import OpenAILLMConfig
from evoagentx.agents import CustomizeAgent

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Configure LLM
openai_config = OpenAILLMConfig(
    model="gpt-4o-mini", 
    openai_key=OPENAI_API_KEY, 
    stream=True
)
``` 

现在，让我们创建一个打印 hello world 的简单代理。有两种方法可以创建 CustomizeAgent：

### 方法 1：直接初始化
你可以直接使用 `CustomizeAgent` 类初始化代理：
```python
first_agent = CustomizeAgent(
    name="FirstAgent",
    description="A simple agent that prints hello world",
    prompt="Print 'hello world'", 
    llm_config=openai_config # specify the LLM configuration 
)
```

### 方法 2：从字典创建

你也可以通过定义字典中的配置来创建代理：

```python
agent_data = {
    "name": "FirstAgent",
    "description": "A simple agent that prints hello world",
    "prompt": "Print 'hello world'",
    "llm_config": openai_config
}
first_agent = CustomizeAgent.from_dict(agent_data) # use .from_dict() to create an agent. 
```

### 使用代理

创建完成后，你可以使用代理来打印 hello world。

```python
# Execute the agent without input. The agent will return a Message object containing the results. 
message = first_agent()

print(f"Response from {first_agent.name}:")
print(message.content.content) # the content of a Message object is a LLMOutputParser object, where the `content` attribute is the raw LLM output. 
```

有关完整示例，请参考 [CustomizeAgent 示例](https://github.com/EvoAgentX/EvoAgentX/blob/main/examples/customize_agent.py)。

CustomizeAgent 还提供其他功能，包括结构化输入/输出和多种解析策略。有关详细信息，请参阅 [CustomizeAgent 文档](../modules/customize_agent.md)。

## 2. 创建具有多个动作的代理

在 EvoAgentX 中，你可以创建一个具有多个预定义动作的代理。这允许你构建可以执行多个任务的更复杂的代理。以下是一个示例，展示如何创建一个具有 `TestCodeGeneration` 和 `TestCodeReview` 动作的代理：

### 定义动作
首先，我们需要定义动作，它们是 `Action` 的子类：
```python
from evoagentx.agents import Agent
from evoagentx.actions import Action, ActionInput, ActionOutput

# Define the CodeGeneration action inputs
class TestCodeGenerationInput(ActionInput):
    requirement: str = Field(description="The requirement for the code generation")

# Define the CodeGeneration action outputs
class TestCodeGenerationOutput(ActionOutput):
    code: str = Field(description="The generated code")

# Define the CodeGeneration action
class TestCodeGeneration(Action): 

    def __init__(
        self, 
        name: str="TestCodeGeneration", 
        description: str="Generate code based on requirements", 
        prompt: str="Generate code based on requirements: {requirement}",
        inputs_format: ActionInput=None, 
        outputs_format: ActionOutput=None, 
        **kwargs
    ):
        inputs_format = inputs_format or TestCodeGenerationInput
        outputs_format = outputs_format or TestCodeGenerationOutput
        super().__init__(
            name=name, 
            description=description, 
            prompt=prompt, 
            inputs_format=inputs_format, 
            outputs_format=outputs_format, 
            **kwargs
        )
    
    def execute(self, llm: Optional[BaseLLM] = None, inputs: Optional[dict] = None, sys_msg: Optional[str]=None, return_prompt: bool = False, **kwargs) -> TestCodeGenerationOutput:
        action_input_attrs = self.inputs_format.get_attrs() # obtain the attributes of the action input 
        action_input_data = {attr: inputs.get(attr, "undefined") for attr in action_input_attrs}
        prompt = self.prompt.format(**action_input_data) # format the prompt with the action input data 
        output = llm.generate(
            prompt=prompt, 
            system_message=sys_msg, 
            parser=self.outputs_format, 
            parse_mode="str" # specify how to parse the output 
        )
        if return_prompt:
            return output, prompt
        return output


# Define the CodeReview action inputs
class TestCodeReviewInput(ActionInput):
    code: str = Field(description="The code to be reviewed")
    requirements: str = Field(description="The requirements for the code review")

# Define the CodeReview action outputs
class TestCodeReviewOutput(ActionOutput):
    review: str = Field(description="The review of the code")

# Define the CodeReview action
class TestCodeReview(Action):
    def __init__(
        self, 
        name: str="TestCodeReview", 
        description: str="Review the code based on requirements", 
        prompt: str="Review the following code based on the requirements:\n\nRequirements: {requirements}\n\nCode:\n{code}.\n\nYou should output a JSON object with the following format:\n```json\n{{\n'review': '...'\n}}\n```", 
        inputs_format: ActionInput=None, 
        outputs_format: ActionOutput=None, 
        **kwargs
    ):
        inputs_format = inputs_format or TestCodeReviewInput
        outputs_format = outputs_format or TestCodeReviewOutput
        super().__init__(
            name=name, 
            description=description, 
            prompt=prompt, 
            inputs_format=inputs_format, 
            outputs_format=outputs_format, 
            **kwargs
        )
    
    def execute(self, llm: Optional[BaseLLM] = None, inputs: Optional[dict] = None, sys_msg: Optional[str]=None, return_prompt: bool = False, **kwargs) -> TestCodeReviewOutput:
        action_input_attrs = self.inputs_format.get_attrs()
        action_input_data = {attr: inputs.get(attr, "undefined") for attr in action_input_attrs}
        prompt = self.prompt.format(**action_input_data)
        output = llm.generate(
            prompt=prompt, 
            system_message=sys_msg,
            parser=self.outputs_format, 
            parse_mode="json" # specify how to parse the output 
        ) 
        if return_prompt:
            return output, prompt
        return output
```

从上面的示例中，我们可以看到，为了定义一个动作，我们需要：

1. 使用 `ActionInput` 和 `ActionOutput` 类定义动作的输入和输出
2. 创建一个继承自 `Action` 的动作类
3. 实现 `execute` 方法，该方法使用动作输入数据格式化提示，并使用 LLM 生成输出，并通过 `parse_mode` 指定如何解析输出。

### 定义代理

一旦我们定义了动作，我们就可以通过将动作添加到代理中来创建代理：

```python
# Initialize the LLM
openai_config = OpenAILLMConfig(model="gpt-4o-mini", openai_key=os.getenv("OPENAI_API_KEY"))

# Define the agent 
developer = Agent(
    name="Developer", 
    description="A developer who can write code and review code",
    actions=[TestCodeGeneration(), TestCodeReview()], 
    llm_config=openai_config
)
```

### 执行不同的动作

一旦你创建了一个具有多个动作的代理，你可以执行特定的动作：

```python
# List all available actions on the agent
actions = developer.get_all_actions()
print(f"Available actions of agent {developer.name}:")
for action in actions:
    print(f"- {action.name}: {action.description}")

# Generate some code using the CodeGeneration action
generation_result = developer.execute(
    action_name="TestCodeGeneration", # specify the action name
    action_input_data={ 
        "requirement": "Write a function that returns the sum of two numbers"
    }
)

# Access the generated code
generated_code = generation_result.content.code
print("Generated code:")
print(generated_code)

# Review the generated code using the CodeReview action
review_result = developer.execute(
    action_name="TestCodeReview",
    action_input_data={
        "requirements": "Write a function that returns the sum of two numbers",
        "code": generated_code
    }
)

# Access the review results
review = review_result.content.review
print("\nReview:")
print(review)
```

这个示例演示了如何：
1. 列出代理上可用的所有动作
2. 使用 TestCodeGeneration 动作生成代码
3. 使用 TestCodeReview 动作审查生成的代码
4. 访问每个动作执行的结果

有关完整的工作示例，请参考 [Agent 示例](https://github.com/EvoAgentX/EvoAgentX/blob/main/examples/agent_with_multiple_actions.py)。 


## 3. 保存和加载代理

你可以将代理保存到文件并在稍后加载它：

```python
# 保存代理
developer.save_module("examples/output/developer.json")

# 加载代理
loaded_developer = Agent.load_module("examples/output/developer.json", llm_config=openai_config)
```