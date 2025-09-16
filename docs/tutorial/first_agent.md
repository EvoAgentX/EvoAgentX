# Build Your First Agent

In EvoAgentX, agents are intelligent components designed to complete specific tasks autonomously. This tutorial will walk you through the essential concepts of creating and using agents in EvoAgentX:

1. **Creating a Simple Agent with CustomizeAgent**: Learn how to create a basic agent with custom prompts 
2. **Working with Multiple Actions**: Create more complex agents that can perform multiple tasks
3. **Saving and Loading Agents**: Learn how to save and load your agents

By the end of this tutorial, you'll be able to create both simple and complex agents, understand how they process inputs and outputs, and know how to save and reuse them in your projects.

## 1. Creating a Simple Agent with CustomizeAgent

The easiest way to create an agent is using `CustomizeAgent`, which allows you to quickly define an agent with a specific prompt.  

First, let's import the necessary components and setup the LLM:

```python
import os 
from dotenv import load_dotenv
from evoagentx.models import OpenAILLMConfig
from evoagentx.agents import CustomizeAgent
from evoagentx.prompts.template import StringTemplate

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Configure LLM
openai_config = OpenAILLMConfig(
    model="gpt-4o-mini", 
    openai_key=OPENAI_API_KEY, 
    stream=True
)
``` 

Now, let's create a simple agent that prints hello world. There are two ways to create a CustomizeAgent:

### Method 1: Direct Initialization
You can directly initialize the agent with the `CustomizeAgent` class: 
```python
first_agent = CustomizeAgent(
    name="FirstAgent",
    description="A agent that generates a blog post about the topic for the target audience",
    prompt_template=StringTemplate(instruction="Generate a blog post about the topic {topic} for the target audience {target_audience}"), 
    inputs=[{"name": "topic", "type": "string", "description": "The question to answer"}, {"name": "target_audience", "type": "string", "description": "The target audience of the topic"}],
    outputs=[{"name": "blog_post", "type": "string", "description": "The blog post about the topic for the target audience"}, {"name": "outline", "type": "string", "description": "The outline of the blog post"}],
    llm_config=openai_config # specify the LLM configuration 
)
```

**Note:** Make sure to import `StringTemplate` as shown in the imports section above.

### Method 2: Creating from Dictionary

You can also create an agent by defining its configuration in a dictionary:

```python
agent_data = {
    "name": "FirstAgent",
    "description": "A simple agent that prints hello world",
    "prompt": "Print 'hello world'",
    "llm_config": openai_config
}
first_agent = CustomizeAgent.from_dict(agent_data) # use .from_dict() to create an agent. 
```

### Using the Agent

Once created, you can use the agent in different ways:

#### Using the Simple Agent (Method 2)
```python
# Execute the agent without input. The agent will return a Message object containing the results. 
message = first_agent()

print(f"Response from {first_agent.name}:")
print(message.content.content) # the content of a Message object is a LLMOutputParser object, where the `content` attribute is the raw LLM output. 
```

#### Using the Blog Post Agent (Method 1)
```python
# Execute the agent with inputs for the blog post
message = first_agent(inputs={
    "topic": "Python programming", 
    "target_audience": "beginners"
})

print(f"Response from {first_agent.name}:")
print(f"Blog Post: {message.content.blog_post}")
print(f"Outline: {message.content.outline}")
```

For a complete example, please refer to the [CustomizeAgent example](https://github.com/EvoAgentX/EvoAgentX/blob/main/examples/customize_agent.py). 

CustomizeAgent also offers other features including structured inputs/outputs and multiple parsing strategies. For detailed information, see the [CustomizeAgent documentation](../modules/customize_agent.md).

## 2. Creating an Agent with Multiple Actions

In EvoAgentX, you can create an agent with multiple predefined actions. This allows you to build more complex agents that can perform multiple tasks. Here's an example showing how to create an agent with `TestCodeGeneration` and `TestCodeReview` actions:

### Defining Actions
First, we need to define the actions, which are subclasses of `Action`. Make sure to import all necessary dependencies:

```python
from evoagentx.agents import Agent
from evoagentx.actions import Action, ActionInput, ActionOutput
from evoagentx.models.base_model import BaseLLM
from pydantic import Field
from typing import Optional

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

From the above example, we can see that in order to define an action, we need to:

1. Define the action inputs and outputs using `ActionInput` and `ActionOutput` classes
2. Create an action class that inherits from `Action`
3. Implement the `execute` method which formulates the prompt with the action input data and uses the LLM to generate output, and specify how to parse the output using `parse_mode`.

### Defining an Agent 

Once we have defined the actions, we can create an agent by adding the actions to it:

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

### Executing Different Actions

Once you've created an agent with multiple actions, you can execute specific actions:

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

This example demonstrates how to:
1. List all available actions on an agent
2. Generate code using the TestCodeGeneration action
3. Review the generated code using the TestCodeReview action
4. Access the results from each action execution

For a complete working example, please refer to the [Agent example](https://github.com/EvoAgentX/EvoAgentX/blob/main/examples/agent_with_multiple_actions.py). 


## 3. Saving and Loading Agents

You can save an agent to a file and load it later:

```python
# Save the agent to a file
developer.save_module("examples/output/developer.json") # ignore the LLM config to avoid saving the API key 

# Load the agent from a file
developer = Agent.load_module("examples/output/developer.json", llm_config=openai_config)
```
