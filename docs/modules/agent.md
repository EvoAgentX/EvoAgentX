# Agent

## Introduction

The `Agent` class is the fundamental building block for creating intelligent AI agents within the EvoAgentX framework. It provides a structured way to combine language models with actions, and memory management. 

## Architecture

An Agent consists of several key components:

1. **Large Language Model (LLM)**: 

    The LLM is specified through the `llm` or `llm_config` parameter and serve as the building block for the agent. It is responsible for interpreting context, generating responses, and making high-level decisions. The LLM will be passed to an action for executing a specific task. 

2. **Actions**: 

    Actions are the fundamental operational units of an agent. Each Action encapsulates a specific task and is the actual point where the LLM is invoked to reason, generate, or make decisions. While the Agent provides overall orchestration, it is through Actions that the LLM performs its core functions. Each Action is designed to do exactly one thing—such as retrieving knowledge, summarizing input, or calling an API—and can include the following components:

    - **prompt**: The prompt template used to guide the LLM's behavior for this specific task.
    - **inputs_format**: The expected structure and keys of the inputs passed into the action.
    - **outputs_format**: The format used to interpret and parse the LLM's output.
    - **tools**: Optional tools that can be integrated and utilized within the action.

3. **Memory Components**:

    Memory allows the agent to retain and recall relevant information across interactions, enhancing contextual awareness. There are two types of memory within the EvoAgentX framework: 

    - **Short-term memory**: Maintains the intermediate conversation or context for the current task. 
    - **Long-term memory (optional)**: Stores persistent knowledge that can span across sessions or tasks. This enables the agent to learn from past experiences, maintain user preferences, or build knowledge bases over time.


## Usage

### Basic Agent Creation

In order to create an agent, you need to define the actions that the agent will perform. Each action is defined as a class that inherits from the `Action` class. The action class should define the following components: `name`, `description`, `prompt`, `inputs_format`, and `outputs_format`, and implement the `execute` method (and `async_exectue` if you want to use the agent asynchronously). 


```python
from evoagentx.agents import Agent
from evoagentx.models import OpenAILLMConfig
from evoagentx.actions import Action, ActionInput, ActionOutput

# Define a simple action that uses the LLM to answer a question

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

# Configure LLM
llm_config = OpenAILLMConfig(model="gpt-4o-mini", openai_key="your-api-key")

# Create an agent
agent = Agent(
    name="AssistantAgent",
    description="Answers a factual question using the LLM",
    llm_config=llm_config,
    system_prompt="You are a helpful assistant.",
    actions=[AnswerQuestionAction()]
)
```

### Executing Actions

You can directly call the `Agent` instance like a function. This will internally invoke the `execute()` method of the matching action using the specified `action_name` and `action_input_data`.

```python
# Execute an action with input data
message = agent(
    action_name="answer_question",
    action_input_data={"question": "What is the capital of France?"}
)

# Access the output
result = message.content.answer 
```

### Asynchronous Execution

You can also call the `Agent` instance in an asynchronous context. If the action defines an `async_execute` method, it will be used automatically when you `await` the agent.

```python
# Execute an action asynchronously
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

## Memory Management

The Agent maintains a short-term memory for tracking conversation context:

```python
# Access the agent's memory
messages = agent.short_term_memory.get(n=5)  # Get last 5 messages

# Clear memory
agent.clear_short_term_memory()
```

## Agent Profile

You can get a human-readable description of an agent and its capabilities:

```python
# Get description of all actions
profile = agent.get_agent_profile()
print(profile)

# Get description of specific actions
profile = agent.get_agent_profile(action_names=["answer_question"])
print(profile)
```

## Prompt Management

Access and modify the prompts used by an agent:

```python
# Get all prompts
prompts = agent.get_prompts()
# prompts is a dictionary with the structure:
# {'answer_question': {'system_prompt': 'You are a helpful assistant.', 'prompt': 'Answer the following question as accurately as possible:\n\n{question}'}}

# Set a specific prompt
agent.set_prompt(
    action_name="answer_question",
    prompt="Please provide a clear and concise answer to the following query:\n\n{question}",
    system_prompt="You are a helpful assistant." # optional, if not provided, the system prompt will remain unchanged 
)

# Update all prompts
prompts_dict = {
    "answer_question": {
        "system_prompt": "You are an expert in providing concise, accurate information.",
        "prompt": "Please answer this question with precision and clarity:\n\n{question}"
    }
}
agent.set_prompts(prompts_dict)
```

## Saving and Loading Agents

Agents can be persisted and reloaded:

```python
# Save agent
agent.save_module("./agents/my_agent.json")

# Load agent (requires providing llm_config again)
loaded_agent = Agent.from_file(
    "./agents/my_agent.json", 
    llm_config=llm_config
)
```

## Context Extraction

The Agent includes a built-in context extraction mechanism that automatically derives appropriate inputs for actions from conversation history:

```python
# Context is automatically extracted when executing without explicit input data
response = agent.execute(
    action_name="action_name",
    msgs=conversation_history
)

# Get action inputs manually
action = agent.get_action("action_name")
inputs = agent.get_action_inputs(action)
```
