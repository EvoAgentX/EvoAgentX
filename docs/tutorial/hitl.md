# Getting Started with Human-in-the-Loop (HITL)

In EvoAgentX, **Human-in-the-Loop (HITL)** allows you to insert manual approval, user input collection, or multi-turn conversations(Not implemented yet) at critical points of a workflow. This guarantees that generated content meets expectations or lets you inject human feedback when necessary. With the `hitl` module you can:

1. Approve or reject the output of an Action/Agent **before** or **after** it runs.
2. Dynamically collect structured input from an end user and pass it to downstream tasks.
3. Centrally manage every human interaction through a single `HITLManager` instance.

This tutorial walks you through the following steps:

1. Enable HITL and create a `HITLManager`.
2. Insert a pre-execution approval step with `HITLInterceptorAgent`.
3. Collect user input with `HITLUserInputCollectorAgent`.
4. Combine both kinds of HITL agents into a workflow and run it.

> Full runnable examples:
> * [`examples/hitl/hitl_example.py`](../../examples/hitl/hitl_example.py) – pre-execution approval
> * [`examples/hitl/hitl_example2.py`](../../examples/hitl/hitl_example2.py) – user input collection

---

## 1. Enabling HITL

In any scenario you must first instantiate `HITLManager` and call `activate()`; otherwise all HITL requests will be auto-approved and won't block execution.

```python
from evoagentx.hitl import HITLManager

# Create the manager
hitl_manager = HITLManager()
# Enable HITL (disabled by default)
hitl_manager.activate()
```

Once enabled, simply pass `hitl_manager` to the `WorkFlow` constructor.

---

## 2. Pre-Execution Approval – `HITLInterceptorAgent`

`HITLInterceptorAgent` intercepts the execution of a target Agent/Action and asks you in the terminal to **approve** or **reject** it. Using `examples/hitl/hitl_example.py` as the base, we prepare three agents:

1. `DataExtractionAgent` – extracts data from raw text.
2. `DataSendingAgent` – sends the extracted data via email.
3. `HITLInterceptorAgent` – asks for human approval before the email is sent.

### 2.1 Business Agents

```python
from evoagentx.agents import CustomizeAgent, Agent
from evoagentx.models import OpenAILLMConfig, OpenAILLM

llm_config = OpenAILLMConfig(model="gpt-4o-mini", openai_key=OPENAI_API_KEY, stream=True)
llm = OpenAILLM(llm_config)

# Extraction agent
extraction_agent = CustomizeAgent(
    name="DataExtractionAgent",
    description="Extract requested data",
    inputs=[{"name": "data_source", "type": "str"}],
    outputs=[{"name": "extracted_data", "type": "str"}],
    prompt="Extract data from source: {data_source}",  # prompt text in Chinese is fine
    llm_config=llm_config,
    parse_mode="str"
)

# Dummy email-sending agent
from evoagentx.actions import Action, ActionInput, ActionOutput

class EmailInput(ActionInput):
    human_verified_data: str

class EmailOutput(ActionOutput):
    send_action_result: str

class DummyEmailSendAction(Action):
    def __init__(self):
        super().__init__(
            name="DummyEmailSendAction",
            description="Send email",
            inputs_format=EmailInput,
            outputs_format=EmailOutput,
        )
    def execute(self, llm, inputs, **kwargs):
        return EmailOutput(send_action_result=f"Email sent with: {inputs['human_verified_data']}")

data_sending_agent = Agent(
    name="DataSendingAgent",
    description="Email-sending Agent",
    actions=[DummyEmailSendAction()],
    llm_config=llm_config
)
```

### 2.2 Creating the Interceptor

```python
from evoagentx.hitl import HITLInterceptorAgent, HITLInteractionType, HITLMode

interceptor_agent = HITLInterceptorAgent(
    target_agent_name="DataSendingAgent",      # name of the business agent to intercept
    target_action_name="DummyEmailSendAction", # action to intercept
    interaction_type=HITLInteractionType.APPROVE_REJECT,
    mode=HITLMode.PRE_EXECUTION                 # pre-execution approval
)
```


### 2.3 Mapping Input and Output Fields

The interceptor itself does not transform data, so we must tell `HITLManager` that the interceptor's output field `human_verified_data` corresponds to the upstream field `extracted_data`. This ensures the downstream `DataSendingAgent` receives the approved data.

```python
hitl_manager.hitl_input_output_mapping = {
    "human_verified_data": "extracted_data"
}
```

### 2.4 Building and Running the Workflow

```python
from evoagentx.workflow import WorkFlow, WorkFlowGraph
from evoagentx.workflow.workflow_graph import WorkFlowNode, WorkFlowEdge
from evoagentx.agents import AgentManager

nodes = [
    WorkFlowNode(
        name="extract_node",
        description="Data extraction",
        agents=[extraction_agent],
        inputs=[{"name": "data_source", "type": "str"}],
        outputs=[{"name": "extracted_data", "type": "str"}]
    ),
    WorkFlowNode(
        name="intercept_node",
        description="Human approval",
        agents=[interceptor_agent],
        inputs=[{"name": "extracted_data", "type": "str"}],
        outputs=[{"name": "human_verified_data", "type": "str"}]
    ),
    WorkFlowNode(
        name="send_node",
        description="Send email",
        agents=[data_sending_agent],
        inputs=[{"name": "human_verified_data", "type": "str"}],
        outputs=[{"name": "send_action_result", "type": "str"}]
    )
]

edges = [
    WorkFlowEdge(source="extract_node", target="intercept_node"),
    WorkFlowEdge(source="intercept_node", target="send_node")
]

graph = WorkFlowGraph(goal="Extract data and send an email", nodes=nodes, edges=edges)
manager = AgentManager(agents=[extraction_agent, interceptor_agent, data_sending_agent])
workflow = WorkFlow(graph=graph, llm=llm, agent_manager=manager, hitl_manager=hitl_manager)

result = await workflow.async_execute(inputs={
    "data_source": "2025Q2 financial report ..."
})
```

When the interceptor runs you will see a prompt like below. Type `a` (approve) or `r` (reject):

```
🔔 Human-in-the-Loop approval request
================================================================================
Task: send_node
Agent: DataSendingAgent
Action: DummyEmailSendAction (PRE-EXECUTION)
...
Please select [a]pprove / [r]eject: _
```

* Approve → workflow continues.
* Reject → the current run terminates.

---

## 3. Collecting User Input – `HITLUserInputCollectorAgent`

When you need full user input instead of a simple yes/no approval, use `HITLUserInputCollectorAgent`. It asks for each field in the terminal and maps the collected data to downstream tasks.

Based on [`examples/hitl/hitl_example2.py`](../../examples/hitl/hitl_example2.py):

### 3.1 Define the Fields

```python
user_input_fields = {
    "user_name": {
        "type": "string",
        "description": "Please enter your name",
        "required": True
    },
    "user_age": {
        "type": "int",
        "description": "Please enter your age",
        "required": True
    },
    "user_email": {
        "type": "string",
        "description": "Please enter your email",
        "required": True
    },
    "user_preferences": {
        "type": "string",
        "description": "Preferences (optional)",
        "required": False,
        "default": "No special preferences"
    }
}
```

### 3.2 Create the Collector Agent

```python
from evoagentx.hitl import HITLUserInputCollectorAgent

collector_agent = HITLUserInputCollectorAgent(
    name="UserProfileCollector",
    input_fields=user_input_fields,
    llm_config=llm_config
)
```

### 3.3 Downstream Processor

```python
profile_processor = CustomizeAgent(
    name="ProfileProcessor",
    description="Generate recommendations from user info",
    inputs=[
        {"name": "user_name", "type": "string"},
        {"name": "user_age", "type": "int"},
        {"name": "user_email", "type": "string"},
        {"name": "user_preferences", "type": "string"}
    ],
    outputs=[
        {"name": "profile_summary", "type": "string"},
        {"name": "recommendations", "type": "string"}
    ],
    prompt="Generate profile summary and personalized recommendations based on the following user information:\nName: {user_name}\nAge: {user_age}\nEmail: {user_email}\nPreferences: {user_preferences}\n\nPlease provide profile summary and personalized recommendations. The results should be presented in json format and have field of 'profile_summary' and 'recommendations'",
    llm_config=llm_config,
    parse_mode="json"
)
```

### 3.4 Stitching the Workflow

```python
nodes = [
    WorkFlowNode(
        name="collect_node",
        description="Collect user input",
        agents=[collector_agent],
        inputs=[],  # no external input
        outputs=[
            {"name": "user_name", "type": "string"},
            {"name": "user_age", "type": "int"},
            {"name": "user_email", "type": "string"},
            {"name": "user_preferences", "type": "string"}
        ]
    ),
    WorkFlowNode(
        name="process_node",
        description="Generate personalised recommendations",
        agents=[profile_processor],
        inputs=[
            {"name": "user_name", "type": "string"},
            {"name": "user_age", "type": "int"},
            {"name": "user_email", "type": "string"},
            {"name": "user_preferences", "type": "string"}
        ],
        outputs=[
            {"name": "profile_summary", "type": "string"},
            {"name": "recommendations", "type": "string"}
        ]
    )
]

edges = [WorkFlowEdge(source="collect_node", target="process_node")]
```

Run the workflow after injecting `HITLManager` and the mapping. You will be prompted for every field:

```
User input fields to be collected:

- user_name (string): please input your name
- user_age (int): please input your age
- user_email (string): please input your email address
- user_preferences (string): please input your preferences (optional) [optional] [default: no special preferences]
================================================================================

Please provide the following inputs:

user_name (please input your name): david

user_age (please input your age): 26

user_email (please input your email address): david@gmail.com
...
```

---

## 4. Interaction Types & Modes

| Enum | Description |
| --- | --- |
| `HITLInteractionType.APPROVE_REJECT` | Simple approval / rejection |
| `HITLInteractionType.COLLECT_USER_INPUT` | Collect user input |
| `HITLInteractionType.REVIEW_EDIT_STATE` | Review and edit intermediate state *(coming soon)* |
| `HITLInteractionType.REVIEW_TOOL_CALLS` | Review tool calls *(coming soon)* |
| `HITLInteractionType.MULTI_TURN_CONVERSATION` | Multi-turn conversation guidance *(coming soon)* |

**Mode** (`HITLMode`) decides *when* interception happens:

- `HITLMode.PRE_EXECUTION` — before the target Action runs.
- `HITLMode.POST_EXECUTION` — after the target Action runs *(not fully implemented yet).* 

---
