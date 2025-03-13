
# EvoAgentX 
An automatic agentic workflow generation and evolving framework. 

## News

## Get Started

### Installation
create environment: 
```
conda create -n agent python=3.10 
pip install -r requirements.txt 
```
### Configuration
Todos:
1. How to set up keys
2. others

### Examples: Automatic WorkFlow Generation 
```python
from evoagentx.models import OpenAILLMConfig, OpenAILLM
from evoagentx.agents import AgentManager
from evoagentx.workflow import WorkFlowGenerator, WorkFlowGraph, WorkFlow

OPENAI_API_KEY = "OPENAI_API_KEY" 
# set output_response=True to see LLM outputs 
openai_config = OpenAILLMConfig(model="gpt-4o-mini", openai_key=OPENAI_API_KEY, stream=True, output_response=False)
model = OpenAILLM(config=openai_config)

agent_manager = AgentManager()
wf_generator = WorkFlowGenerator(llm=model)

# generate workflow & agents
workflow_graph: WorkFlowGraph = wf_generator.generate_workflow(goal="Generate a python code for greedy snake game")

# [optional] display workflow
workflow_graph.display()
# [optional] save workflow 
workflow_graph.save_module("debug/workflow_demo.json")
#[optional] load saved workflow 
workflow_graph: WorkFlowGraph = WorkFlowGraph.from_file("debug/workflow_demo.json")

agent_manager.add_agents_from_workflow(workflow_graph)
# execute workflow
workflow = WorkFlow(graph=workflow_graph, agent_manager=agent_manager, llm=model)
output = workflow.execute()
print(output)
```

### QuickStart & Demo Video
Todos

## Tutorial
Todos
## Support

### Discord Join US

📢 Join Our [Discord Channel](https://discord.gg/q5hBjHVz)! Looking forward to seeing you there! 🎉

### Contact Information

If you have any questions or feedback about this project, please feel free to contact us. We highly appreciate your suggestions!

- **Email:** clayx.ai.co@gmail.com

We will respond to all questions within 2-3 business days.

## Contributing to EvoAgentX
We appreciate your interest in contributing to our open-source initiative. We provide a document of [contributing guidelines](https://github.com/clayxai/EvoAgentX/blob/main/CONTRIBUTING.md) which outlines the steps for contributing to EvoAgentX. Please refer to this guide to ensure smooth collaboration and successful contributions. 🤝🚀
