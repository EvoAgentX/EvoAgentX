import asyncio

from evoagentx.agents import AgentManager, CustomizeAgent
from evoagentx.core.logging import logger
from evoagentx.models import OpenAILLM, OpenAILLMConfig
from evoagentx.tools import (
    BrowserUseToolkit,
    FileToolkit,
    GoogleFreeSearchToolkit,
    RequestToolkit,
    RSSToolkit,
)
from evoagentx.workflow import WorkFlow, WorkFlowGenerator
from evoagentx.core.base_config import Parameter

llm_config = OpenAILLMConfig(model="gpt-4o", stream=True, output_response=True)

#----------------------------------------------------------------------
# Create prebuilt agent
#----------------------------------------------------------------------
logger.info("Creating prebuilt agents...")

inputs = [
      Parameter(name="article_content", type="string", description="Full article text content", required=True),
      Parameter(name="article_title", type="string", description="Article title", required=True),
      Parameter(name="source_category", type="string", description="Source RSS feed category hint", required=True),
    
]

outputs = [
      Parameter(name="categories", type="array", description="Extracted categories with confidence scores"),
      Parameter(name="key_topics", type="array", description="Main topics and keywords"),
      Parameter(name="sentiment_score", type="number", description="Sentiment analysis score (-1 to 1)"),
      Parameter(name="summary", type="string", description="AI-generated article summary")
]

# agent_description = "An agent that can find information online."
# prompt = """You are a web agent who can find information online using the tools provided.
# Always try to find the latest information on a topic using web search.

# ## Task:
# {task}
# """

# web_agent = CustomizeAgent(
#     name="WebAgent",
#     description=agent_description,
#     inputs=inputs,
#     outputs=outputs,
#     tools=[GoogleFreeSearchToolkit(), BrowserUseToolkit(), RequestToolkit(), RSSToolkit()],
#     llm_config=llm_config,
#     prompt=prompt
# )

#----------------------------------------------------------------------
# Set up workflow generator
#----------------------------------------------------------------------
llm = OpenAILLM(llm_config)

tools = [
    FileToolkit(),
    GoogleFreeSearchToolkit(),
    BrowserUseToolkit(),
    RequestToolkit(),
    RSSToolkit()
]

workflow_generator = WorkFlowGenerator(
    llm=llm,
    tools=tools,
    # prebuilt_agents=[web_agent],
    # workflow_folder="./examples/workflows"  # folder that contains workflow json files for RAG
)

goal = "Analyze RSS news content to extract key topics, sentiment, and auto-categorize articles"

logger.info("Generating workflow...")
workflow_graph = workflow_generator.generate_workflow(goal, workflow_inputs=inputs, workflow_outputs=outputs)
workflow_graph.save_module("./newsletter_workflow.json")

agent_manager = AgentManager()
agent_manager.add_agents_from_workflow(workflow_graph, llm_config)
llm = OpenAILLM(llm_config)
workflow = WorkFlow(graph=workflow_graph, agent_manager=agent_manager, llm=llm)

inputs = {
    "goal": "I'm interested in AI."
}
logger.info("Executing workflow...")
asyncio.run(workflow.async_execute(inputs=inputs))

