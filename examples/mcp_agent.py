## You will need to create a MCP config file first at "examples/output/tests/shares_mcp.config"

import os 
from dotenv import load_dotenv
from evoagentx.models import OpenAILLMConfig
from evoagentx.agents import CustomizeAgent
from evoagentx.prompts import StringTemplate 
from evoagentx.tools.mcp import MCPToolkit
from evoagentx.tools.image_analysis import ImageAnalysisTool
from evoagentx.tools.images_flux_generation import FluxImageGenerationTool

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_ORGANIZATION_ID = os.getenv("OPENAI_ORGANIZATION_ID")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
openai_config = OpenAILLMConfig(model="gpt-4o-mini", openai_key=OPENAI_API_KEY, stream=True, output_response=True)

def test_MCP_server():
    
    mcp_Toolkit = MCPToolkit(config_path="examples/output/mcp_agent/mcp.config")
    tools = mcp_Toolkit.get_toolkits()
    
    mcp_agent = CustomizeAgent(
        name="MCPAgent",
        description="A MCP agent that can use the tools provided by the MCP server",
        prompt_template= StringTemplate(
            instruction="Do some operations based on the user's instruction."
        ), 
        llm_config=openai_config,
        inputs=[
            {"name": "instruction", "type": "string", "description": "The goal you need to achieve"}
        ],
        outputs=[
            {"name": "result", "type": "string", "description": "The result of the operation"}
        ],
        tools=tools
    )
    mcp_agent.save_module("examples/output/mcp_agent/mcp_agent.json")
    mcp_agent.load_module("examples/output/mcp_agent/mcp_agent.json", llm_config=openai_config, tools=tools)

    message = mcp_agent(
        inputs={"instruction": "Summarize all the tools."}
    )
    
    print(f"Response from {mcp_agent.name}:")
    print(message.content.result)

if __name__ == "__main__":
    test_MCP_server()
