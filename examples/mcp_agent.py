## You will need to create a MCP config file first at "examples/output/tests/shares_mcp.config"

import os 
from dotenv import load_dotenv
from evoagentx.models import OpenAILLMConfig
from evoagentx.agents import CustomizeAgent
from evoagentx.prompts import StringTemplate 
from evoagentx.tools.mcp import MCPToolkit

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai_config = OpenAILLMConfig(model="gpt-4o-mini", openai_key=OPENAI_API_KEY, stream=True, output_response=True)

def test_MCP_server():
    
    mcp_toolkit = MCPToolkit(config_path="examples/output/tests/shares_mcp.config")
    tools = mcp_toolkit.get_tools()
    
    code_writer = CustomizeAgent(
        name="CodeWriter",
        description="Writes Python code based on requirements",
        prompt_template= StringTemplate(
            instruction="Do some operations based on the user's instruction."
        ), 
        llm_config=openai_config,
        inputs=[
            {"name": "instruction", "type": "string", "description": "The goal you need to achieve"}
        ],
        outputs=[
            {"name": "result", "type": "string", "description": "The tools you have"}
        ],
        tools=tools
    )

    message = code_writer(
        inputs={"instruction": "lets try **get_stock_info** tool to get the stock info of **AAPL**"}
    )
    print(f"Response from {code_writer.name}:")
    print(message.content.result)

if __name__ == "__main__":
    test_MCP_server()