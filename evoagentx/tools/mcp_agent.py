## You will need to create a MCP config file first at "examples/output/tests/shares_mcp.config"

import os 
from dotenv import load_dotenv
from evoagentx.models import OpenAILLMConfig
from evoagentx.agents import CustomizeAgent
from evoagentx.prompts import StringTemplate 
from evoagentx.tools.mcp import MCPToolkit
from evoagentx.tools.image_analysis import ImageAnalysisTool
from evoagentx.tools.flux_image_generation import FluxImageGenerationTool

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
        description="一个可以使用MCP服务器提供工具的MCP代理",
        prompt_template= StringTemplate(
            instruction="根据用户的指令执行一些操作。"
        ), 
        llm_config=openai_config,
        inputs=[
            {"name": "instruction", "type": "string", "description": "你需要实现的目标"}
        ],
        outputs=[
            {"name": "result", "type": "string", "description": "操作的结果"}
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

def test_image_analysis_tool():
    tools = []
    tools.append(ImageAnalysisTool(api_key=OPENROUTER_API_KEY, model="openai/gpt-4o-mini"))
    
    mcp_agent = CustomizeAgent(
        name="MCPAgent",
        description="一个可以使用MCP服务器提供工具的MCP代理",
        prompt_template= StringTemplate(
            instruction="根据用户的指令执行一些操作。"
        ), 
        llm_config=openai_config,
        inputs=[
            {"name": "prompt", "type": "string", "description": "图像分析的问题或指令。"},
            {"name": "image_url", "type": "string", "description": "要分析的图像的URL。"}
        ],
        outputs=[
            {"name": "content", "type": "string", "description": "分析结果。"},
            {"name": "usage", "type": "object", "description": "令牌使用信息。"}
        ],
        tools=tools
    )
    mcp_agent.save_module("examples/output/mcp_agent/mcp_agent.json")
    mcp_agent.load_module("examples/output/mcp_agent/mcp_agent.json", llm_config=openai_config, tools=tools)

    message = mcp_agent(
        inputs={
            "prompt": "Describe this image.",
            "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
        }
    )
    print(f"Response from {mcp_agent.name}:")
    print(message.content.content)

def test_image_generation_tool():
    from evoagentx.tools.OpenAI_Image_Generation import OpenAI_ImageGenerationTool
    tools = []
    tools.append(OpenAI_ImageGenerationTool(api_key=OPENAI_API_KEY, organization_id=OPENAI_ORGANIZATION_ID, model="gpt-4o", save_path="./imgs"))

    mcp_agent = CustomizeAgent(
        name="MCPAgent",
        description="一个可以使用MCP服务器提供工具的MCP代理",
        prompt_template= StringTemplate(
            instruction="根据用户的指令执行一些操作。"
        ), 
        llm_config=openai_config,
        inputs=[
            {"name": "prompt", "type": "string", "description": "图像生成的提示词。"},
        ],
        outputs=[
            {"name": "file_path", "type": "object", "description": "生成的图像（PIL.Image）。"}
        ],
        tools=tools
    )
    mcp_agent.save_module("examples/output/mcp_agent/mcp_agent.json")
    mcp_agent.load_module("examples/output/mcp_agent/mcp_agent.json", llm_config=openai_config, tools=tools)

    message = mcp_agent(
        inputs={
            "prompt": "画一个阿斯塔特战士，穿着黑色盔甲，手持爆弹枪，站在一个充满火焰的背景前和泰伦虫族战斗",
        }
    )
    from PIL import Image
    img = Image.open(message.content.file_path)
    img.show()

def test_flux_image_generation_tool():
    from evoagentx.tools.flux_image_generation import FluxImageGenerationTool
    
    # 需要设置BFL API密钥
    BFL_API_KEY = os.getenv("BFL_API_KEY")
    if not BFL_API_KEY:
        print("请设置BFL_API_KEY环境变量")
        return
    
    tools = []
    tools.append(FluxImageGenerationTool(api_key=BFL_API_KEY, save_path="./flux_imgs"))

    mcp_agent = CustomizeAgent(
        name="FluxImageAgent",
        description="一个可以使用Flux Kontext Max生成图像的MCP代理",
        prompt_template= StringTemplate(
            instruction="使用Flux Kontext Max根据用户的提示词生成图像。你必须调用flux_image_generation工具来生成图像，不要直接返回描述性文本。"
        ), 
        llm_config=openai_config,
        inputs=[
            {"name": "prompt", "type": "string", "description": "图像生成的提示词。"},
        ],
        outputs=[
            {"name": "file_path", "type": "string", "description": "生成图像的路径。"}
        ],
        tools=tools
    )
    mcp_agent.save_module("examples/output/mcp_agent/flux_agent.json")
    mcp_agent.load_module("examples/output/mcp_agent/flux_agent.json", llm_config=openai_config, tools=tools)

    # 测试生成新图像
    message = mcp_agent(
        inputs={
            "prompt": 
            """生成一副马耳他风景图片，图片中包含马耳他首府mosta以及海岸边风景""",
        }
    )
    print(f"Generated image path: {message.content.file_path}")
    
    # 显示生成的图像
    from PIL import Image
    img = Image.open(message.content.file_path)
    img.show()

if __name__ == "__main__":
    # test_MCP_server()
    # test_image_analysis_tool()
    # 
    # 
    # test_image_generation_tool()
    test_flux_image_generation_tool()