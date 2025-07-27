import os
from dotenv import load_dotenv
from evoagentx.workflow import WorkFlow, WorkFlowGraph
from evoagentx.workflow.workflow_graph import WorkFlowNode, WorkFlowEdge
from evoagentx.agents import CustomizeAgent
from evoagentx.prompts import StringTemplate, ChatTemplate
from evoagentx.models import OpenAILLMConfig
from evoagentx.tools.browser_tool import BrowserToolkit
from evoagentx.tools.image_analysis import ImageAnalysisTool
from evoagentx.tools.flux_image_generation import FluxImageGenerationTool

load_dotenv()

# 需要的API密钥
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BFL_API_KEY = os.getenv("BFL_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# 配置LLM
llm_config = OpenAILLMConfig(
    model="gpt-4o-mini", 
    openai_key=OPENAI_API_KEY, 
    stream=True, 
    output_response=True
)

def create_browser_research_agent():
    """
    Agent 1: Browser Research Agent with Image Analysis
    负责从网络获取热点信息和相关资料，并分析图片内容
    """
    
    # 创建工具
    browser_toolkit = BrowserToolkit(
        browser_type="chrome",
        headless=False,
        timeout=10
    )
    browser_tools = browser_toolkit.get_tools()
    
    # 添加图片分析工具（如果有API密钥）
    all_tools = browser_tools.copy()
    
    if OPENROUTER_API_KEY:
        image_analysis_tool = ImageAnalysisTool(
            api_key=OPENROUTER_API_KEY, 
            model="openai/gpt-4o-mini"
        )
        all_tools.append(image_analysis_tool)
        print("✅ 图片分析工具已启用")
    else:
        print("⚠️  图片分析工具未启用（需要OPENROUTER_API_KEY）")
    
    research_agent = CustomizeAgent(
        name="BrowserResearchAgent",
        description="Web research agent with browser automation and image analysis capabilities",
        prompt_template=ChatTemplate(
            instruction="You are a professional web research assistant specializing in social media content research. Your goal is to gather comprehensive information about trending topics using browser automation tools.",
            context="You have access to browser automation tools (navigate, click, input text, screenshot) and optional image analysis capabilities. Use these systematically to research current trends and popular content.",
            constraints=[
                "Focus on recent and trending information",
                "Gather data from multiple reliable sources",
                "Include visual content insights when possible",
                "Organize findings for content creation use",
                "Prioritize social media platforms and viral content"
            ],
            demonstrations=[
                {
                    "topic": "AI productivity tools",
                    "research_info": "AI tools gaining popularity among professionals for productivity. Key findings: 73% of professionals use AI daily, 40% faster task completion, reduced burnout levels, enhanced creativity. Trending platforms: ChatGPT, Notion AI, Claude. Visual content shows modern workspaces with AI interfaces."
                },
                {
                    "topic": "remote work trends",
                    "research_info": "Remote work trends and home office setups. Popular hashtags: #WFHLife #HomeOffice #RemoteWork. Visual content shows laptop setups, coffee cups, Zoom backgrounds. Key insights: flexible schedules, work-life balance, tech setup importance."
                },
                {
                    "topic": "sustainable technology",
                    "research_info": "Green tech innovations and eco-friendly solutions. Key trends: renewable energy adoption, electric vehicles, smart home sustainability. Popular hashtags: #GreenTech #Sustainability #CleanEnergy. Visual content shows solar panels, electric cars, smart home devices."
                }
            ]
        ),
        llm_config=llm_config,
        inputs=[
            {"name": "topic", "type": "string", "description": "The topic to research"}
        ],
        outputs=[
            {"name": "research_info", "type": "string", "description": "Comprehensive research report with text and visual insights"}
        ],
        tools=all_tools
    )
    
    return research_agent

def create_content_generation_agent():
    """
    Agent 2: Content Generation Agent
    基于研究信息生成社交媒体推文内容
    """
    
    content_agent = CustomizeAgent(
        name="ContentGenerationAgent", 
        description="Social media content creation specialist",
        prompt_template=ChatTemplate(
            instruction="You are an expert social media content creator who transforms research insights into engaging, viral-worthy posts.",
            context="You specialize in creating platform-specific content that drives engagement. You understand current social media trends, algorithm preferences, and audience psychology.",
            constraints=[
                "Content must be original and authentic",
                "Match the specified style and platform requirements",
                "Include engaging hooks and clear calls-to-action",
                "Incorporate trending insights from research data",
                "Optimize for maximum engagement and shareability"
            ],
            demonstrations=[
                {
                    "research_info": "AI tools gaining popularity among professionals for productivity. Key findings: 73% of professionals use AI daily, 40% faster task completion, reduced burnout levels, enhanced creativity. Trending platforms: ChatGPT, Notion AI, Claude. Visual content shows modern workspaces with AI interfaces.",
                    "style": "professional",
                    "platform": "LinkedIn",
                    "post_content": "🚀 The productivity revolution is here! New data shows 73% of professionals are now using AI tools daily.\n\nKey insights:\n• 40% faster task completion\n• Reduced burnout levels\n• Enhanced creativity\n\nWhich AI tool has transformed your workflow? Share your experience below! 👇\n\n#AI #Productivity #FutureOfWork #Innovation"
                },
                {
                    "research_info": "Remote work trends and home office setups. Popular hashtags: #WFHLife #HomeOffice #RemoteWork. Visual content shows laptop setups, coffee cups, Zoom backgrounds. Key insights: flexible schedules, work-life balance, tech setup importance.",
                    "style": "casual",
                    "platform": "Instagram",
                    "post_content": "Working from home got you feeling like a tech guru? 😎\n\nMy setup: laptop + coffee + endless Zoom calls\n\nWhat's your WFH essential? Drop your setup below! 👇\n\n#RemoteWork #HomeOffice #WFHLife #TechSetup"
                },
                {
                    "research_info": "Green tech innovations and eco-friendly solutions. Key trends: renewable energy adoption, electric vehicles, smart home sustainability. Popular hashtags: #GreenTech #Sustainability #CleanEnergy. Visual content shows solar panels, electric cars, smart home devices.",
                    "style": "informative",
                    "platform": "Twitter",
                    "post_content": "🌱 The future is green! Solar adoption up 40% this year.\n\nKey stats:\n• 2.5M new solar installations\n• EV sales doubled\n• Smart home energy savings: 25%\n\nWhat's your green tech story? 🌍\n\n#GreenTech #Sustainability #CleanEnergy #ClimateAction"
                }
            ]
        ),
        llm_config=llm_config,
        inputs=[
            {"name": "research_info", "type": "string", "description": "Research information from previous step"},
            {"name": "style", "type": "string", "description": "Content style (professional, casual, humorous, etc.)"},
            {"name": "platform", "type": "string", "description": "Target social media platform"}
        ],
        outputs=[
            {"name": "post_content", "type": "string", "description": "Generated social media post content"}
        ]
    )
    
    return content_agent

def create_image_generation_agent():
    """
    Agent 3: Image Generation Agent
    基于内容生成配套的社交媒体图片
    """
    
    # 创建图片生成工具
    image_gen_tool = FluxImageGenerationTool(
        api_key=BFL_API_KEY, 
        save_path="./social_media_images"
    )
    
    image_agent = CustomizeAgent(
        name="ImageGenerationAgent",
        description="Social media image creation specialist",
        prompt_template=ChatTemplate(
            instruction="You are a professional visual content creator specializing in social media imagery. Your role is to generate detailed prompts for creating compelling visuals that complement social media posts.",
            context="You understand visual trends across different social media platforms and know how to create images that stop the scroll and drive engagement. Your images should be optimized for mobile viewing and social media algorithms.",
            constraints=[
                "Generate detailed English prompts for image creation",
                "Ensure visuals are optimized for social media feeds",
                "No text elements in images (text overlay handled separately)",
                "High contrast and vibrant colors for mobile viewing",
                "Professional yet approachable aesthetic",
                "Platform-appropriate aspect ratios and compositions"
            ],
            demonstrations=[
                {
                    "post_content": "🚀 The productivity revolution is here! New data shows 73% of professionals are now using AI tools daily.",
                    "research_info": "AI tools gaining popularity among professionals for productivity. Key findings: 73% of professionals use AI daily, 40% faster task completion, reduced burnout levels, enhanced creativity. Trending platforms: ChatGPT, Notion AI, Claude. Visual content shows modern workspaces with AI interfaces.",
                    "image_path": "./social_media_images/ai_workspace_001.png"
                },
                {
                    "post_content": "Working from home got you feeling like a tech guru? 😎",
                    "research_info": "Remote work trends and home office setups. Popular hashtags: #WFHLife #HomeOffice #RemoteWork. Visual content shows laptop setups, coffee cups, Zoom backgrounds. Key insights: flexible schedules, work-life balance, tech setup importance.",
                    "image_path": "./social_media_images/home_office_002.jpg"
                },
                {
                    "post_content": "🌱 The future is green! Solar adoption up 40% this year.",
                    "research_info": "Green tech innovations and eco-friendly solutions. Key trends: renewable energy adoption, electric vehicles, smart home sustainability. Popular hashtags: #GreenTech #Sustainability #CleanEnergy. Visual content shows solar panels, electric cars, smart home devices.",
                    "image_path": "./social_media_images/green_tech_003.png"
                }
            ]
        ),
        llm_config=llm_config,
        inputs=[
            {"name": "post_content", "type": "string", "description": "Social media post content"},
            {"name": "research_info", "type": "string", "description": "Research context for visual inspiration"}
        ],
        outputs=[
            {"name": "image_path", "type": "string", "description": "Path to generated image file"}
        ],
        tools=[image_gen_tool]
    )
    
    return image_agent

def create_social_media_workflow():
    """
    创建完整的社交媒体内容生成工作流
    包含三个Agent：Research -> Content Generation -> Image Generation
    """
    
    # 1. 创建三个Agent
    research_agent = create_browser_research_agent()
    content_agent = create_content_generation_agent()
    image_agent = create_image_generation_agent()
    
    # 2. 创建工作流节点
    research_node = WorkFlowNode(name="research", module=research_agent)
    content_node = WorkFlowNode(name="content_generation", module=content_agent)
    image_node = WorkFlowNode(name="image_generation", module=image_agent)
    
    # 3. 定义工作流边（数据流向）
    edges = [
        # Research -> Content Generation
        WorkFlowEdge(
            source="research",
            target="content_generation",
            mappings={
                "research_info": "research_info",
                "style": "style",
                "platform": "platform"
            }
        ),
        # Content Generation -> Image Generation
        WorkFlowEdge(
            source="content_generation",
            target="image_generation", 
            mappings={
                "post_content": "post_content"
            }
        ),
        # Research -> Image Generation (为图片生成提供额外上下文)
        WorkFlowEdge(
            source="research",
            target="image_generation",
            mappings={
                "research_info": "research_info"
            }
        )
    ]
    
    # 4. 创建工作流图
    workflow_graph = WorkFlowGraph(
        nodes=[research_node, content_node, image_node],
        edges=edges
    )
    
    # 5. 创建完整工作流
    workflow = WorkFlow(
        name="SocialMediaContentWorkflow",
        description="Complete social media content creation workflow: Research -> Content -> Image",
        graph=workflow_graph,
        inputs=[
            {"name": "topic", "type": "string", "description": "Content topic to research"},
            {"name": "style", "type": "string", "description": "Content style (professional, casual, humorous, etc.)"},
            {"name": "platform", "type": "string", "description": "Target social media platform (Twitter, Instagram, LinkedIn, etc.)"}
        ],
        outputs=[
            {"name": "research_info", "type": "string", "description": "Research findings"},
            {"name": "post_content", "type": "string", "description": "Generated social media content"},
            {"name": "image_path", "type": "string", "description": "Path to generated image"}
        ]
    )
    
    return workflow

def test_complete_workflow():
    """
    测试完整的三节点工作流
    """
    # 检查API密钥
    required_keys = {
        "OPENAI_API_KEY": OPENAI_API_KEY,
        "BFL_API_KEY": BFL_API_KEY,
    }
    
    # OPENROUTER_API_KEY是可选的，用于图片分析
    optional_keys = {
        "OPENROUTER_API_KEY": OPENROUTER_API_KEY
    }
    
    missing_keys = [key for key, value in required_keys.items() if not value]
    if missing_keys:
        print("❌ 缺少必需的API密钥:")
        for key in missing_keys:
            print(f"   - {key}")
        print("\n请在.env文件中添加所有必需的API密钥")
        return
    
    # 检查可选API密钥
    missing_optional = [key for key, value in optional_keys.items() if not value]
    if missing_optional:
        print("⚠️  可选API密钥未配置（图片分析功能将不可用）:")
        for key in missing_optional:
            print(f"   - {key}")
        print()
    
    print("🚀 测试完整的社交媒体内容工作流")
    print("=" * 60)
    
    try:
        # 创建工作流
        workflow = create_social_media_workflow()
        print("✅ 完整工作流创建成功!")
        
        # 测试案例
        test_cases = [
            {
                "topic": "AI productivity tools for professionals",
                "style": "professional",
                "platform": "LinkedIn"
            },
            {
                "topic": "Healthy morning routines",
                "style": "casual",
                "platform": "Instagram"
            },
            {
                "topic": "Remote work productivity tips",
                "style": "humorous",
                "platform": "Twitter"
            }
        ]
        
        print("\n🎯 可选测试案例：")
        for i, case in enumerate(test_cases, 1):
            print(f"{i}. {case['topic']} ({case['style']} style for {case['platform']})")
        
        print("\n请选择测试案例（输入数字1-3）:")
        try:
            choice = int(input().strip())
            if 1 <= choice <= len(test_cases):
                selected_case = test_cases[choice - 1]
                print(f"\n🎯 执行案例: {selected_case['topic']}")
                print(f"   风格: {selected_case['style']}")
                print(f"   平台: {selected_case['platform']}")
                
                # 执行完整工作流
                result = workflow.run(**selected_case)
                
                print("\n" + "="*60)
                print("🎉 工作流执行完成！")
                print("="*60)
                
                print("\n📊 1. 研究结果:")
                print("-" * 40)
                print(result['research_info'])
                
                print("\n📝 2. 生成的内容:")
                print("-" * 40)
                print(result['post_content'])
                
                print(f"\n🖼️ 3. 生成的图片:")
                print("-" * 40)
                print(f"图片路径: {result['image_path']}")
                
                # 尝试显示图片
                if os.path.exists(result['image_path']):
                    try:
                        from PIL import Image
                        img = Image.open(result['image_path'])
                        img.show()
                        print("✅ 图片已自动打开")
                    except ImportError:
                        print("💡 安装PIL查看图片: pip install Pillow")
                    except Exception as e:
                        print(f"❌ 显示图片失败: {e}")
                else:
                    print("❌ 图片文件未找到")
                
                return result
                
            else:
                print("❌ 无效选择")
                
        except (ValueError, KeyboardInterrupt):
            print("\n💡 你也可以直接运行:")
            print('result = workflow.run(topic="your topic", style="professional", platform="LinkedIn")')
            
    except Exception as e:
        print(f"❌ 工作流执行失败: {e}")
        print("\n💡 可能的问题:")
        print("1. 依赖安装: pip install browser-use")
        print("2. 浏览器: 需要Chrome/Chromium")
        print("3. 网络连接问题")
        print("4. API密钥配置错误")

def test_direct_agent():
    """
    直接测试单个Agent（不通过workflow）
    """
    print("🧪 直接测试Browser Research Agent")
    print("=" * 40)
    
    try:
        agent = create_browser_research_agent()
        
        # 简单测试
        topic = "Today's trending topics"
        print(f"🎯 测试主题: {topic}")
        
        result = agent(inputs={"topic": topic})
        
        print("\n📋 Agent直接输出:")
        print(result.content.research_info)
        
    except Exception as e:
        print(f"❌ 直接测试失败: {e}")

if __name__ == "__main__":
    """
    🔧 配置说明：
    
    完整的三节点社交媒体内容工作流
    
    1. 必需环境变量：
       - OPENAI_API_KEY: OpenAI API密钥 (用于LLM)
       - BFL_API_KEY: Black Forest Labs API密钥 (用于图片生成)
    
    2. 可选环境变量：
       - OPENROUTER_API_KEY: OpenRouter API密钥 (用于图片分析，可选)
    
    3. 依赖安装：
       - pip install selenium (浏览器自动化)
       - pip install Pillow (用于图片显示)
       - 需要安装Chrome浏览器
    
    4. 工作流节点：
       - Node 1: Browser Research Agent (Selenium浏览器 + 可选图片分析)
       - Node 2: Content Generation Agent (创建推文内容)
       - Node 3: Image Generation Agent (生成配图)
    
    4. 数据流向：
       Research -> Content Generation -> Image Generation
       Research -> Image Generation (提供额外上下文)
    """
    
    print("📱 完整社交媒体内容工作流")
    print("三节点工作流：Research -> Content -> Image")
    print("=" * 60)
    
    # 运行完整工作流测试
    # test_complete_workflow()
    
    # 可选：直接测试单个Agent
    test_direct_agent() 