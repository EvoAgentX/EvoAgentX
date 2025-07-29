import os
import asyncio
import textwrap
import datetime
from dotenv import load_dotenv
from evoagentx.models.openai_model import OpenAILLM
from evoagentx.models.model_configs import LLMConfig
from evoagentx.workflow import WorkFlow, WorkFlowGraph
from evoagentx.workflow.workflow_graph import WorkFlowNode, WorkFlowEdge
from evoagentx.agents import CustomizeAgent, AgentManager
from evoagentx.prompts import StringTemplate, ChatTemplate
from evoagentx.models import OpenAILLMConfig
from evoagentx.tools.search_ddgs import DDGSSearchToolkit
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

def create_prompt_analysis_agent():
    """
    Agent 0: Prompt Analysis Agent
    负责解析用户输入的prompt，确定研究主题、是否需要图片生成以及图片数量和内容
    """
    
    prompt_analysis_agent = CustomizeAgent(
        name="PromptAnalysisAgent",
        description="Analyzes user prompts to determine research topics and image generation requirements",
        prompt_template=ChatTemplate(
            instruction="You are a prompt analysis specialist who breaks down user requests into structured research and content generation tasks.",
            context="You analyze user prompts to determine: 1) The main research topic, 2) How many images are required, 3) What each image should contain. You should also preserve the original user prompt for downstream processing.",
            constraints=[
                "Always provide a clear research topic summary",
                "Specify exact number of images needed",
                "Describe what each image should contain"
            ],
            demonstrations=[]
        ),
        llm_config=llm_config,
        inputs=[
            {"name": "user_prompt", "type": "string", "description": "User's original input prompt"}
        ],
        outputs=[
            {"name": "research_topic", "type": "string", "description": "Summarized research topic"},
            {"name": "image_count", "type": "integer", "description": "Number of images to generate"},
            {"name": "image_descriptions", "type": "string", "description": "Descriptions for each image"},
            {"name": "user_prompt", "type": "string", "description": "Original user prompt"}
        ]
    )
    
    return prompt_analysis_agent

def create_research_agent():
    """
    Agent 1: Research Agent with Image Analysis
    负责使用 DDGS 搜索获取热点信息和相关资料，并分析图片内容
    """
    
    # 创建 DDGS 搜索工具包
    all_tools = []
    
    ddgs_toolkit = DDGSSearchToolkit(
        name="DDGSSearchToolkit",
        num_search_pages=5,
        max_content_words=500,
        backend="auto",
        region="us-en"
    )
    search_tools = ddgs_toolkit.get_tools()
    all_tools.extend(search_tools)
    
    # 添加图片分析工具（如果有API密钥）
    if OPENROUTER_API_KEY:
        image_analysis_tool = ImageAnalysisTool(
            api_key=OPENROUTER_API_KEY, 
            model="openai/gpt-4o-mini"
        )
        all_tools.append(image_analysis_tool)
    
    research_agent = CustomizeAgent(
        name="SearchResearchAgent",
        description="Web research agent using DDGS search and image analysis capabilities",
        prompt_template=ChatTemplate(
            instruction="You are a professional web research assistant specializing in social media content research. Your goal is to gather comprehensive information about trending topics",
            context="You have access to search the web for current information. You can also analyze images to understand visual content trends. CRITICAL: You MUST use the ddgs_search tool first to gather comprehensive information before proceeding with any analysis. Never provide information without searching first. Always search for current trends, statistics, and recent developments related to the research topic.",
            constraints=[
                "CRITICAL: You MUST use the ddgs_search tool to search for current information",
                "Search for content specifically relevant to the target platform",
                "Always search for recent trends, statistics, and developments",
                "Never provide information without first searching the web"
            ],
            demonstrations=[]
        ),
        llm_config=llm_config,
        inputs=[
            {"name": "research_topic", "type": "string", "description": "The topic to research"},
            {"name": "platform", "type": "string", "description": "Target social media platform"}
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
                "Include engaging hooks and clear calls-to-action"
            ],
            demonstrations=[]
        ),
        llm_config=llm_config,
        inputs=[
            {"name": "research_info", "type": "string", "description": "Research information from previous step"},
            {"name": "user_prompt", "type": "string", "description": "User input prompt for content generation"}
        ],
        outputs=[
            {"name": "post_content", "type": "string", "description": "Generated social media post content"}
        ]
    )
    
    return content_agent

def create_image_generation_agent(save_path: str = "./social_media_output"):
    """
    Agent 3: Image Generation Agent
    基于内容生成配套的社交媒体图片
    
    Args:
        save_path: 保存路径，默认为"./social_media_output"
    """
    
    # 创建输出文件夹
    os.makedirs(save_path, exist_ok=True)
    
    # 创建图片生成工具
    image_gen_tool = FluxImageGenerationTool(
        api_key=BFL_API_KEY, 
        save_path=save_path
    )
    
    image_agent = CustomizeAgent(
        name="ImageGenerationAgent",
        description="Social media image creation specialist",
        prompt_template=ChatTemplate(
            instruction="You are a professional visual content creator specializing in social media imagery. Your role is to generate images using the flux_image_generation tool based on the provided descriptions and research information.",
            context="You understand visual trends across different social media platforms and know how to create images that stop the scroll and drive engagement. Your images should be optimized for mobile viewing and social media algorithms. CRITICAL: You MUST use the flux_image_generation tool to create actual images.",
            constraints=[
                "CRITICAL: You MUST use the flux_image_generation tool to create actual images",
                "Generate images based on the provided image descriptions and research information",
                "Create high-quality, engaging visuals optimized for social media",
                "Always call the flux_image_generation tool with appropriate prompts"
            ],
            demonstrations=[]
        ),
        llm_config=llm_config,
        inputs=[
            {"name": "research_info", "type": "string", "description": "Research information from search results"},
            {"name": "image_descriptions", "type": "string", "description": "Image descriptions from prompt analysis"},
        ],
        outputs=[
            {"name": "image_path", "type": "string", "description": "Path to generated image file"}
        ],
        tools=[image_gen_tool]
    )
    
    return image_agent

def create_social_media_workflow(save_path: str = "./social_media_output", load_from_file: str = None):
    """
    创建完整的社交媒体内容生成工作流
    包含四个Agent：Prompt Analysis -> Research -> Content Generation -> Image Generation
    
    Args:
        save_path: 保存路径，默认为"./social_media_output"
    """
    
    # 1. 创建四个Agent
    prompt_analysis_agent = create_prompt_analysis_agent()
    research_agent = create_research_agent()
    content_agent = create_content_generation_agent()
    image_agent = create_image_generation_agent(save_path)
    
    # 2. 创建工作流节点
    nodes = [
        WorkFlowNode(
            name="prompt_analysis",
            description="Analyze user prompt to determine research topic and image requirements",
            agents=[prompt_analysis_agent],
            inputs=[
                {"name": "user_prompt", "type": "string", "description": "User's original input", "required": True}
            ],
            outputs=[
                {"name": "research_topic", "type": "string", "description": "Research topic", "required": True},
                {"name": "image_count", "type": "integer", "description": "Number of images", "required": True},
                {"name": "image_descriptions", "type": "string", "description": "Image descriptions", "required": True},
                {"name": "user_prompt", "type": "string", "description": "Original user prompt", "required": True}
            ]
        ),
        WorkFlowNode(
            name="research",
            description="Research trending topics using DDGS search",
            agents=[research_agent],
            inputs=[
                {"name": "research_topic", "type": "string", "description": "Topic to research", "required": True},
                {"name": "platform", "type": "string", "description": "Target platform", "required": True}
            ],
            outputs=[
                {"name": "research_info", "type": "string", "description": "Research findings", "required": True}
            ]
        ),
        WorkFlowNode(
            name="content_generation",
            description="Generate social media content based on research",
            agents=[content_agent],
            inputs=[
                {"name": "research_info", "type": "string", "description": "Research information", "required": True},
                {"name": "user_prompt", "type": "string", "description": "User input prompt", "required": True}
            ],
            outputs=[
                {"name": "post_content", "type": "string", "description": "Generated content", "required": True}
            ]
        ),
        WorkFlowNode(
            name="image_generation",
            description="Generate images for social media content",
            agents=[image_agent],
            inputs=[
                {"name": "research_info", "type": "string", "description": "Research context", "required": True},
                {"name": "image_descriptions", "type": "string", "description": "Image descriptions", "required": True}
            ],
            outputs=[
                {"name": "image_path", "type": "string", "description": "Path to generated image", "required": True}
            ]
        )
    ]
    
    # 3. 定义工作流边（数据流向）
    edges = [
        # Prompt Analysis -> Research
        WorkFlowEdge(source="prompt_analysis", target="research"),
        # Prompt Analysis -> Content Generation (传递user_prompt)
        WorkFlowEdge(source="prompt_analysis", target="content_generation"),
        # Prompt Analysis -> Image Generation (传递image_descriptions)
        WorkFlowEdge(source="prompt_analysis", target="image_generation"),
        # Research -> Content Generation
        WorkFlowEdge(source="research", target="content_generation"),
        # Research -> Image Generation (为图片生成提供额外上下文)
        WorkFlowEdge(source="research", target="image_generation")
    ]
    
    # 4. 创建工作流图
    if load_from_file and os.path.exists(load_from_file):
        print(f"📂 从文件加载workflow: {load_from_file}")
        graph = WorkFlowGraph.from_file(load_from_file)
    else:
        graph = WorkFlowGraph(
            goal="Create social media content with prompt analysis, research, content generation, and image creation",
            nodes=nodes,
            edges=edges
        )
    
    # 5. 创建Agent Manager
    agents = [prompt_analysis_agent, research_agent, content_agent, image_agent]
    agent_manager = AgentManager(agents=agents)
    
    # 6. 创建完整工作流
    workflow = WorkFlow(
        graph=graph,
        llm= OpenAILLM(llm_config),
        agent_manager=agent_manager
    )
    
    # 保存workflow配置到JSON文件
    os.makedirs("examples/output", exist_ok=True)
    save_path = "examples/output/social_media_workflow.json"
    graph.save_module(save_path)
    print(f"✅ Workflow已保存到: {save_path}")
    
    return workflow

async def test_complete_workflow(save_path: str = "./social_media_output", load_workflow: bool = False):
    """
    测试完整的四节点工作流
    
    Args:
        save_path: 保存路径，默认为"./social_media_output"
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
    
    # 创建工作流
    if load_workflow:
        workflow_file = "examples/output/social_media_workflow.json"
        if os.path.exists(workflow_file):
            print(f"🔄 从保存的文件加载workflow: {workflow_file}")
            workflow = create_social_media_workflow(save_path, load_from_file=workflow_file)
        else:
            print(f"⚠️  未找到保存的workflow文件: {workflow_file}")
            print("📝 创建新的workflow...")
            workflow = create_social_media_workflow(save_path)
    else:
        workflow = create_social_media_workflow(save_path)
        print("✅ 完整工作流创建成功!")
        print(f"📁 保存路径: {save_path}")
        
        # 测试案例
        test_cases = [
            {
                "user_prompt": "Create a professional post about AI productivity tools for LinkedIn with 2 images. Search for current trends and statistics about AI adoption in the workplace, productivity improvements, and popular AI tools used by professionals. Generate images showing professionals using AI tools, productivity dashboards, and modern workplace technology.",
                "platform": "LinkedIn"
            },
            {
                "user_prompt": "Create a casual post about healthy morning routines for Instagram with 1 image. Search for current wellness trends, morning routine statistics, and popular health practices. Generate an image showing a peaceful morning routine with healthy breakfast, meditation, and wellness activities.",
                "platform": "Instagram"
            },
            {
                "user_prompt": "Create a humorous post about remote work tips for Twitter with 2 images. Search for funny remote work anecdotes, common work-from-home challenges, and viral remote work memes. Generate images showing humorous remote work situations, pets interrupting meetings, and creative home office setups.",
                "platform": "Twitter"
            },
            {
                "user_prompt": "Create an informative post about sustainable living practices for Facebook with 3 images. Search for current trends and statistics about eco-friendly lifestyle choices, renewable energy adoption, and zero-waste living. Generate images showing sustainable home practices, green energy solutions, and eco-friendly daily routines.",
                "platform": "Facebook"
            }
        ]
        
        print("\n🎯 可选测试案例：")
        print("=" * 60)
        for i, case in enumerate(test_cases, 1):
            print(f"\n{i}. 平台: {case['platform']}")
            print(f"   主题: {case['user_prompt'][:100]}...")
            print(f"   完整描述:")
            
            # 将长文本按行分割并添加缩进
            wrapped_text = textwrap.fill(case['user_prompt'], width=70, initial_indent="      ", subsequent_indent="      ")
            print(wrapped_text)
            print("-" * 60)
        
        print("\n请选择测试案例（输入数字1-4）:")
        try:
            choice = int(input().strip())
            if 1 <= choice <= len(test_cases):
                selected_case = test_cases[choice - 1]
                print(f"\n🎯 执行案例: {selected_case['user_prompt']}")
                print(f"   平台: {selected_case['platform']}")
                
                # 执行完整工作流
                result = await workflow.async_execute(
                    inputs=selected_case,
                    task_name="social media content creation",
                    goal="Create social media content with prompt analysis, research, content generation, and image creation"
                )
                
                print("\n" + "="*60)
                print("🎉 工作流执行完成！")
                print("="*60)
                
                print("\n📋 完整结果:")
                print("-" * 40)
                print(result)
                
                # 保存内容到文件
                try:
                    # 创建保存目录
                    os.makedirs(save_path, exist_ok=True)
                    
                    # 生成文件名（基于时间戳）
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    content_filename = f"social_media_content_{timestamp}.txt"
                    content_filepath = os.path.join(save_path, content_filename)
                    
                    # 保存内容
                    with open(content_filepath, "w", encoding="utf-8") as f:
                        f.write("=== 社交媒体内容生成结果 ===\n")
                        f.write(f"生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                        f.write(f"测试案例: {selected_case['user_prompt']}\n")
                        f.write(f"目标平台: {selected_case['platform']}\n")
                        f.write("\n" + "="*50 + "\n")
                        f.write(result)
                    
                    print(f"\n💾 内容已保存到: {content_filepath}")
                    
                except Exception as e:
                    print(f"⚠️ 保存内容时出错: {e}")
                
                return result
                
            else:
                print("❌ 无效选择")
                
        except (ValueError, KeyboardInterrupt):
            print("\n💡 你也可以直接运行:")
            print('result = workflow.run(user_prompt="your prompt", platform="LinkedIn")')
                    
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
    
    try:
        agent = create_content_generation_agent()
        
        # 模拟更真实的research_info和prompt输入
        research_info = """
        AI Productivity Tools Market Analysis 2024:
        
        Key Findings:
        - 73% of professionals now use AI tools daily for productivity
        - Average time savings: 40% faster task completion
        - Top AI tools: ChatGPT (45%), Notion AI (32%), Grammarly (28%)
        - 67% report reduced burnout levels with AI assistance
        - 89% say AI enhances creativity and problem-solving
        
        Trending Topics:
        - AI-powered project management tools
        - Automated content creation platforms
        - Smart calendar and scheduling assistants
        - AI-driven data analysis tools
        
        Platform-Specific Insights:
        - LinkedIn: Professional development and career growth focus
        - High engagement with thought leadership content
        - Preference for data-driven insights and actionable tips
        """
        
        prompt = "Create a professional LinkedIn post about AI productivity tools that drives engagement and positions the author as a thought leader"
        
        result = agent(inputs={
            "research_info": research_info,
            "prompt": prompt
        })
        
        print("\n📋 Content Generation Agent 测试结果:")
        print("=" * 50)
        print("输入 research_info:")
        print("-" * 30)
        print(research_info[:200] + "...")
        print("\n输入 prompt:")
        print("-" * 30)
        print(prompt)
        print("\n输出 post_content:")
        print("-" * 30)
        print(result.content.post_content)
        
    except Exception as e:
        print(f"❌ 直接测试失败: {e}")

def test_prompt_analysis_agent():
    """
    测试Prompt Analysis Agent
    """
    
    try:
        agent = create_prompt_analysis_agent()
        
        test_prompts = [
            "Create a professional post about AI productivity tools for LinkedIn",
            "Create a casual post about healthy morning routines for Instagram with 2 images",
            "Create a humorous post about remote work tips for Twitter"
        ]
        
        for i, prompt in enumerate(test_prompts, 1):
            print(f"\n🧪 测试案例 {i}: {prompt}")
            print("-" * 50)
            
            result = agent(inputs={"user_prompt": prompt})
            
            print(f"Research Topic: {result.content.research_topic}")
            print(f"Image Count: {result.content.image_count}")
            print(f"Image Descriptions: {result.content.image_descriptions}")
            print()
            
    except Exception as e:
        print(f"❌ Prompt Analysis Agent 测试失败: {e}")

def test_tools():
    """
    测试工具是否正常工作
    """
    print("🧪 测试工具功能")
    print("=" * 50)
    
    # 测试DDGS搜索工具
    print("1. 测试DDGS搜索工具...")
    try:
        ddgs_toolkit = DDGSSearchToolkit(
            name="DDGSSearchToolkit",
            num_search_pages=2,
            max_content_words=200,
            backend="auto",
            region="us-en"
        )
        search_tools = ddgs_toolkit.get_tools()
        if search_tools:
            search_tool = search_tools[0]
            result = search_tool(query="AI productivity tools", num_search_pages=1)
            print(f"✅ DDGS搜索工具测试成功")
            print(f"搜索结果数量: {len(result.get('results', []))}")
            if result.get('error'):
                print(f"⚠️ 搜索错误: {result['error']}")
        else:
            print("❌ 没有找到DDGS搜索工具")
    except Exception as e:
        print(f"❌ DDGS搜索工具测试失败: {e}")
    
    # 测试图片生成工具
    print("\n2. 测试图片生成工具...")
    try:
        if not BFL_API_KEY:
            print("❌ BFL_API_KEY未配置，跳过图片生成测试")
        else:
            image_gen_tool = FluxImageGenerationTool(
                api_key=BFL_API_KEY, 
                save_path="./test_output"
            )
            result = image_gen_tool(prompt="A simple test image", aspect_ratio="1:1")
            print(f"✅ 图片生成工具测试成功")
            print(f"生成图片路径: {result.get('file_path', 'N/A')}")
    except Exception as e:
        print(f"❌ 图片生成工具测试失败: {e}")

if __name__ == "__main__":
    """
    🔧 配置说明：
    
    完整的四节点社交媒体内容工作流
    
    1. 必需环境变量：
       - OPENAI_API_KEY: OpenAI API密钥 (用于LLM)
       - BFL_API_KEY: Black Forest Labs API密钥 (用于图片生成)
    
    2. 可选环境变量：
       - OPENROUTER_API_KEY: OpenRouter API密钥 (用于图片分析，可选)
    
    3. 依赖安装：
       - pip install selenium (浏览器自动化)
       - pip install Pillow (用于图片显示)
       - pip install ddgs (用于搜索功能)
       - 需要安装Chrome浏览器
    
    4. 工作流节点：
       - Node 0: Prompt Analysis Agent (解析用户输入)
       - Node 1: Research Agent (DDGS搜索 + 可选图片分析)
       - Node 2: Content Generation Agent (创建推文内容)
       - Node 3: Image Generation Agent (生成配图)
    
    5. 数据流向：
       Prompt Analysis -> Research -> Content Generation
       Research -> Image Generation (提供额外上下文)
    """
    
    print("📱 完整社交媒体内容工作流")
    print("四节点工作流：Prompt Analysis -> Research -> Content -> Image")
    print("=" * 60)
    
    # 测试工具功能
    # test_tools()
    
    # 运行完整工作流测试
    # 可以通过修改这个路径来自定义保存位置
    save_path = "./social_media_output"
    
    # 选择运行模式
    print("\n🎯 选择运行模式:")
    print("1. 创建新的workflow (默认)")
    print("2. 从保存的文件加载workflow")
    print("3. 测试工具功能")
    print("4. 测试Prompt Analysis Agent")
    print("5. 测试Content Generation Agent")
    
    try:
        choice = input("\n请输入选择 (1-5，默认为1): ").strip()
        if not choice:
            choice = "1"
        
        if choice == "1":
            print("\n🚀 创建新的workflow...")
            asyncio.run(test_complete_workflow(save_path, load_workflow=False))
        elif choice == "2":
            print("\n🔄 从保存的文件加载workflow...")
            asyncio.run(test_complete_workflow(save_path, load_workflow=True))
        elif choice == "3":
            print("\n🧪 测试工具功能...")
            test_tools()
        elif choice == "4":
            print("\n🧪 测试Prompt Analysis Agent...")
            test_prompt_analysis_agent()
        elif choice == "5":
            print("\n🧪 测试Content Generation Agent...")
            test_direct_agent()
        else:
            print("❌ 无效选择，使用默认模式")
            asyncio.run(test_complete_workflow(save_path, load_workflow=False))
            
    except KeyboardInterrupt:
        print("\n👋 程序已退出")
    except Exception as e:
        print(f"❌ 运行出错: {e}")
    
    # 测试Prompt Analysis Agent
    # test_prompt_analysis_agent()
    
    # 测试Content Generation Agent
    # test_direct_agent() 