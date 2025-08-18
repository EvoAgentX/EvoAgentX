import os
import asyncio
import textwrap
import datetime
from dotenv import load_dotenv
from evoagentx.models.openai_model import OpenAILLM
from evoagentx.models.model_configs import LLMConfig
from evoagentx.workflow import WorkFlow, WorkFlowGraph
from evoagentx.workflow.workflow_graph import WorkFlowNode, WorkFlowEdge
from evoagentx.agents import CustomizeAgent, AgentManager, ActionAgent
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
OPENAI_ORGANIZATION_ID = os.getenv("OPENAI_ORGANIZATION_ID")

# 配置LLM
llm_config = OpenAILLMConfig(
    model="gpt-4.1", 
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
        description="分析用户提示词，确定研究主题和图片生成需求",
        prompt_template=ChatTemplate(
            instruction="你是一个专业的提示词分析专家，负责将用户请求分解为结构化的研究和内容生成任务。你必须返回一个有效的JSON格式字符串，包含图片描述和图片上的文字内容。请用中文回复所有内容。",
            context="你分析用户提示词来确定：1) 主要研究主题，2) 需要生成多少张图片，3) 每张图片应该包含什么内容，4) 每张图片上需要添加什么文字内容。你还应该保留原始用户提示词供后续处理使用。",
            constraints=[
                "必须返回有效的JSON格式字符串",
                "JSON结构必须严格按照指定格式：{\"image_1\": {\"description\": \"图片描述\", \"add_on\": \"图片上的文字内容\"}, \"image_2\": {...}}",
                "image_descriptions_json字段必须是一个可以直接被json.loads()解析的字符串，而不是dict对象",
                "请严格按照json.dumps的格式输出image_descriptions_json：最外层必须用英文双引号包裹，内部key和value也都用英文双引号",
                "不要输出Python风格的字典对象或数组，不要省略引号",
                "图片编号从image_1开始递增",
                "description字段包含图片的主要内容描述（场景、风格、色调、构图等）",
                "add_on字段专门包含需要在图片上添加的文字内容（标题、标语、关键词等），必须详细说明：1) 文字内容，2) 字体颜色，3) 字体粗细，4) 对齐方式，5) 字体大小（占图片高度的百分比），6) 文字位置（顶部、中央、底部、左上角、右下角等）",
                "始终提供清晰的研究主题总结",
                "所有输出内容必须使用中文",
                "JSON字符串必须是有效的，可以被json.loads()解析",
                "不要使用markdown代码块格式，直接返回纯JSON字符串",
                "不要添加```json或```标记"
            ],
            demonstrations=[
                {
                    "input": "为小红书创建一个关于AI生产力工具的专业帖子，包含2张图片",
                    "output": {
                        "research_topic": "AI生产力工具在小红书平台的专业内容创作",
                        "image_descriptions_json": "{\"image_1\": {\"description\": \"专业人士在现代化办公环境中使用AI工具的场景，包括智能助手、自动化工作流程和数据分析界面，专业商务风格，明亮色调，俯视角度构图，体现科技感和效率\", \"add_on\": \"在图片上方添加大标题：'AI生产力革命'，使用白色字体，粗体，居中显示，字体大小占图片高度的15%。在图片下方添加副标题：'提升工作效率 | 智能办公新时代'，使用浅蓝色字体，中等粗细，居中显示，字体大小占图片高度的8%。在图片右下角添加标签：'#AI工具 #效率提升'，使用白色字体，细体，右对齐，字体大小占图片高度的5%\"}, \"image_2\": {\"description\": \"一个现代化的办公桌，上面有笔记本电脑、咖啡杯、绿植和整洁的文件，体现高效办公环境，自然光线，温暖色调，45度角度构图，突出工作氛围\", \"add_on\": \"\"}}",
                        "user_prompt": "为小红书创建一个关于AI生产力工具的专业帖子，包含2张图片"
                    }
                },
                {
                    "input": "为小红书创建一个关于健康晨间习惯的轻松帖子，包含1张图片",
                    "output": {
                        "research_topic": "健康晨间习惯在小红书平台的生活方式内容创作",
                        "image_descriptions_json": "{\"image_1\": {\"description\": \"宁静的晨间场景，包含营养丰富的早餐（如燕麦粥、新鲜水果、坚果）、冥想角落（瑜伽垫、香薰蜡烛、绿植）、以及健康活动（如晨跑装备、水杯、维生素），温暖自然风格，柔和色调，45度角度构图，体现生活品质和健康理念\", \"add_on\": \"在图片顶部添加主标题：'健康晨间习惯'，使用深绿色字体，粗体，居中显示，字体大小占图片高度的12%。在图片中央偏上添加副标题：'美好生活从早晨开始'，使用浅棕色字体，中等粗细，居中显示，字体大小占图片高度的8%。在图片右下角添加标签：'#健康生活 #晨间routine'，使用白色字体，细体，右对齐，字体大小占图片高度的5%。在图片左下角添加小提示：'每日坚持，改变生活'，使用浅灰色字体，细体，左对齐，字体大小占图片高度的4%\"}}",
                        "user_prompt": "为小红书创建一个关于健康晨间习惯的轻松帖子，包含1张图片"
                    }
                },
                {
                    "input": "为小红书创建一个关于远程工作技巧的幽默帖子，包含3张图片",
                    "output": {
                        "research_topic": "远程工作技巧在小红书平台的职场内容创作",
                        "image_descriptions_json": "{\"image_1\": {\"description\": \"宠物打断视频会议的搞笑场景，包括猫咪坐在键盘上、狗狗好奇地看着屏幕、以及主人无奈的表情，轻松幽默风格，温暖色调，平视角度构图，体现居家办公的真实场景\", \"add_on\": \"在图片顶部添加标题：'远程工作的真实写照'，使用橙色字体，粗体，居中显示，字体大小占图片高度的10%。在图片中央添加幽默文字：'宠物也是同事'，使用深蓝色字体，中等粗细，居中显示，字体大小占图片高度的8%。在图片右下角添加标签：'#远程工作 #居家办公'，使用白色字体，细体，右对齐，字体大小占图片高度的5%。在图片左上角添加表情符号：'😸🐕'，使用黑色字体，粗体，左对齐，字体大小占图片高度的6%\"}, \"image_2\": {\"description\": \"创意居家办公设置，包括多屏显示器、人体工学椅、绿植装饰、以及咖啡杯等生活化元素，现代简约风格，中性色调，45度角度构图，突出工作效率和舒适性\", \"add_on\": \"\"}, \"image_3\": {\"description\": \"远程工作的时间管理工具和技巧展示，包括日程表、番茄钟、任务清单等，实用工具风格，蓝色科技色调，俯视角度构图，体现时间管理的重要性\", \"add_on\": \"在图片顶部添加标题：'时间管理大师'，使用深蓝色字体，粗体，居中显示，字体大小占图片高度的10%。在图片中央添加副标题：'高效工作技巧'，使用浅蓝色字体，中等粗细，居中显示，字体大小占图片高度的7%。在图片右下角添加标签：'#时间管理 #工作效率'，使用白色字体，细体，右对齐，字体大小占图片高度的5%。在图片左下角添加小提示：'25分钟专注，5分钟休息'，使用浅灰色字体，细体，左对齐，字体大小占图片高度的4%\"}}",
                        "user_prompt": "为小红书创建一个关于远程工作技巧的幽默帖子，包含3张图片"
                    }
                }
            ]
        ),
        llm_config=llm_config,
        inputs=[
            {"name": "user_prompt", "type": "string", "description": "用户的原始输入提示词"}
        ],
        outputs=[
            {"name": "research_topic", "type": "string", "description": "总结的研究主题"},
            {"name": "image_descriptions_json", "type": "string", "description": "JSON格式的图片描述和文字内容"},
            {"name": "user_prompt", "type": "string", "description": "原始用户提示词"}
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
        region="cn-zh"
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
        description="使用DDGS搜索和图片分析功能的网络研究代理",
        prompt_template=ChatTemplate(
            instruction="你是一个专业的网络研究助手，专门从事社交媒体内容研究。你的目标是收集关于热门话题的全面信息。请用中文回复所有内容。",
            context="你可以搜索网络获取当前信息，也可以分析图片来了解视觉内容趋势。你可以根据研究主题和可用信息来决定是否使用搜索工具。对于需要当前数据、趋势或最新发展的主题，你应该使用ddgs_search工具。对于你有足够知识或不需要实时信息的主题，你可以直接进行而不搜索。",
            constraints=[
                "需要当前信息或趋势时使用搜索工具",
                "无论是否搜索都要提供准确信息",
                "所有输出内容必须使用中文",
                "不要返回research topic的内容"
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
        description="社交媒体内容创作专家",
        prompt_template=ChatTemplate(
            instruction="你是一个专业的社交媒体内容创作者，负责将研究洞察转化为引人入胜、具有传播价值的内容。请用中文创作所有内容。",
            context="你专门创作针对特定平台的内容，以提升用户参与度。你了解当前社交媒体趋势、算法偏好和受众心理。",
            constraints=[
                "内容必须原创且真实",
                "符合指定的风格和平台要求",
                "包含引人入胜的开头和清晰的行动号召",
                "所有输出内容必须使用中文"
            ],
            demonstrations=[]
        ),
        llm_config=llm_config,
        inputs=[
            {"name": "research_info", "type": "string", "description": "Research information from previous step"},
        ],
        outputs=[
            {"name": "post_content", "type": "string", "description": "Generated social media post content"}
        ]
    )
    
    return content_agent

def create_image_generation_agent(save_path: str = "./social_media_output"):
    """
    Agent 3: Image Generation Agent (使用ActionAgent实现FluxOpenAIEditingActionAgent)
    基于内容生成配套的社交媒体图片，支持Flux生成和OpenAI文字编辑
    
    Args:
        save_path: 保存路径，默认为"./social_media_output"
    """
    
    def execute_flux_openai_editing(image_descriptions_json: str) -> dict:
        """
        执行Flux to OpenAI Editing流程
        
        Args:
            image_descriptions_json: JSON格式的图片描述，来自xhs workflow的analyze agent
            
        Returns:
            dict: 包含图片路径信息的字典
        """
        import json
        import base64
        
        # 验证环境变量
        if not all([OPENAI_API_KEY, OPENAI_ORGANIZATION_ID, BFL_API_KEY]):
            raise ValueError("请设置OPENAI_API_KEY、OPENAI_ORGANIZATION_ID和BFL_API_KEY环境变量")
        
        # 创建保存目录
        os.makedirs(save_path, exist_ok=True)
        
        try:
            # 解析JSON输入 - 使用更详细的解析方式
            if isinstance(image_descriptions_json, str):
                # 处理可能包含markdown代码块的JSON字符串
                json_string = image_descriptions_json.strip()
                
                # 如果包含markdown代码块，提取其中的JSON内容
                if json_string.startswith('```'):
                    # 移除markdown代码块标记
                    lines = json_string.split('\n')
                    json_lines = []
                    in_json = False
                    for line in lines:
                        if line.strip().startswith('```'):
                            if not in_json:
                                in_json = True
                            else:
                                break
                        elif in_json:
                            json_lines.append(line)
                    json_string = '\n'.join(json_lines)
                
                # 解析JSON
                try:
                    image_descriptions = json.loads(json_string)
                except json.JSONDecodeError as e:
                    raise ValueError(f"JSON解析失败: {str(e)}，原始内容: {image_descriptions_json[:200]}...")
            else:
                image_descriptions = image_descriptions_json
            
            # 验证JSON结构
            if not isinstance(image_descriptions, dict):
                raise ValueError("image_descriptions_json必须是包含图片描述的字典")
            
            result = {}
            
            # 处理每个图片描述
            for image_key, image_info in image_descriptions.items():
                if not isinstance(image_info, dict):
                    print(f"警告: {image_key} 不是有效的字典格式，跳过")
                    continue
                
                description = image_info.get("description", "")
                add_on = image_info.get("add_on", "")
                
                if not description:
                    print(f"警告: {image_key} 缺少description字段")
                    continue
                
                print(f"开始处理 {image_key}: {description}")
                
                # 使用Flux生成图片
                flux_tool = FluxImageGenerationTool(api_key=BFL_API_KEY, save_path=save_path)
                flux_result = flux_tool(prompt=description)
                generated_image_path = flux_result.get("file_path")
                
                if not generated_image_path or not os.path.exists(generated_image_path):
                    raise Exception(f"Flux图片生成失败: {image_key}")
                
                print(f"Flux生成的图片已保存到: {generated_image_path}")
                
                # 如果有add_on信息，使用OpenAI添加文字
                if add_on:
                    from openai import OpenAI
                    
                    # 创建 OpenAI 客户端
                    client = OpenAI(api_key=OPENAI_API_KEY, organization=OPENAI_ORGANIZATION_ID)
                    
                    # 编辑图片
                    response = client.images.edit(
                        model="gpt-image-1",
                        image=open(generated_image_path, "rb"),
                        prompt=add_on
                    )
                    
                    # 设置输出文件名
                    output_name = f"{image_key}_edited.jpeg"
                    edited_image_path = os.path.join(save_path, output_name)
                    
                    # 保存编辑后的图片
                    image_base64 = response.data[0].b64_json
                    image_bytes = base64.b64decode(image_base64)
                    
                    with open(edited_image_path, "wb") as f:
                        f.write(image_bytes)
                    
                    print(f"✅ OpenAI添加文字完成！保存在: {edited_image_path}")
                    
                    result[image_key] = {
                        "generated_image_path": generated_image_path,
                        "edited_image_path": edited_image_path,
                        "description": description,
                        "add_on": add_on
                    }
                else:
                    result[image_key] = {
                        "generated_image_path": generated_image_path,
                        "edited_image_path": generated_image_path,  # 没有编辑时，使用原图路径
                        "description": description,
                        "add_on": ""
                    }
            
            return {"image_paths_json": json.dumps(result, ensure_ascii=False, indent=2)}
            
        except Exception as e:
            print(f"执行Flux to OpenAI Editing时发生错误: {str(e)}")
            error_result = {
                "error": f"执行Flux to OpenAI Editing时发生错误: {str(e)}",
                "status": "failed"
            }
            return {"image_paths_json": json.dumps(error_result, ensure_ascii=False, indent=2)}
    
    return ActionAgent(
        name="FluxOpenAIEditingAgent",
        description="基于Flux生成图片并使用OpenAI添加文字的Action Agent，专门处理xhs workflow中的图片生成需求",
        inputs=[
            {
                "name": "image_descriptions_json",
                "type": "string",
                "description": "JSON格式的图片描述，来自xhs workflow的analyze agent，包含图片描述和需要添加的文字内容",
                "required": True
            }
        ],
        outputs=[
            {
                "name": "image_paths_json",
                "type": "string",
                "description": "JSON格式的图片路径信息，包含每个图片的生成路径和编辑后的路径",
                "required": True
            }
        ],
        execute_func=execute_flux_openai_editing
    )

def create_post_content_writer_agent(save_path: str = "./social_media_output"):
    """
    Agent 4: Post Content Writer Agent (使用ActionAgent)
    负责将生成的社交媒体内容写入文件
    
    Args:
        save_path: 保存路径，默认为"./social_media_output"
    """
    
    def execute_post_content_writer(post_content: str) -> dict:
        """
        执行post content写入功能
        
        Args:
            post_content: 要写入的post content
            
        Returns:
            dict: 包含写入结果的字典
        """
        import json
        
        try:
            # 创建保存目录
            os.makedirs(save_path, exist_ok=True)
            
            # 生成文件名（基于时间戳）
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"post_content_{timestamp}.txt"
            filepath = os.path.join(save_path, filename)
            
            # 写入文件
            with open(filepath, "w", encoding="utf-8") as f:
                f.write("=== 社交媒体内容 ===\n")
                f.write(f"生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("\n" + "="*50 + "\n")
                f.write(post_content)
                f.write("\n\n" + "="*50 + "\n")
                f.write("内容生成完成")
            
            result = {
                "status": "success",
                "file_path": filepath,
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "content_length": len(post_content)
            }
            
            print(f"✅ Post content已保存到: {filepath}")
            return {"post_content_result": json.dumps(result, ensure_ascii=False, indent=2)}
            
        except Exception as e:
            print(f"❌ 写入post content时发生错误: {str(e)}")
            error_result = {
                "status": "failed",
                "error": f"写入post content时发生错误: {str(e)}",
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            return {"post_content_result": json.dumps(error_result, ensure_ascii=False, indent=2)}
    
    return ActionAgent(
        name="PostContentWriterAgent",
        description="专门负责接收post content并写入文件的Action Agent",
        inputs=[
            {
                "name": "post_content",
                "type": "string",
                "description": "要写入的post content内容",
                "required": True
            }
        ],
        outputs=[
            {
                "name": "post_content_result",
                "type": "string",
                "description": "JSON格式的写入结果信息，包含文件路径和状态",
                "required": True
            }
        ],
        execute_func=execute_post_content_writer
    )

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
    post_writer_agent = create_post_content_writer_agent(save_path)
    
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
                {"name": "image_descriptions_json", "type": "string", "description": "JSON格式的图片描述和附加信息", "required": True},
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
            description="Generate images for social media content using Flux and OpenAI editing",
            agents=[image_agent],
            inputs=[
                {"name": "image_descriptions_json", "type": "string", "description": "JSON格式的图片描述，来自prompt_analysis节点", "required": True}
            ],
            outputs=[
                {"name": "image_paths_json", "type": "string", "description": "JSON格式的图片路径信息，包含每个图片的生成路径和编辑后的路径", "required": True}
            ]
        ),
        WorkFlowNode(
            name="post_content_writer",
            description="Write post content to file",
            agents=[post_writer_agent],
            inputs=[
                {"name": "post_content", "type": "string", "description": "Generated post content from content_generation node", "required": True}
            ],
            outputs=[
                {"name": "post_content_result", "type": "string", "description": "JSON格式的写入结果信息", "required": True}
            ]
        )
    ]
    
    # 3. 定义工作流边（数据流向）
    edges = [
        # Prompt Analysis -> Research
        WorkFlowEdge(source="prompt_analysis", target="research"),
        # Prompt Analysis -> Content Generation (传递user_prompt)
        WorkFlowEdge(source="prompt_analysis", target="content_generation"),
        # Prompt Analysis -> Image Generation (传递image_descriptions_json)
        WorkFlowEdge(source="prompt_analysis", target="image_generation"),
        # Research -> Content Generation
        WorkFlowEdge(source="research", target="content_generation"),
        # Content Generation -> Post Content Writer
        WorkFlowEdge(source="content_generation", target="post_content_writer")
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
    agents = [prompt_analysis_agent, research_agent, content_agent, image_agent, post_writer_agent]
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
