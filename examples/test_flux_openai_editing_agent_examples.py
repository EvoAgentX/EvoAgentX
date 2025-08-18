#!/usr/bin/env python3
"""
测试FluxOpenAIEditingActionAgent的简单例子
"""

import os
import json
from dotenv import load_dotenv

# 导入我们的action agent
from flux_openai_editing_action_agent import create_flux_openai_editing_agent

load_dotenv()

def test_single_image_with_addon():
    """测试单张图片带文字添加"""
    print("=== 测试1: 单张图片带文字添加 ===")
    
    # 创建agent
    agent = create_flux_openai_editing_agent(save_path="./test_output/single_with_addon")
    
    # 示例输入
    example_input = {
        "image_1": {
            "description": "一只可爱的小猫坐在窗台上，阳光透过窗户洒在它身上，温暖的自然光，高清摄影风格",
            "add_on": "在图片顶部添加标题：'阳光下的猫咪'，使用白色字体，粗体，居中显示，字体大小占图片高度的10%"
        }
    }
    
    print(f"输入: {json.dumps(example_input, ensure_ascii=False, indent=2)}")
    
    # 执行agent
    try:
        result = agent.execute(
            action_name=agent.main_action_name,
            action_input_data={"image_descriptions_json": json.dumps(example_input)}
        )
        
        print("✅ 执行成功!")
        print(f"结果: {result.content}")
        
        # 解析结果
        if isinstance(result.content, dict) and "image_paths_json" in result.content:
            image_paths = json.loads(result.content["image_paths_json"])
            for image_key, image_info in image_paths.items():
                print(f"\n{image_key}:")
                print(f"  生成图片路径: {image_info.get('generated_image_path', 'N/A')}")
                print(f"  编辑后图片路径: {image_info.get('edited_image_path', 'N/A')}")
        
    except Exception as e:
        print(f"❌ 执行失败: {str(e)}")

def test_single_image_without_addon():
    """测试单张图片不带文字添加"""
    print("\n=== 测试2: 单张图片不带文字添加 ===")
    
    # 创建agent
    agent = create_flux_openai_editing_agent(save_path="./test_output/single_without_addon")
    
    # 示例输入
    example_input = {
        "image_1": {
            "description": "一片美丽的向日葵花田，金黄色的花朵在阳光下绽放，蓝天白云背景，自然风光摄影风格",
            "add_on": ""  # 空字符串，不添加文字
        }
    }
    
    print(f"输入: {json.dumps(example_input, ensure_ascii=False, indent=2)}")
    
    # 执行agent
    try:
        result = agent.execute(
            action_name=agent.main_action_name,
            action_input_data={"image_descriptions_json": json.dumps(example_input)}
        )
        
        print("✅ 执行成功!")
        print(f"结果: {result.content}")
        
        # 解析结果
        if isinstance(result.content, dict) and "image_paths_json" in result.content:
            image_paths = json.loads(result.content["image_paths_json"])
            for image_key, image_info in image_paths.items():
                print(f"\n{image_key}:")
                print(f"  生成图片路径: {image_info.get('generated_image_path', 'N/A')}")
                print(f"  编辑后图片路径: {image_info.get('edited_image_path', 'N/A')}")
        
    except Exception as e:
        print(f"❌ 执行失败: {str(e)}")

def test_multiple_images_mixed():
    """测试多张图片，部分带文字部分不带"""
    print("\n=== 测试3: 多张图片混合测试 ===")
    
    # 创建agent
    agent = create_flux_openai_editing_agent(save_path="./test_output/multiple_mixed")
    
    # 示例输入
    example_input = {
        "image_1": {
            "description": "一杯热咖啡，上面有漂亮的拉花，放在木质桌面上，温暖的光线，美食摄影风格",
            "add_on": "在图片底部添加文字：'美好的一天从咖啡开始'，使用深棕色字体，中等粗细，居中显示，字体大小占图片高度的8%"
        },
        "image_2": {
            "description": "一座古老的石桥横跨在小溪上，周围是绿色的森林，自然风光，风景摄影风格",
            "add_on": ""  # 不添加文字
        },
        "image_3": {
            "description": "一本打开的书，旁边放着一副眼镜和一杯茶，温馨的阅读场景，生活摄影风格",
            "add_on": "在图片右上角添加小标签：'阅读时光'，使用黑色字体，细体，右对齐，字体大小占图片高度的5%"
        }
    }
    
    print(f"输入: {json.dumps(example_input, ensure_ascii=False, indent=2)}")
    
    # 执行agent
    try:
        result = agent.execute(
            action_name=agent.main_action_name,
            action_input_data={"image_descriptions_json": json.dumps(example_input)}
        )
        
        print("✅ 执行成功!")
        print(f"结果: {result.content}")
        
        # 解析结果
        if isinstance(result.content, dict) and "image_paths_json" in result.content:
            image_paths = json.loads(result.content["image_paths_json"])
            for image_key, image_info in image_paths.items():
                print(f"\n{image_key}:")
                print(f"  生成图片路径: {image_info.get('generated_image_path', 'N/A')}")
                print(f"  编辑后图片路径: {image_info.get('edited_image_path', 'N/A')}")
                print(f"  描述: {image_info.get('description', 'N/A')[:50]}...")
                print(f"  添加内容: {image_info.get('add_on', 'N/A')[:30] if image_info.get('add_on') else '无'}")
        
    except Exception as e:
        print(f"❌ 执行失败: {str(e)}")

def test_multiple_images_all_with_addon():
    """测试多张图片都带文字添加"""
    print("\n=== 测试4: 多张图片都带文字添加 ===")
    
    # 创建agent
    agent = create_flux_openai_editing_agent(save_path="./test_output/multiple_all_addon")
    
    # 示例输入
    example_input = {
        "image_1": {
            "description": "一个现代化的办公桌，上面有笔记本电脑、咖啡杯和绿植，专业商务风格",
            "add_on": "在图片顶部添加标题：'高效办公'，使用深蓝色字体，粗体，居中显示，字体大小占图片高度的12%"
        },
        "image_2": {
            "description": "一个创意工作台，上面有各种设计工具、颜料和画布，艺术创作风格",
            "add_on": "在图片中央添加文字：'创意无限'，使用橙色字体，粗体，居中显示，字体大小占图片高度的15%"
        }
    }
    
    print(f"输入: {json.dumps(example_input, ensure_ascii=False, indent=2)}")
    
    # 执行agent
    try:
        result = agent.execute(
            action_name=agent.main_action_name,
            action_input_data={"image_descriptions_json": json.dumps(example_input)}
        )
        
        print("✅ 执行成功!")
        print(f"结果: {result.content}")
        
        # 解析结果
        if isinstance(result.content, dict) and "image_paths_json" in result.content:
            image_paths = json.loads(result.content["image_paths_json"])
            for image_key, image_info in image_paths.items():
                print(f"\n{image_key}:")
                print(f"  生成图片路径: {image_info.get('generated_image_path', 'N/A')}")
                print(f"  编辑后图片路径: {image_info.get('edited_image_path', 'N/A')}")
        
    except Exception as e:
        print(f"❌ 执行失败: {str(e)}")

if __name__ == "__main__":
    print("开始测试FluxOpenAIEditingActionAgent...")
    print("=" * 50)
    
    # 检查环境变量
    required_env_vars = ["OPENAI_API_KEY", "OPENAI_ORGANIZATION_ID", "BFL_API_KEY"]
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"❌ 缺少必要的环境变量: {missing_vars}")
        print("请设置以下环境变量:")
        for var in missing_vars:
            print(f"  - {var}")
        exit(1)
    
    # 运行测试
    test_single_image_with_addon()
    test_single_image_without_addon()
    test_multiple_images_mixed()
    test_multiple_images_all_with_addon()
    
    print("\n" + "=" * 50)
    print("所有测试完成!")
