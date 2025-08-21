#!/usr/bin/env python3
"""
测试更新后的XHS Workflow（集成FluxOpenAIEditingActionAgent）
"""

import os
import json
import asyncio
from dotenv import load_dotenv

# 导入更新后的workflow
from examples.xhs_workflow import create_social_media_workflow

load_dotenv()

async def test_updated_workflow():
    """测试更新后的workflow"""
    print("🚀 测试更新后的XHS Workflow（集成FluxOpenAIEditingActionAgent）")
    print("=" * 80)
    
    # 检查环境变量
    required_env_vars = ["OPENAI_API_KEY", "OPENAI_ORGANIZATION_ID", "BFL_API_KEY"]
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"❌ 缺少必要的环境变量: {missing_vars}")
        print("请设置以下环境变量:")
        for var in missing_vars:
            print(f"  - {var}")
        return
    
    # 创建workflow
    save_path = "./test_updated_workflow_output"
    workflow = create_social_media_workflow(save_path=save_path)
    
    print("✅ Workflow创建成功!")
    print(f"📁 保存路径: {save_path}")
    
    # 测试案例
    test_case = {
        "user_prompt": """1. 推文内容生成要求
主题：“畅游马耳他：蓝色地中海的慢生活体验指南”
文案风格：专业、亲切、信息丰富，适合小红书，逻辑清晰，细节具体，适度插入emoji和话题标签。我需要超过500字的文案内容

开头点题，引发兴趣
🌊✨一场说走就走的地中海之旅，你想体验真正的“慢生活”吗？马耳他，这座隐藏在地中海的宝石，正在悄悄改变我的旅行观。

概览介绍目的地亮点
马耳他不仅以壮丽的海岸线和湛蓝的海水著称，更有着丰富的历史遗迹、缤纷的城市街景与独特的地中海美食。从古城瓦莱塔的巷弄到蓝窗的壮观，再到Mostar大教堂的宁静，每一步都能感受到时光的流转与自然的馈赠。

结合配图1，描绘真实的旅游场景和细节
如图所示，在马耳他的Mosta大教堂前，我选择清晨出发，避开人流，独享古建筑的静谧。宽阔的石板路，精致的咖啡馆，街头艺术与明媚的阳光交织，组成了最理想的城市漫步体验。沿海骑行道上，蓝色大海与橄榄树林交相辉映，随手一拍都是大片。

引用具体旅行数据与效率提升（对标生产力指标）
旅行前我使用了定制行程App，路线时间节省约40%，热门景点排队时间减少一半。三天之内，核心地标全部打卡，拍照打卡率高达90%，每天步行超1.5万步却一点都不疲惫，真正实现了“轻松游+高效体验”的完美结合！

结尾鼓励互动
如果你也向往慢节奏的深度旅行，欢迎在评论区留言或私信我获取我的专属马耳他行程规划清单和路线推荐，和我一起开启不一样的地中海探索之旅吧！

结尾加话题标签
#马耳他旅游 #地中海旅行 #慢生活攻略 #高效出游 #欧洲小众目的地

1. 图片1内容生成要求
生成一张超高分辨率、写实风格的马耳他实景旅行照：

画面中央是一位穿着夏日休闲装的年轻游客（背包/墨镜/轻便鞋），正站在马耳他Mostar大教堂前的广场上，侧身眺望远方，表情自然放松。

框中可见远处的Mostar大教堂圆顶、宏伟建筑细节。

游客身旁的石板路边，有咖啡馆露天座椅、地中海绿植和街头艺术元素点缀。

背景是明亮的蓝天、远处能隐约看到海岸线，阳光自然洒落，整体光影真实、细节丰富。

整体画面清新、现代，无卡通特效，突出真实的度假体验与地中海风情。

2. 图片2内容生成要求
生成一张超高分辨率、写实风格的马耳他实景旅行照：
展示马耳他的海景风光
""",
        "platform": "小红书"
    }
    
    print(f"\n🎯 测试案例:")
    print(f"用户提示: {test_case['user_prompt']}")
    print(f"目标平台: {test_case['platform']}")
    
    try:
        # 执行workflow
        print("\n🔄 执行workflow...")
        result = await workflow.async_execute(
            inputs=test_case,
            task_name="social media content creation with FluxOpenAIEditingActionAgent",
            goal="Create social media content with prompt analysis, research, content generation, and image creation using Flux and OpenAI editing"
        )
        
        print("\n" + "="*80)
        print("🎉 Workflow执行完成！")
        print("="*80)
        
        # 显示结果
        print("\n📋 完整结果:")
        print("-" * 40)
        print(result)
    
        
        return result
        
    except Exception as e:
        print(f"❌ Workflow执行失败: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    import datetime
    
    print("开始测试更新后的XHS Workflow...")
    
    # 测试完整workflow
    asyncio.run(test_updated_workflow())
    
    print("\n🎉 所有测试完成!")
