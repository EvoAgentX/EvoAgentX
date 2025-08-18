## Flux生成图片 + OpenAI添加文字

import os 
from dotenv import load_dotenv
from PIL import Image
import base64

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_ORGANIZATION_ID = os.getenv("OPENAI_ORGANIZATION_ID")
BFL_API_KEY = os.getenv("BFL_API_KEY")

if not all([OPENAI_API_KEY, OPENAI_ORGANIZATION_ID, BFL_API_KEY]):
    print("请设置OPENAI_API_KEY、OPENAI_ORGANIZATION_ID和BFL_API_KEY环境变量")
    exit(1)

class FluxToOpenAITextTool:
    """Flux生成图片 + OpenAI添加文字的组合工具"""
    
    def __init__(self, flux_api_key: str, openai_api_key: str, openai_org_id: str, save_path: str = "./flux_openai_text_images"):
        self.flux_api_key = flux_api_key
        self.openai_api_key = openai_api_key
        self.openai_org_id = openai_org_id
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)
        
    def generate_with_flux(self, prompt: str) -> str:
        """使用Flux生成图片"""
        from evoagentx.tools.flux_image_generation import FluxImageGenerationTool
        
        flux_tool = FluxImageGenerationTool(api_key=self.flux_api_key, save_path=self.save_path)
        result = flux_tool(prompt=prompt)
        generated_image_path = result.get("file_path")
        
        if not generated_image_path or not os.path.exists(generated_image_path):
            raise Exception("Flux图片生成失败")
            
        print(f"Flux生成的图片已保存到: {generated_image_path}")
        return generated_image_path
    
    def add_text_with_openai(self, image_path: str, text_prompt: str, output_name: str = None) -> str:
        """使用OpenAI在图片上添加文字"""
        from openai import OpenAI
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"输入图片不存在: {image_path}")
        
        # 创建 OpenAI 客户端
        client = OpenAI(api_key=self.openai_api_key, organization=self.openai_org_id)
        
        # 编辑图片（使用 gpt-image-1 模型，支持多种格式）
        response = client.images.edit(
            model="gpt-image-1",
            image=open(image_path, "rb"),
            prompt=text_prompt
        )
        
        # 设置输出文件名
        output_name = output_name or "image_with_text.jpeg"
        if not output_name.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
            output_name += ".jpeg"
        
        edited_image_path = os.path.join(self.save_path, output_name)
        
        # 保存编辑后的图片
        image_base64 = response.data[0].b64_json
        image_bytes = base64.b64decode(image_base64)
        
        with open(edited_image_path, "wb") as f:
            f.write(image_bytes)
        
        print(f"✅ OpenAI添加文字完成！保存在: {edited_image_path}")
        return edited_image_path
    
    def _resize_to_square(self, image_path: str, size: int = 1024) -> str:
        """将图片调整为正方形"""
        with Image.open(image_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            width, height = img.size
            if width > height:
                new_width = size
                new_height = int(height * size / width)
            else:
                new_height = size
                new_width = int(width * size / height)
            
            img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            square_img = Image.new('RGB', (size, size), (255, 255, 255))
            
            x_offset = (size - new_width) // 2
            y_offset = (size - new_height) // 2
            square_img.paste(img_resized, (x_offset, y_offset))
            
            temp_path = image_path.replace('.', '_square.')
            square_img.save(temp_path, format='PNG')
            return temp_path
    
    def generate_and_add_text(self, generation_prompt: str, text_prompt: str, output_name: str = None) -> dict:
        """完整的生成+添加文字流程"""
        print(f"开始生成图片: {generation_prompt}")
        generated_image_path = self.generate_with_flux(generation_prompt)
        
        print(f"开始添加文字: {text_prompt}")
        edited_image_path = self.add_text_with_openai(
            image_path=generated_image_path,
            text_prompt=text_prompt,
            output_name=output_name
        )
        
        return {
            "generated_image_path": generated_image_path,
            "edited_image_path": edited_image_path,
            "generation_prompt": generation_prompt,
            "text_prompt": text_prompt
        }

def example():
    """简单示例 - Flux自动生成图片地址，用户只需指定最终保存地址"""
    tool = FluxToOpenAITextTool(
        flux_api_key=BFL_API_KEY,
        openai_api_key=OPENAI_API_KEY,
        openai_org_id=OPENAI_ORGANIZATION_ID
    )
    
    # 用户只需要指定最终保存的图片地址
    final_output_name = "my_final_landscape.jpeg"
    
    result = tool.generate_and_add_text(
        generation_prompt="生成一副马耳他的风景照，包括马耳他首付mosta和海岸风光",
        text_prompt="在图片上添加简体中文广告标语，‘马耳他，地中海的明珠’，艺术风格的字体，广告风格",
        output_name=final_output_name  # 只指定最终保存的图片名称
    )
    
    print(f"\n🎉 工作流完成！")
    print(f"📁 Flux生成的原始图片: {result['generated_image_path']}")
    print(f"📁 最终保存的图片: {result['edited_image_path']}")
    print(f"🎨 生成提示: {result['generation_prompt']}")
    print(f"✍️ 文字提示: {result['text_prompt']}")

def openai_example():
    """简单的 OpenAI 图片编辑示例 - 编辑 flux_42_1.jpeg"""
    from openai import OpenAI
    
    # 输入图片路径
    image_path = "flux_openai_text_images/flux_42_1.jpeg"
    
    # 检查图片是否存在
    if not os.path.exists(image_path):
        print(f"图片不存在: {image_path}")
        return
    
    # 创建 OpenAI 客户端
    client = OpenAI(api_key=OPENAI_API_KEY, organization=OPENAI_ORGANIZATION_ID)
    
    # 编辑图片（简化参数）
    response = client.images.edit(
        model="gpt-image-1",
        image=open(image_path, "rb"),
        prompt="在图片顶部中央添加白色文字'AI Generated Art - 2024'，字体要大且清晰可见"
    )
    
    # 保存编辑后的图片
    output_path = "flux_openai_text_images/flux_42_1_edited.jpeg"
    image_base64 = response.data[0].b64_json
    image_bytes = base64.b64decode(image_base64)
    
    with open(output_path, "wb") as f:
        f.write(image_bytes)
    
    print(f"✅ 图片编辑完成！保存在: {output_path}")

if __name__ == "__main__":
    example()  # 注释掉原来的示例
    # openai_example()  # 运行新的 OpenAI 编辑示例 