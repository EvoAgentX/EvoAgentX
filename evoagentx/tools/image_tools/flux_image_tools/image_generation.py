from typing import Dict, List, Optional
from ...tool import Tool
from ...storage_handler import FileStorageHandler, LocalStorageHandler
from .flux_image_postprocessor import FluxImagePostProcessor
import requests
import time


class FluxImageGenerationTool(Tool):
    name: str = "flux_image_generation"
    description: str = (
        "使用 bfl.ai flux-kontext-max API 从文本生成图像。"
    )

    inputs: Dict[str, Dict] = {
        "prompt": {"type": "string", "description": "描述要生成的图像的提示词。"},
        "seed": {"type": "integer", "description": "随机种子，默认为 42。", "default": 42},
        "aspect_ratio": {"type": "string", "description": "宽高比，例如 '1:1', '16:9' 等。"},
        "output_format": {"type": "string", "description": "图像格式，默认为 jpeg。", "default": "jpeg"},
        "output_size": {"type": "string", "description": "输出图像尺寸，例如 '1024x768'（需要后处理）或 '16:9'（原生支持）。"},
        "output_quality": {"type": "integer", "description": "输出质量 (0-100)，用于 JPEG/WEBP 压缩，默认 95。", "default": 95},
        "prompt_upsampling": {"type": "boolean", "description": "启用提示词增强，默认为 false。", "default": False},
        "safety_tolerance": {"type": "integer", "description": "安全容忍级别，默认为 2。", "default": 2},
        "image_name": {"type": "string", "description": "可选的保存名称。"},
    }
    required: List[str] = ["prompt"]

    def __init__(self, api_key: str, storage_handler: Optional[FileStorageHandler] = None, 
                 base_path: str = "./imgs", save_path: str = None, auto_postprocess: bool = False):
        super().__init__()
        self.api_key = api_key
        
        # 处理向后兼容性：如果提供了 save_path，使用它作为 base_path
        if save_path is not None:
            base_path = save_path
            
        # 初始化存储处理器
        if storage_handler is None:
            self.storage_handler = LocalStorageHandler(base_path=base_path)
        else:
            self.storage_handler = storage_handler
        
        self.auto_postprocess = auto_postprocess
        self.postprocessor = FluxImagePostProcessor()

    def __call__(
        self,
        prompt: str,
        seed: int = 42,
        aspect_ratio: str = None,
        output_format: str = "jpeg",
        output_size: str = None,
        output_quality: int = 95,
        prompt_upsampling: bool = False,
        safety_tolerance: int = 2,
        image_name: str = None,
    ):
        try:
            # 检查是否需要后处理
            needs_pp = self.postprocessor.needs_postprocessing(
                output_size or aspect_ratio, 
                output_format
            )
            
            target_size = output_size
            target_format = output_format
            target_quality = output_quality
            
            if self.auto_postprocess and (needs_pp["need_resize"] or needs_pp["need_format_conversion"]):
                print("🔄 检测到不兼容的参数，将启用自动后处理：")
                for reason in needs_pp["reason"]:
                    print(f"   • {reason}")
                
                # 获取兼容的 API 参数
                compat_params = self.postprocessor.get_compatible_params(
                    output_size or aspect_ratio, 
                    output_format
                )
                api_aspect_ratio = compat_params["api_params"].get("aspect_ratio")
                api_format = compat_params["api_params"].get("output_format")
                
                print(f"📝 API 将使用: aspect_ratio={api_aspect_ratio}, format={api_format}")
                print(f"🎯 后处理目标: size={target_size}, format={target_format}")
            else:
                api_aspect_ratio = aspect_ratio or output_size
                api_format = output_format
            
            # 构建请求载荷
            payload = {
                "prompt": prompt,
                "seed": seed,
                "output_format": api_format if self.auto_postprocess and needs_pp["need_format_conversion"] else output_format,
                "prompt_upsampling": prompt_upsampling,
                "safety_tolerance": safety_tolerance,
            }
            
            # 处理宽高比
            if self.auto_postprocess and needs_pp["need_resize"]:
                payload["aspect_ratio"] = api_aspect_ratio
            elif aspect_ratio:
                payload["aspect_ratio"] = aspect_ratio
            elif output_size and ":" in output_size:
                payload["aspect_ratio"] = output_size
            
            headers = {
                "accept": "application/json",
                "x-key": self.api_key,
                "Content-Type": "application/json",
            }

            # 发起生成请求
            response = requests.post("https://api.bfl.ai/v1/flux-kontext-max", json=payload, headers=headers)
            response.raise_for_status()
            request_data = response.json()

            request_id = request_data["id"]
            polling_url = request_data["polling_url"]

            # 轮询结果
            while True:
                time.sleep(2)
                result = requests.get(
                    polling_url,
                    headers={
                        "accept": "application/json",
                        "x-key": self.api_key,
                    },
                    params={"id": request_id},
                ).json()

                if result["status"] == "Ready":
                    image_url = result["result"]["sample"]
                    break
                elif result["status"] in ["Error", "Failed"]:
                    return {"error": f"生成失败: {result}"}

            # 下载图像
            image_response = requests.get(image_url)
            image_response.raise_for_status()
            image_content = image_response.content
            
            # 应用后处理（如果需要）
            if self.auto_postprocess and (needs_pp["need_resize"] or needs_pp["need_format_conversion"]):
                print("🔧 正在后处理图像...")
                image_content, ext = self.postprocessor.process_image(
                    image_content,
                    target_size=target_size,
                    target_format=target_format,
                    compression_quality=target_quality
                )
            else:
                ext = output_format if output_format != "jpeg" else "jpg"

            # 生成唯一文件名
            filename = self._get_unique_filename(image_name, seed, ext)
            
            # 使用存储处理器保存图像
            result = self.storage_handler.save(filename, image_content)
            
            if result["success"]:
                return {
                    "results": [filename],
                    "count": 1,
                    "file_path": filename,  # 向后兼容
                    "full_path": result.get("full_path", filename),
                }
            else:
                return {
                    "error": f"保存图像失败: {result.get('error', '未知错误')}"
                }
        except Exception as e:
            return {"error": f"图像生成失败: {e}"}
    
    def _get_unique_filename(self, image_name: str, seed: int, ext: str = "jpg") -> str:
        """生成图像的唯一文件名"""
        
        if image_name:
            base = image_name.rsplit(".", 1)[0]
            filename = f"{base}.{ext}"
        else:
            filename = f"flux_{seed}.{ext}"
        
        counter = 1
        
        # 检查文件是否存在并生成唯一名称
        while self.storage_handler.exists(filename):
            if image_name:
                base = image_name.rsplit(".", 1)[0]
                filename = f"{base}_{counter}.{ext}"
            else:
                filename = f"flux_{seed}_{counter}.{ext}"
            counter += 1
            
        return filename

