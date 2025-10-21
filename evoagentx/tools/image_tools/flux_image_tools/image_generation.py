import os
import requests
import time
from typing import Dict, List, Optional
from ...tool import Tool
from ...storage_handler import FileStorageHandler, LocalStorageHandler
from .image_postprocessor import FluxImagePostProcessor


class FluxImageGenerationTool(Tool):
    name: str = "flux_image_generation"
    description: str = "Flux image generation supporting models like flux-kontext-max. It supports automatic postprocessing for unsupported sizes/formats."

    inputs: Dict[str, Dict[str, str]] = {
        "prompt": {"type": "string", "description": "Text prompt for image generation."},
        "seed": {"type": "integer", "description": "Random seed, default is 42.", "default": 42},
        "aspect_ratio": {"type": "string", "description": "Aspect ratio, e.g. '1:1', '16:9' etc.", "default": None},
        "output_format": {"type": "string", "description": "Image format, default is jpeg.", "default": "jpeg"},
        "output_size": {"type": "string", "description": "Output image size, e.g. '1024x768' (needs postprocessing) or '16:9' (native support).", "default": None},
        "output_quality": {"type": "integer", "description": "Output quality (0-100), for JPEG/WEBP compression, default is 95.", "default": 95},
        "prompt_upsampling": {"type": "boolean", "description": "Enable prompt upsampling, default is false.", "default": False},
        "safety_tolerance": {"type": "integer", "description": "Safety tolerance level, default is 2.", "default": 2},
        "image_name": {"type": "string", "description": "Optional save name.", "default": None},
    }
    required: List[str] = ["prompt"]

    def __init__(self, api_key: str = None, model: str = "flux-kontext-max",
                 save_path: str = "./flux_generated_images", storage_handler: Optional[FileStorageHandler] = None, auto_postprocess: bool = False):
        super().__init__()
        self.api_key = api_key or os.getenv("FLUX_API_KEY")
        self.model = model
        self.save_path = save_path
        self.storage_handler = storage_handler or LocalStorageHandler(base_path=save_path)
        self.auto_postprocess = auto_postprocess
        self.postprocessor = FluxImagePostProcessor()

    def __call__(
        self,
        prompt: str,
        seed: int = 42,
        model: str = None,
        aspect_ratio: str = None,
        output_format: str = "jpeg",
        output_size: str = None,
        output_quality: int = 95,
        prompt_upsampling: bool = False,
        safety_tolerance: int = 2,
        image_name: str = None,
    ):
        try:
            actual_model = model if model else self.model
            # Check if postprocessing is needed
            needs_pp = self.postprocessor.needs_postprocessing(
                output_size or aspect_ratio, 
                output_format
            )
            
            target_size = output_size
            target_format = output_format
            target_quality = output_quality
            
            if self.auto_postprocess and (needs_pp["need_resize"] or needs_pp["need_format_conversion"]):
                print("🔄 Detected incompatible parameters, automatic postprocessing will be enabled:")
                for reason in needs_pp["reason"]:
                    print(f"   • {reason}")
                
                # Get compatible API parameters
                compat_params = self.postprocessor.get_compatible_params(
                    output_size or aspect_ratio, 
                    output_format
                )
                api_aspect_ratio = compat_params["api_params"].get("aspect_ratio")
                api_format = compat_params["api_params"].get("output_format")
                
                print(f"📝 API will use: aspect_ratio={api_aspect_ratio}, format={api_format}")
                print(f"🎯 Postprocessing target: size={target_size}, format={target_format}")
            else:
                api_aspect_ratio = aspect_ratio or output_size
                api_format = output_format
            
            # Build request payload
            payload = {
                "prompt": prompt,
                "seed": seed,
                "output_format": api_format if self.auto_postprocess and needs_pp["need_format_conversion"] else output_format,
                "prompt_upsampling": prompt_upsampling,
                "safety_tolerance": safety_tolerance,
            }
            
            # Handle aspect ratio
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

            # Send generation request
            response = requests.post(f"https://api.bfl.ai/v1/{actual_model}", json=payload, headers=headers)
            response.raise_for_status()
            request_data = response.json()

            request_id = request_data["id"]
            polling_url = request_data["polling_url"]

            # Poll result
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
                    return {"error": f"Generation failed: {result}"}

            # Download image
            image_response = requests.get(image_url)
            image_response.raise_for_status()
            image_content = image_response.content
            
            # Apply postprocessing (if needed)
            if self.auto_postprocess and (needs_pp["need_resize"] or needs_pp["need_format_conversion"]):
                print("🔧 Postprocessing image...")
                image_content, ext = self.postprocessor.process_image(
                    image_content,
                    target_size=target_size,
                    target_format=target_format,
                    compression_quality=target_quality
                )
            else:
                ext = output_format if output_format != "jpeg" else "jpg"

            # Generate unique filename
            filename = self._get_unique_filename(image_name, seed, ext)
            
            # Use storage handler to save image
            result = self.storage_handler.save(filename, image_content)
            
            if result["success"]:
                return {
                    "results": [filename],
                    "count": 1,
                    "file_path": filename,  # Backward compatibility
                    "full_path": result.get("full_path", filename),
                }
            else:
                return {
                    "error": f"Failed to save image: {result.get('error', 'Unknown error')}"
                }
        except Exception as e:
            return {"error": f"Image generation failed: {e}"}
    
    def _get_unique_filename(self, image_name: str, seed: int, ext: str = "jpg") -> str:
        """Generate unique filename for image"""
        
        if image_name:
            base = image_name.rsplit(".", 1)[0]
            filename = f"{base}.{ext}"
        else:
            filename = f"flux_{seed}.{ext}"
        
        counter = 1
        
        # Check if file exists and generate unique name
        while self.storage_handler.exists(filename):
            if image_name:
                base = image_name.rsplit(".", 1)[0]
                filename = f"{base}_{counter}.{ext}"
            else:
                filename = f"flux_{seed}_{counter}.{ext}"
            counter += 1
            
        return filename

