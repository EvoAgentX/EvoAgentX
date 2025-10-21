from typing import Dict, List, Optional
import os
import requests
import base64
from ...tool import Tool
from ...storage_handler import FileStorageHandler, LocalStorageHandler
from .image_postprocessor import OpenRouterImagePostProcessor


class OpenRouterImageGenerationTool(Tool):
    name: str = "openrouter_image_generation"
    description: str = " OpenRouter image generation supporting models like google/gemini-2.5-flash-image. It supports automatic postprocessing for unsupported sizes/formats."

    inputs: Dict[str, Dict[str, str]] = {
        "prompt": {"type": "string", "description": "Text prompt for image generation."},
        "model": {"type": "string", "description": "OpenRouter model id.", "default": "google/gemini-2.5-flash-image"},
        "api_key": {"type": "string", "description": "OpenRouter API key (fallback to env OPENROUTER_API_KEY)."},
        "image_name": {"type": "string", "description": "Base filename for outputs.", "default": "or_gen"},
        "output_size": {"type": "string", "description": "Output image size as 'WIDTHxHEIGHT' (e.g., '512x512', '1024x768'). If not specified, uses model's default size."},
        "output_format": {"type": "string", "description": "Output format: 'png', 'jpeg', 'webp' etc. If not specified, uses PNG.", "default": "png"},
        "output_quality": {"type": "integer", "description": "JPEG/WEBP quality (1-100). Only used for jpeg/webp formats.", "default": 95}
    }
    required: List[str] = ["prompt"]

    def __init__(self, name: str = None, api_key: str = None, model: str = "google/gemini-2.5-flash-image",
                 save_path: str = "./openrouter_generated_images", storage_handler: Optional[FileStorageHandler] = None, auto_postprocess: bool = False):
        super().__init__()
        self.name = name or self.name
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.model = model
        self.save_path = save_path
        self.storage_handler = storage_handler or LocalStorageHandler(base_path=save_path)
        self.auto_postprocess = auto_postprocess
        self.postprocessor = OpenRouterImagePostProcessor()

    def __call__(
        self,
        prompt: str,
        model: str = None,
        api_key: str = None,
        image_name: str = None,
        output_size: str = None,
        output_format: str = None,
        output_quality: int = None,
    ):
        try:
            # Get actual parameters
            actual_model = model if model else self.model
            actual_api_key = api_key if api_key else self.api_key
            if not actual_api_key:
                return {"error": "OPENROUTER_API_KEY not provided."}

            # Check if postprocessing is needed
            needs_pp = self.postprocessor.needs_postprocessing(actual_model, output_size, output_format)
            target_size = output_size
            target_format = output_format
            target_quality = output_quality if output_quality is not None else 95
            
            if self.auto_postprocess and (needs_pp["need_resize"] or needs_pp["need_format_conversion"]):
                print("🔄 Detected incompatible parameters, automatic postprocessing will be enabled:")
                for reason in needs_pp["reason"]:
                    print(f"   • {reason}")
                
                # Get compatible API parameters
                compat_params = self.postprocessor.get_compatible_params(actual_model, output_size, output_format)
                api_format = compat_params["api_params"].get("output_format")
                
                print(f"📝 API will use: format={api_format}")
                print(f"🎯 Postprocessing target: size={target_size or 'original'}, format={target_format}")
            
            # Build API request
            messages = [{"role": "user", "content": prompt}]
            payload = {
                "model": actual_model, 
                "messages": messages, 
                "modalities": ["image", "text"]
            }

            headers = {"Authorization": f"Bearer {actual_api_key}", "Content-Type": "application/json"}
            url = "https://openrouter.ai/api/v1/chat/completions"
            
            # Call API
            resp = requests.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()

            # Process and save images
            results = []
            if data.get("choices"):
                msg = data["choices"][0]["message"]
                
                # Try to get images from different possible locations in response
                images = msg.get("images") or []
                
                # If no images in direct field, check content for image_url parts
                if not images and msg.get("content"):
                    content = msg.get("content")
                    # Handle content as list of parts
                    if isinstance(content, list):
                        for part in content:
                            if isinstance(part, dict) and part.get("type") == "image_url":
                                images.append(part)
                
                if not images:
                    # Return warning with response structure for debugging
                    return {
                        "warning": "No images found in response",
                        "message_keys": list(msg.keys()),
                        "content_type": type(msg.get("content")).__name__ if msg.get("content") else "None",
                        "raw": data
                    }
                
                for idx, im in enumerate(images):
                    try:
                        image_url = im.get("image_url", {}).get("url") if isinstance(im.get("image_url"), dict) else im.get("image_url")
                        if not image_url:
                            continue
                        
                        # Handle data URL
                        if image_url.startswith("data:") and "," in image_url:
                            header, b64data = image_url.split(",", 1)
                            
                            # Detect MIME type
                            mime = "image/png"
                            if ";" in header:
                                mime = header.split(":", 1)[1].split(";", 1)[0] or mime
                            
                            # Default extension based on MIME
                            ext = "png"
                            if mime == "image/jpeg":
                                ext = "jpg"
                            elif mime == "image/webp":
                                ext = "webp"
                            elif mime == "image/heic":
                                ext = "heic"
                            elif mime == "image/heif":
                                ext = "heif"
                            
                            # Decode image data
                            image_bytes = base64.b64decode(b64data)
                            
                            # Apply postprocessing if needed
                            if self.auto_postprocess and (needs_pp["need_resize"] or needs_pp["need_format_conversion"]):
                                print(f"🔧 Postprocessing image {idx+1}/{len(images)}...")
                                image_bytes, ext = self.postprocessor.process_image(
                                    image_bytes,
                                    target_size=target_size,
                                    target_format=target_format,
                                    compression_quality=target_quality
                                )
                            
                            # Generate unique filename
                            filename = self._get_unique_filename(image_name or "or_gen", idx, ext)
                            
                            # Save using storage handler
                            save_result = self.storage_handler.save(filename, image_bytes)
                            
                            if save_result["success"]:
                                results.append(filename)
                            else:
                                results.append(f"Error saving image {idx+1}: {save_result.get('error', 'Unknown error')}")
                    except Exception as e:
                        results.append(f"Error saving image {idx+1}: {e}")

            if results:
                return {"results": results, "count": len(results)}
            return {"error": "No images returned or saved."}
        except Exception as e:
            return {"error": f"Image generation failed: {e}"}

    def _get_unique_filename(self, image_name: str, index: int, ext: str = "png") -> str:
        """Generate a unique filename for the image"""
        import time
        
        if image_name:
            base = image_name.rsplit(".", 1)[0]
            # Try without index suffix first (for single image case)
            if index == 0:
                filename = f"{base}.{ext}"
                if not self.storage_handler.exists(filename):
                    return filename
            # If first attempt failed or not first image, use index suffix
            filename = f"{base}_{index+1}.{ext}"
        else:
            ts = int(time.time())
            filename = f"generated_{ts}_{index+1}.{ext}"
        
        # Check if file exists and generate unique name
        counter = 1
        while self.storage_handler.exists(filename):
            if image_name:
                base = image_name.rsplit(".", 1)[0]
                filename = f"{base}_{index+1}_{counter}.{ext}"
            else:
                filename = f"generated_{ts}_{index+1}_{counter}.{ext}"
            counter += 1
            
        return filename
