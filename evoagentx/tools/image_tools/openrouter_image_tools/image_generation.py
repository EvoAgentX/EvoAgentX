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

    def __init__(self, api_key: str = None, model: str = "google/gemini-2.5-flash-image",
                 save_path: str = "./openrouter_generated_images", storage_handler: Optional[FileStorageHandler] = None, auto_postprocess: bool = False):
        super().__init__()
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
            actual_model = model if model else self.model
            key = api_key or self.api_key
            if not key:
                return {"error": "OPENROUTER_API_KEY not provided."}

            # Check if postprocessing is needed
            needs_pp = self.postprocessor.needs_postprocessing(actual_model, output_size, output_format)
            target_size = output_size
            target_format = output_format
            target_quality = output_quality if output_quality is not None else 95
            
            if self.auto_postprocess and (needs_pp["need_resize"] or needs_pp["need_format_conversion"]):
                print("🔄 Detected parameters requiring postprocessing:")
                for reason in needs_pp["reason"]:
                    print(f"   • {reason}")
                
                # Get compatible API parameters
                compat_params = self.postprocessor.get_compatible_params(actual_model, output_size, output_format)
                api_format = compat_params["api_params"].get("output_format")
                
                print(f"📝 API will generate with format: {api_format}")
                print(f"🎯 Postprocessing target: size={target_size or 'original'}, format={target_format}")
            
            # Build API request
            messages = [{"role": "user", "content": prompt}]
            payload = {
                "model": actual_model, 
                "messages": messages, 
                "modalities": ["image", "text"]
            }

            headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
            url = "https://openrouter.ai/api/v1/chat/completions"
            
            resp = requests.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()

            saved_paths: List[str] = []
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
                        ext = ".png"
                        if mime == "image/jpeg":
                            ext = ".jpg"
                        elif mime == "image/webp":
                            ext = ".webp"
                        elif mime == "image/heic":
                            ext = ".heic"
                        elif mime == "image/heif":
                            ext = ".heif"
                        
                        # Generate unique filename
                        filename = self._get_unique_filename(image_name or "or_gen", ext)
                        
                        # Decode image data
                        image_content = base64.b64decode(b64data)
                        
                        # Apply postprocessing if needed
                        if self.auto_postprocess and (needs_pp["need_resize"] or needs_pp["need_format_conversion"]):
                            print(f"🔧 Postprocessing image {idx+1}/{len(images)}...")
                            processed_content, new_ext = self.postprocessor.process_image(
                                image_content,
                                target_size=target_size,
                                target_format=target_format,
                                compression_quality=target_quality
                            )
                            # Update filename extension if format changed
                            filename = self._change_extension(filename, new_ext)
                            image_content = processed_content
                        
                        # Save using storage handler
                        result = self.storage_handler.save(filename, image_content)
                        
                        if result["success"]:
                            saved_paths.append(filename)
                        else:
                            return {"error": f"Failed to save image: {result.get('error', 'Unknown error')}"}

            if saved_paths:
                return {"results": saved_paths, "count": len(saved_paths)}
            return {"warning": "No image returned or saved.", "raw": data}
        except Exception as e:
            return {"error": f"OpenRouter image generation failed: {e}"}

    def _get_unique_filename(self, base_name: str, extension: str) -> str:
        """Generate a unique filename for the image"""
        filename = f"{base_name}{extension}"
        counter = 1
        
        # Check if file exists and generate unique name
        while self.storage_handler.exists(filename):
            filename = f"{base_name}_{counter}{extension}"
            counter += 1
            
        return filename
    
    def _change_extension(self, filename: str, new_ext: str) -> str:
        """Change file extension to match new format"""
        # Remove leading dot if present
        if new_ext.startswith('.'):
            new_ext = new_ext[1:]
        
        base = filename.rsplit('.', 1)[0] if '.' in filename else filename
        return f"{base}.{new_ext}"
