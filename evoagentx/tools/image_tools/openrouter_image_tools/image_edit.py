from typing import Dict, List, Optional
import os
import requests
import base64
import mimetypes
from ...tool import Tool
from ...storage_handler import FileStorageHandler, LocalStorageHandler
from .image_postprocessor import ImagePostProcessor


class OpenRouterImageEditTool(Tool):
    name: str = "openrouter_image_edit"
    description: str = (
        "Edit or compose images using OpenRouter models (e.g., google/gemini-2.5-flash-image). "
        "Provide image URLs or local paths along with a text prompt to modify or enhance the image. "
        "Supports custom output sizes and formats with automatic postprocessing."
    )

    inputs: Dict[str, Dict] = {
        "prompt": {"type": "string", "description": "Text prompt for image editing/composition."},
        "image_urls": {"type": "array", "description": "Remote image URLs (optional)."},
        "image_paths": {"type": "array", "description": "Local image paths (optional)."},
        "model": {"type": "string", "description": "OpenRouter model id.", "default": "google/gemini-2.5-flash-image"},
        "api_key": {"type": "string", "description": "OpenRouter API key (fallback to env OPENROUTER_API_KEY)."},
        "save_path": {"type": "string", "description": "Directory to save edited images.", "default": "./openrouter_images"},
        "output_basename": {"type": "string", "description": "Base filename for outputs.", "default": "or_edit"},
        "output_size": {"type": "string", "description": "Output image size as 'WIDTHxHEIGHT' (e.g., '512x512', '1024x768'). If not specified, keeps original size."},
        "output_format": {"type": "string", "description": "Output format: 'png', 'jpeg', 'webp' etc. If not specified, uses PNG.", "default": "png"},
        "output_quality": {"type": "integer", "description": "JPEG/WEBP quality (1-100). Only used for jpeg/webp formats.", "default": 95}
    }
    required: List[str] = ["prompt"]

    def __init__(self, api_key: str = None, storage_handler: Optional[FileStorageHandler] = None, 
                 base_path: str = "./openrouter_images", auto_postprocess: bool = False):
        super().__init__()
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.storage_handler = storage_handler or LocalStorageHandler(base_path=base_path)
        self.auto_postprocess = auto_postprocess
        self.postprocessor = ImagePostProcessor()

    def __call__(
        self,
        prompt: str,
        image_urls: list = None,
        image_paths: list = None,
        model: str = "google/gemini-2.5-flash-image",
        api_key: str = None,
        save_path: str = "./openrouter_images",
        output_basename: str = "or_edit",
        output_size: str = None,
        output_format: str = "png",
        output_quality: int = 95,
    ):
        key = api_key or self.api_key
        if not key:
            return {"error": "OPENROUTER_API_KEY not provided."}

        # Validate that at least one image source is provided
        if not image_urls and not image_paths:
            return {"error": "At least one of image_urls or image_paths must be provided for editing."}

        # Check if postprocessing is needed
        needs_pp = self.postprocessor.needs_postprocessing(model, output_size, output_format)
        target_size = output_size
        target_format = output_format
        target_quality = output_quality
        
        if self.auto_postprocess and (needs_pp["need_resize"] or needs_pp["need_format_conversion"]):
            print("🔄 Detected parameters requiring postprocessing:")
            for reason in needs_pp["reason"]:
                print(f"   • {reason}")
            
            # Get compatible API parameters
            compat_params = self.postprocessor.get_compatible_params(model, output_size, output_format)
            api_format = compat_params["api_params"].get("output_format", "png")
            
            print(f"📝 API will generate with format: {api_format}")
            print(f"🎯 Postprocessing target: size={target_size or 'original'}, format={target_format}")

        # Build content parts from URLs and/or local paths
        content_parts = [{"type": "text", "text": prompt}]
        
        if image_urls:
            content_parts.extend(self._urls_to_image_parts(image_urls))
        
        if image_paths:
            content_parts.extend(self._paths_to_image_parts(image_paths))
        
        # Build API request
        messages = [{"role": "user", "content": content_parts}]
        payload = {
            "model": model,
            "messages": messages,
            "modalities": ["image", "text"]
        }

        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        url = "https://openrouter.ai/api/v1/chat/completions"
        
        try:
            resp = requests.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.HTTPError as e:
            # Log error details for debugging
            try:
                error_data = resp.json()
                return {"error": f"OpenRouter API error: {error_data}", "status_code": resp.status_code}
            except Exception:
                return {"error": f"OpenRouter API error: {e}", "status_code": resp.status_code}
        except Exception as e:
            return {"error": f"Request failed: {e}"}

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
                    filename = self._get_unique_filename(output_basename or "or_edit", ext)
                    
                    # Decode image data
                    image_content = base64.b64decode(b64data)
                    
                    # Apply postprocessing if needed
                    if self.auto_postprocess and (needs_pp["need_resize"] or needs_pp["need_format_conversion"]):
                        try:
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
                        except Exception as e:
                            return {"error": f"Failed to postprocess image: {str(e)}"}
                    
                    # Save using storage handler
                    result = self.storage_handler.save(filename, image_content)
                    
                    if result["success"]:
                        # Return the translated path that was actually used for saving
                        translated_path = self.storage_handler.translate_in(filename)
                        saved_paths.append(translated_path)
                    else:
                        return {"error": f"Failed to save image: {result.get('error', 'Unknown error')}"}

        if saved_paths:
            return {"results": saved_paths, "count": len(saved_paths)}
        return {"warning": "No image returned or saved.", "raw": data}

    # --- Helper methods ---
    
    def _url_to_image_part(self, url: str) -> Dict:
        """Convert URL to image_url content part"""
        return {"type": "image_url", "image_url": {"url": url}}

    def _guess_mime_from_name(self, name: str, default: str = "image/png") -> str:
        """Guess MIME type from filename"""
        guess, _ = mimetypes.guess_type(name)
        return guess or default

    def _path_to_data_url(self, path: str) -> str:
        """Convert local image path to data URL"""
        mime = self._guess_mime_from_name(path)
        
        # Use storage handler to read raw bytes directly
        try:
            # Translate user path to system path first
            system_path = self.storage_handler.translate_in(path)
            content = self.storage_handler._read_raw(system_path)
        except Exception as e:
            raise FileNotFoundError(f"Could not read file {path}: {str(e)}")
        
        b64 = base64.b64encode(content).decode("utf-8")
        return f"data:{mime};base64,{b64}"
    
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

    def _paths_to_image_parts(self, paths: list) -> List[Dict]:
        """Convert list of local paths to image_url content parts"""
        parts: List[Dict] = []
        for p in paths:
            try:
                parts.append(self._url_to_image_part(self._path_to_data_url(p)))
            except Exception as e:
                print(f"⚠️ Failed to read image path {p}: {e}")
                # Skip unreadable path
                continue
        return parts

    def _urls_to_image_parts(self, urls: list) -> List[Dict]:
        """Convert list of URLs to image_url content parts"""
        return [self._url_to_image_part(u) for u in urls]

