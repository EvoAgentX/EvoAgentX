from typing import Dict, Optional, List, Any
import os
import base64
import requests
import time
from dotenv import load_dotenv
from .tool import Tool, Toolkit
from .storage_handler import FileStorageHandler, LocalStorageHandler

load_dotenv()


class FluxImageBase:
    """Base class for Flux image provider with shared functionality."""
    
    def __init__(self, api_key: Optional[str] = None, storage_handler: Optional[FileStorageHandler] = None,
                 base_path: str = "./flux_images"):
        self.api_key = api_key or os.getenv("FLUX_API_KEY")
        self.storage_handler = storage_handler or LocalStorageHandler(base_path=base_path)
        self.base_path = base_path

    def get_unique_filename(self, base_name: str, extension: str = ".png") -> str:
        """Generate a unique, sanitized filename for saving images."""
        import re
        # Replace disallowed characters, collapse whitespace to underscores
        safe_base = re.sub(r"[^A-Za-z0-9._-]+", "_", (base_name or "image").strip())
        if not safe_base:
            safe_base = "image"
        # Limit length to avoid path issues
        if len(safe_base) > 64:
            safe_base = safe_base[:64].rstrip("._-") or "image"
        filename = f"{safe_base}{extension}"
        counter = 1
        
        while self.storage_handler.exists(filename):
            filename = f"{safe_base}_{counter}{extension}"
            counter += 1
            
        return filename

    def save_image_from_url(self, image_url: str, filename: str) -> Dict[str, Any]:
        """Download and save image from URL."""
        try:
            response = requests.get(image_url)
            response.raise_for_status()
            return self.save_image_bytes(response.content, filename)
        except Exception as e:
            return {"success": False, "error": f"Failed to download image: {e}"}

    def save_image_from_base64(self, b64_data: str, filename: str) -> Dict[str, Any]:
        """Save image from base64 encoded data."""
        try:
            image_bytes = base64.b64decode(b64_data)
            return self.save_image_bytes(image_bytes, filename)
        except Exception as e:
            return {"success": False, "error": f"Failed to decode base64 image: {e}"}

    def save_image_bytes(self, image_bytes: bytes, filename: str) -> Dict[str, Any]:
        """Save image bytes using storage handler."""
        try:
            result = self.storage_handler.save(filename, image_bytes)
            if result["success"]:
                return {
                    "success": True,
                    "filename": filename,
                    "full_path": result.get("full_path", filename),
                    **({"url": result.get("url")} if isinstance(result.get("url"), str) else {})
                }
            else:
                return {"success": False, "error": result.get("error", "Unknown storage error")}
        except Exception as e:
            return {"success": False, "error": f"Failed to save image: {e}"}

    def read_image_as_base64(self, image_path: str) -> Dict[str, Any]:
        """Read image file and convert to base64.
        Handles:
        - http(s) URLs -> downloads bytes
        - Local paths -> reads raw bytes using translate_in
        - storage_handler.read returning bytes/str/PIL.Image.Image
        Returns base64 string without data URL prefix.
        """
        try:
            if not image_path or not isinstance(image_path, str):
                return {"success": False, "error": "Invalid image path"}

            s = image_path.strip()

            # Case 1: Remote URL
            if s.startswith("http://") or s.startswith("https://"):
                resp = requests.get(s)
                resp.raise_for_status()
                b64_data = base64.b64encode(resp.content).decode("utf-8")
                return {"success": True, "base64": b64_data}

            # Case 2: Try direct disk read via translate_in for raw bytes
            try:
                local_path = self.storage_handler.translate_in(s)
                with open(local_path, "rb") as f:
                    raw = f.read()
                b64_data = base64.b64encode(raw).decode("utf-8")
                return {"success": True, "base64": b64_data}
            except Exception:
                # Fall through to storage handler read
                pass

            # Case 3: Use storage handler which may return bytes, str (base64), or PIL.Image
            result = self.storage_handler.read(s)
            if not result.get("success"):
                return {"success": False, "error": f"Failed to read image: {result.get('error')}"}

            content = result.get("content")

            # If bytes/bytearray -> base64
            if isinstance(content, (bytes, bytearray)):
                b64_data = base64.b64encode(content).decode("utf-8")
                return {"success": True, "base64": b64_data}

            # If string, assume it's already base64; validate if possible
            if isinstance(content, str):
                try:
                    base64.b64decode(content, validate=True)
                    return {"success": True, "base64": content}
                except Exception:
                    # Not valid base64; attempt to treat as path as a last try
                    try:
                        with open(content, "rb") as f:
                            raw = f.read()
                        b64_data = base64.b64encode(raw).decode("utf-8")
                        return {"success": True, "base64": b64_data}
                    except Exception as e:
                        return {"success": False, "error": f"Failed to process image: {e}"}

            # If PIL.Image.Image -> convert to bytes then base64
            try:
                from PIL import Image  # type: ignore
                from io import BytesIO
                if isinstance(content, Image.Image):
                    fmt = content.format
                    if not fmt:
                        # Infer from extension; default to PNG
                        _, ext = os.path.splitext(s)
                        fmt = ext.lstrip(".").upper() if ext else "PNG"
                        if fmt == "JPG":
                            fmt = "JPEG"
                    buf = BytesIO()
                    content.save(buf, format=fmt)
                    raw = buf.getvalue()
                    b64_data = base64.b64encode(raw).decode("utf-8")
                    return {"success": True, "base64": b64_data}
            except Exception:
                # Continue to final fallback
                pass

            # Final fallback: attempt to open the provided path directly
            try:
                with open(s, "rb") as f:
                    raw = f.read()
                b64_data = base64.b64encode(raw).decode("utf-8")
                return {"success": True, "base64": b64_data}
            except Exception as e:
                return {"success": False, "error": f"Failed to process image: {e}"}
        except Exception as e:
            return {"success": False, "error": f"Failed to process image: {e}"}

    def _make_flux_request(self, prompt: str, input_image: Optional[str] = None, 
                          seed: Optional[int] = None, aspect_ratio: Optional[str] = None,
                          output_format: Optional[str] = None, prompt_upsampling: Optional[bool] = None,
                          safety_tolerance: Optional[int] = None) -> Dict[str, Any]:
        """Make request to Flux API and handle polling."""
        if not self.api_key:
            return {"success": False, "error": "Flux API key not provided"}
            
        headers = {
            "accept": "application/json",
            "x-key": self.api_key,
            "Content-Type": "application/json",
        }
        
        # Build payload with only provided parameters
        payload = {"prompt": prompt}
        if input_image is not None:
            payload["input_image"] = input_image
        if seed is not None:
            payload["seed"] = seed
        if aspect_ratio is not None:
            payload["aspect_ratio"] = aspect_ratio
        if output_format is not None:
            payload["output_format"] = output_format
        if prompt_upsampling is not None:
            payload["prompt_upsampling"] = prompt_upsampling
        if safety_tolerance is not None:
            payload["safety_tolerance"] = safety_tolerance
        
        try:
            # Submit request
            response = requests.post("https://api.bfl.ai/v1/flux-kontext-max", 
                                   json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            request_data = response.json()
            
            request_id = request_data["id"]
            polling_url = request_data["polling_url"]
            
            # Poll for completion with timeout
            max_polls = 150  # 5 minutes at 2-second intervals
            poll_count = 0
            
            while poll_count < max_polls:
                time.sleep(2)
                poll_count += 1
                
                try:
                    result = requests.get(
                        polling_url,
                        headers={"accept": "application/json", "x-key": self.api_key},
                        params={"id": request_id},
                        timeout=10
                    ).json()
                    
                    if result["status"] == "Ready":
                        return {"success": True, "image_url": result["result"]["sample"]}
                    elif result["status"] in ["Error", "Failed"]:
                        return {"success": False, "error": f"Flux API failed: {result}"}
                except requests.exceptions.RequestException as e:
                    return {"success": False, "error": f"Polling request failed: {e}"}
                    
            return {"success": False, "error": "Request timed out after 5 minutes"}
                    
        except requests.exceptions.RequestException as e:
            return {"success": False, "error": f"Flux API request failed: {e}"}
        except ValueError as e:
            return {"success": False, "error": f"Invalid JSON response: {e}"}
        except Exception as e:
            return {"success": False, "error": f"Unexpected error: {e}"}


class FluxImageProvider(FluxImageBase):
    """Flux-specific image provider implementation."""
    
    def generate_image(self, prompt: str,
                      aspect_ratio: Optional[str] = None, output_format: Optional[str] = None,
                      prompt_upsampling: Optional[bool] = None, safety_tolerance: Optional[int] = None,
                      image_name: Optional[str] = None, seed: Optional[int] = None) -> Dict[str, Any]:
        """Generate image using Flux AI."""
        try:
            # Make request with individual parameters
            result = self._make_flux_request(
                prompt=prompt,
                seed=seed,
                aspect_ratio=aspect_ratio,
                output_format=output_format,
                prompt_upsampling=prompt_upsampling,
                safety_tolerance=safety_tolerance
            )
            if not result["success"]:
                return result

            # Download and save image
            output_format_for_filename = output_format if output_format is not None else "jpeg"
            # Use provided image_name as base name if given; else default
            base_name = (image_name or "flux").strip() or "flux"
            filename = self.get_unique_filename(base_name, f".{output_format_for_filename}")
            save_result = self.save_image_from_url(result["image_url"], filename)
            
            if save_result["success"]:
                return {
                    "success": True,
                    "file_path": save_result["filename"],
                    "full_path": save_result.get("full_path"),
                    **({"url": save_result.get("url")} if isinstance(save_result.get("url"), str) else {}),
                    "message": f"Image saved successfully as {save_result['filename']}"
                }
            else:
                return save_result
                
        except Exception as e:
            return {"success": False, "error": f"Flux image generation failed: {e}"}

    def edit_image(self, prompt: str, image_input: str,
                  aspect_ratio: Optional[str] = None, output_format: Optional[str] = None,
                  prompt_upsampling: Optional[bool] = None, safety_tolerance: Optional[int] = None,
                  image_name: Optional[str] = None, seed: Optional[int] = None) -> Dict[str, Any]:
        """Edit image using Flux AI."""
        try:
            # Handle input image (accept base64/data URLs or local file paths)
            if not isinstance(image_input, str) or not image_input:
                return {"success": False, "error": "Invalid image input format"}

            input_image_b64: Optional[str] = None
            s = image_input.strip()

            # data URL prefix case
            if s.startswith("data:"):
                comma = s.find(",")
                if comma != -1:
                    s = s[comma + 1 :]
                input_image_b64 = s
            else:
                # Try to validate as base64; if fails, treat as path and read
                try:
                    import base64 as _b64
                    _b64.b64decode(s, validate=True)
                    input_image_b64 = s
                except Exception:
                    b64_result = self.read_image_as_base64(s)
                    if not b64_result["success"]:
                        return {"success": False, "error": b64_result["error"]}
                    input_image_b64 = b64_result["base64"]
            
            # Make request with individual parameters
            result = self._make_flux_request(
                prompt=prompt,
                input_image=input_image_b64,
                seed=seed,
                aspect_ratio=aspect_ratio,
                output_format=output_format,
                prompt_upsampling=prompt_upsampling,
                safety_tolerance=safety_tolerance
            )
            if not result["success"]:
                return result
            
            # Download and save image
            output_format_for_filename = output_format if output_format is not None else "jpeg"
            # Use provided image_name as base name if given; else default
            base_name = (image_name or "flux_edited").strip() or "flux_edited"
            filename = self.get_unique_filename(base_name, f".{output_format_for_filename}")
            save_result = self.save_image_from_url(result["image_url"], filename)
            
            if save_result["success"]:
                return {
                    "success": True,
                    "file_path": save_result["filename"],
                    "full_path": save_result.get("full_path"),
                    **({"url": save_result.get("url")} if isinstance(save_result.get("url"), str) else {}),
                    "message": f"Edited image saved successfully as {save_result['filename']}"
                }
            else:
                return save_result
        except Exception as e:
            return {"success": False, "error": f"Flux image editing failed: {e}"}


class FluxImageGenerationTool(Tool):
    name: str = "flux_image_generation"
    description: str = "Generate images using Flux AI models"

    inputs: Dict[str, Dict[str, str]] = {
        "prompt": {"type": "string", "description": "Text prompt for image generation"},
        "aspect_ratio": {"type": "string", "description": "Aspect ratio (e.g., '1:1', '16:9')"},
        "output_format": {"type": "string", "description": "Output format (jpeg, png)"},
        "prompt_upsampling": {"type": "boolean", "description": "Enable prompt upsampling"},
        "safety_tolerance": {"type": "integer", "description": "Safety tolerance level"},
        "image_name": {"type": "string", "description": "Base filename for saved output"},
        "seed": {"type": "integer", "description": "Random seed for reproducibility"},
    }
    required: Optional[List[str]] = ["prompt"]

    def __init__(self, provider: FluxImageProvider):
        super().__init__()
        self.provider = provider

    def __call__(self, prompt: str, aspect_ratio: str = None, 
                 output_format: str = None, prompt_upsampling: bool = None, 
                 safety_tolerance: int = None, image_name: str = None, seed: int = None) -> Dict[str, Any]:
        if not prompt:
            return {"error": "Missing required parameter 'prompt'"}
        
        # Call provider directly with explicit parameters
        result = self.provider.generate_image(
            prompt=prompt,
            aspect_ratio=aspect_ratio,
            output_format=output_format,
            prompt_upsampling=prompt_upsampling,
            safety_tolerance=safety_tolerance,
            image_name=image_name,
            seed=seed,
        )
        
        # Convert provider response to tool response format
        if result["success"]:
            return {
                "success": True,
                "file_path": result["file_path"],
                "full_path": result.get("full_path"),
                "message": result["message"]
            }
        else:
            return {"success": False, "error": result["error"]}


class FluxImageEditTool(Tool):
    name: str = "flux_image_edit"
    description: str = "Edit images using Flux AI models"

    inputs: Dict[str, Dict[str, str]] = {
        "prompt": {"type": "string", "description": "Edit instruction"},
        "input_image": {"type": "string", "description": "Base64 encoded input image"},
        "image_path": {"type": "string", "description": "Local path to input image"},
        "aspect_ratio": {"type": "string", "description": "Aspect ratio (e.g., '1:1', '16:9')"},
        "output_format": {"type": "string", "description": "Output format (jpeg, png)"},
        "prompt_upsampling": {"type": "boolean", "description": "Enable prompt upsampling"},
        "safety_tolerance": {"type": "integer", "description": "Safety tolerance level"},
        "image_name": {"type": "string", "description": "Base filename for saved output"},
        "seed": {"type": "integer", "description": "Random seed for reproducibility"},
    }
    required: Optional[List[str]] = ["prompt"]

    def __init__(self, provider: FluxImageProvider):
        super().__init__()
        self.provider = provider

    def __call__(self, prompt: str, input_image: str = None, image_path: str = None,
                 aspect_ratio: str = None, 
                 output_format: str = None, prompt_upsampling: bool = None, 
                 safety_tolerance: int = None, image_name: str = None, seed: int = None) -> Dict[str, Any]:
        if not prompt:
            return {"error": "Missing required parameter 'prompt'"}
        
        # Determine image input
        image_input = input_image or image_path
        if not image_input:
            return {"error": "No input image provided. Please specify input_image or image_path."}
        
        # Call provider directly with explicit parameters
        result = self.provider.edit_image(
            prompt=prompt,
            image_input=image_input,
            aspect_ratio=aspect_ratio,
            output_format=output_format,
            prompt_upsampling=prompt_upsampling,
            safety_tolerance=safety_tolerance,
            image_name=image_name,
            seed=seed,
        )
        
        # Convert provider response to tool response format
        if result["success"]:
            return {
                "success": True,
                "file_path": result["file_path"],
                "full_path": result.get("full_path"),
                "message": result["message"]
            }
        else:
            return {"success": False, "error": result["error"]}


class FluxImageToolkit(Toolkit):
    """Flux Image Toolkit combining generation and editing capabilities."""

    def __init__(
        self,
        name: str = "FluxImageToolkit",
        api_key: Optional[str] = None,
        save_path: str = "./images",
        storage_handler: Optional[FileStorageHandler] = None,
    ):
        provider = FluxImageProvider(
            api_key=api_key,
            storage_handler=storage_handler,
            base_path=save_path
        )
           
        # Initialize tools with provider
        generation_tool = FluxImageGenerationTool(provider)
        edit_tool = FluxImageEditTool(provider)

        super().__init__(name=name, tools=[generation_tool, edit_tool])
        
        
        self.provider = provider
        # Store configuration
        self.api_key = api_key
        self.save_path = save_path
        self.storage_handler = storage_handler