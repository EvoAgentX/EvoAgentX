from typing import Dict, Optional, List
import os
import base64
import requests
import time
from dotenv import load_dotenv
from .tool import Tool, Toolkit
from .storage_handler import FileStorageHandler, LocalStorageHandler

load_dotenv()


class OpenAIImageGenerationTool(Tool):
    name: str = "openai_image_generation"
    description: str = "OpenAI image generation supporting dall-e-2, dall-e-3, gpt-image-1 (with validation)."

    inputs: Dict[str, Dict[str, str]] = {
        "prompt": {"type": "string", "description": "Prompt text. Required."},
        "image_name": {"type": "string", "description": "Optional save name."},
        "model": {"type": "string", "description": "dall-e-2 | dall-e-3 | gpt-image-1"},
        "size": {"type": "string", "description": "Model-specific size."},
        "quality": {"type": "string", "description": "quality for gpt-image-1/dall-e-3"},
        "n": {"type": "integer", "description": "1-10 (1 for dalle-3)"},
        "background": {"type": "string", "description": "gpt-image-1 only"},
        "moderation": {"type": "string", "description": "gpt-image-1 only"},
        "output_compression": {"type": "integer", "description": "gpt-image-1 jpeg/webp"},
        "output_format": {"type": "string", "description": "gpt-image-1 png/jpeg/webp"},
        "partial_images": {"type": "integer", "description": "gpt-image-1 streaming partials"},
        "response_format": {"type": "string", "description": "url | b64_json for dalle-2/3"},
        "stream": {"type": "boolean", "description": "gpt-image-1 streaming"},
        "style": {"type": "string", "description": "dall-e-3 vivid|natural"},
    }
    required: Optional[List[str]] = ["prompt"]

    def __init__(self, api_key: str, organization_id: str = None, model: str = "dall-e-3", 
                 save_path: str = "./generated_images", storage_handler: Optional[FileStorageHandler] = None):
        super().__init__()
        self.api_key = api_key
        self.organization_id = organization_id
        self.model = model
        self.save_path = save_path
        self.storage_handler = storage_handler or LocalStorageHandler(base_path=save_path)

    def __call__(
        self,
        prompt: str,
        image_name: str = None,
        model: str = None,
        size: str = None,
        quality: str = None,
        n: int = None,
        background: str = None,
        moderation: str = None,
        output_compression: int = None,
        output_format: str = None,
        partial_images: int = None,
        response_format: str = None,
        stream: bool = None,
        style: str = None,
    ):
        try:
            client = self._create_openai_client()
            actual_model = model if model else self.model

            # Build parameters with validation (simplified)
            params = {
                "model": actual_model,
                "prompt": prompt,
                "response_format": response_format or "b64_json"
            }
            
            # Add optional parameters if provided
            for param in ["size", "quality", "n", "style"]:
                if locals()[param] is not None:
                    params[param] = locals()[param]

            response = client.images.generate(**params)

            # Save results using storage handler
            results = []
            for i, image_data in enumerate(response.data):
                try:
                    if hasattr(image_data, "b64_json") and image_data.b64_json:
                        image_bytes = base64.b64decode(image_data.b64_json)
                    elif hasattr(image_data, "url") and image_data.url:
                        r = requests.get(image_data.url)
                        r.raise_for_status()
                        image_bytes = r.content
                    else:
                        raise Exception("No valid image data in response")

                    # Generate unique filename
                    filename = self._get_unique_filename(image_name, i)
                    
                    # Save using storage handler
                    result = self.storage_handler.save(filename, image_bytes)
                    
                    if result["success"]:
                        results.append(filename)
                    else:
                        results.append(f"Error saving image {i+1}: {result.get('error', 'Unknown error')}")
                except Exception as e:
                    results.append(f"Error saving image {i+1}: {e}")

            return {"results": results, "count": len(results)}
        except Exception as e:
            return {"error": f"Image generation failed: {e}"}
    
    def _create_openai_client(self):
        try:
            from openai import OpenAI
            return OpenAI(api_key=self.api_key, organization=self.organization_id)
        except ImportError:
            raise ImportError("OpenAI package not installed. Please install with: pip install openai")
    
    def _get_unique_filename(self, image_name: str, index: int) -> str:
        """Generate a unique filename for the image"""
        if image_name:
            base = image_name.rsplit(".", 1)[0]
            filename = f"{base}_{index+1}.png"
        else:
            ts = int(time.time())
            filename = f"generated_{ts}_{index+1}.png"
        
        # Check if file exists and generate unique name
        counter = 1
        while self.storage_handler.exists(filename):
            if image_name:
                base = image_name.rsplit(".", 1)[0]
                filename = f"{base}_{index+1}_{counter}.png"
            else:
                filename = f"generated_{ts}_{index+1}_{counter}.png"
            counter += 1
            
        return filename


class OpenAIImageEditTool(Tool):
    name: str = "openai_image_edit"
    description: str = "Edit images using OpenAI gpt-image-1 (direct, minimal validation)."

    inputs: Dict[str, Dict[str, str]] = {
        "prompt": {"type": "string", "description": "Edit instruction. Required."},
        "images": {"type": "array", "description": "Image path(s) png/webp/jpg <50MB. Required. Single string accepted and normalized to array."},
        "mask_path": {"type": "string", "description": "Optional PNG mask path (same size as first image)."},
        "size": {"type": "string", "description": "1024x1024 | 1536x1024 | 1024x1536 | auto"},
        "n": {"type": "integer", "description": "1-10"},
        "background": {"type": "string", "description": "transparent | opaque | auto"},
        "input_fidelity": {"type": "string", "description": "high | low"},
        "output_compression": {"type": "integer", "description": "0-100 for jpeg/webp"},
        "output_format": {"type": "string", "description": "png | jpeg | webp (default png)"},
        "partial_images": {"type": "integer", "description": "0-3 partial streaming"},
        "quality": {"type": "string", "description": "auto | high | medium | low"},
        "stream": {"type": "boolean", "description": "streaming mode"},
        "image_name": {"type": "string", "description": "Optional output base name"},
    }
    required: Optional[List[str]] = ["prompt", "images"]

    def __init__(self, api_key: str, organization_id: str = None, save_path: str = "./edited_images", 
                 storage_handler: Optional[FileStorageHandler] = None):
        super().__init__()
        self.api_key = api_key
        self.organization_id = organization_id
        self.save_path = save_path
        self.storage_handler = storage_handler or LocalStorageHandler(base_path=save_path)

    def __call__(
        self,
        prompt: str,
        images: list,
        mask_path: str = None,
        size: str = None,
        n: int = None,
        background: str = None,
        input_fidelity: str = None,
        output_compression: int = None,
        output_format: str = None,
        partial_images: int = None,
        quality: str = None,
        stream: bool = None,
        image_name: str = None,
    ):
        try:
            client = self._create_openai_client()

            # Accept either list[str] or a single string at runtime
            if isinstance(images, str):
                image_paths = [images]
            else:
                image_paths = list(images)

            # For simplicity, just use the first image for editing
            if not image_paths:
                return {"error": "No images provided for editing"}
            
            # Open image files using translated system paths so the OpenAI SDK can infer filename/content-type
            opened_images = []
            mask_fh = None
            try:
                for p in image_paths:
                    sys_path = self.storage_handler.translate_in(p)
                    opened_images.append(open(sys_path, "rb"))

                api_kwargs = {
                    "model": "gpt-image-1",
                    "prompt": prompt,
                    "image": opened_images if len(opened_images) > 1 else opened_images[0],
                }
                
                # Add optional parameters
                if size is not None:
                    api_kwargs["size"] = size
                if n is not None:
                    api_kwargs["n"] = n
                if output_format is not None:
                    api_kwargs["output_format"] = output_format

                if mask_path:
                    try:
                        mask_sys_path = self.storage_handler.translate_in(mask_path)
                        mask_fh = open(mask_sys_path, "rb")
                        api_kwargs["mask"] = mask_fh
                    except Exception as e:
                        # Proceed without mask if it fails to open
                        pass

                response = client.images.edit(**api_kwargs)

                # Save results
                results = []
                for i, img in enumerate(response.data):
                    try:
                        if hasattr(img, "b64_json") and img.b64_json:
                            img_bytes = base64.b64decode(img.b64_json)
                        elif hasattr(img, "url") and img.url:
                            r = requests.get(img.url)
                            r.raise_for_status()
                            img_bytes = r.content
                        else:
                            raise Exception("No valid image data in response")
                        
                        ts = int(time.time())
                        if image_name:
                            filename = f"{image_name.rsplit('.', 1)[0]}_{i+1}.png"
                        else:
                            filename = f"image_edit_{ts}_{i+1}.png"
                        
                        # Save using storage handler
                        result = self.storage_handler.save(filename, img_bytes)
                        
                        if result["success"]:
                            results.append(filename)
                        else:
                            results.append(f"Error saving image {i+1}: {result.get('error', 'Unknown error')}")
                    except Exception as e:
                        results.append(f"Error saving image {i+1}: {e}")

                return {"results": results, "count": len(results)}
            finally:
                # Ensure file handles are closed
                try:
                    for fh in opened_images:
                        try:
                            fh.close()
                        except Exception:
                            pass
                except Exception:
                    pass
                if mask_fh is not None:
                    try:
                        mask_fh.close()
                    except Exception:
                        pass
        except Exception as e:
            return {"error": f"gpt-image-1 editing failed: {e}"}

    def _create_openai_client(self):
        try:
            from openai import OpenAI
            return OpenAI(api_key=self.api_key, organization=self.organization_id)
        except ImportError:
            raise ImportError("OpenAI package not installed. Please install with: pip install openai")


class OpenAIImageAnalysisTool(Tool):
    name: str = "openai_image_analysis"
    description: str = "Simple image analysis via OpenAI Responses API (input_text + input_image)."

    inputs: Dict[str, Dict[str, str]] = {
        "prompt": {"type": "string", "description": "User question/instruction. Required."},
        "image_url": {"type": "string", "description": "HTTP(S) image URL. Optional if image_path provided."},
        "image_path": {"type": "string", "description": "Local image path; converted to data URL internally."},
        "model": {"type": "string", "description": "OpenAI model for responses.create (e.g., gpt-4o-mini, gpt-4.1, gpt-5). Optional."},
    }
    required: Optional[List[str]] = ["prompt"]

    def __init__(self, api_key: str, organization_id: str = None, model: str = "gpt-4o-mini", 
                 storage_handler: Optional[FileStorageHandler] = None):
        super().__init__()
        self.api_key = api_key
        self.organization_id = organization_id
        self.model = model
        self.storage_handler = storage_handler or LocalStorageHandler()

    def __call__(
        self,
        prompt: str,
        image_url: str = None,
        image_path: str = None,
        model: str = None,
    ):
        try:
            client = self._create_openai_client()
            actual_model = model if model else self.model

            # Resolve image source: prefer URL, else local path to data URL
            final_image_url = image_url
            if not final_image_url and image_path:
                import mimetypes
                mime, _ = mimetypes.guess_type(image_path)
                mime = mime or "image/png"
                
                # Use storage handler to read raw bytes directly
                try:
                    system_path = self.storage_handler.translate_in(image_path)
                    content = self.storage_handler._read_raw(system_path)
                except Exception as e:
                    return {"error": f"Could not read image {image_path}: {str(e)}"}
                
                b64 = base64.b64encode(content).decode("utf-8")
                final_image_url = f"data:{mime};base64,{b64}"

            response = client.responses.create(
                model=actual_model,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": prompt},
                            {"type": "input_image", "image_url": final_image_url},
                        ],
                    }
                ],
            )

            # Prefer unified output_text when present
            text = getattr(response, "output_text", None)
            if text is None:
                # Fallback: try to assemble from content if SDK shape differs
                try:
                    choices = getattr(response, "output", None) or getattr(response, "choices", None)
                    if choices and isinstance(choices, list):
                        first = choices[0]
                        text = getattr(first, "message", {}).get("content", "") if isinstance(first, dict) else ""
                except Exception:
                    text = ""

            return {"content": text or ""}
        except Exception as e:
            return {"error": f"OpenAI image analysis failed: {e}"}
    
    def _create_openai_client(self):
        try:
            from openai import OpenAI
            return OpenAI(api_key=self.api_key, organization=self.organization_id)
        except ImportError:
            raise ImportError("OpenAI package not installed. Please install with: pip install openai")


class OpenAIImageToolkit(Toolkit):
    def __init__(self, name: str = "OpenAIImageToolkit", api_key: str = None, organization_id: str = None,
                 generation_model: str = "dall-e-3", save_path: str = "./generated_images", 
                 storage_handler: Optional[FileStorageHandler] = None):
        gen_tool = OpenAIImageGenerationTool(api_key=api_key, organization_id=organization_id,
                                             model=generation_model, save_path=save_path, 
                                             storage_handler=storage_handler)
        edit_tool = OpenAIImageEditTool(api_key=api_key, organization_id=organization_id,
                                        save_path=save_path, storage_handler=storage_handler)
        analysis_tool = OpenAIImageAnalysisTool(api_key=api_key, organization_id=organization_id, 
                                               storage_handler=storage_handler)
        super().__init__(name=name, tools=[gen_tool, edit_tool, analysis_tool])
        self.api_key = api_key
        self.organization_id = organization_id
        self.generation_model = generation_model
        self.save_path = save_path
        self.storage_handler = storage_handler