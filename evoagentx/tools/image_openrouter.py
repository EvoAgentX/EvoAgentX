from re import S
from typing import Dict, List, Optional
import os
import requests

from evoagentx.core.module import BaseModule
from .tool import Tool, Toolkit
from .storage_handler import FileStorageHandler, LocalStorageHandler


# Shared base to reduce duplication across OpenRouter image tools
class OpenRouterImageGenerationEditBase(BaseModule):
    """Common helper methods for OpenRouter image tools (generation/edit)."""

    def _url_to_image_part(self, url: str) -> Dict:
        return {"type": "image_url", "image_url": {"url": url}}

    def _guess_mime_from_name(self, name: str, default: str = "image/png") -> str:
        import mimetypes
        guess, _ = mimetypes.guess_type(name)
        return guess or default

    def _get_extension_from_mime(self, mime: str) -> str:
        if mime == "image/jpeg":
            return ".jpg"
        if mime == "image/webp":
            return ".webp"
        if mime == "image/heic":
            return ".heic"
        if mime == "image/heif":
            return ".heif"
        return ".png"

    def _path_to_data_url(self, path: str) -> str:
        import base64
        mime = self._guess_mime_from_name(path)
        try:
            system_path = self.storage_handler.translate_in(path)
            content = self.storage_handler._read_raw(system_path)
        except Exception as e:
            raise FileNotFoundError(f"Could not read file {path}: {str(e)}")
        b64 = base64.b64encode(content).decode("utf-8")
        return f"data:{mime};base64,{b64}"

    def _get_unique_filename(self, base_name: str, extension: str) -> str:
        filename = f"{base_name}{extension}"
        counter = 1
        while self.storage_handler.exists(filename):
            filename = f"{base_name}_{counter}{extension}"
            counter += 1
        return filename

    def _paths_to_image_parts(self, paths: list) -> List[Dict]:
        parts: List[Dict] = []
        for p in paths or []:
            try:
                parts.append(self._url_to_image_part(self._path_to_data_url(p)))
            except Exception:
                continue
        return parts

    def _urls_to_image_parts(self, urls: list) -> List[Dict]:
        return [self._url_to_image_part(u) for u in (urls or [])]

    def _save_images_from_choices(self, data: Dict, output_basename: str) -> Dict[str, List[str]]:
        import base64
        saved_paths: List[str] = []
        urls: List[str] = []
        if not data or not data.get("choices"):
            return {"saved_paths": saved_paths, "urls": urls}
        msg = data["choices"][0].get("message", {})
        images = msg.get("images") or []
        for im in images:
            image_url = (im.get("image_url") or {}).get("url")
            if not image_url:
                continue
            if image_url.startswith("data:") and "," in image_url:
                header, b64data = image_url.split(",", 1)
                mime = "image/png"
                if ";" in header:
                    try:
                        mime = header.split(":", 1)[1].split(";", 1)[0] or mime
                    except Exception:
                        pass
                ext = self._get_extension_from_mime(mime)
                filename = self._get_unique_filename(output_basename or "or_gen", ext)
                image_content = base64.b64decode(b64data)
                result = self.storage_handler.save(filename, image_content)
                if result.get("success"):
                    saved_paths.append(filename)
                    if isinstance(result.get("url"), str):
                        urls.append(result["url"])
                else:
                    # stop early on save error to surface issue
                    return {"saved_paths": [], "urls": []}
        return {"saved_paths": saved_paths, "urls": urls}


# Deprecated combined tool retained for reference only (not registered in toolkit)
class OpenRouterImageGenerationEditTool(OpenRouterImageGenerationEditBase, Tool):
    name: str = "openrouter_image_generation_edit"
    description: str = (
        "Text-to-image and image-editing via OpenRouter models (e.g., google/gemini-2.5-flash-image-preview). "
        "No images → generate; with images (URLs or local paths) → edit/compose."
    )

    inputs: Dict[str, Dict] = {
        "prompt": {"type": "string", "description": "Text prompt."},
        "image_urls": {"type": "array", "description": "Remote image URLs (optional)."},
        "image_paths": {"type": "array", "description": "Local image paths (optional)."},
        "model": {"type": "string", "description": "OpenRouter model id.", "default": "google/gemini-2.5-flash-image-preview"},
        "api_key": {"type": "string", "description": "OpenRouter API key (fallback to env OPENROUTER_API_KEY)."},
        "save_path": {"type": "string", "description": "Directory to save images (when data URLs).", "default": "./openrouter_images"},
        "output_basename": {"type": "string", "description": "Base filename for outputs.", "default": "or_gen"}
    }
    required: List[str] = ["prompt"]

    def __init__(self, api_key: str = None, storage_handler: Optional[FileStorageHandler] = None, 
                 base_path: str = "./openrouter_images"):
        super().__init__()
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.storage_handler = storage_handler or LocalStorageHandler(base_path=base_path)

    def __call__(
        self,
        prompt: str,
        image_urls: list = None,
        image_paths: list = None,
        model: str = "google/gemini-2.5-flash-image-preview",
        api_key: str = None,
        save_path: str = "./openrouter_images",
        output_basename: str = "or_gen",
    ):
        key = api_key or self.api_key
        if not key:
            return {"error": "OPENROUTER_API_KEY not provided."}

        messages = [{"role": "user", "content": prompt}]
        payload = {"model": model, "messages": messages, "modalities": ["image", "text"]}

        # Build content parts from URLs and/or local paths
        content_parts = [{"type": "text", "text": prompt}]
        if image_urls:
            content_parts.extend(self._urls_to_image_parts(image_urls))
        if image_paths:
            content_parts.extend(self._paths_to_image_parts(image_paths))
        if len(content_parts) > 1:
            payload["messages"][0] = {"role": "user", "content": content_parts}

        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        url = "https://openrouter.ai/api/v1/chat/completions"
        
        try:
            resp = requests.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.HTTPError as e:
            # Log the error details for debugging
            try:
                error_data = resp.json()
                return {"error": f"OpenRouter API error: {error_data}", "status_code": resp.status_code}
            except Exception:
                return {"error": f"OpenRouter API error: {e}", "status_code": resp.status_code}
        except Exception as e:
            return {"error": f"Request failed: {e}"}

        saved = self._save_images_from_choices(data, output_basename)
        saved_paths = saved.get("saved_paths", []) if isinstance(saved, dict) else (saved or [])
        urls = saved.get("urls", []) if isinstance(saved, dict) else []
        if saved_paths:
            resp = {"saved_paths": saved_paths}
            if urls:
                resp["urls"] = urls
            return resp
        return {"warning": "No image returned or saved.", "raw": data}


# New: dedicated generation-only tool
class OpenRouterImageGenerationTool(OpenRouterImageGenerationEditBase, Tool):
    name: str = "openrouter_image_generation"
    description: str = (
        "Text-to-image via OpenRouter models (e.g., google/gemini-2.5-flash-image-preview)."
    )

    inputs: Dict[str, Dict] = {
        "prompt": {"type": "string", "description": "Text prompt."},
        "output_basename": {"type": "string", "description": "Base filename for outputs.", "default": "or_gen"}
    }
    required: List[str] = ["prompt"]

    def __init__(
        self, 
        name: str = None, 
        api_key: str = None, 
        model: str = "google/gemini-2.5-flash-image-preview", 
        storage_handler: Optional[FileStorageHandler] = None, 
        save_path: str = "./openrouter_images"
    ):
        super().__init__()
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.model = model
        self.storage_handler = storage_handler or LocalStorageHandler(base_path=save_path)
        self.name = name or self.name 

    def __call__(
        self,
        prompt: str,
        output_basename: str = "default_or_generated_image",
    ):
        key = self.api_key
        model = self.model 

        if not key:
            return {"error": "OPENROUTER_API_KEY not provided."}

        messages = [{"role": "user", "content": prompt}]
        payload = {"model": model, "messages": messages, "modalities": ["image", "text"]}

        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        url = "https://openrouter.ai/api/v1/chat/completions"
        
        try:
            resp = requests.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.HTTPError as e:
            try:
                error_data = resp.json()
                return {"error": f"OpenRouter API error: {error_data}", "status_code": resp.status_code}
            except Exception:
                return {"error": f"OpenRouter API error: {e}", "status_code": resp.status_code}
        except Exception as e:
            return {"error": f"Request failed: {e}"}

        saved = self._save_images_from_choices(data, output_basename)
        saved_paths = saved.get("saved_paths", []) if isinstance(saved, dict) else (saved or [])
        urls = saved.get("urls", []) if isinstance(saved, dict) else []
        if saved_paths:
            resp = {"saved_paths": saved_paths}
            if urls:
                resp["urls"] = urls
            return resp
        return {"warning": "No image returned or saved.", "raw": data}


# New: dedicated edit-only tool
class OpenRouterImageEditTool(OpenRouterImageGenerationEditBase, Tool):
    name: str = "openrouter_image_edit"
    description: str = (
        "Image editing/compositing via OpenRouter models. Provide image_urls and/or image_paths."
    )

    inputs: Dict[str, Dict] = {
        "prompt": {"type": "string", "description": "Text prompt describing the edit."},
        "image_urls": {"type": "array", "items": {"type": "string", "description": "HTTP(S) image URL"}, "description": "Remote image URLs (optional)."},
        "image_paths": {"type": "array", "items": {"type": "string", "description": "Local image file path"}, "description": "Local image paths (optional)."},
        "output_basename": {"type": "string", "description": "Base filename for outputs.", "default": "or_gen"}
    }
    required: List[str] = ["prompt"]

    def __init__(
        self, 
        name: str = None, 
        api_key: str = None, 
        model: str = "google/gemini-2.5-flash-image-preview", 
        storage_handler: Optional[FileStorageHandler] = None, 
        save_path: str = "./openrouter_images"
    ):
        super().__init__()
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.model = model
        self.storage_handler = storage_handler or LocalStorageHandler(base_path=save_path)
        self.name = name or self.name  

    def __call__(
        self,
        prompt: str,
        image_urls: list = None,
        image_paths: list = None,
        output_basename: str = "default_or_edited_image",
    ):
        key = self.api_key
        model = self.model 
        if not key:
            return {"error": "OPENROUTER_API_KEY not provided."}

        if not (image_urls or image_paths):
            return {"error": "No image input provided. Provide image_urls or image_paths."}

        # Build content parts from URLs and/or local paths
        content_parts = [{"type": "text", "text": prompt}]
        if image_urls:
            content_parts.extend(self._urls_to_image_parts(image_urls)) 
        if image_paths:
            content_parts.extend(self._paths_to_image_parts(image_paths))

        payload = {"model": model, "messages": [{"role": "user", "content": content_parts}], "modalities": ["image", "text"]}

        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        url = "https://openrouter.ai/api/v1/chat/completions"
        
        try:
            resp = requests.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.HTTPError as e:
            try:
                error_data = resp.json()
                return {"error": f"OpenRouter API error: {error_data}", "status_code": resp.status_code}
            except Exception:
                return {"error": f"OpenRouter API error: {e}", "status_code": resp.status_code}
        except Exception as e:
            return {"error": f"Request failed: {e}"}
        # Save returned images and propagate URLs if the storage handler provides them
        saved = self._save_images_from_choices(data, output_basename)
        saved_paths = saved.get("saved_paths", []) if isinstance(saved, dict) else (saved or [])
        urls = saved.get("urls", []) if isinstance(saved, dict) else []
        if saved_paths:
            resp = {"saved_paths": saved_paths}
            if urls:
                resp["urls"] = urls
            return resp
        return {"warning": "No image returned or saved.", "raw": data}


class OpenRouterImageAnalysisTool(Tool):
    name: str = "openrouter_image_analysis"
    description: str = "Analyze images using OpenRouter vision models"

    inputs: Dict[str, Dict[str, str]] = {
        "prompt": {"type": "string", "description": "Analysis question or instruction"},
        "image_urls": {"type": "array", "items": {"type": "string", "description": "HTTP(S) image URL"}, "description": "HTTP(S) image URL"},
        "image_paths": {"type": "array", "items": {"type": "string", "description": "Local image file path"}, "description": "Local image file path"},
    }
    required: Optional[List[str]] = ["prompt"]

    def __init__(
        self, 
        name: str = None, 
        api_key: str = None, 
        model: str = "openai/gpt-4o", 
        storage_handler: Optional[FileStorageHandler] = None
    ):
        super().__init__()
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.model = model
        self.name = name or self.name 
        self.storage_handler = storage_handler or LocalStorageHandler()

    def __call__(
        self, 
        prompt: str = None, 
        image_urls: list = None, 
        image_paths: list = None,
    ):
        key = self.api_key
        if not key:
            return {"error": "OPENROUTER_API_KEY not provided."}

        actual_model = self.model

        # Determine image input
        if not image_urls and not image_paths:
            return {"error": "No image input provided. Please provide one of the following: 1) image_urls - a list of one or more HTTP/HTTPS URL(s) to image files, 2) image_paths - a list of one or more local file path(s) to image files. The image will be analyzed according to your prompt."}

        content_parts = [{"type": "text", "text": prompt or ""}]
        
        if image_urls:
            for url in image_urls:
                content_parts.append({"type": "image_url", "image_url": {"url": url}})
        
        if image_paths:
            for image_path in image_paths:
                try:
                    data_url = self._path_to_data_url(image_path)
                    content_parts.append({"type": "image_url", "image_url": {"url": data_url}})
                except Exception as e:
                    return {"error": f"Failed to read local image file '{image_path}': {e}. Please check that the file exists and is a valid image file."}

        payload = {
            "model": actual_model,
            "messages": [{"role": "user", "content": content_parts}],
            "modalities": ["image", "text"]
        }

        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        
        try:
            resp = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            return {"error": f"OpenRouter analysis failed: {e}"}

        if data.get("choices"):
            content = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
            return {"content": content, "usage": usage}
        
        return {"error": "No response from model", "raw": data}

    def _path_to_data_url(self, path: str) -> str:
        import base64
        import mimetypes
        
        mime, _ = mimetypes.guess_type(path)
        mime = mime or "image/png"
        
        try:
            system_path = self.storage_handler.translate_in(path)
            content = self.storage_handler._read_raw(system_path)
        except Exception as e:
            raise FileNotFoundError(f"Could not read file {path}: {str(e)}")
        
        b64 = base64.b64encode(content).decode("utf-8")
        return f"data:{mime};base64,{b64}"


class OpenRouterImageToolkit(Toolkit):
    def __init__(self, name: str = "OpenRouterImageToolkit", api_key: Optional[str] = None, 
                 storage_handler: Optional[FileStorageHandler] = None):
        analysis = OpenRouterImageAnalysisTool(api_key=api_key, storage_handler=storage_handler)
        generation = OpenRouterImageGenerationTool(api_key=api_key, storage_handler=storage_handler)
        edit = OpenRouterImageEditTool(api_key=api_key, storage_handler=storage_handler)
        # Only register analysis, generation, and edit; omit deprecated combined tool
        super().__init__(name=name, tools=[analysis, generation, edit])
        self.api_key = api_key
        self.storage_handler = storage_handler