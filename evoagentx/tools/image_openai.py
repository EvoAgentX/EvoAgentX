from typing import Dict, Optional, List, Tuple
import base64
import requests
import time
from dotenv import load_dotenv
from .tool import Tool, Toolkit
from .storage_handler import FileStorageHandler, LocalStorageHandler

load_dotenv()


def download_image(url: str, timeout: float = 20.0) -> Tuple[bytes, str]:
    """Download an image from HTTP(S) URL and return (bytes, ext).

    Many CDNs (including Wikimedia) block requests without a browser-like User-Agent.
    We send minimal headers to improve compatibility.

    ext is inferred from Content-Type header or URL path; defaults to ".png".
    This function is intentionally independent of storage handlers/classes.
    """
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
            ),
            "Accept": "image/*,*/*;q=0.8",
        }
        resp = requests.get(url, timeout=timeout, stream=True, headers=headers)
        resp.raise_for_status()
        content_type = resp.headers.get("content-type", "").lower()
        ext = ".png"
        if "image/" in content_type:
            ct = content_type.split(";")[0].strip()
            if ct == "image/png":
                ext = ".png"
            elif ct in ("image/jpeg", "image/jpg"):
                ext = ".jpg"
            elif ct == "image/webp":
                ext = ".webp"
            elif ct == "image/gif":
                ext = ".gif"
        else:
            # Fallback from URL suffix
            lower_url = url.split("?")[0].lower()
            for cand in (".png", ".jpg", ".jpeg", ".webp", ".gif"):
                if lower_url.endswith(cand):
                    ext = ".jpg" if cand == ".jpeg" else cand
                    break
        return resp.content, ext
    except Exception as e:
        raise Exception(f"Failed to download image {url}: {e}")


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

    def __init__(
        self, 
        name: str = None, 
        api_key: str = None, 
        organization_id: str = None, 
        model: str = "dall-e-3", 
        save_path: str = "./generated_images", 
        storage_handler: Optional[FileStorageHandler] = None
    ):
        super().__init__()
        self.name = name or self.name
        self.api_key = api_key
        self.organization_id = organization_id
        self.model = model
        self.save_path = save_path
        self.storage_handler = storage_handler or LocalStorageHandler(base_path=save_path)

    def __call__(
        self,
        prompt: str,
        image_name: str = "default_openai_image_generation",
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
            urls: List[str] = []
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
                    
                    if result.get("success"):
                        results.append(filename)
                        if isinstance(result.get("url"), str):
                            urls.append(result["url"])
                    else:
                        results.append(f"Error saving image {i+1}: {result.get('error', 'Unknown error')}")
                except Exception as e:
                    results.append(f"Error saving image {i+1}: {e}")

            response = {"results": results, "count": len(results)}
            if urls:
                response["urls"] = urls
            return response
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
        "image_urls": {"type": "array", "items": {"type": "string", "description": "HTTP(S) image URL"}, "description": "HTTP(S) image URL"},
        "image_paths": {"type": "array", "items": {"type": "string", "description": "Image path(s) png/webp/jpg <50MB"}, "description": "Image path(s) png/webp/jpg <50MB. Required. Single string accepted and normalized to array."},
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
    required: Optional[List[str]] = ["prompt"]

    def __init__(
        self, 
        name: str = None, 
        api_key: str = None, 
        organization_id: str = None,
        model: str = "dall-e-2",
        save_path: str = "./edited_images", 
        storage_handler: Optional[FileStorageHandler] = None
    ):
        super().__init__()
        self.api_key = api_key
        self.organization_id = organization_id
        self.model = model
        self.name = name or self.name
        self.save_path = save_path
        self.storage_handler = storage_handler or LocalStorageHandler(base_path=save_path)

    def __call__(
        self,
        prompt: str,
        image_urls: list = None,
        image_paths: list = None,
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
        image_name: str = "default_openai_image_edit",
    ):
        try:
            client = self._create_openai_client()

            # Normalize local path inputs (`image_paths` only)
            normalized_local_paths: List[str] = []
            if image_paths is not None:
                if isinstance(image_paths, str):
                    normalized_local_paths.append(image_paths)
                elif isinstance(image_paths, list):
                    normalized_local_paths.extend([p for p in image_paths if isinstance(p, str)])

            # Prepare temp dir for URL downloads if needed
            tmp_dir_rel: Optional[str] = None
            downloaded_rel_paths: List[str] = []
            if image_urls:
                ts = int(time.time())
                tmp_dir_rel = f"_tmp_openai_edit/{ts}"
                # Ensure temp directory exists via storage handler
                self.storage_handler.create_directory(tmp_dir_rel)
                # Download each URL to temp dir
                for idx, url in enumerate(image_urls):
                    try:
                        content, ext = download_image(url)
                        filename = f"url_{idx+1}{ext}"
                        rel_path = f"{tmp_dir_rel}/{filename}"
                        save_res = self.storage_handler.save(rel_path, content)
                        if not save_res.get("success"):
                            raise Exception(save_res.get("error", "save failed"))
                        downloaded_rel_paths.append(rel_path)
                    except Exception:
                        # Skip failed download; continue if others exist
                        continue

            # Merge all candidate paths
            all_rel_paths: List[str] = normalized_local_paths + downloaded_rel_paths

            if not all_rel_paths:
                return {"error": "No images provided for editing (need local paths or downloadable URLs)"}

            # Open image files using translated system paths so the OpenAI SDK can infer filename/content-type
            opened_images = []
            mask_fh = None
            try:
                for p in all_rel_paths:
                    sys_path = self.storage_handler.translate_in(p)
                    opened_images.append(open(sys_path, "rb"))

                api_kwargs = {
                    "model": self.model,
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
                urls: List[str] = []
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
                        
                        if result.get("success"):
                            results.append(filename)
                            if isinstance(result.get("url"), str):
                                urls.append(result["url"])
                        else:
                            results.append(f"Error saving image {i+1}: {result.get('error', 'Unknown error')}")
                    except Exception as e:
                        results.append(f"Error saving image {i+1}: {e}")
                response = {"results": results, "count": len(results)}
                if urls:
                    response["urls"] = urls
                return response
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
                # Cleanup downloaded temp files/dir
                try:
                    if downloaded_rel_paths:
                        for rel in downloaded_rel_paths:
                            # Delete files
                            try:
                                self.storage_handler.delete(rel)
                            except Exception:
                                pass
                    if tmp_dir_rel:
                        try:
                            self.storage_handler.delete(tmp_dir_rel)
                        except Exception:
                            pass
                except Exception:
                    pass
        except Exception as e:
            return {"error": f"{self.model} editing failed: {e}"}

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
        "image_urls": {"type": "array", "items": {"type": "string", "description": "HTTP(S) image URL"}, "description": "HTTP(S) image URL. Optional if image_path provided."},
        "image_paths": {"type": "array", "items": {"type": "string", "description": "Local image path; converted to data URL internally."}, "description": "Local image path; converted to data URL internally."},
        "model": {"type": "string", "description": "OpenAI model for responses.create (e.g., gpt-4o-mini, gpt-4.1, gpt-5). Optional."},
    }
    required: Optional[List[str]] = ["prompt"]

    def __init__(
        self, 
        api_key: str, 
        organization_id: str = None, 
        model: str = "gpt-4o-mini", 
        storage_handler: Optional[FileStorageHandler] = None, 
        name: str = None
    ):
        super().__init__()
        self.api_key = api_key
        self.organization_id = organization_id
        self.model = model
        self.name = name or self.name
        self.storage_handler = storage_handler or LocalStorageHandler()

    def __call__(
        self,
        prompt: str,
        image_urls: list = None,
        image_paths: list = None,
        model: str = None,
    ):
        try:
            client = self._create_openai_client()
            actual_model = model if model else self.model

            # Check if both image_urls and image_paths are provided
            if image_urls is None and image_paths is None:
                return {"error": "At least one of `image_urls` or `image_paths` must be provided"}

            # Build content list starting with text
            content = [{"type": "input_text", "text": prompt}]
            
            # Add images from URLs
            if image_urls:
                for image_url in image_urls:
                    content.append({"type": "input_image", "image_url": image_url})
            
            # Add images from paths (convert to base64)
            if image_paths:
                import mimetypes
                import base64
                
                for image_path in image_paths:
                    mime, _ = mimetypes.guess_type(image_path)
                    mime = mime or "image/png"
                    
                    # Use storage handler to read raw bytes directly
                    try:
                        system_path = self.storage_handler.translate_in(image_path)
                        file_content = self.storage_handler._read_raw(system_path)
                    except Exception as e:
                        return {"error": f"Could not read image {image_path}: {str(e)}"}
                    
                    b64 = base64.b64encode(file_content).decode("utf-8")
                    data_url = f"data:{mime};base64,{b64}"
                    content.append({"type": "input_image", "image_url": data_url})

            response = client.responses.create(
                model=actual_model,
                input=[{"role": "user","content": content}],
            )

            # Extract text robustly across SDK shapes
            def _extract_text(resp):
                # 1) Preferred: Responses API unified text
                text = getattr(resp, "output_text", None)
                if text:
                    return text

                # 2) Responses API structured output → output[*].content[*].text
                try:
                    output = getattr(resp, "output", None)
                    if output and isinstance(output, list):
                        parts = []
                        for block in output:
                            content_items = getattr(block, "content", None)
                            if content_items and isinstance(content_items, list):
                                for item in content_items:
                                    val = getattr(item, "text", None)
                                    if val:
                                        parts.append(val)
                        if parts:
                            return "\n".join(parts)
                except Exception:
                    pass

                # 3) Try dict-like serialization and parse similarly
                try:
                    as_dict = resp if isinstance(resp, dict) else None
                    if not as_dict and hasattr(resp, "model_dump"):
                        as_dict = resp.model_dump()
                    if not as_dict and hasattr(resp, "to_dict"):
                        as_dict = resp.to_dict()
                    if as_dict:
                        parts = []
                        for block in as_dict.get("output", []) or []:
                            for item in block.get("content", []) or []:
                                t = item.get("text")
                                if t:
                                    parts.append(t)
                        if parts:
                            return "\n".join(parts)
                        # Chat-style fallback if present
                        choices = as_dict.get("choices") or []
                        if choices:
                            msg = choices[0].get("message") or {}
                            c = msg.get("content")
                            if isinstance(c, str):
                                return c
                except Exception:
                    pass

                # 4) Chat Completions typed fallback
                try:
                    choices = getattr(resp, "choices", None)
                    if choices and isinstance(choices, list):
                        first = choices[0]
                        if isinstance(first, dict):
                            return first.get("message", {}).get("content", "")
                        message = getattr(first, "message", None)
                        content = getattr(message, "content", None)
                        if isinstance(content, str):
                            return content
                except Exception:
                    pass

                return ""

            text = _extract_text(response)
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