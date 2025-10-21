import os
from typing import Dict, Optional, List
from ...tool import Tool
from ...storage_handler import FileStorageHandler, LocalStorageHandler
from .openai_utils import create_openai_client


class OpenAIImageAnalysisTool(Tool):
    name: str = "openai_image_analysis"
    description: str = "OpenAI image analysis supporting models like gpt-4o-mini, gpt-4.1, gpt-5. It supports image URLs, local image files."

    inputs: Dict[str, Dict[str, str]] = {
        "prompt": {"type": "string", "description": "User question/instruction. Required."},
        "image_urls": {"type": "array", "items": {"type": "string"}, "description": "Array of HTTP(S) image URLs. Optional if image_paths provided."},
        "image_paths": {"type": "array", "items": {"type": "string"}, "description": "Array of local image paths; converted to data URLs internally."},
        "model": {"type": "string", "description": "OpenAI model for responses.create (e.g., gpt-4o-mini, gpt-4.1, gpt-5). Optional."},
    }
    required: Optional[List[str]] = ["prompt"]

    def __init__(self, name: str = None, api_key: str = None, organization_id: str = None, 
                model: str = "gpt-4o-mini", storage_handler: Optional[FileStorageHandler] = None):
        super().__init__()
        self.name = name or self.name
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.organization_id = organization_id or os.getenv("OPENAI_ORGANIZATION_ID")
        self.model = model
        self.storage_handler = storage_handler or LocalStorageHandler()

    def __call__(
        self,
        prompt: str,
        image_urls: List[str] = None,
        image_paths: List[str] = None,
        model: str = None,
    ):
        try:
            client = create_openai_client(self.api_key, self.organization_id)
            actual_model = model if model else self.model

            # Build content list starting with prompt
            content = [{"type": "input_text", "text": prompt}]
            
            # Add images from URLs
            if image_urls:
                for url in image_urls:
                    content.append({"type": "input_image", "image_url": url})
            
            # Add images from local paths (convert to data URLs)
            if image_paths:
                import base64
                import mimetypes
                
                for image_path in image_paths:
                    mime, _ = mimetypes.guess_type(image_path)
                    mime = mime or "image/png"
                    
                    # Use storage handler to read raw bytes directly
                    try:
                        # Translate user path to system path first
                        system_path = self.storage_handler.translate_in(image_path)
                        file_content = self.storage_handler._read_raw(system_path)
                    except Exception as e:
                        return {"error": f"Could not read image {image_path}: {str(e)}"}
                    
                    b64 = base64.b64encode(file_content).decode("utf-8")
                    data_url = f"data:{mime};base64,{b64}"
                    content.append({"type": "input_image", "image_url": data_url})

            response = client.responses.create(
                model=actual_model,
                input=[
                    {
                        "role": "user",
                        "content": content,
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


