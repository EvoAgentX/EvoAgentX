import os
import requests
import base64
import mimetypes
from typing import Dict, Optional, List
from ...tool import Tool
from ...storage_handler import FileStorageHandler, LocalStorageHandler


class OpenRouterImageAnalysisTool(Tool):
    name: str = "image_analysis"
    description: str = "OpenRouter image analysis supporting models like openai/gpt-4o. It supports image URLs, local image files, and local PDF files."

    inputs: Dict[str, Dict[str, str]] = {
        "prompt": {"type": "string", "description": "Question or instruction for image/PDF analysis."},
        "image_url": {"type": "string", "description": "URL of the image (optional)."},
        "image_path": {"type": "string", "description": "Local image file path (optional)."},
        "pdf_path": {"type": "string", "description": "Local PDF file path (optional)."},
        "model": {"type": "string", "description": "OpenRouter model for analysis.", "default": "openai/gpt-4o"},
    }
    required: Optional[List[str]] = ["prompt"]

    def __init__(self, api_key: str = None, model: str = "openai/gpt-4o", storage_handler: Optional[FileStorageHandler] = None):
        super().__init__()
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.model = model
        self.storage_handler = storage_handler or LocalStorageHandler()

    def _read_file_to_base64(self, file_path: str) -> str:
        """Read file and convert to base64 string."""
        system_path = self.storage_handler.translate_in(file_path)
        content = self.storage_handler._read_raw(system_path)
        return base64.b64encode(content).decode("utf-8")

    def _build_image_content(self, image_url: str = None, image_path: str = None):
        """Build image content for API request."""
        # Prefer URL over local path
        if image_url:
            return {
                "type": "image_url",
                "image_url": {"url": image_url}
            }
        
        if image_path:
            # Guess MIME type from filename
            mime, _ = mimetypes.guess_type(image_path)
            mime = mime or "image/png"
            
            # Read and encode image
            base64_image = self._read_file_to_base64(image_path)
            data_url = f"data:{mime};base64,{base64_image}"
            
            return {
                "type": "image_url",
                "image_url": {"url": data_url}
            }
        
        return None

    def _build_pdf_content(self, pdf_path: str):
        """Build PDF content for API request."""
        base64_pdf = self._read_file_to_base64(pdf_path)
        data_url = f"data:application/pdf;base64,{base64_pdf}"
        
        return {
            "type": "file",
            "file": {
                "filename": pdf_path.split("/")[-1],
                "file_data": data_url
            }
        }

    def _parse_response(self, response: requests.Response, actual_model: str) -> Dict:
        """Parse API response and extract content."""
        try:
            data = response.json()
            
            # Check for API errors
            if "error" in data:
                return {"error": f"OpenRouter API error: {data['error']}", "raw": data}
            
            # Extract response content
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            
            # If content is empty, provide debug information
            if not content or content.strip() == "":
                return {
                    "error": "Empty response from API",
                    "debug_info": {
                        "model": actual_model,
                        "choices_count": len(data.get("choices", [])),
                        "message_keys": list(data.get("choices", [{}])[0].get("message", {}).keys()) if data.get("choices") else [],
                        "response_status": response.status_code,
                        "full_response": data
                    }
                }
            
            return {
                "content": content,
                "usage": data.get("usage", {})
            }
        except Exception as e:
            return {"error": f"Failed to parse OpenRouter response: {e}", "raw": response.text}

    def __call__(
        self,
        prompt: str,
        image_url: str = None,
        image_path: str = None,
        pdf_path: str = None,
        model: str = None,
    ):
        try:
            actual_model = model or self.model
            
            # Build message with text prompt
            messages = [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}]
                }
            ]

            # Add media content (image or PDF)
            if pdf_path:
                # Handle PDF files
                media_content = self._build_pdf_content(pdf_path)
            else:
                # Handle images (URL or local path)
                media_content = self._build_image_content(image_url, image_path)
            
            if media_content:
                messages[0]["content"].append(media_content)

            # Build API request
            payload = {"model": actual_model, "messages": messages}
            url = "https://openrouter.ai/api/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            # Send request and parse response
            response = requests.post(url, headers=headers, json=payload)
            return self._parse_response(response, actual_model)
            
        except Exception as e:
            return {"error": f"OpenRouter image analysis failed: {e}"}
