import requests
import base64
from typing import Dict, Optional, List
from ...tool import Tool
from ...storage_handler import FileStorageHandler, LocalStorageHandler


class ImageAnalysisTool(Tool):
    name: str = "image_analysis"
    description: str = (
        "Analyze and understand images and PDF documents using a multimodal LLM (via OpenRouter). "
        "Supports image URLs, local image files, and local PDF files."
    )

    inputs: Dict[str, Dict[str, str]] = {
        "prompt": {"type": "string", "description": "Question or instruction for image/PDF analysis."},
        "image_url": {"type": "string", "description": "URL of the image (optional)."},
        "image_path": {"type": "string", "description": "Local image file path (optional)."},
        "pdf_path": {"type": "string", "description": "Local PDF file path (optional)."},
        "model": {"type": "string", "description": "OpenRouter model for analysis.", "default": "openai/gpt-4o"},
    }
    required: Optional[List[str]] = ["prompt"]

    def __init__(self, api_key, model="openai/gpt-4o", storage_handler: Optional[FileStorageHandler] = None):
        super().__init__()
        self.api_key = api_key
        self.model = model
        self.storage_handler = storage_handler or LocalStorageHandler()

    def __call__(
        self,
        prompt: str,
        image_url: str = None,
        image_path: str = None,
        pdf_path: str = None,
        model: str = None,
    ):
        # Use provided model or default to instance model
        actual_model = model or self.model
        
        # Build message with text prompt
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        # Add image from URL
        if image_url:
            messages[0]["content"].append({
                "type": "image_url",
                "image_url": {"url": image_url}
            })
        # Add image from local path
        elif image_path:
            try:
                import mimetypes
                
                # Guess MIME type from filename
                mime, _ = mimetypes.guess_type(image_path)
                mime = mime or "image/png"
                
                # Use storage handler to read raw bytes directly
                # This bypasses the high-level read() method that processes images
                try:
                    # Translate user path to system path first
                    system_path = self.storage_handler.translate_in(image_path)
                    image_content = self.storage_handler._read_raw(system_path)
                except Exception as e:
                    return {"error": f"Could not read image {image_path}: {str(e)}"}
                
                # Encode to base64
                base64_image = base64.b64encode(image_content).decode("utf-8")
                
                # Create data URL with correct MIME type
                data_url = f"data:{mime};base64,{base64_image}"
                messages[0]["content"].append({
                    "type": "image_url",
                    "image_url": {"url": data_url}
                })
            except Exception as e:
                return {"error": f"Failed to read image: {e}"}
        # Add PDF from local path
        elif pdf_path:
            try:
                # Read PDF using storage handler
                result = self.storage_handler.read(pdf_path)
                if not result["success"]:
                    return {"error": f"Failed to read PDF: {result.get('error', 'Unknown error')}"}
                
                # Get PDF content as bytes
                if isinstance(result["content"], bytes):
                    pdf_content = result["content"]
                else:
                    # If content is not bytes, convert to bytes
                    pdf_content = str(result["content"]).encode('utf-8')
                
                # Encode to base64
                base64_pdf = base64.b64encode(pdf_content).decode("utf-8")
            except Exception as e:
                return {"error": f"Failed to read PDF: {e}"}
            
            # Create data URL for PDF
            data_url = f"data:application/pdf;base64,{base64_pdf}"
            messages[0]["content"].append({
                "type": "file",
                "file": {"filename": pdf_path.split("/")[-1], "file_data": data_url}
            })

        # Build API request payload
        payload = {"model": actual_model, "messages": messages}
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        
        # Send request
        response = requests.post(url, headers=headers, json=payload)
        
        try:
            data = response.json()
            
            # Check for API errors
            if "error" in data:
                return {"error": f"OpenRouter API error: {data['error']}", "raw": data}
            
            # Extract response content and usage info
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
            
            result = {
                "content": content,
                "usage": data.get("usage", {})
            }
            return result
        except Exception as e:
            return {"error": f"Failed to parse OpenRouter response: {e}", "raw": response.text}
