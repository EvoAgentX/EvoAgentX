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
    ):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        if image_url:
            messages[0]["content"].append({
                "type": "image_url",
                "image_url": {"url": image_url}
            })
        elif image_path:
            try:
                result = self.storage_handler.read(image_path)
                if not result["success"]:
                    return {"error": f"Failed to read image: {result.get('error', 'Unknown error')}"}
                
                # Get image content as bytes
                if isinstance(result["content"], bytes):
                    image_content = result["content"]
                else:
                    # If content is not bytes, convert to bytes
                    image_content = str(result["content"]).encode('utf-8')
                
                base64_image = base64.b64encode(image_content).decode("utf-8")
            except Exception as e:
                return {"error": f"Failed to read image: {e}"}
            data_url = f"data:image/jpeg;base64,{base64_image}"
            messages[0]["content"].append({
                "type": "image_url",
                "image_url": {"url": data_url}
            })
        elif pdf_path:
            try:
                result = self.storage_handler.read(pdf_path)
                if not result["success"]:
                    return {"error": f"Failed to read PDF: {result.get('error', 'Unknown error')}"}
                
                # Get PDF content as bytes
                if isinstance(result["content"], bytes):
                    pdf_content = result["content"]
                else:
                    # If content is not bytes, convert to bytes
                    pdf_content = str(result["content"]).encode('utf-8')
                
                base64_pdf = base64.b64encode(pdf_content).decode("utf-8")
            except Exception as e:
                return {"error": f"Failed to read PDF: {e}"}
            data_url = f"data:application/pdf;base64,{base64_pdf}"
            messages[0]["content"].append({
                "type": "file",
                "file": {"filename": pdf_path.split("/")[-1], "file_data": data_url}
            })

        payload = {"model": self.model, "messages": messages}
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        response = requests.post(url, headers=headers, json=payload)
        try:
            data = response.json()
            result = {
                "content": data.get("choices", [{}])[0].get("message", {}).get("content", ""),
                "usage": data.get("usage", {})
            }
            return result
        except Exception as e:
            return {"error": f"Failed to parse OpenRouter response: {e}", "raw": response.text}


## ImageAnalysisToolkit moved to toolkit.py to consolidate toolkits


