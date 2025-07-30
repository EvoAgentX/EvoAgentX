import requests
import base64
from typing import Dict, Optional, List
from .tool import Tool

class ImageAnalysisTool(Tool):
    name: str = "image_analysis"
    description: str = (
        "Analyze and understand images and PDF documents using a multimodal LLM (e.g., OpenRouter gpt-4o-mini). "
        "Supports image URLs, local image files, and local PDF files."
    )

    inputs: Dict[str, Dict[str, str]] = {
        "prompt": {
            "type": "string",
            "description": "The question or instruction for image or PDF analysis. Required."
        },
        "image_url": {
            "type": "string",
            "description": "The URL of the image to analyze. Optional."
        },
        "image_path": {
            "type": "string",
            "description": "The local file path of the image to analyze. Optional."
        },
        "pdf_path": {
            "type": "string",
            "description": "The local file path of the PDF document to analyze. Optional."
        }
    }
    required: Optional[List[str]] = ["prompt"]

    def __init__(self, api_key, model="openai/gpt-4o"):
        super().__init__()
        self.api_key = api_key
        self.model = model

    def __call__(
        self,
        prompt: str,
        image_url: str = None,
        image_path: str = None,
        pdf_path: str = None
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
            with open(image_path, "rb") as f:
                base64_image = base64.b64encode(f.read()).decode("utf-8")
            data_url = f"data:image/jpeg;base64,{base64_image}"
            messages[0]["content"].append({
                "type": "image_url",
                "image_url": {"url": data_url}
            })
        elif pdf_path:
            with open(pdf_path, "rb") as f:
                base64_pdf = base64.b64encode(f.read()).decode("utf-8")
            data_url = f"data:application/pdf;base64,{base64_pdf}"
            messages[0]["content"].append({
                "type": "file",
                "file": {
                    "filename": pdf_path.split("/")[-1],
                    "file_data": data_url
                }
            })

        payload = {
            "model": self.model,
            "messages": messages
        }

        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        response = requests.post(url, headers=headers, json=payload)
        try:
            data = response.json()
            # 只提取 message.content 和 usage
            result = {
                "content": data.get("choices", [{}])[0].get("message", {}).get("content", ""),
                "usage": data.get("usage", {})
            }
            return result
        except Exception as e:
            return {"error": f"Failed to parse OpenRouter response: {e}", "raw": response.text} 