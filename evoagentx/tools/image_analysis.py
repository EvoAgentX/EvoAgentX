import requests
import base64
from typing import Dict, Optional, List
from .tool import Tool, Toolkit
from .storage_handler import FileStorageHandler

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

    def __init__(self, api_key, model="openai/gpt-4o", storage_handler: FileStorageHandler = None):
        super().__init__()
        self.api_key = api_key
        self.model = model
        self.storage_handler = storage_handler

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
            # Read image using storage handler
            result = self.storage_handler.read(image_path)
            if result["success"]:
                image_content = result["content"]
                if isinstance(image_content, str):
                    # If content is string, encode it
                    image_content = image_content.encode('utf-8')
                base64_image = base64.b64encode(image_content).decode("utf-8")
            else:
                return {"error": f"Failed to read image: {result.get('error', 'Unknown error')}"}
            data_url = f"data:image/jpeg;base64,{base64_image}"
            messages[0]["content"].append({
                "type": "image_url",
                "image_url": {"url": data_url}
            })
        elif pdf_path:
            # Read PDF using storage handler
            result = self.storage_handler.read(pdf_path)
            if result["success"]:
                pdf_content = result["content"]
                if isinstance(pdf_content, str):
                    # If content is string, encode it
                    pdf_content = pdf_content.encode('utf-8')
                base64_pdf = base64.b64encode(pdf_content).decode("utf-8")
            else:
                return {"error": f"Failed to read PDF: {result.get('error', 'Unknown error')}"}
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


class ImageAnalysisToolkit(Toolkit):
    """
    Toolkit for image analysis with storage handler integration.
    """
    
    def __init__(self, name: str = "ImageAnalysisToolkit", api_key: str = None, model: str = "openai/gpt-4o", storage_handler: FileStorageHandler = None):
        """
        Initialize the image analysis toolkit.
        
        Args:
            name: Name of the toolkit
            api_key: API key for OpenRouter
            model: Model to use for image analysis
            storage_handler: Storage handler for file operations
        """
        # Initialize storage handler if not provided
        if storage_handler is None:
            from .storage_file import LocalStorageHandler
            storage_handler = LocalStorageHandler(base_path="./workplace/analysis")
        
        # Create the image analysis tool
        tool = ImageAnalysisTool(
            api_key=api_key,
            model=model,
            storage_handler=storage_handler
        )
        
        # Create tools list
        tools = [tool]
        
        # Initialize parent with tools
        super().__init__(name=name, tools=tools)
        
        # Store instance variables
        self.api_key = api_key
        self.model = model
        self.storage_handler = storage_handler 