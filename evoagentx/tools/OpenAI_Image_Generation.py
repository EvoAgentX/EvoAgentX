from typing import Dict, Optional, List
from .tool import Tool

class OpenAI_ImageGenerationTool(Tool):
    name: str = "image_generation"
    description: str = "Generate images from text prompts using an image model."

    inputs: Dict[str, Dict[str, str]] = {
        "prompt": {
            "type": "string",
            "description": "The prompt describing the image to generate. Required."
        },
        "image_name": {
            "type": "string",
            "description": "The name of the image to generate. Optional."
        },
        "size": {
            "type": "string",
            "description": "Image dimensions (e.g., 1024x1024, 1536x1024, 1024x1536). Optional."
        },
        "quality": {
            "type": "string",
            "description": "Rendering quality (low, medium, high). Optional."
        },
        "output_format": {
            "type": "string",
            "description": "File output format (png, jpeg, webp). Optional."
        },
        "output_compression": {
            "type": "integer",
            "description": "Compression level (0-100) for jpeg/webp. Optional."
        },
        "background": {
            "type": "string",
            "description": "Background: transparent or opaque or auto. Optional."
        }
    }
    required: Optional[List[str]] = ["prompt"]

    def __init__(self, api_key: str, organization_id: str, model: str = "gpt-4o", save_path: str = "./"):
        super().__init__()
        self.api_key = api_key
        self.organization_id = organization_id
        self.model = model
        self.save_path = save_path

    def __call__(
        self,
        prompt: str,
        image_name: str = None,
        size: str = None,
        quality: str = None,
        output_format: str = None,
        output_compression: int = None,
        background: str = None
    ):
        from openai import OpenAI
        import base64        
        from PIL import Image
        import os
        import io

        tool_dict = {
            "type": "image_generation"
        }
        if size:
            tool_dict["size"] = size
        if quality:
            tool_dict["quality"] = quality
        if output_format:
            tool_dict["output_format"] = output_format
        if output_compression:
            tool_dict["output_compression"] = output_compression
        if background:
            tool_dict["background"] = background

        client = OpenAI(api_key = self.api_key,
                        organization = self.organization_id,
                        )

        response = client.responses.create(
            model=self.model,
            input=prompt,
            tools=[tool_dict]
        )
        
        image_data = [output.result
                      for output in response.output
                      if output.type == "image_generation_call"
                      ]

        if image_data:
            image_base64 = image_data[0]

            os.makedirs(self.save_path, exist_ok=True)

            image_name = image_name or "image.png"

            if not image_name.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                image_name += ".png"

            file_path = os.path.join(self.save_path, image_name)
            
            with open(file_path, "wb") as f:
                f.write(base64.b64decode(image_base64))
        
        return {"file_path": file_path}