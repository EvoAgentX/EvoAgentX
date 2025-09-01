from typing import Dict, Optional, List
from .tool import Tool, Toolkit
from .storage_handler import FileStorageHandler

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

    def __init__(self, api_key: str, organization_id: str, model: str = "gpt-4o", save_path: str = "./", storage_handler: FileStorageHandler = None):
        super().__init__()
        self.api_key = api_key
        self.organization_id = organization_id
        self.model = model
        self.save_path = save_path
        self.storage_handler = storage_handler

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
            image_content = base64.b64decode(image_base64)

            image_name = image_name or "image.png"

            if not image_name.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                image_name += ".png"

            # Save the image using storage handler
            result = self.storage_handler.save(image_name, image_content)
            if result["success"]:
                return {"file_path": image_name, "storage_handler": type(self.storage_handler).__name__}
            else:
                return {"error": f"Failed to save image: {result.get('error', 'Unknown error')}"}
        
        return {"error": "No image data received"}


class OpenAIImageGenerationToolkit(Toolkit):
    """
    Toolkit for OpenAI image generation with storage handler integration.
    """
    
    def __init__(self, name: str = "OpenAIImageGenerationToolkit", api_key: str = None, organization_id: str = None, model: str = "gpt-4o", save_path: str = "./", storage_handler: FileStorageHandler = None):
        """
        Initialize the OpenAI image generation toolkit.
        
        Args:
            name: Name of the toolkit
            api_key: API key for OpenAI
            organization_id: Organization ID for OpenAI
            model: Model to use for image generation
            save_path: Default save path for images
            storage_handler: Storage handler for file operations
        """
        # Initialize storage handler if not provided
        if storage_handler is None:
            from .storage_file import LocalStorageHandler
            storage_handler = LocalStorageHandler(base_path="./workplace/images")
        
        # Create the image generation tool
        tool = OpenAI_ImageGenerationTool(
            api_key=api_key,
            organization_id=organization_id,
            model=model,
            save_path=save_path,
            storage_handler=storage_handler
        )
        
        # Create tools list
        tools = [tool]
        
        # Initialize parent with tools
        super().__init__(name=name, tools=tools)
        
        # Store instance variables
        self.api_key = api_key
        self.organization_id = organization_id
        self.model = model
        self.save_path = save_path
        self.storage_handler = storage_handler