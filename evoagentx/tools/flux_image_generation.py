from typing import Dict, Optional, List
from .tool import Tool, Toolkit
from .storage_handler import FileStorageHandler
import requests
import os
import base64
import time

class FluxImageGenerationTool(Tool):
    name: str = "flux_image_generation"
    description: str = "Generate images from text prompts using the bfl.ai flux-kontext-max API."

    inputs: Dict[str, Dict] = {
        "prompt": {"type": "string", "description": "The prompt describing the image to generate."},
        "input_image": {"type": "string", "description": "Base64 encoded input image for editing, optional."},
        "seed": {"type": "integer", "description": "Random seed, default is 42.", "default": 42},
        "aspect_ratio": {"type": "string", "description": "Aspect ratio, e.g. '1:1', optional."},
        "output_format": {"type": "string", "description": "Image format, default is jpeg.", "default": "jpeg"},
        "prompt_upsampling": {"type": "boolean", "description": "Enable prompt upsampling, default is false.", "default": False},
        "safety_tolerance": {"type": "integer", "description": "Safety tolerance level, default is 2.", "default": 2},
    }
    required: List[str] = ["prompt"]

    def __init__(self, api_key: str, save_path: str = "./imgs", storage_handler: FileStorageHandler = None):
        super().__init__()
        self.api_key = api_key
        self.save_path = save_path
        self.storage_handler = storage_handler

    def __call__(self, prompt: str, input_image: str = None, seed: int = 42, aspect_ratio: str = None, output_format: str = "jpeg", prompt_upsampling: bool = False, safety_tolerance: int = 2):
        # Create request
        payload = {
            "prompt": prompt,
            "seed": seed,
            "output_format": output_format,
            "prompt_upsampling": prompt_upsampling,
            "safety_tolerance": safety_tolerance,
        }
        if aspect_ratio:
            payload["aspect_ratio"] = aspect_ratio
        if input_image:
            payload["input_image"] = input_image

        headers = {
            "accept": "application/json",
            "x-key": self.api_key,
            "Content-Type": "application/json"
        }

        # Step 1: Create request
        response = requests.post("https://api.bfl.ai/v1/flux-kontext-max", json=payload, headers=headers)
        response.raise_for_status()
        request_data = response.json()
        
        request_id = request_data["id"]
        polling_url = request_data["polling_url"]

        # Step 2: Poll for result
        while True:
            time.sleep(2)  # 添加延迟避免过度轮询
            result = requests.get(
                polling_url,
                headers={
                    "accept": "application/json",
                    "x-key": self.api_key,
                },
                params={"id": request_id}
            ).json()
            
            if result["status"] == "Ready":
                # Get the image URL from result
                image_url = result["result"]["sample"]
                break
            elif result["status"] in ["Error", "Failed"]:
                raise ValueError(f"Generation failed: {result}")

        # Download the image
        image_response = requests.get(image_url)
        image_response.raise_for_status()
        image_content = image_response.content
        
        # Generate filename
        filename = f"flux_{seed}.{output_format}"
        i = 1
        while self.storage_handler.exists(filename):
            filename = f"flux_{seed}_{i}.{output_format}"
            i += 1
        
        # Save the image using storage handler with save_path
        if self.save_path:
            # If save_path is specified and different from default, prepend it to filename
            full_filename = f"{self.save_path}/{filename}"
        else:
            # Use the filename as is
            full_filename = filename
        
        result = self.storage_handler.save(full_filename, image_content)
        if result["success"]:
            return {"file_path": full_filename, "storage_handler": type(self.storage_handler).__name__}
        else:
            return {"error": f"Failed to save image: {result.get('error', 'Unknown error')}"}


class FluxImageGenerationToolkit(Toolkit):
    """
    Toolkit for Flux image generation with storage handler integration.
    """
    
    def __init__(self, name: str = "FluxImageGenerationToolkit", api_key: str = None, save_path: str = "./imgs", storage_handler: FileStorageHandler = None):
        """
        Initialize the Flux image generation toolkit.
        
        Args:
            name: Name of the toolkit
            api_key: API key for Flux image generation
            save_path: Default save path for images
            storage_handler: Storage handler for file operations
        """
        # Initialize storage handler if not provided
        if storage_handler is None:
            from .storage_file import LocalStorageHandler
            storage_handler = LocalStorageHandler(base_path=save_path)
        
        # Create the image generation tool
        tool = FluxImageGenerationTool(
            api_key=api_key,
            save_path=save_path,
            storage_handler=storage_handler
        )
        
        # Create tools list
        tools = [tool]
        
        # Initialize parent with tools
        super().__init__(name=name, tools=tools)
        
        # Store instance variables
        self.api_key = api_key
        self.save_path = save_path
        self.storage_handler = storage_handler
