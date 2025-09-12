from typing import Dict, List, Optional
from ...tool import Tool
from ...storage_handler import FileStorageHandler, LocalStorageHandler
import requests
import time


class FluxImageGenerationEditTool(Tool):
    name: str = "flux_image_generation_edit"
    description: str = (
        "Text-to-image and image-editing using the bfl.ai flux-kontext-max API. "
        "Without input_image: generate from prompt. With input_image (base64): edit/transform."
    )

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

    def __init__(self, api_key: str, storage_handler: Optional[FileStorageHandler] = None, 
                 base_path: str = "./imgs", save_path: str = None):
        super().__init__()
        self.api_key = api_key
        
        # Handle backward compatibility: if save_path is provided, use it as base_path
        if save_path is not None:
            base_path = save_path
            
        # Initialize storage handler
        if storage_handler is None:
            self.storage_handler = LocalStorageHandler(base_path=base_path)
        else:
            self.storage_handler = storage_handler

    def __call__(
        self,
        prompt: str,
        input_image: str = None,
        seed: int = 42,
        aspect_ratio: str = None,
        output_format: str = "jpeg",
        prompt_upsampling: bool = False,
        safety_tolerance: int = 2,
    ):
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
            "Content-Type": "application/json",
        }

        response = requests.post("https://api.bfl.ai/v1/flux-kontext-max", json=payload, headers=headers)
        response.raise_for_status()
        request_data = response.json()

        request_id = request_data["id"]
        polling_url = request_data["polling_url"]

        while True:
            time.sleep(2)
            result = requests.get(
                polling_url,
                headers={
                    "accept": "application/json",
                    "x-key": self.api_key,
                },
                params={"id": request_id},
            ).json()

            if result["status"] == "Ready":
                image_url = result["result"]["sample"]
                break
            elif result["status"] in ["Error", "Failed"]:
                raise ValueError(f"Generation failed: {result}")

        image_response = requests.get(image_url)
        image_response.raise_for_status()
        image_content = image_response.content

        # Generate unique filename using storage handler
        filename = self._get_unique_filename(seed, output_format)
        
        # Save image using storage handler
        result = self.storage_handler.save(filename, image_content)
        
        if result["success"]:
            return {
                "success": True,
                "file_path": filename,
                "full_path": result.get("full_path", filename),
                "message": f"Image saved successfully as {filename}"
            }
        else:
            return {
                "success": False,
                "error": f"Failed to save image: {result.get('error', 'Unknown error')}"
            }
    
    def _get_unique_filename(self, seed: int, output_format: str) -> str:
        """Generate a unique filename for the image"""
        base_filename = f"flux_{seed}.{output_format}"
        filename = base_filename
        counter = 1
        
        # Check if file exists and generate unique name
        while self.storage_handler.exists(filename):
            filename = f"flux_{seed}_{counter}.{output_format}"
            counter += 1
            
        return filename


