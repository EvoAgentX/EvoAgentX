from typing import Dict, List
from ...tool import Tool
from ...storage_handler import FileStorageHandler
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

    def __init__(self, api_key: str, save_path: str = "./imgs", storage_handler: FileStorageHandler = None):
        super().__init__()
        self.api_key = api_key
        self.save_path = save_path
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

        filename = f"flux_{seed}.{output_format}"
        i = 1
        while self.storage_handler.exists(filename):
            filename = f"flux_{seed}_{i}.{output_format}"
            i += 1

        storage_relative_path = filename
        result = self.storage_handler.save(storage_relative_path, image_content)
        if result.get("success"):
            public_path = f"{self.save_path}/{filename}" if self.save_path else filename
            return {"file_path": public_path, "storage_handler": type(self.storage_handler).__name__}
        else:
            return {"error": f"Failed to save image: {result.get('error', 'Unknown error')}"}


