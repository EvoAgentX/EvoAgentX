from typing import Dict, Optional, List
from ...tool import Tool
from ...storage_handler import FileStorageHandler, LocalStorageHandler
from .openai_utils import (
    create_openai_client,
    build_validation_params,
    validate_parameters,
    handle_validation_result,
)


class OpenAIImageGenerationTool(Tool):
    name: str = "openai_image_generation"
    description: str = "OpenAI image generation supporting dall-e-2, dall-e-3, gpt-image-1 (with validation)."

    inputs: Dict[str, Dict[str, str]] = {
        "prompt": {"type": "string", "description": "Prompt text. Required."},
        "image_name": {"type": "string", "description": "Optional save name."},
        "model": {"type": "string", "description": "dall-e-2 | dall-e-3 | gpt-image-1"},
        "size": {"type": "string", "description": "Model-specific size."},
        "quality": {"type": "string", "description": "quality for gpt-image-1/dall-e-3"},
        "n": {"type": "integer", "description": "1-10 (1 for dalle-3)"},
        "background": {"type": "string", "description": "gpt-image-1 only"},
        "moderation": {"type": "string", "description": "gpt-image-1 only"},
        "output_compression": {"type": "integer", "description": "gpt-image-1 jpeg/webp"},
        "output_format": {"type": "string", "description": "gpt-image-1 png/jpeg/webp"},
        "partial_images": {"type": "integer", "description": "gpt-image-1 streaming partials"},
        "response_format": {"type": "string", "description": "url | b64_json for dalle-2/3"},
        "stream": {"type": "boolean", "description": "gpt-image-1 streaming"},
        "style": {"type": "string", "description": "dall-e-3 vivid|natural"},
    }
    required: Optional[List[str]] = ["prompt"]

    def __init__(self, api_key: str, organization_id: str = None, model: str = "dall-e-3", 
                 save_path: str = "./generated_images", storage_handler: Optional[FileStorageHandler] = None):
        super().__init__()
        self.api_key = api_key
        self.organization_id = organization_id
        self.model = model
        self.save_path = save_path
        self.storage_handler = storage_handler or LocalStorageHandler(base_path=save_path)

    def __call__(
        self,
        prompt: str,
        image_name: str = None,
        model: str = None,
        size: str = None,
        quality: str = None,
        n: int = None,
        background: str = None,
        moderation: str = None,
        output_compression: int = None,
        output_format: str = None,
        partial_images: int = None,
        response_format: str = None,
        stream: bool = None,
        style: str = None,
    ):
        try:
            client = create_openai_client(self.api_key, self.organization_id)
            actual_model = model if model else self.model

            params_to_validate = build_validation_params(
                model=actual_model,
                prompt=prompt,
                size=size,
                quality=quality,
                n=n,
                background=background,
                moderation=moderation,
                output_compression=output_compression,
                output_format=output_format,
                partial_images=partial_images,
                response_format=response_format,
                stream=stream,
                style=style,
            )

            validation_result = validate_parameters(actual_model, params_to_validate, "generation")
            error = handle_validation_result(validation_result)
            if error:
                return error

            api_params = validation_result["validated_params"].copy()
            api_params.pop("image_name", None)

            response = client.images.generate(**api_params)

            # Save results using storage handler
            import base64
            results = []
            for i, image_data in enumerate(response.data):
                try:
                    if hasattr(image_data, "b64_json") and image_data.b64_json:
                        image_bytes = base64.b64decode(image_data.b64_json)
                    elif hasattr(image_data, "url") and image_data.url:
                        import requests
                        r = requests.get(image_data.url)
                        r.raise_for_status()
                        image_bytes = r.content
                    else:
                        raise Exception("No valid image data in response")

                    # Generate unique filename
                    filename = self._get_unique_filename(image_name, i)
                    
                    # Save using storage handler
                    result = self.storage_handler.save(filename, image_bytes)
                    
                    if result["success"]:
                        results.append(filename)
                    else:
                        results.append(f"Error saving image {i+1}: {result.get('error', 'Unknown error')}")
                except Exception as e:
                    results.append(f"Error saving image {i+1}: {e}")

            return {"results": results, "count": len(results)}
        except Exception as e:
            return {"error": f"Image generation failed: {e}"}
    
    def _get_unique_filename(self, image_name: str, index: int) -> str:
        """Generate a unique filename for the image"""
        import time
        
        if image_name:
            base = image_name.rsplit(".", 1)[0]
            filename = f"{base}_{index+1}.png"
        else:
            ts = int(time.time())
            filename = f"generated_{ts}_{index+1}.png"
        
        # Check if file exists and generate unique name
        counter = 1
        while self.storage_handler.exists(filename):
            if image_name:
                base = image_name.rsplit(".", 1)[0]
                filename = f"{base}_{index+1}_{counter}.png"
            else:
                filename = f"generated_{ts}_{index+1}_{counter}.png"
            counter += 1
            
        return filename


