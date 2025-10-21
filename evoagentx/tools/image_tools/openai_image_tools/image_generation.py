import os
from typing import Dict, Optional, List
from ...tool import Tool
from ...storage_handler import FileStorageHandler, LocalStorageHandler
from .openai_utils import (
    create_openai_client,
    build_validation_params,
    validate_parameters,
    handle_validation_result,
)
from .image_postprocessor import OpenAIImagePostProcessor


class OpenAIImageGenerationTool(Tool):
    name: str = "openai_image_generation"
    description: str = "OpenAI image generation supporting dall-e-2, dall-e-3, gpt-image-1 (with validation). It supports automatic postprocessing for unsupported sizes/formats."

    inputs: Dict[str, Dict[str, str]] = {
        "prompt": {"type": "string", "description": "Prompt text. Required."},
        "image_name": {"type": "string", "description": "Optional save name."},
        "model": {"type": "string", "description": "dall-e-2 | dall-e-3 | gpt-image-1. Default: dall-e-3"},
        "size": {"type": "string", "description": "Image size. dall-e-2: 256x256|512x512|1024x1024. dall-e-3: 1024x1024|1792x1024|1024x1792. gpt-image-1: 1024x1024|1536x1024|1024x1536|auto"},
        "quality": {"type": "string", "description": "Image quality. dall-e-3: standard|hd. gpt-image-1: auto|high|medium|low"},
        "n": {"type": "integer", "description": "Number of images. 1-10 for dall-e-2/gpt-image-1, only 1 for dall-e-3"},
        "background": {"type": "string", "description": "Background mode (gpt-image-1 only): auto|transparent|opaque"},
        "moderation": {"type": "string", "description": "Content moderation (gpt-image-1 only): auto|low"},
        "output_compression": {"type": "integer", "description": "Compression quality 0-100 (gpt-image-1 jpeg/webp only)"},
        "output_format": {"type": "string", "description": "Output format (gpt-image-1 only): png|jpeg|webp"},
        "partial_images": {"type": "integer", "description": "Streaming partial images 0-3 (gpt-image-1 only)"},
        "response_format": {"type": "string", "description": "Response format (dall-e-2/3 only): url|b64_json"},
        "stream": {"type": "boolean", "description": "Enable streaming (gpt-image-1 only)"},
        "style": {"type": "string", "description": "Image style (dall-e-3 only): vivid|natural"},
    }
    required: Optional[List[str]] = ["prompt"]

    def __init__(self, name: str = None, api_key: str = None, organization_id: str = None, model: str = "dall-e-3", 
                 save_path: str = "./openai_generated_images", storage_handler: Optional[FileStorageHandler] = None,
                 auto_postprocess: bool = False):
        super().__init__()
        self.name = name or self.name
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.organization_id = organization_id or os.getenv("OPENAI_ORGANIZATION_ID")
        self.model = model
        self.save_path = save_path
        self.storage_handler = storage_handler or LocalStorageHandler(base_path=save_path)
        self.auto_postprocess = auto_postprocess  # auto postprocess images if needed
        self.postprocessor = OpenAIImagePostProcessor()

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
            # Get actual parameters
            actual_model = model if model else self.model
            client = create_openai_client(self.api_key, self.organization_id)

            # Check if postprocessing is needed
            needs_pp = self.postprocessor.needs_postprocessing(actual_model, size, output_format)
            target_size = size
            target_format = output_format
            target_compression = output_compression if output_compression is not None else 95
            
            if self.auto_postprocess and (needs_pp["need_resize"] or needs_pp["need_format_conversion"]):
                print("🔄 Detected incompatible parameters, automatic postprocessing will be enabled:")
                for reason in needs_pp["reason"]:
                    print(f"   • {reason}")
                
                # Get compatible API parameters for postprocessing
                compat_params = self.postprocessor.get_compatible_params(actual_model, size, output_format)
                api_size = compat_params["api_params"].get("size")
                api_format = compat_params["api_params"].get("output_format")
                
                print(f"📝 API will use: size={api_size}, format={api_format}")
                print(f"🎯 Postprocessing target: size={target_size}, format={target_format}")
            else:
                api_size = size
                api_format = output_format

            # Build and validate parameters
            params_to_validate = build_validation_params(
                model=actual_model,
                prompt=prompt,
                size=api_size if self.auto_postprocess and needs_pp["need_resize"] else size,
                quality=quality,
                n=n,
                background=background,
                moderation=moderation,
                output_compression=output_compression,
                output_format=api_format if self.auto_postprocess and needs_pp["need_format_conversion"] else output_format,
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

            # Call API
            response = client.images.generate(**api_params)

            # Process and save images
            import base64
            results = []
            for i, image_data in enumerate(response.data):
                try:
                    # Get image bytes from response
                    if hasattr(image_data, "b64_json") and image_data.b64_json:
                        image_bytes = base64.b64decode(image_data.b64_json)
                    elif hasattr(image_data, "url") and image_data.url:
                        import requests
                        r = requests.get(image_data.url)
                        r.raise_for_status()
                        image_bytes = r.content
                    else:
                        raise Exception("No valid image data in response")

                    # Apply postprocessing if needed
                    if self.auto_postprocess and (needs_pp["need_resize"] or needs_pp["need_format_conversion"]):
                        print(f"🔧 Postprocessing image {i+1}/{len(response.data)}...")
                        image_bytes, ext = self.postprocessor.process_image(
                            image_bytes,
                            target_size=target_size,
                            target_format=target_format,
                            compression_quality=target_compression
                        )
                    else:
                        # Determine file extension
                        ext = (output_format or "png").lower()
                        if ext == "jpeg":
                            ext = "jpg"

                    # Generate unique filename
                    filename = self._get_unique_filename(image_name, i, ext)
                    
                    # Save using storage handler
                    save_result = self.storage_handler.save(filename, image_bytes)
                    
                    if save_result["success"]:
                        results.append(filename)
                    else:
                        results.append(f"Error saving image {i+1}: {save_result.get('error', 'Unknown error')}")
                except Exception as e:
                    results.append(f"Error saving image {i+1}: {e}")

            return {"results": results, "count": len(results)}
        except Exception as e:
            return {"error": f"Image generation failed: {e}"}
    
    def _get_unique_filename(self, image_name: str, index: int, ext: str = "png") -> str:
        """Generate a unique filename for the image"""
        import time
        
        if image_name:
            base = image_name.rsplit(".", 1)[0]
            # Try without index suffix first (for single image case)
            if index == 0:
                filename = f"{base}.{ext}"
                if not self.storage_handler.exists(filename):
                    return filename
            # If first attempt failed or not first image, use index suffix
            filename = f"{base}_{index+1}.{ext}"
        else:
            ts = int(time.time())
            filename = f"generated_{ts}_{index+1}.{ext}"
        
        # Check if file exists and generate unique name
        counter = 1
        while self.storage_handler.exists(filename):
            if image_name:
                base = image_name.rsplit(".", 1)[0]
                filename = f"{base}_{index+1}_{counter}.{ext}"
            else:
                filename = f"generated_{ts}_{index+1}_{counter}.{ext}"
            counter += 1
            
        return filename


