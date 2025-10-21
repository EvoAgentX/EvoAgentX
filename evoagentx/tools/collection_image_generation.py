from typing import Dict, List, Optional, Any
import os
from dotenv import load_dotenv
from .tool import Toolkit
from .collection_base import ToolCollection
from .storage_handler import FileStorageHandler, LocalStorageHandler

from .image_tools.openrouter_image_tools.image_generation import OpenRouterImageGenerationTool 
from .image_tools.openai_image_tools.image_generation import OpenAIImageGenerationTool 
from .image_tools.flux_image_tools.image_generation import FluxImageGenerationTool 

load_dotenv()

IMAGE_GENERATION_COLLECTION_DESC = "Generates an image based on a natural language prompt. The tool processes the text input, creates the image, and returns the actual stored image filename and file path."


def map_args_for_openai(inputs: Dict[str, Any], outputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Map arguments for OpenAI image generation.
    - prompt: str (required)
    - image_name: str (optional)
    """
    mapped = {
        "prompt": inputs.get("prompt"),
        "image_name": inputs.get("image_name", None),
    }
    if "size" in inputs:
        mapped["size"] = inputs["size"]
    if "output_format" in inputs:
        mapped["output_format"] = inputs["output_format"]
    return {k: v for k, v in mapped.items() if v is not None}


def map_args_for_openrouter(inputs: Dict[str, Any], outputs: Dict[str, Any]) -> Dict[str, Any]:
    mapped = {
        "prompt": inputs.get("prompt"),
        "image_name": inputs.get("image_name", None),
    }
    if "size" in inputs:
        mapped["output_size"] = inputs["size"]
    if "output_format" in inputs:
        # mapped["output_format"] = "b64_json"
        mapped["output_format"] = inputs["output_format"]
    return {k: v for k, v in mapped.items() if v is not None}


def _gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return a


def map_args_for_flux(inputs: Dict[str, Any], outputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Map arguments for Flux image generation.
    - prompt: str (required)
    - aspect_ratio: str (e.g., '1:1') derived from size if provided
    - output_format: str ('jpeg'|'png')
    - image_name: str (optional)
    """
    mapped = {
        "prompt": inputs.get("prompt"),
        "image_name": inputs.get("image_name", None),
    }
    if "size" in inputs:
        mapped["output_size"] = inputs["size"]
    if "aspect_ratio" in inputs:
        mapped["aspect_ratio"] = inputs["aspect_ratio"]
    if "output_format" in inputs:
        mapped["output_format"] = inputs["output_format"]
    return {k: v for k, v in mapped.items() if v is not None}


def convert_output_from_openai(result: Dict[str, Any]) -> Dict[str, Any]:
    if "error" in result:
        return {"success": False, "error": result["error"], "provider": "openai"}
    resp = {
        "success": True,
        "images": result.get("results", []),
        "count": result.get("count", 0),
        "provider": "openai",
    }
    urls = result.get("urls")
    if isinstance(urls, list) and urls:
        resp["urls"] = urls
    return resp


def convert_output_from_openrouter(result: Dict[str, Any]) -> Dict[str, Any]:
    if "error" in result:
        return {"success": False, "error": result["error"], "provider": "openrouter"}
    
    # Handle unified format: {"results": [...], "count": n}
    if "results" in result:
        resp = {
            "success": True,
            "images": result["results"],
            "count": result.get("count", len(result["results"])),
            "provider": "openrouter",
        }
        urls = result.get("urls")
        if isinstance(urls, list) and urls:
            resp["urls"] = urls
        return resp
    
    # Legacy format support
    if "saved_paths" in result:
        resp = {
            "success": True,
            "images": result["saved_paths"],
            "count": len(result["saved_paths"]),
            "provider": "openrouter",
        }
        urls = result.get("urls")
        if isinstance(urls, list) and urls:
            resp["urls"] = urls
        return resp
    
    return {"success": False, "error": "Unknown OpenRouter output format", "provider": "openrouter"}


def convert_output_from_flux(result: Dict[str, Any]) -> Dict[str, Any]:
    # Handle error case first
    if "error" in result:
        return {"success": False, "error": result["error"], "provider": "flux"}
    
    # Handle unified format: {"results": [...], "count": n}
    if "results" in result:
        out = {
            "success": True,
            "images": result["results"],
            "count": result.get("count", len(result["results"])),
            "provider": "flux",
        }
        urls = result.get("urls")
        if isinstance(urls, list) and urls:
            out["urls"] = urls
        return out
    
    # Legacy format support
    if result.get("success"):
        out = {
            "success": True,
            "images": [result.get("file_path")] if result.get("file_path") else [],
            "count": 1 if result.get("file_path") else 0,
            "provider": "flux",
        }
        if isinstance(result.get("url"), str):
            out["urls"] = [result["url"]]
        return out
    
    return {"success": False, "error": result.get("error", "Flux operation failed"), "provider": "flux"}


class ImageGenerationCollection(ToolCollection):
    """
    Generate images from text prompts with automatic provider fallback.
    Supports OpenAI DALL·E/GPT-Image, OpenRouter vision models, and Flux.
    """
    name: str = "image_generation"
    description: str = IMAGE_GENERATION_COLLECTION_DESC

    inputs: Dict[str, Dict[str, str]] = {
        "prompt": {"type": "string", "description": "Required. A natural language description of the desired image."},
        "image_name": {"type": "string", "description": "Optional. A custom base filename (without extension) to use when saving the generated image (e.g., `cyberpunk_city`). If not provided, a default unique name will be assigned."}, 
    }
    required: Optional[List[str]] = ["prompt"]

    def __init__(
        self,
        name: str = "image_generation",
        openai_org_api_key: Optional[str] = None,
        openai_org_id: Optional[str] = None, 
        openrouter_api_key: Optional[str] = None,
        flux_api_key: Optional[str] = None,
        storage_handler: Optional[FileStorageHandler] = None,
        save_path: str = "./images/generated",
        per_tool_timeout: Optional[float] = None,
        auto_postprocess: bool = False,
    ):
        openai_model_is_available = openai_org_api_key is not None and openai_org_id is not None
        openrouter_model_is_available = openrouter_api_key is not None
        flux_model_is_available = flux_api_key is not None

        if not openai_model_is_available and not openrouter_model_is_available and not flux_model_is_available:
            raise ValueError(f"At least one of `openai_org_api_key` (with `openai_org_id`), `openrouter_api_key`, or `flux_api_key` is required for {type(self).__name__}!")

        # Initialize storage handler
        if storage_handler is None:
            storage_handler = LocalStorageHandler(base_path=save_path)

        toolkits, tool_names = [], []
        if openrouter_model_is_available:
            # Google Nano Banana Image Generation 
            nano_banana_image_generation = OpenRouterImageGenerationTool(
                name="openrouter_image_generation_nano_banana",
                api_key=openrouter_api_key, 
                model="google/gemini-2.5-flash-image-preview",
                save_path=save_path,
                storage_handler=storage_handler,
                auto_postprocess=auto_postprocess,
            )
            tool_names.append(nano_banana_image_generation.name)
            toolkits.append(nano_banana_image_generation)

        if openai_model_is_available:
            openai_image_generation = OpenAIImageGenerationTool(
                name="openai_image_generation",
                api_key=openai_org_api_key,
                organization_id=openai_org_id,
                model="dall-e-3",
                save_path=save_path,
                storage_handler=storage_handler,
                auto_postprocess=auto_postprocess,
            )
            tool_names.append(openai_image_generation.name)
            toolkits.append(openai_image_generation)

        if flux_model_is_available:
            flux_image_generation = FluxImageGenerationTool(
                name="flux_image_generation",
                api_key=flux_api_key,
                model="flux-kontext-max",
                save_path=save_path,
                storage_handler=storage_handler,
                auto_postprocess=auto_postprocess,
            )
            tool_names.append(flux_image_generation.name)
            toolkits.append(flux_image_generation)

        super().__init__(
            name=name,
            description=IMAGE_GENERATION_COLLECTION_DESC,
            kits=toolkits,
            execution_order=tool_names,
            argument_mapping_function={
                "openrouter_image_generation_nano_banana": map_args_for_openrouter,
                "openai_image_generation": map_args_for_openai,
                "flux_image_generation": map_args_for_flux,
            },
            output_mapping_function={
                "openrouter_image_generation_nano_banana": convert_output_from_openrouter,
                "openai_image_generation": convert_output_from_openai,
                "flux_image_generation": convert_output_from_flux,
            },
            per_tool_timeout=per_tool_timeout,
        )

        self.storage_handler = storage_handler

    def _get_next_execute(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> Optional[str]:
        for tool_name in self.execution_order:
            if tool_name not in outputs:
                return tool_name
        return None

    def __call__(
        self,
        prompt: str,
        image_name: str = None,
        **kwargs,
    ) -> Dict[str, Any]:
        inputs = {
            "prompt": prompt,
            "image_name": image_name,
            **kwargs,
        }
        result = self._run_pipeline(inputs)

        # check image name and the actual saved path 
        if isinstance(result, dict) and image_name is not None and result.get("count", 0) == 1 and isinstance(result.get("images"), list): 
            saved_image_path = result["images"][0]
            saved_image_name, saved_image_extension = os.path.basename(saved_image_path).rsplit(".", 1)
            if saved_image_name != image_name:
                result["warning"] = f"The actual saved filename `{saved_image_name}.{saved_image_extension}` does not match the requested image_name `{image_name}.{saved_image_extension}`, likely due to a name collision and auto-renaming. Use the actual filename returned in images to avoid errors."
        return result


class ImageGenerationCollectionToolkit(Toolkit):
    """Toolkit exposing only image generation via provider fallback."""

    def __init__(
        self,
        name: str = "ImageGenerationCollectionToolkit",
        openai_org_api_key: Optional[str] = None,
        openai_org_id: Optional[str] = None, 
        openrouter_api_key: Optional[str] = None,
        flux_api_key: Optional[str] = None,
        storage_handler: Optional[FileStorageHandler] = None,
        save_path: str = "./images/generated",
        auto_postprocess: bool = False,
    ):
        if not openai_org_api_key or not openai_org_id:
            openai_org_api_key = os.getenv("OPENAI_IMAGE_ORG_API_KEY")
            openai_org_id = os.getenv("OPENAI_IMAGE_ORG_ID")
        if not openrouter_api_key:
            openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        if not flux_api_key:
            flux_api_key = os.getenv("FLUX_API_KEY")

        if storage_handler is None:
            storage_handler = LocalStorageHandler(base_path=save_path)

        generation_collection = ImageGenerationCollection(
            openai_org_api_key=openai_org_api_key,
            openai_org_id=openai_org_id,
            openrouter_api_key=openrouter_api_key,
            flux_api_key=flux_api_key,
            storage_handler=storage_handler,
            save_path=save_path,
            auto_postprocess=auto_postprocess,
        )

        super().__init__(name=name, tools=[generation_collection])
        # Store configuration for reference (consistent with other toolkits)
        self.openai_org_api_key = openai_org_api_key
        self.openai_org_id = openai_org_id
        self.openrouter_api_key = openrouter_api_key
        self.flux_api_key = flux_api_key
        self.storage_handler = storage_handler
        self.save_path = save_path
        self.auto_postprocess = auto_postprocess