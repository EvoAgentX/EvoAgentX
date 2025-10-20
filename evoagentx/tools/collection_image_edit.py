from typing import Dict, List, Optional, Any
import os
from dotenv import load_dotenv
from .tool import Toolkit
from .collection_base import ToolCollection
from .storage_handler import FileStorageHandler, LocalStorageHandler

from .image_openai import OpenAIImageEditTool
from .image_openrouter import OpenRouterImageEditTool
from .image_flux import FluxImageProvider, FluxImageEditTool

load_dotenv()

IMAGE_EDIT_COLLECTION_DESC = (
    "Edits or combines one or more existing images based on a natural language prompt. "
    "The tool can perform modifications (e.g., color adjustments, object removal, style transfer), or merge multiple images into a new image, "
    "and returns the actual saved image filename and file path."
)

def map_args_for_openai(inputs: Dict[str, Any], outputs: Dict[str, Any]) -> Dict[str, Any]:
    mapped = {
        "prompt": inputs.get("prompt"),
        "image_urls": inputs.get("image_urls"),
        "image_name": inputs.get("image_name", None),
    }
    return {k: v for k, v in mapped.items() if v is not None}

def map_args_for_openrouter(inputs: Dict[str, Any], outputs: Dict[str, Any]) -> Dict[str, Any]:
    mapped = {
        "prompt": inputs.get("prompt"),
        "image_urls": inputs.get("image_urls"), 
        "output_basename": inputs.get("image_name", "edited"),
    }
    return {k: v for k, v in mapped.items() if v is not None}

def map_args_for_flux(inputs: Dict[str, Any], outputs: Dict[str, Any]) -> Dict[str, Any]:
    mapped = {
        "prompt": inputs.get("prompt"),
        "image_urls": inputs.get("image_urls"),
        "image_name": inputs.get("image_name"),
    }
    return {k: v for k, v in mapped.items() if v is not None}

def convert_output_from_openai(result: Dict[str, Any]) -> Dict[str, Any]:
    if "error" in result:
        return {"success": False, "error": result["error"], "provider": "openai"}
    
    if "results" in result:
        resp = {
            "success": True,
            "images": result["results"],
            "count": len(result["results"]),
            "provider": "openai",
        }
        urls = result.get("urls")
        if isinstance(urls, list) and urls:
            resp["urls"] = urls
        return resp
    return {"success": False, "error": "Unknown OpenAI output format", "provider": "openai"}

def convert_output_from_openrouter(result: Dict[str, Any]) -> Dict[str, Any]:
    if "error" in result:
        return {"success": False, "error": result["error"], "provider": "openrouter"}
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
    return {"success": False, "error": result.get("error", "Flux edit failed"), "provider": "flux"}


class ImageEditingCollection(ToolCollection):
    """
    Edit and modify images with automatic provider fallback.
    Supports OpenAI GPT-Image, OpenRouter image models, and Flux.
    """
    name: str = "image_editing"
    description: str = IMAGE_EDIT_COLLECTION_DESC

    inputs: Dict[str, Dict[str, str]] = {
        "prompt": {"type": "string", "description": "Required. A natural language description of the edits or transformations to apply to the input image(s)."},
        "image_urls": {"type": "array", "items": {"type": "string", "description": "HTTP(S) image URL"}, "description": "Required. A list of one or more HTTP(S) image URLs to be modified according to the prompt. Example: ['https://example.com/photo1.png'], or multiple: ['https://example.com/photo1.png', 'https://example.com/photo2.jpg']."},
        "image_name": {"type": "string", "description": "Optional. A custom base filename (without extension) for saving the output image (e.g., 'starry_portrait'). If not provided, a default unique name will be assigned."}, 
    }
    required: Optional[List[str]] = ["prompt", "image_urls"]

    def __init__(
        self,
        name: str = "image_editing",
        openai_org_api_key: Optional[str] = None,
        openai_org_id: Optional[str] = None,
        openrouter_api_key: Optional[str] = None,
        flux_api_key: Optional[str] = None,
        storage_handler: Optional[FileStorageHandler] = None,
        base_path: str = "./images/edited",
        per_tool_timeout: Optional[float] = None,
    ):
        openai_model_is_available = openai_org_api_key is not None and openai_org_id is not None
        openrouter_model_is_available = openrouter_api_key is not None
        flux_model_is_available = flux_api_key is not None

        if not openai_model_is_available and not openrouter_model_is_available and not flux_model_is_available:
            raise ValueError(f"At least one of `openai_org_api_key` (with `openai_org_id`), `openrouter_api_key`, or `flux_api_key` is required for {type(self).__name__}!")

        if storage_handler is None:
            storage_handler = LocalStorageHandler(base_path=base_path)

        toolkits, tool_names = [], [] 
        if openrouter_model_is_available:
            nano_banana_image_edit = OpenRouterImageEditTool(
                name="openrouter_image_edit_nano_banana",
                api_key=openrouter_api_key,
                model="google/gemini-2.5-flash-image-preview",
                storage_handler=storage_handler,
            )
            tool_names.append(nano_banana_image_edit.name)
            toolkits.append(nano_banana_image_edit)

        if openai_model_is_available:
            openai_image_edit = OpenAIImageEditTool(
                name="openai_image_edit",
                api_key=openai_org_api_key,
                organization_id=openai_org_id,
                model="gpt-image-1", 
                storage_handler=storage_handler,
            )
            tool_names.append(openai_image_edit.name)
            toolkits.append(openai_image_edit)

        # if flux_model_is_available:
        #     provider = FluxImageProvider(api_key=flux_api_key, storage_handler=storage_handler, model="flux-kontext-pro")
        #     flux_image_edit = FluxImageEditTool(provider, name="flux_image_edit")
        #     tool_names.append(flux_image_edit.name)
        #     toolkits.append(flux_image_edit)

        super().__init__(
            name=name,
            description=IMAGE_EDIT_COLLECTION_DESC,
            kits=toolkits,
            execution_order=tool_names,
            argument_mapping_function={
                "openrouter_image_edit_nano_banana": map_args_for_openrouter,
                "openai_image_edit": map_args_for_openai,
                "flux_image_edit": map_args_for_flux,
            },
            output_mapping_function={
                "openrouter_image_edit_nano_banana": convert_output_from_openrouter,
                "openai_image_edit": convert_output_from_openai,
                "flux_image_edit": convert_output_from_flux,
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
        image_urls: list = None,
        image_name: str = None,
        **kwargs,
    ) -> Dict[str, Any]:
        inputs = {
            "prompt": prompt,
            "image_urls": image_urls,
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


class ImageEditingCollectionToolkit(Toolkit):
    """Toolkit exposing only image editing via provider fallback."""

    def __init__(
        self,
        name: str = "ImageEditingCollectionToolkit",
        openai_org_api_key: Optional[str] = None,
        openai_org_id: Optional[str] = None,
        openrouter_api_key: Optional[str] = None,
        flux_api_key: Optional[str] = None,
        storage_handler: Optional[FileStorageHandler] = None,
        base_path: str = "./images/edited",
    ):
        if not openai_org_api_key or not openai_org_id:
            openai_org_api_key = os.getenv("OPENAI_IMAGE_ORG_API_KEY")
            openai_org_id = os.getenv("OPENAI_IMAGE_ORG_ID")
        if not openrouter_api_key:
            openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        if not flux_api_key:
            flux_api_key = os.getenv("FLUX_API_KEY")

        if storage_handler is None:
            storage_handler = LocalStorageHandler(base_path=base_path)

        editing_collection = ImageEditingCollection(
            openai_org_api_key=openai_org_api_key,
            openai_org_id=openai_org_id,
            openrouter_api_key=openrouter_api_key,
            flux_api_key=flux_api_key,
            storage_handler=storage_handler,
            base_path=base_path,
        )

        super().__init__(name=name, tools=[editing_collection])
        # Store configuration for reference (consistent with other toolkits)
        self.openai_org_api_key = openai_org_api_key
        self.openai_org_id = openai_org_id
        self.openrouter_api_key = openrouter_api_key
        self.flux_api_key = flux_api_key
        self.storage_handler = storage_handler
        self.base_path = base_path