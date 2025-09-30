from typing import Dict, List, Optional, Any
import os
from dotenv import load_dotenv
from .tool import Toolkit
from .collection_base import ToolCollection
from .storage_handler import FileStorageHandler, LocalStorageHandler

from .image_openai import OpenAIImageToolkit
from .image_openrouter import OpenRouterImageToolkit
from .image_flux import FluxImageToolkit

load_dotenv()


class ImageEditingCollection(ToolCollection):
    """
    Edit and modify images with automatic provider fallback.
    Supports OpenAI GPT-Image, OpenRouter image models, and Flux.
    """
    name: str = "image_editing"
    description: str = "Edit and modify images with automatic provider fallback"

    inputs: Dict[str, Dict[str, str]] = {
        "prompt": {"type": "string", "description": "Required. Edit instruction or description of changes."},
        "images": {"type": "array", "description": "Optional. List of image paths or URLs (normalized to provider-specific parts)."},
        "image_path": {"type": "string", "description": "Optional. Single local image path."},
        "image_url": {"type": "string", "description": "Optional. Single HTTP(S) image URL."},
        "size": {"type": "string", "description": "Optional. Output size in 'WxH' format (e.g., '1024x1024')."},
        "output_format": {"type": "string", "description": "Optional. Output format (e.g., 'png'|'jpeg'|'webp')."},
        "image_name": {"type": "string", "description": "Optional. Base filename for saved outputs."},
    }
    required: Optional[List[str]] = ["prompt"]

    def __init__(
        self,
        name: str = "image_editing",
        openai_api_key: Optional[str] = None,
        openrouter_api_key: Optional[str] = None,
        flux_api_key: Optional[str] = None,
        storage_handler: Optional[FileStorageHandler] = None,
        base_path: str = "./images/edited",
        execution_order: Optional[List[str]] = None,
        per_tool_timeout: Optional[float] = None,
    ):
        if not openai_api_key:
            openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openrouter_api_key:
            openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        if not flux_api_key:
            flux_api_key = os.getenv("FLUX_API_KEY")

        if storage_handler is None:
            storage_handler = LocalStorageHandler(base_path=base_path)

        toolkits = []
        if openai_api_key:
            toolkits.append(OpenAIImageToolkit(
                api_key=openai_api_key,
                storage_handler=storage_handler,
                save_path=base_path
            ))
        if openrouter_api_key:
            toolkits.append(OpenRouterImageToolkit(
                api_key=openrouter_api_key,
                storage_handler=storage_handler
            ))
        if flux_api_key:
            toolkits.append(FluxImageToolkit(
                api_key=flux_api_key,
                storage_handler=storage_handler,
                save_path=base_path
            ))

        if execution_order is None:
            execution_order = ["openai_image_edit", "openrouter_image_edit", "flux_image_edit"]

        super().__init__(
            name=name,
            description="Edit and modify images with automatic provider fallback",
            kits=toolkits,
            execution_order=execution_order,
            argument_mapping_function={
                "openai_image_edit": self._map_args_for_openai,
                "openrouter_image_edit": self._map_args_for_openrouter,
                "flux_image_edit": self._map_args_for_flux,
            },
            output_mapping_function={
                "openai_image_edit": self._convert_output_from_openai,
                "openrouter_image_edit": self._convert_output_from_openrouter,
                "flux_image_edit": self._convert_output_from_flux,
            },
            per_tool_timeout=per_tool_timeout,
        )

        self.storage_handler = storage_handler

    def _map_args_for_openai(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> Dict[str, Any]:
        mapped = {
            "prompt": inputs.get("prompt"),
            "size": inputs.get("size"),
            "output_format": inputs.get("output_format"),
            "partial_images": inputs.get("partial_images"),
            "stream": inputs.get("stream"),
            "image_name": inputs.get("image_name"),
        }
        # Normalize images list
        images = inputs.get("images")
        single_path = inputs.get("image_path")
        single_url = inputs.get("image_url")
        if images and isinstance(images, list):
            mapped["images"] = images
        elif single_path:
            mapped["images"] = [single_path]
        elif single_url:
            mapped["images"] = [single_url]
        return {k: v for k, v in mapped.items() if v is not None}

    def _map_args_for_openrouter(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> Dict[str, Any]:
        mapped = {
            "prompt": inputs.get("prompt"),
            "output_basename": inputs.get("image_name", "edited"),
        }
        urls = inputs.get("images") if isinstance(inputs.get("images"), list) else None
        if inputs.get("image_url"):
            urls = (urls or []) + [inputs["image_url"]]
        paths = None
        if isinstance(inputs.get("images"), list):
            paths = [p for p in inputs["images"] if isinstance(p, str) and not p.startswith("http")]
        if inputs.get("image_path"):
            paths = (paths or []) + [inputs["image_path"]]
        if urls:
            mapped["image_urls"] = urls
        if paths:
            mapped["image_paths"] = paths
        return {k: v for k, v in mapped.items() if v is not None}

    def _map_args_for_flux(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> Dict[str, Any]:
        mapped = {
            "prompt": inputs.get("prompt"),
            "output_format": inputs.get("output_format"),
            "image_name": inputs.get("image_name"),
        }
        # Flux accepts either base64 in input_image or a file path
        input_image = inputs.get("input_image")
        image_path = inputs.get("image_path") or (inputs.get("images")[0] if isinstance(inputs.get("images"), list) else None)
        if input_image:
            mapped["input_image"] = input_image
        elif image_path:
            mapped["image_path"] = image_path
        # Map size to aspect_ratio if provided
        if "size" in inputs and isinstance(inputs["size"], str) and "x" in inputs["size"]:
            try:
                w, h = inputs["size"].split("x")
                w, h = int(w), int(h)
                g = self._gcd(w, h)
                mapped["aspect_ratio"] = f"{w//g}:{h//g}"
            except Exception:
                pass
        return {k: v for k, v in mapped.items() if v is not None}

    def _convert_output_from_openai(self, result: Dict[str, Any]) -> Dict[str, Any]:
        if "error" in result:
            return {"success": False, "error": result["error"], "provider": "openai"}
        return {
            "success": True,
            "images": result.get("results", []),
            "count": result.get("count", 0),
            "provider": "openai",
        }

    def _convert_output_from_openrouter(self, result: Dict[str, Any]) -> Dict[str, Any]:
        if "error" in result:
            return {"success": False, "error": result["error"], "provider": "openrouter"}
        if "saved_paths" in result:
            return {
                "success": True,
                "images": result["saved_paths"],
                "count": len(result["saved_paths"]),
                "provider": "openrouter",
            }
        return {"success": False, "error": "Unknown OpenRouter output format", "provider": "openrouter"}

    def _convert_output_from_flux(self, result: Dict[str, Any]) -> Dict[str, Any]:
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

    def _gcd(self, a: int, b: int) -> int:
        while b:
            a, b = b, a % b
        return a

    def _get_next_execute(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> Optional[str]:
        for tool_name in self.execution_order:
            if tool_name not in outputs:
                return tool_name
        return None

    def __call__(
        self,
        prompt: str,
        images: list = None,
        image_path: str = None,
        image_url: str = None,
        size: str = None,
        output_format: str = None,
        image_name: str = None,
        **kwargs,
    ) -> Dict[str, Any]:
        inputs = {
            "prompt": prompt,
            "images": images,
            "image_path": image_path,
            "image_url": image_url,
            "size": size,
            "output_format": output_format,
            "image_name": image_name,
            **kwargs,
        }
        return self._run_pipeline(inputs)


class ImageEditingCollectionToolkit(Toolkit):
    """Toolkit exposing only image editing via provider fallback."""

    def __init__(
        self,
        name: str = "ImageEditingCollectionToolkit",
        openai_api_key: Optional[str] = None,
        openrouter_api_key: Optional[str] = None,
        flux_api_key: Optional[str] = None,
        storage_handler: Optional[FileStorageHandler] = None,
        base_path: str = "./images/edited",
    ):
        if not openai_api_key:
            openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openrouter_api_key:
            openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        if not flux_api_key:
            flux_api_key = os.getenv("FLUX_API_KEY")

        if storage_handler is None:
            storage_handler = LocalStorageHandler(base_path=base_path)

        editing_collection = ImageEditingCollection(
            openai_api_key=openai_api_key,
            openrouter_api_key=openrouter_api_key,
            flux_api_key=flux_api_key,
            storage_handler=storage_handler,
            base_path=base_path,
        )

        super().__init__(name=name, tools=[editing_collection])
        # Store configuration for reference (consistent with other toolkits)
        self.openai_api_key = openai_api_key
        self.openrouter_api_key = openrouter_api_key
        self.flux_api_key = flux_api_key
        self.storage_handler = storage_handler
        self.base_path = base_path