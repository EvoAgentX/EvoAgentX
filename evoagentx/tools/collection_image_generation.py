from typing import Dict, List, Optional, Any
import os
from dotenv import load_dotenv
from .tool import Toolkit
from .collection_base import ToolCollection
from .storage_handler import FileStorageHandler, LocalStorageHandler

# Provider toolkits
from .image_openai import OpenAIImageToolkit
from .image_openrouter import OpenRouterImageToolkit
from .image_flux import FluxImageToolkit

load_dotenv()


class ImageGenerationCollection(ToolCollection):
    """
    Generate images from text prompts with automatic provider fallback.
    Supports OpenAI DALL·E/GPT-Image, OpenRouter vision models, and Flux.
    """
    name: str = "image_generation"
    description: str = "Generate images from text prompts with automatic provider fallback"

    inputs: Dict[str, Dict[str, str]] = {
        "prompt": {"type": "string", "description": "Required. Text prompt for image generation."},
        "size": {"type": "string", "description": "Optional. Output size in 'WxH' format (e.g., '1024x1024')."},
        "output_format": {"type": "string", "description": "Optional. Output format (e.g., 'png'|'jpeg'|'webp')."},
        "image_name": {"type": "string", "description": "Optional. Base filename (without extension) for saved images."},
    }
    required: Optional[List[str]] = ["prompt"]

    def __init__(
        self,
        name: str = "image_generation",
        openai_api_key: Optional[str] = None,
        openrouter_api_key: Optional[str] = None,
        flux_api_key: Optional[str] = None,
        storage_handler: Optional[FileStorageHandler] = None,
        base_path: str = "./images/generated",
        execution_order: Optional[List[str]] = None,
        per_tool_timeout: Optional[float] = None,
    ):
        # Load environment variables if no keys provided
        if not openai_api_key:
            openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openrouter_api_key:
            openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        if not flux_api_key:
            flux_api_key = os.getenv("FLUX_API_KEY")

        # Initialize storage handler
        if storage_handler is None:
            storage_handler = LocalStorageHandler(base_path=base_path)

        # Initialize provider toolkits
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

        # Default execution order uses provider tool names
        if execution_order is None:
            execution_order = ["openai_image_generation", "openrouter_image_generation", "flux_image_generation"]

        super().__init__(
            name=name,
            description="Generate images from text prompts with automatic provider fallback",
            kits=toolkits,
            execution_order=execution_order,
            argument_mapping_function={
                "openai_image_generation": self._map_args_for_openai,
                "openrouter_image_generation": self._map_args_for_openrouter,
                "flux_image_generation": self._map_args_for_flux,
            },
            output_mapping_function={
                "openai_image_generation": self._convert_output_from_openai,
                "openrouter_image_generation": self._convert_output_from_openrouter,
                "flux_image_generation": self._convert_output_from_flux,
            },
            per_tool_timeout=per_tool_timeout,
        )

        self.storage_handler = storage_handler

    def _map_args_for_openai(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Map arguments for OpenAI image generation.
        - prompt: str (required)
        - size: str (provider-supported size)
        - response_format: str ('url'|'b64_json') — mapped to 'b64_json'
        - image_name: str
        """
        mapped = {
            "prompt": inputs.get("prompt"),
            "image_name": inputs.get("image_name"),
        }
        if "size" in inputs:
            mapped["size"] = inputs["size"]
        if "output_format" in inputs:
            mapped["response_format"] = "b64_json"
        return {k: v for k, v in mapped.items() if v is not None}

    def _map_args_for_openrouter(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Map arguments for OpenRouter image generation.
        - prompt: str (required)
        - output_basename: str base filename
        """
        mapped = {
            "prompt": inputs.get("prompt"),
            "output_basename": inputs.get("image_name", "generated"),
        }
        return {k: v for k, v in mapped.items() if v is not None}

    def _map_args_for_flux(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Map arguments for Flux image generation.
        - prompt: str (required)
        - aspect_ratio: str (e.g., '1:1')
        - output_format: str ('jpeg'|'png')
        - prompt_upsampling: bool
        - safety_tolerance: int
        """
        mapped = {
            "prompt": inputs.get("prompt"),
            "output_format": inputs.get("output_format", "jpeg"),
            "image_name": inputs.get("image_name"),
        }
        if "size" in inputs:
            size = inputs["size"]
            if isinstance(size, str) and "x" in size:
                w, h = size.split("x")
                try:
                    ratio_w, ratio_h = int(w), int(h)
                    gcd = self._gcd(ratio_w, ratio_h)
                    mapped["aspect_ratio"] = f"{ratio_w//gcd}:{ratio_h//gcd}"
                except ValueError:
                    pass
        return {k: v for k, v in mapped.items() if v is not None}

    def _convert_output_from_openai(self, result: Dict[str, Any]) -> Dict[str, Any]:
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

    def _convert_output_from_openrouter(self, result: Dict[str, Any]) -> Dict[str, Any]:
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
        return {"success": False, "error": result.get("error", "Flux generation failed"), "provider": "flux"}

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
        size: str = None,
        output_format: str = None,
        image_name: str = None,
        **kwargs,
    ) -> Dict[str, Any]:
        inputs = {
            "prompt": prompt,
            "size": size,
            "output_format": output_format,
            "image_name": image_name,
            **kwargs,
        }
        return self._run_pipeline(inputs)


class ImageGenerationCollectionToolkit(Toolkit):
    """Toolkit exposing only image generation via provider fallback."""

    def __init__(
        self,
        name: str = "ImageGenerationCollectionToolkit",
        openai_api_key: Optional[str] = None,
        openrouter_api_key: Optional[str] = None,
        flux_api_key: Optional[str] = None,
        storage_handler: Optional[FileStorageHandler] = None,
        base_path: str = "./images/generated",
    ):
        if not openai_api_key:
            openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openrouter_api_key:
            openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        if not flux_api_key:
            flux_api_key = os.getenv("FLUX_API_KEY")

        if storage_handler is None:
            storage_handler = LocalStorageHandler(base_path=base_path)

        generation_collection = ImageGenerationCollection(
            openai_api_key=openai_api_key,
            openrouter_api_key=openrouter_api_key,
            flux_api_key=flux_api_key,
            storage_handler=storage_handler,
            base_path=base_path,
        )

        super().__init__(name=name, tools=[generation_collection])

        # Store configuration for reference (consistent with other toolkits)
        self.openai_api_key = openai_api_key
        self.openrouter_api_key = openrouter_api_key
        self.flux_api_key = flux_api_key
        self.storage_handler = storage_handler
        self.base_path = base_path