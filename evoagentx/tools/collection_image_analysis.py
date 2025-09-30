from typing import Dict, List, Optional, Any
import os
from dotenv import load_dotenv
from .tool import Toolkit
from .collection_base import ToolCollection
from .storage_handler import FileStorageHandler, LocalStorageHandler

from .image_openai import OpenAIImageToolkit
from .image_openrouter import OpenRouterImageToolkit

load_dotenv()


class ImageAnalysisCollection(ToolCollection):
    """
    Analyze and understand images with automatic provider fallback.
    Supports OpenAI vision models and OpenRouter vision-capable models.
    """
    name: str = "image_analysis"
    description: str = "Analyze and understand images with automatic provider fallback"

    inputs: Dict[str, Dict[str, str]] = {
        "prompt": {"type": "string", "description": "Required. Question or instruction for analysis."},
        "image_url": {"type": "string", "description": "Optional. HTTP(S) image URL."},
        "image_path": {"type": "string", "description": "Optional. Local image file path (converted internally)."},
        "pdf_path": {"type": "string", "description": "Optional. Local PDF path (converted when supported)."},
    }
    required: Optional[List[str]] = ["prompt"]

    def __init__(
        self,
        name: str = "image_analysis",
        openai_api_key: Optional[str] = None,
        openrouter_api_key: Optional[str] = None,
        storage_handler: Optional[FileStorageHandler] = None,
        base_path: str = "./images/analysis_cache",
        execution_order: Optional[List[str]] = None,
        per_tool_timeout: Optional[float] = None,
    ):
        if not openai_api_key:
            openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openrouter_api_key:
            openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

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

        if execution_order is None:
            execution_order = ["openai_image_analysis", "openrouter_image_analysis"]

        super().__init__(
            name=name,
            description="Analyze and understand images with automatic provider fallback",
            kits=toolkits,
            execution_order=execution_order,
            argument_mapping_function={
                "openai_image_analysis": self._map_args_for_openai,
                "openrouter_image_analysis": self._map_args_for_openrouter,
            },
            output_mapping_function={
                "openai_image_analysis": self._convert_output_from_openai,
                "openrouter_image_analysis": self._convert_output_from_openrouter,
            },
            per_tool_timeout=per_tool_timeout,
        )

        self.storage_handler = storage_handler

    def _map_args_for_openai(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> Dict[str, Any]:
        mapped = {
            "prompt": inputs.get("prompt"),
            "image_url": inputs.get("image_url"),
            "image_path": inputs.get("image_path"),
        }
        return {k: v for k, v in mapped.items() if v is not None}

    def _map_args_for_openrouter(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> Dict[str, Any]:
        mapped = {
            "prompt": inputs.get("prompt"),
            "image_url": inputs.get("image_url"),
            "image_path": inputs.get("image_path"),
            "pdf_path": inputs.get("pdf_path"),
        }
        return {k: v for k, v in mapped.items() if v is not None}

    def _convert_output_from_openai(self, result: Dict[str, Any]) -> Dict[str, Any]:
        if "error" in result:
            return {"success": False, "error": result["error"], "provider": "openai"}
        # OpenAI analysis returns text content
        content = None
        if isinstance(result, dict):
            content = result.get("content") or result.get("text")
        return {"success": True, "content": content or "", "provider": "openai"}

    def _convert_output_from_openrouter(self, result: Dict[str, Any]) -> Dict[str, Any]:
        if "error" in result:
            return {"success": False, "error": result["error"], "provider": "openrouter"}
        text = None
        if isinstance(result, dict):
            text = result.get("content") or result.get("text")
        return {"success": True, "content": text or "", "provider": "openrouter"}

    def _get_next_execute(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> Optional[str]:
        for tool_name in self.execution_order:
            if tool_name not in outputs:
                return tool_name
        return None

    def __call__(
        self,
        prompt: str,
        image_url: str = None,
        image_path: str = None,
        pdf_path: str = None,
        **kwargs,
    ) -> Dict[str, Any]:
        inputs = {
            "prompt": prompt,
            "image_url": image_url,
            "image_path": image_path,
            "pdf_path": pdf_path,
            **kwargs,
        }
        return self._run_pipeline(inputs)


class ImageAnalysisCollectionToolkit(Toolkit):
    """Toolkit exposing only image analysis via provider fallback."""

    def __init__(
        self,
        name: str = "ImageAnalysisCollectionToolkit",
        openai_api_key: Optional[str] = None,
        openrouter_api_key: Optional[str] = None,
        storage_handler: Optional[FileStorageHandler] = None,
        base_path: str = "./images/analysis_cache",
    ):
        if not openai_api_key:
            openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openrouter_api_key:
            openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

        if storage_handler is None:
            storage_handler = LocalStorageHandler(base_path=base_path)

        analysis_collection = ImageAnalysisCollection(
            openai_api_key=openai_api_key,
            openrouter_api_key=openrouter_api_key,
            storage_handler=storage_handler,
            base_path=base_path,
        )

        super().__init__(name=name, tools=[analysis_collection])
        # Store configuration for reference (consistent with other toolkits)
        self.openai_api_key = openai_api_key
        self.openrouter_api_key = openrouter_api_key
        self.storage_handler = storage_handler
        self.base_path = base_path