from typing import Dict, List, Optional, Any
import os
from dotenv import load_dotenv
from .tool import Toolkit
from .collection_base import ToolCollection
from .storage_handler import FileStorageHandler, LocalStorageHandler

# from .image_openai import OpenAIImageAnalysisTool 
from .image_tools.openrouter_image_tools.image_analysis import OpenRouterImageAnalysisTool

load_dotenv()

IMAGE_ANALYSIS_COLLECTION_DESC = (
    "Analyzes and interprets one or more images according to a natural language prompt. "
    "Generates a text response (e.g., description, identification, classification, reasoning) based on the provided images."
)

def map_args_for_openai(inputs: Dict[str, Any], outputs: Dict[str, Any]) -> Dict[str, Any]:
    mapped = {
        "prompt": inputs.get("prompt"),
        "image_urls": inputs.get("image_urls"),
    }
    return {k: v for k, v in mapped.items() if v is not None}


def map_args_for_openrouter(inputs: Dict[str, Any], outputs: Dict[str, Any]) -> Dict[str, Any]:
    mapped = {
        "prompt": inputs.get("prompt"),
        "image_urls": inputs.get("image_urls"),
    }
    return {k: v for k, v in mapped.items() if v is not None}


def convert_output_from_openai(result: Dict[str, Any]) -> Dict[str, Any]:
    if "error" in result:
        return {"success": False, "error": result["error"], "provider": "openai"}
    # OpenAI analysis returns text content
    content = None
    if isinstance(result, dict):
        content = result.get("content") or result.get("text")
    return {"success": True, "content": content or "", "provider": "openai"}


def convert_output_from_openrouter(result: Dict[str, Any]) -> Dict[str, Any]:
    if "error" in result:
        return {"success": False, "error": result["error"], "provider": "openrouter"}
    text = None
    if isinstance(result, dict):
        text = result.get("content") or result.get("text")
    return {"success": True, "content": text or "", "provider": "openrouter"}


class ImageAnalysisCollection(ToolCollection):
    """
    Analyze and understand images with automatic provider fallback.
    Supports OpenAI vision models and OpenRouter vision-capable models.
    """
    name: str = "image_analysis"
    description: str = IMAGE_ANALYSIS_COLLECTION_DESC

    inputs: Dict[str, Dict[str, str]] = {
        "prompt": {"type": "string", "description": "Required. A natural language question or instruction for analyzing/understanding the input image(s)."},
        "image_urls": {"type": "array", "items": {"type": "string", "description": "HTTP(S) image URL"}, "description": "Required. An array of one or more HTTP(S) image URLs to provide as input for analysis. Example: ['https://example.com/photo1.png'], or multiple: ['https://example.com/photo1.png', 'https://example.com/photo2.jpg']."},
    }
    required: Optional[List[str]] = ["prompt", "image_urls"]

    def __init__(
        self,
        name: str = "image_analysis",
        openrouter_api_key: Optional[str] = None,
        storage_handler: Optional[FileStorageHandler] = None,
        base_path: str = "./images/analysis_cache",
        per_tool_timeout: Optional[float] = None,
    ):
        if not openrouter_api_key:
            raise ValueError(f"`openrouter_api_key` is required for {type(self).__name__}!")

        if storage_handler is None:
            storage_handler = LocalStorageHandler(base_path=base_path)

        toolkits, tool_names = [], []
        # Google Nano Banana Image Analysis 
        nano_banana_image_analysis = OpenRouterImageAnalysisTool(
            name="openrouter_image_analysis_nano_banana",
            api_key=openrouter_api_key, 
            model="google/gemini-2.5-flash-image-preview",
            storage_handler=storage_handler,
        )
        tool_names.append(nano_banana_image_analysis.name)
        toolkits.append(nano_banana_image_analysis)

        # OpenAI GPT-4o Image Analysis
        gpt4o_image_analysis = OpenRouterImageAnalysisTool(
            name="openrouter_image_analysis_gpt4o",
            api_key=openrouter_api_key, 
            model="openai/gpt-4o",
            storage_handler=storage_handler,
        )
        tool_names.append(gpt4o_image_analysis.name)
        toolkits.append(gpt4o_image_analysis)

        # # Qwen Image Analysis
        qwen_image_analysis = OpenRouterImageAnalysisTool(
            name="openrouter_image_analysis_qwen",
            api_key=openrouter_api_key, 
            model="qwen/qwen3-vl-235b-a22b-instruct",
            storage_handler=storage_handler,
        )
        tool_names.append(qwen_image_analysis.name)
        toolkits.append(qwen_image_analysis)

        super().__init__(
            name=name,
            description=IMAGE_ANALYSIS_COLLECTION_DESC,
            kits=toolkits,
            execution_order=tool_names,
            argument_mapping_function={
                "openrouter_image_analysis_nano_banana": map_args_for_openrouter,
                "openrouter_image_analysis_gpt4o": map_args_for_openrouter,
                "openrouter_image_analysis_qwen": map_args_for_openrouter,
            },
            output_mapping_function={
                "openrouter_image_analysis_nano_banana": convert_output_from_openrouter,
                "openrouter_image_analysis_gpt4o": convert_output_from_openrouter,
                "openrouter_image_analysis_qwen": convert_output_from_openrouter,
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
        **kwargs,
    ) -> Dict[str, Any]:
        inputs = {
            "prompt": prompt,
            "image_urls": image_urls,
            **kwargs,
        }
        return self._run_pipeline(inputs)


class ImageAnalysisCollectionToolkit(Toolkit):
    """Toolkit exposing only image analysis via provider fallback."""

    def __init__(
        self,
        name: str = "ImageAnalysisCollectionToolkit",
        openrouter_api_key: Optional[str] = None,
        storage_handler: Optional[FileStorageHandler] = None,
        base_path: str = "./images/analysis_cache",
    ):
        if not openrouter_api_key:
            openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

        if storage_handler is None:
            storage_handler = LocalStorageHandler(base_path=base_path)

        analysis_collection = ImageAnalysisCollection(
            openrouter_api_key=openrouter_api_key,
            storage_handler=storage_handler,
            base_path=base_path,
        )

        super().__init__(name=name, tools=[analysis_collection])
        # Store configuration for reference (consistent with other toolkits)
        self.openrouter_api_key = openrouter_api_key
        self.storage_handler = storage_handler
        self.base_path = base_path