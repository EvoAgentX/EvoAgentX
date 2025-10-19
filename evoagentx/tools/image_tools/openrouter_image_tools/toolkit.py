from typing import Optional
from ...tool import Toolkit
from ...storage_handler import FileStorageHandler
from .image_generation import OpenRouterImageGenerationTool
from .image_edit import OpenRouterImageEditTool
from .image_analysis import ImageAnalysisTool


class OpenRouterImageToolkit(Toolkit):
    def __init__(self, name: str = "OpenRouterImageToolkit", api_key: Optional[str] = None, 
                 save_path: str = "./openrouter_images", storage_handler: Optional[FileStorageHandler] = None, 
                 auto_postprocess: bool = False):
        """
        OpenRouter Image Toolkit combining generation, editing, and analysis capabilities.
        
        Args:
            name: Toolkit name
            api_key: OpenRouter API key (fallback to env OPENROUTER_API_KEY)
            save_path: Directory to save images (default: "./openrouter_images")
            storage_handler: File storage handler for managing images
            auto_postprocess: Enable automatic postprocessing for unsupported sizes/formats
        """
        generation = OpenRouterImageGenerationTool(
            api_key=api_key,
            base_path=save_path,
            storage_handler=storage_handler, 
            auto_postprocess=auto_postprocess
        )
        edit = OpenRouterImageEditTool(
            api_key=api_key,
            base_path=save_path,
            storage_handler=storage_handler, 
            auto_postprocess=auto_postprocess
        )
        analysis = ImageAnalysisTool(
            api_key=api_key, 
            storage_handler=storage_handler
        )
        super().__init__(name=name, tools=[generation, edit, analysis])
        self.api_key = api_key
        self.save_path = save_path
        self.storage_handler = storage_handler
        self.auto_postprocess = auto_postprocess


class ImageAnalysisToolkit(Toolkit):
    def __init__(self, name: str = "ImageAnalysisToolkit", api_key: Optional[str] = None, 
                 model: str = "openai/gpt-4o", storage_handler: Optional[FileStorageHandler] = None):
        """
        Lightweight toolkit for image analysis only.
        
        Args:
            name: Toolkit name
            api_key: OpenRouter API key (fallback to env OPENROUTER_API_KEY)
            model: Model to use for analysis (default: openai/gpt-4o)
            storage_handler: File storage handler for managing images
        """
        analysis = ImageAnalysisTool(api_key=api_key, model=model, storage_handler=storage_handler)
        super().__init__(name=name, tools=[analysis])
        self.api_key = api_key
        self.model = model
        self.storage_handler = storage_handler
