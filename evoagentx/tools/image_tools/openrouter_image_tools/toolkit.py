from typing import Optional
from ...tool import Toolkit
from ...storage_handler import FileStorageHandler
from .image_generation import OpenRouterImageGenerationEditTool
from .image_analysis import ImageAnalysisTool


class OpenRouterImageToolkit(Toolkit):
    def __init__(self, name: str = "OpenRouterImageToolkit", api_key: Optional[str] = None, 
                 storage_handler: Optional[FileStorageHandler] = None):
        analysis = ImageAnalysisTool(api_key=api_key, storage_handler=storage_handler)
        generation = OpenRouterImageGenerationEditTool(api_key=api_key, storage_handler=storage_handler)
        super().__init__(name=name, tools=[analysis, generation])
        self.api_key = api_key
        self.storage_handler = storage_handler


class ImageAnalysisToolkit(Toolkit):
    def __init__(self, name: str = "ImageAnalysisToolkit", api_key: Optional[str] = None, 
                 model: str = "openai/gpt-4o", storage_handler: Optional[FileStorageHandler] = None):
        # Lightweight re-export in this consolidated file
        analysis = ImageAnalysisTool(api_key=api_key, model=model, storage_handler=storage_handler)
        super().__init__(name=name, tools=[analysis])
        self.api_key = api_key
        self.model = model
        self.storage_handler = storage_handler


