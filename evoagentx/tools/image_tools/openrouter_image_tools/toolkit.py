from typing import Optional
from ...tool import Toolkit
from ...storage_handler import FileStorageHandler
from .image_generation import OpenRouterImageGenerationTool
from .image_edit import OpenRouterImageEditTool
from .image_analysis import OpenRouterImageAnalysisTool


class OpenRouterImageToolkit(Toolkit):
    def __init__(self, name: str = "OpenRouterImageToolkit", api_key: Optional[str] = None, 
                 model: str = "google/gemini-2.5-flash-image", save_path: str = "./openrouter_images", 
                 storage_handler: Optional[FileStorageHandler] = None, auto_postprocess: bool = False):
        # Create storage handler if not provided (use save_path as base)
        from ...storage_handler import LocalStorageHandler
        if storage_handler is None:
            storage_handler = LocalStorageHandler(base_path=save_path)
        
        gen_tool = OpenRouterImageGenerationTool(api_key=api_key, model=model, save_path=save_path, 
                                                 storage_handler=storage_handler, auto_postprocess=auto_postprocess)
        edit_tool = OpenRouterImageEditTool(api_key=api_key, model=model, save_path=save_path, 
                                            storage_handler=storage_handler, auto_postprocess=auto_postprocess)
        analysis_tool = OpenRouterImageAnalysisTool(api_key=api_key, model=model, storage_handler=storage_handler)
        super().__init__(name=name, tools=[gen_tool, edit_tool, analysis_tool])
        self.api_key = api_key
        self.model = model
        self.save_path = save_path
        self.storage_handler = storage_handler
        self.auto_postprocess = auto_postprocess


class ImageAnalysisToolkit(Toolkit):
    def __init__(self, name: str = "ImageAnalysisToolkit", api_key: Optional[str] = None, 
                 model: str = "openai/gpt-4o", storage_handler: Optional[FileStorageHandler] = None):
        analysis_tool = OpenRouterImageAnalysisTool(api_key=api_key, model=model, storage_handler=storage_handler)
        super().__init__(name=name, tools=[analysis_tool])
        self.api_key = api_key
        self.model = model
        self.storage_handler = storage_handler
