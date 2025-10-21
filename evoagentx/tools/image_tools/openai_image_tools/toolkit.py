from typing import Optional
from ...tool import Toolkit
from ...storage_handler import FileStorageHandler
from .image_generation import OpenAIImageGenerationTool
from .image_edit import OpenAIImageEditTool
from .image_analysis import OpenAIImageAnalysisTool


class OpenAIImageToolkit(Toolkit):
    def __init__(self, name: str = "OpenAIImageToolkit", api_key: Optional[str] = None, organization_id: Optional[str] = None,
                 model: str = "dall-e-3", save_path: str = "./openai_images", 
                 storage_handler: Optional[FileStorageHandler] = None, auto_postprocess: bool = False):
        # Create storage handler if not provided (use save_path as base)
        from ...storage_handler import LocalStorageHandler
        if storage_handler is None:
            storage_handler = LocalStorageHandler(base_path=save_path)
        
        gen_tool = OpenAIImageGenerationTool(api_key=api_key, organization_id=organization_id,
                                             model=model, save_path=save_path, 
                                             storage_handler=storage_handler, auto_postprocess=auto_postprocess)
        edit_tool = OpenAIImageEditTool(api_key=api_key, organization_id=organization_id,
                                        save_path=save_path, storage_handler=storage_handler, auto_postprocess=auto_postprocess)
        analysis_tool = OpenAIImageAnalysisTool(api_key=api_key, organization_id=organization_id, 
                                                model=model, storage_handler=storage_handler)
        super().__init__(name=name, tools=[gen_tool, edit_tool, analysis_tool])
        self.api_key = api_key
        self.organization_id = organization_id
        self.model = model
        self.save_path = save_path
        self.storage_handler = storage_handler
        self.auto_postprocess = auto_postprocess

