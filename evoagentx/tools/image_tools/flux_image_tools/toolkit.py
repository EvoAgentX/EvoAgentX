from typing import Optional
from ...tool import Toolkit
from ...storage_handler import FileStorageHandler
from .image_generation import FluxImageGenerationTool
from .image_edit import FluxImageEditTool
from ..openrouter_image_tools.image_analysis import OpenRouterImageAnalysisTool


class FluxImageToolkit(Toolkit):
    """
    Flux toolkit, combines image generation, edit and analysis tools.
    """

    def __init__(self, name: str = "FluxImageToolkit", flux_api_key: Optional[str] = None, openrouter_api_key: Optional[str] = None,
                 model: str = "flux-kontext-max", save_path: str = "./flux_images", storage_handler: Optional[FileStorageHandler] = None,
                 analysis_model: str = "openai/gpt-4o-mini", auto_postprocess: bool = False):
        gen_tool = FluxImageGenerationTool(api_key=flux_api_key, model=model, storage_handler=storage_handler, save_path=save_path, auto_postprocess=auto_postprocess)
        edit_tool = FluxImageEditTool(api_key=flux_api_key, model=model, storage_handler=storage_handler, save_path=save_path, auto_postprocess=auto_postprocess)
        analysis_tool = OpenRouterImageAnalysisTool(api_key=openrouter_api_key, model=analysis_model, storage_handler=storage_handler)
        super().__init__(name=name, tools=[gen_tool, edit_tool, analysis_tool])
        self.flux_api_key = flux_api_key
        self.openrouter_api_key = openrouter_api_key
        self.model = model
        self.analysis_model = analysis_model
        self.save_path = save_path
        self.storage_handler = storage_handler
        self.auto_postprocess = auto_postprocess


