from typing import Optional
import os
from ...tool import Toolkit
from ...storage_handler import FileStorageHandler
from .image_generation import FluxImageGenerationTool
from .image_edit import FluxImageEditTool
from ..openrouter_image_tools.image_analysis import ImageAnalysisTool


class FluxImageToolkit(Toolkit):
    """
    Flux toolkit, combines image generation, edit and analysis tools.
    """

    def __init__(
        self,
        name: str = "FluxImageToolkit",
        api_key: Optional[str] = None,
        save_path: str = "./flux_images",
        storage_handler: Optional[FileStorageHandler] = None,
        analysis_model: str = "openai/gpt-4o-mini",
        auto_postprocess: bool = False,
    ):
        tools = []

        # add image generation tool
        gen_tool = FluxImageGenerationTool(
            api_key=api_key,
            storage_handler=storage_handler,
            base_path=save_path,
            auto_postprocess=auto_postprocess,
        )
        tools.append(gen_tool)

        # add image edit tool
        edit_tool = FluxImageEditTool(
            api_key=api_key,
            storage_handler=storage_handler,
            base_path=save_path,
            auto_postprocess=auto_postprocess,
        )
        tools.append(edit_tool)

        # add image analysis tool (if OPENROUTER_API_KEY is set)
        try:
            resolved_key = os.getenv("OPENROUTER_API_KEY")
            if resolved_key:
                analysis_tool = ImageAnalysisTool(
                    api_key=resolved_key,
                    model=analysis_model,
                )
                tools.append(analysis_tool)
        except Exception:
            pass

        super().__init__(name=name, tools=tools)
        self.api_key = api_key
        self.save_path = save_path
        self.storage_handler = storage_handler
        self.auto_postprocess = auto_postprocess


