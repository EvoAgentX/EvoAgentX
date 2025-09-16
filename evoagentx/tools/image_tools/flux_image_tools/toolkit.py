from typing import Optional
import os
from ...tool import Toolkit
from ...storage_handler import FileStorageHandler
from .image_generation_edit import FluxImageGenerationEditTool
from ..openrouter_image_tools.image_analysis import ImageAnalysisTool


class FluxImageGenerationToolkit(Toolkit):
    """
    Flux toolkit combining generation tool and a common image analysis tool.
    """

    def __init__(
        self,
        name: str = "FluxImageGenerationToolkit",
        api_key: Optional[str] = None,
        save_path: str = "./imgs",
        storage_handler: Optional[FileStorageHandler] = None,
        analysis_model: str = "openai/gpt-4o-mini",
    ):
        tools = []

        gen_tool = FluxImageGenerationEditTool(
            api_key=api_key,
            storage_handler=storage_handler,
            base_path=save_path,
        )
        tools.append(gen_tool)

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


