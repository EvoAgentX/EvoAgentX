from typing import Optional
from ...tool import Toolkit
from .image_analysis import ImageAnalysisTool
from .image_generation import OpenRouterImageGenerationEditTool


class OpenRouterImageToolkit(Toolkit):
    def __init__(self, name: str = "OpenRouterImageToolkit", api_key: Optional[str] = None):
        analysis = ImageAnalysisTool(api_key=api_key)
        generation = OpenRouterImageGenerationEditTool(api_key=api_key)
        super().__init__(name=name, tools=[analysis, generation])
        self.api_key = api_key


