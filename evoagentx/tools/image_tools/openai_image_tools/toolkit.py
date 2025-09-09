from typing import Optional
from ...tool import Toolkit
from .image_generation import OpenAIImageGenerationV2
from .image_edit import OpenAIGPTImage1EditV2
from .image_analysis_openai import OpenAIImageAnalysisV2


class OpenAIImageToolkitV2(Toolkit):
    def __init__(self, name: str = "OpenAIImageToolkitV2", api_key: str = None, organization_id: str = None,
                 generation_model: str = "dall-e-3", save_path: str = "./generated_images"):
        gen_tool = OpenAIImageGenerationV2(api_key=api_key, organization_id=organization_id,
                                           model=generation_model, save_path=save_path)
        edit_tool = OpenAIGPTImage1EditV2(api_key=api_key, organization_id=organization_id,
                                          save_path=save_path)
        analysis_tool = OpenAIImageAnalysisV2(api_key=api_key, organization_id=organization_id)
        super().__init__(name=name, tools=[gen_tool, edit_tool, analysis_tool])
        self.api_key = api_key
        self.organization_id = organization_id
        self.generation_model = generation_model
        self.save_path = save_path


