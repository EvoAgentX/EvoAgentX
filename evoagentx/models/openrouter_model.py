from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
from typing import List

from .openai_model import OpenAILLM
from .model_configs import OpenRouterConfig
from ..core.registry import register_model
from openai import OpenAI
from .openai_model import OpenAILLM

@register_model(config_cls=OpenRouterConfig, alias=["openrouter"])
class OpenRouterLLM(OpenAILLM):
    def init_model(self):
        config: OpenRouterConfig = self.config
        self._client = OpenAI(api_key=config.openrouter_key, base_url="https://openrouter.ai/api/v1")
        self._default_ignore_fields = ["llm_type", "output_response", "openrouter_key", "deepseek_key", "anthropic_key"] # parameters in OpenAILLMConfig that are not OpenAI models' input parameters 


