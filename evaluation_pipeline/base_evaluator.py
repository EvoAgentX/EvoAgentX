from .config import EvaluationConfig
from evoagentx.models import OpenAILLM, OpenAILLMConfig

class BaseEvaluator:
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.layer_num = None
        self.init_llm_object()
        self.init_tools_object()

    def init_llm_object(self):
        if self.config.evaluation_llm_type == "OpenAILLM":
            llm_config = OpenAILLMConfig(model=self.config.evaluation_llm_model, temperature=self.config.evaluation_llm_temperature)
            llm = OpenAILLM(config=llm_config)
            self.llm = llm
            self.llm_config = llm_config
        else:
            raise ValueError(f"Unsupported LLM type: {self.config.evaluation_llm_type}")
        
    def init_tools_object(self):
        self.tools = self.config.tools