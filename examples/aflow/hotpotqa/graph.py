import evoagentx.workflow.operators as operator
import examples.aflow.hotpotqa.prompt as prompt_custom # noqa: F401
from evoagentx.models.model_configs import LLMConfig
from evoagentx.benchmark.benchmark import Benchmark
from evoagentx.models.model_utils import create_llm_instance

class Workflow:
    
    def __init__(
        self,
        name: str,
        llm_config: LLMConfig,
        benchmark: Benchmark
    ):
        self.name = name
        self.llm = create_llm_instance(llm_config)
        self.benchmark = benchmark
        self.custom = operator.Custom(self.llm)
        self.answer_generate = operator.AnswerGenerate(self.llm) 
    
    async def __call__(self, problem: str):
        """
        Implementation of the workflow
        """
        solution = await self.answer_generate(input=problem)
        return solution['answer']

