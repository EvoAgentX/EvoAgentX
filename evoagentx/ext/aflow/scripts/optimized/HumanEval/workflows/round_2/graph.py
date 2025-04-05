from typing import Literal
import evoagentx.workflow.operators as operator
from evoagentx.workflow.workflow_generator import WorkFlowGenerator
from evoagentx.models.base_model import BaseLLM
from evoagentx.models.model_utils import CostManager
import evoagentx.prompts.aflow_optimize_prompt as prompt

DatasetType = Literal["HumanEval", "MBPP", "GSM8K", "MATH", "HotpotQA", "DROP"]

class AFlowWorkflowGenerator(WorkFlowGenerator):
    
    def __init__(self, *, 
                 name: str,
                 llm: BaseLLM,
                 dataset: DatasetType,
                 **kwargs):
        super().__init__(name=name, llm=llm, **kwargs)
        
        self.llm = llm
        self.dataset = dataset
        self.name = name
        self.cost_manager = CostManager()
        self.custom = operator.Custom(self.llm)
        self.custom_code_generate = operator.CustomCodeGenerate(self.llm)
        self.answer_generate = operator.AnswerGenerate(self.llm)
        self.test = operator.Test(self.llm)
        self.sc_ensemble = operator.ScEnsemble(self.llm)
        
        
    async def __call__(self, problem: str, entry_point: str):
        solution = await self.custom_code_generate.execute_async(input=problem, 
                                             instruction=entry_point)
        testing_result = await self.test.execute_async(problem=problem, solution=solution['response'], entry_point=entry_point)
        if testing_result['result']:  # If test passes
            return solution['response'], float(self.cost_manager.total_cost) if isinstance(self.cost_manager.total_cost, (int, float, str)) else 0.0
        else:  # If test fails
            return "Solution is incorrect", 0.0
