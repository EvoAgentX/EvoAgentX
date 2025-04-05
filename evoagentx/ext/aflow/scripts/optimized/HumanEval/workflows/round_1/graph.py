# -*- coding: utf-8 -*-
# @Date    : 6/27/2024 22:07 PM
# @Author  : didi
# @Desc    : Basic Graph Class

from typing import Literal
# import evoagentx.ext.aflow.scripts.optimized.HumanEval.workflows.template.operator as operator
import evoagentx.workflow.operators as operator
from evoagentx.workflow.workflow_generator import WorkFlowGenerator
from evoagentx.models.base_model import BaseLLM
from evoagentx.models.model_utils import CostManager

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
        total_cost = float(self.cost_manager.total_cost) if isinstance(self.cost_manager.total_cost, (int, float, str)) else 0.0
        return solution['response'], total_cost
        