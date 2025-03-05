from typing import Union, Optional
from pydantic import Field 

from ..core.module import BaseModule
from ..models.base_model import BaseLLM
from ..benchmark.benchmark import Benchmark
from ..evaluators.evaluator import Evaluator
from ..workflow.action_graph import ActionGraph 
from ..workflow.workflow_graph import WorkFlowGraph


class Optimizer(BaseModule):
    
    workflow_graph: Optional[Union[WorkFlowGraph, ActionGraph]] = Field(default=None, description="The workflow to optimize.")
    evaluator: Optional[Evaluator] = Field(default=None, description="The evaluator to use for optimization.")
    max_steps: Optional[int] = Field(default=None, description="The maximum number of optimization steps to take.")

    llm: Optional[BaseLLM] = Field(default=None, description="The LLM to use for optimization and evaluation.")
    eval_every_n_steps: int = Field(default=1, description="Evaluate the workflow every `eval_every_n_steps` steps.")
    eval_rounds: int = Field(default=1, description="Run evaluation for `eval_rounds` times and compute the average score.")

    def optimize(self, dataset: Benchmark, **kwargs):
        raise NotImplementedError(f"``optimize`` function for {type(self).__name__} is not implemented!")

    def step(self):
        raise NotImplementedError(f"``step`` function for {type(self).__name__} is not implemented!")
    
