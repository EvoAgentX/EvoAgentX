from typing import Union, Optional
from pydantic import Field 

from ..core.module import BaseModule
from ..models.base_model import BaseLLM
from ..benchmark.benchmark import Benchmark
from ..evaluators.evaluator import Evaluator
from ..workflow.action_graph import ActionGraph 
from ..workflow.workflow_graph import WorkFlowGraph


class Optimizer(BaseModule):
    
    graph: Union[WorkFlowGraph, ActionGraph] = Field(description="The workflow to optimize.")
    evaluator: Evaluator = Field(description="The evaluator to use for optimization.")

    llm: BaseLLM = Field(default=None, description="The LLM to use for optimization and evaluation.")
    max_steps: int = Field(default=5, description="The maximum number of optimization steps to take.")
    eval_every_n_steps: int = Field(default=1, description="Evaluate the workflow every `eval_every_n_steps` steps.")
    eval_rounds: int = Field(default=1, description="Run evaluation for `eval_rounds` times and compute the average score.")
    convergence_threshold: int = Field(default=5, description="If the optimization has not improved the score for `convergence_threshold` steps, the optimization will be stopped.")

    def optimize(self, dataset: Benchmark, **kwargs):
        """
        Optimize the workflow.
        """
        raise NotImplementedError(f"``optimize`` function for {type(self).__name__} is not implemented!")

    def step(self, **kwargs):
        """
        Take a step of optimization.
        """
        raise NotImplementedError(f"``step`` function for {type(self).__name__} is not implemented!")
    
    def evaluate(self, dataset: Benchmark, eval_mode: str = "test", graph: Optional[Union[WorkFlowGraph, ActionGraph]] = None, **kwargs) -> dict:
        """
        Evaluate the workflow. If `graph` is provided, use the provided graph for evaluation. Otherwise, use the graph in the optimizer.
        """
        raise NotImplementedError(f"``evaluate`` function for {type(self).__name__} is not implemented!")
    
    def convergence_check(self, *args, **kwargs) -> bool:
        """
        Check if the optimization has converged.
        """
        raise NotImplementedError(f"``convergence_check`` function for {type(self).__name__} is not implemented!")