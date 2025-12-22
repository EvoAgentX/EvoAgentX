from typing import List
from pydantic import Field 

from ..core.module import BaseModule
from ..agents.agent_manager import AgentManager
from ..optimizers.optimizer import Optimizer
from .workflow import WorkFlow


class WorkFlowController(BaseModule):

    agent_manager: AgentManager
    workflow: WorkFlow
    optimizers: List[Optimizer] = Field(default_factory=list) 

    def start(self, **kwargs):
        """
        start executing the workflow. 
        """
        pass 



