from typing import List

from ..core.module import BaseModule
from ..agents.agent_manager import AgentManager
from ..optimizers.optimizer import Optimizer
from .workflow import WorkFlow


class WorkFlowController(BaseModule):

    agent_manager: AgentManager
    workflow: WorkFlow
    optimizers: List[Optimizer] = []

    def start(self, **kwargs):
        """
        start executing the workflow. 
        """
        pass 



