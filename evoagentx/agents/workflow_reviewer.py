from typing import Optional, List
from .agent import Agent
from ..core.message import Message # MessageType


class WorkFlowReviewer(Agent):

    """
    Placeholder for the Agent that is responsible for reviewing workflow plans and agents.
    """

    def execute(self, action_name: str, msgs: Optional[List[Message]] = None, action_input_data: Optional[dict] = None, **kwargs) -> Message:

        raise NotImplementedError("WorkflowReviewer is not implemented yet.") 