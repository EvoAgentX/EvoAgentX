from typing import Optional, List
from .agent import Agent
from ..core.message import Message, MessageType


class WorkFlowReviewer(Agent):

    def execute(self, action_name: str, msgs: Optional[List[Message]] = None, action_input_data: Optional[dict] = None, **kwargs) -> Message:

        """
        Plan and decompose high-level tasks into executable task configurations.
        """
        message = super().execute(
            action_name=action_name, 
            action_input_data=action_input_data, 
            msgs=msgs, 
            return_msg_type=MessageType.RESPONSE, 
            **kwargs  
        )
        return message