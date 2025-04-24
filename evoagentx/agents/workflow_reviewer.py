from typing import Optional, List
from .agent import Agent
from ..core.message import Message, MessageType


class WorkFlowReviewer(Agent):
    """Agent responsible for reviewing workflow plans and execution results.
    
    The WorkFlowReviewer analyzes workflow plans and execution results to ensure
    quality, completeness, and correctness. It can identify potential issues,
    suggest improvements, and validate that the workflow meets its objectives.
    
    This agent extends the base Agent class with specialized functionality
    for workflow review purposes.
    """

    def execute(self, action_name: str, msgs: Optional[List[Message]] = None, action_input_data: Optional[dict] = None, **kwargs) -> Message:

        """Execute a workflow review action and return the results.
        
        This method overrides the parent execute method to specifically handle
        workflow review actions. It processes incoming messages or action input data,
        executes the specified review action, and returns a response message.
        
        Args:
            action_name: Name of the action to execute
            msgs: Optional list of messages providing context for the action
            action_input_data: Optional pre-extracted input data for the action
            **kwargs: Additional parameters passed to the parent execute method
            
        Returns:
            A message containing the execution results with MessageType.RESPONSE
            
        Notes:
            - Sets the return message type to RESPONSE by default
            - Passes all parameters to the parent execute method
        """
        message = super().execute(
            action_name=action_name, 
            action_input_data=action_input_data, 
            msgs=msgs, 
            return_msg_type=MessageType.RESPONSE, 
            **kwargs  
        )
        return message