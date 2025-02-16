from enum import Enum
from pydantic import Field
from datetime import datetime
from typing import Optional, Callable, Any, List

from .module import BaseModule
from .module_utils import generate_id, get_timestamp

class MessageType(Enum):
    
    REQUEST = "request"
    RESPONSE = "response"
    COMMAND = "command"
    ERROR = "error"
    UNKNOWN = "unknown"
    INPUT = "input"


class Message(BaseModule):

    """
    the base class for message. 

    Args: 
        content (Any): the content of the message, need to implement str() function. 
        agent (str): the sender of the message, normally set as the agent name.
        action (str): the trigger of the message, normally set as the action name.
        prompt (str): the prompt used to obtain the generated text. 
        next_actions (List[str]): the following actions. 
        msg_type (str): the type of the message, such as "request", "response", "command" etc. 
        wf_goal (str): the goal of the whole workflow. 
        wf_task (str): the name of a task in the workflow, i.e., the ``name`` of a WorkFlowNode instance. 
        wf_task_desc (str): the description of a task in the workflow, i.e., the ``description`` of a WorkFlowNode instance.
        message_id (str): the unique identifier of the message. 
        timestamp (str): the timestame of the message. 
    """
    
    content: Any
    agent: Optional[str] = None
    # receivers: Optional[Union[str, List[str]]] = None
    action: Optional[str] = None
    prompt: Optional[str] = None
    next_actions: Optional[List[str]] = None
    msg_type: Optional[MessageType] = MessageType.UNKNOWN
    wf_goal: Optional[str] = None
    wf_task: Optional[str] = None
    wf_task_desc: Optional[str] = None
    message_id: Optional[str] = Field(default_factory=generate_id)
    timestamp: Optional[str] = Field(default_factory=get_timestamp)
    
    def __str__(self) -> str:
        return self.to_str()
    
    def __eq__(self, other: "Message"):
        return self.message_id == other.message_id

    def __hash__(self):
        return self.message_id
    
    def to_str(self) -> str:

        msg_part = []
        if self.timestamp:
            msg_part.append(f"[{self.timestamp}]")
        if self.agent:
            msg_part.append(f"Agent: {self.agent}")
        if self.msg_type and self.msg_type != MessageType.UNKNOWN:
            msg_part.append(f"Type: {self.msg_type}")
        if self.action:
            msg_part.append(f"Action: {self.action}")
        if self.wf_goal:
            msg_part.append(f"Goal: {self.wf_goal}")
        if self.wf_task:
            msg_part.append(f"Task: {self.wf_task} ({self.wf_task_desc or 'No description'})")
        if self.content:
            msg_part.append(f"Content: {str(self.content)}")
                
        msg = "\n".join(msg_part)
        return msg 

    @classmethod
    def sort_by_timestamp(cls, messages: List['Message'], reverse: bool = False) -> List['Message']:
        """
        sort the messages based on the timestamp. 

        Args: 
            messages (List[Message]): the messages to be sorted. 
            reverse (bool): If True, sort the messages in descending order. Otherwise, sort the messages in ascending order.
        """
        messages.sort(key=lambda msg: datetime.strptime(msg.timestamp, "%Y-%m-%d %H:%M:%S"), reverse=reverse)
        return messages

    @classmethod
    def sort(cls, messages: List['Message'], key: Optional[Callable[['Message'], Any]] = None, reverse: bool = False) -> List['Message']:
        """
        sort the messages using key or timestamp (by default). 

        Args:
            messages (List[Message]): the messages to be sorted. 
            key (Optional[Callable[['Message'], Any]]): the function used to sort messages. 
            reverse (bool): If True, sort the messages in descending order. Otherwise, sort the messages in ascending order.
        """
        if key is None:
            return cls.sort_by_timestamp(messages, reverse=reverse)
        messages.sort(key=key, reverse=reverse)
        return messages

    @classmethod
    def merge(cls, messages: List[List['Message']], sort: bool=False, key: Optional[Callable[['Message'], Any]] = None, reverse: bool=False) -> List['Message']:
        """
        merge different message list. 

        Args:
            messages (List[List[Message]]): the message lists to be merged. 
            sort (bool): whether to sort the merged messages.
            key (Optional[Callable[['Message'], Any]]): the function used to sort messages. 
            reverse (bool): If True, sort the messages in descending order. Otherwise, sort the messages in ascending order.
        """
        merged_messages = sum(messages, [])
        if sort:
            merged_messages = cls.sort(merged_messages, key=key, reverse=reverse)
        return merged_messages
    

