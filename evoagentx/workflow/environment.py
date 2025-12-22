from enum import Enum
from pydantic import Field
from typing import Union, Optional, List
from ..core.module import BaseModule
from ..core.message import Message, MessageType
from ..models.base_model import LLMOutputParser


class TrajectoryState(str, Enum):
    """
    Enum representing the status of a trajectory step.
    """
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class TrajectoryStep(BaseModule):

    message: Message = None
    status: TrajectoryState
    error: Optional[str] = None


class Environment(BaseModule):

    """
    Responsible for storing and managing intermediate states of execution.
    """
    trajectory: List[TrajectoryStep] = Field(default_factory=list)
    task_execution_history: List[str] = Field(default_factory=list)
    execution_data: dict = Field(default_factory=dict)

    def update(self, message: Message, state: TrajectoryState = None, error: str = None, **kwargs):
        """
        Add a message to the shared memory and optionally to a specific task's message list.

        Args:
            message (Message): The message to be added.
            task_name (str, optional): The name of the task this message is related to. If None, the message is considered global.
        """
        state = state or TrajectoryState.COMPLETED
        step = TrajectoryStep(message=message, status=state, error=error)
        self.trajectory.append(step)
        self.update_task_execution_history(message=message)
        self.update_execution_data(message=message)
        
    def update_task_execution_history(self, message: Message):
        if message.wf_task is not None and message.msg_type in [MessageType.RESPONSE]:
            # if there are multiple actions for a task, only record once
            if not self.task_execution_history or message.wf_task != self.task_execution_history[-1]:
                self.task_execution_history.append(message.wf_task)

    def update_execution_data(self, message: Message):
        if isinstance(message.content, LLMOutputParser):
            data = message.content.get_structured_data()
            self.execution_data.update(data)
        if isinstance(message.content, dict):
            data = message.content
            self.execution_data.update(data)
    
    def update_execution_data_from_context_extraction(self, extracted_data: dict):
        for key, value in extracted_data.items():
            if key not in self.execution_data:
                self.execution_data[key] = value
    
    def get_task_messages(self, tasks: Union[str, List[str]], n: int = None, include_inputs: bool = False, **kwargs) -> List[Message]:
        """
        Retrieve all messages related to specified tasks

        Returns:
            List[Message]: A list of messages related to the task.
        """
        if isinstance(tasks, str):
            tasks = [tasks]
        message_list = [] 
        for step in self.trajectory:
            message = step.message
            if message.wf_task is not None and message.wf_task in tasks:
                message_list.append(message)
            if include_inputs and message.msg_type == MessageType.INPUT and message not in message_list:
                message_list.append(message)
        message_list = message_list if n is None else message_list[-n:]
        return message_list

    def get(self, n: int=None) -> List[Message]:
        """
        return the most recent n messages
        """
        assert n is None or n>=0, "n must be None or a positive int"
        all_messages = [step.message for step in self.trajectory]
        messages = all_messages if n is None else all_messages[-n:]
        return messages
    
    def get_last_executed_task(self) -> str:
        if self.task_execution_history:
            return self.task_execution_history[-1]
        return None
    
    def get_all_execution_data(self) -> dict:
        return self.execution_data
    
    def get_execution_data(self, params: Union[str, List[str]]) -> dict:
        if isinstance(params, str):
            params = [params]
        data = {}
        for param in params:
            if param not in self.execution_data:
                raise KeyError(f"Couldn't find execution data with key '{param}'. Available execution data: {list(self.execution_data.keys())}")
            data[param] = self.execution_data[param]
        return data

