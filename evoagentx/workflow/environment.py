from enum import Enum
from typing import Union, Optional, List
from ..core.module import BaseModule
from ..core.message import Message, MessageType
from ..models.base_model import LLMOutputParser


class TrajectoryState(str, Enum):
    """Enumeration representing the status of a trajectory step.
    
    Defines possible states for recording the outcome of workflow execution steps,
    allowing the system to track successful and failed operations.
    
    Attributes:
        COMPLETED: Indicates the step was successfully completed
        FAILED: Indicates the step failed during execution
    """
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class TrajectoryStep(BaseModule):
    """Represents a single step in an execution trajectory.
    
    Encapsulates the details of a single step in the workflow execution,
    including the message that was processed, the resulting status,
    and any error information if the step failed.
    
    Attributes:
        message: The message that was processed in this step
        status: The outcome status of this step (COMPLETED or FAILED)
        error: Optional error information if the step failed
    """
    message: Message = None
    status: TrajectoryState
    error: Optional[str] = None


class Environment(BaseModule):
    """Manages the execution environment and state for workflow execution.
    
    Responsible for storing and managing intermediate states during workflow
    execution, including the trajectory of execution steps, task execution
    history, and execution data.
    
    Serves as a shared memory space for the workflow, providing access to
    execution history and data for decision making and state tracking.
    
    Attributes:
        trajectory: Sequential list of execution steps representing the full execution path
        task_execution_history: History of executed task names in order of execution
        execution_data: Dictionary of accumulated execution data from messages
    """
    trajectory: List[TrajectoryStep] = []
    task_execution_history: List[str] = []
    execution_data: dict = {}

    def update(self, message: Message, state: TrajectoryState = None, error: str = None, **kwargs):
        """Add a message to the execution trajectory with its status.
        
        Records a new step in the execution trajectory by creating a TrajectoryStep
        with the provided message, state, and optional error information. Also 
        updates the task execution history and execution data.
        
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
        """Update the task execution history based on a message.
        
        Records task names in the execution history when a response message 
        is processed. Ensures each task is only recorded once in sequence
        (no duplicates of the same task in a row).
        
        Args:
            message: The message to extract task information from
        """
        if message.wf_task is not None and message.msg_type in [MessageType.RESPONSE]:
            # if there are multiple actions for a task, only record once
            if not self.task_execution_history or message.wf_task != self.task_execution_history[-1]:
                self.task_execution_history.append(message.wf_task)

    def update_execution_data(self, message: Message):
        """Update execution data from message content.
        
        Extracts structured data from message content and adds it to the
        execution data dictionary. Handles both LLMOutputParser objects
        and plain dictionaries as message content.
        
        Args:
            message: The message containing data to be extracted
        """
        if isinstance(message.content, LLMOutputParser):
            data = message.content.get_structured_data()
            self.execution_data.update(data)
        if isinstance(message.content, dict):
            data = message.content
            self.execution_data.update(data)

    def get_task_messages(self, tasks: Union[str, List[str]], n: int = None, **kwargs) -> List[Message]:
        """Retrieve messages related to specified tasks.
        
        Filters the trajectory to find messages associated with the specified 
        task names. Can return all matching messages or just the most recent ones.
        
        Args:
            tasks: A single task name or list of task names to filter by
            n: Optional number of most recent messages to return (returns all if None)
            **kwargs: Additional parameters (unused)
            
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
        message_list = message_list if n is None else message_list[-n:]
        return message_list

    def get(self, n: int=None) -> List[Message]:
        """Retrieve the most recent messages from the trajectory.
        
        Gets all messages from the trajectory or just the n most recent ones.
        
        Args:
            n: Optional number of most recent messages to return (returns all if None)
            
        Returns:
            A list of the most recent n messages (or all if n is None)
            
        Raises:
            AssertionError: If n is negative
        """
        assert n is None or n>=0, "n must be None or a positive int"
        all_messages = [step.message for step in self.trajectory]
        messages = all_messages if n is None else all_messages[-n:]
        return messages
    
    def get_last_executed_task(self) -> str:
        """Get the name of the most recently executed task.
        
        Returns the name of the last task recorded in the task execution history,
        or None if no tasks have been executed.
        
        Returns:
            The name of the last executed task, or None if no tasks executed
        """
        if self.task_execution_history:
            return self.task_execution_history[-1]
        return None
    
    def get_all_execution_data(self) -> dict:
        """Get all accumulated execution data.
        
        Returns the complete dictionary of execution data that has been
        collected during workflow execution.
        
        Returns:
            Dictionary containing all execution data
        """
        return self.execution_data
    
    def get_execution_data(self, params: Union[str, List[str]]) -> dict:
        """Get specific execution data by parameter names.
        
        Retrieves specific execution data entries by their parameter names.
        
        Args:
            params: A single parameter name or list of parameter names to retrieve
            
        Returns:
            Dictionary containing the requested execution data
            
        Raises:
            KeyError: If any requested parameter is not found in the execution data
        """
        if isinstance(params, str):
            params = [params]
        data = {}
        for param in params:
            if param not in self.execution_data:
                raise KeyError(f"Couldn't find execution data with key '{param}'. Available execution data: {list(self.execution_data.keys())}")
            data[param] = self.execution_data[param]
        return data

