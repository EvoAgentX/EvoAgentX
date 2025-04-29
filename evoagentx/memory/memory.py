from collections import defaultdict
from pydantic import Field, PositiveInt
from typing import Union, Optional, List, Dict

from ..core.module import BaseModule
from ..core.module_utils import generate_id, get_timestamp
from ..core.message import Message
from ..utils.utils import safe_remove


class BaseMemory(BaseModule):
    """Base class for memory implementations in the EvoAgentX framework.
    
    BaseMemory provides core functionality for storing, retrieving, and 
    filtering messages. It maintains a chronological list of messages while 
    also providing indices for efficient retrieval by action or workflow goal.
    
    Attributes:
        messages: List of stored Message objects.
        memory_id: Unique identifier for this memory instance.
        timestamp: Creation timestamp of this memory instance.
        capacity: Maximum number of messages that can be stored, or None for unlimited.
    """

    messages: List[Message] = []
    memory_id: str = Field(default_factory=generate_id)
    timestamp: str = Field(default_factory=get_timestamp)
    capacity: Optional[PositiveInt] = Field(default=None, description="maximum of messages, None means there is no limit to the message number")

    def init_module(self):
        """Initialize memory indices.
        
        Creates default dictionaries for indexing messages by action and workflow goal.
        """
        self._by_action = defaultdict(list)
        self._by_wf_goal = defaultdict(list)

    @property
    def size(self):
        """Returns the current number of messages in memory.
        
        Returns:
            int: Number of messages currently stored.
        """
        return len(self.messages)
    
    def clear(self):
        """Clear all messages from memory.
        
        Removes all messages and resets all indices.
        """
        self.messages.clear()
        self._by_action.clear()
        self._by_wf_goal.clear()
    
    def remove_message(self, message: Message):
        """Remove a single message from memory.
        
        Removes the specified message from the main message list and all indices.
        If the message is not found in memory, no action is taken.
        
        Args:
            message: The message to be removed. The message will be removed from 
                   self.messages, self._by_action, and self._by_wf_goal.
        """
        if not message:
            return
        if message not in self.messages:
            return
        safe_remove(self.messages, message)
        if self._by_action and not message.action:
            safe_remove(self._by_action[message.action], message)
        if self._by_wf_goal and not message.wf_goal:
            safe_remove(self._by_wf_goal[message.wf_goal], message)

    def add_message(self, message: Message):
        """Store a single message in memory.
        
        Adds the message to the main list and relevant indices if it's not already stored.
        
        Args:
            message (Message): the message to be stored. 
        """
        if not message:
            return
        if message in self.messages:
            return
        self.messages.append(message)
        if self._by_action and not message.action:
            self._by_action[message.action].append(message)
        if self._by_wf_goal and not message.wf_goal:
            self._by_wf_goal[message.wf_goal].append(message)
    
    def add_messages(self, messages: Union[Message, List[Message]], **kwargs):
        """
        store (a) message(s) to the memory. 

        Args:
            messages (Union[Message, List[Message]]): the input messages can be a single message or a list of message.
        """
        if not isinstance(messages, list):
            messages = [messages]
        for message in messages:
            self.add_message(message)
    
    def get(self, n: int=None, **kwargs) -> List[Message]:
        """Retrieve recent messages from memory.
        
        Returns the most recent messages, up to the specified limit.
        
        Args: 
            n: The maximum number of messages to return. If None, returns all messages.
            **kwargs: Additional parameters (unused in base implementation).
            
        Returns:
            A list of Message objects, ordered from oldest to newest.
            
        Raises:
            AssertionError: If n is negative.
        """
        assert n is None or n>=0, "n must be None or a positive int"
        messages = self.messages if n is None else self.messages[-n:]
        return messages

    def get_by_type(self, data: Dict[str, list], key: str, n: int = None, **kwargs) -> List[Message]:
        """
        Retrieve a list of Message objects from a given data dictionary `data` based on a specified type `key`.

        This function looks up the value associated with `key` in the `data` dictionary, which should be a list of messages. It then returns a subset of these messages according to the specified parameters.
        If `n` is provided, it limits the number of messages returned; otherwise, it may return the entire list. Additional keyword arguments (**kwargs) can be used to further filter or process the resulting messages.

        Args:
            data (Dict[str, list]): A dictionary where keys are type strings and values are lists of messages.
            key (str): The key in `data` identifying the specific list of messages to retrieve.
            n (int, optional): The maximum number of messages to return. If not provided, all messages under the given `key` may be returned.
            **kwargs: Additional parameters for filtering or processing the messages.

        Returns:
            List[Message]: A list of messages corresponding to the given `key`, possibly filtered or truncated according to `n` and other provided keyword arguments.
        """
        if not data or key not in data:
            return []
        assert n is None or n>=0, "n must be None or a positive int"
        messages = data[key] if n is None else data[key][-n:]
        return messages
    
    def get_by_action(self, actions: Union[str, List[str]], n: int=None, **kwargs) -> List[Message]:
        """
        return messages triggered by `actions` in the memory. 

        Args:
            actions: A single action name or list of action names to filter by.
            n: Maximum number of messages to return per action. If None, returns all matching messages.
            **kwargs: Additional parameters (unused in base implementation).
            
        Returns:
            A list of Message objects, sorted by timestamp.
        """
        if isinstance(actions, str):
            actions = [actions]
        messages = []
        for action in actions:
            messages.extend(self.get_by_type(self._by_action, key=action, n=n, **kwargs))
        messages = Message.sort_by_timestamp(messages)
        return messages
    
    def get_by_wf_goal(self, wf_goals: Union[str, List[str]], n: int=None, **kwargs) -> List[Message]:
        """
        return messages related to `wf_goals` in the memory. 

        Args:
            wf_goals: A single workflow goal or list of workflow goals to filter by.
            n: Maximum number of messages to return per workflow goal. If None, returns all matching messages.
            **kwargs: Additional parameters (unused in base implementation).
            
        Returns:
            A list of Message objects, sorted by timestamp.
        """
        if isinstance(wf_goals, str):
            wf_goals = [wf_goals]
        messages = []
        for wf_goal in wf_goals:
            messages.append(self.get_by_type(self._by_wf_goal, key=wf_goal, n=n, **kwargs))
        messages = Message.sort_by_timestamp(messages)
        return messages


class ShortTermMemory(BaseMemory):
    """Short-term memory implementation.
    
    This class extends BaseMemory to represent a temporary, short-term memory
    storage. In the current implementation, it inherits all functionality from
    BaseMemory without modifications, but it provides a semantic distinction
    for different memory usage patterns in the framework.
    """
    pass


