from abc import ABC, abstractmethod
from typing import Dict, Any

from ..core.module import BaseModule


class StorageHandler(BaseModule, ABC):
    """An interface for all storage handlers.
    
    StorageHandler defines an abstraction of storage used for reading and writing data 
    (such as memory, agents, workflow, etc.). It can be implemented in various ways 
    such as file storage, database storage, cloud storage, etc.
    """

    @abstractmethod
    def load(self, *args, **kwargs):
        """
        Load all data from the underlying storage.
        """
        pass 

    @abstractmethod
    def save(self, *args, **kwargs):
        """
        Save all data to the underlying storage at once.
        """
        pass 

    @abstractmethod
    def load_memory(self, memory_id: str, **kwargs) -> Dict[str, Any]:
        """
        Load a single long term memory data entry.
        """
        pass

    @abstractmethod
    def save_memory(self, memory_data: Dict[str, Any], **kwargs):
        """
        Save or update a single memory.
        """
        pass 

    @abstractmethod
    def load_agent(self, agent_name: str, **kwargs) -> Dict[str, Any]:
        """
        Load a single agent's data.
        """
        pass 

    @abstractmethod
    def remove_agent(self, agent_name: str, **kwargs):
        """
        Remove an agent from storage if the agent exists.
        """
        pass

    @abstractmethod
    def save_agent(self, agent_data: Dict[str, Any], **kwargs):
        """
        Save or update a single agent's data.
        """
        pass 

    @abstractmethod
    def load_workflow(self, workflow_id: str, **kwargs) -> Dict[str, Any]:
        """
        Load a single workflow's data.
        """
        pass 

    @abstractmethod
    def save_workflow(self, workflow_data: Dict[str, Any], **kwargs):
        """
        Save or update a workflow's data.
        """
        pass


