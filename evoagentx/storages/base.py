from abc import ABC, abstractmethod
from typing import Dict, Any

from ..core.module import BaseModule


class StorageHandler(BaseModule, ABC):

    """
    An interface for all the storage handlers. 
    
    StorageHandler defines an abstraction of storage used for reading and writing data (such as memory, agents, workflow, ect.). 
    It can be implemented in various ways such as file storage, database storage, cloud storage, etc.
    """

    @abstractmethod
    def load(self, *args, **kwargs):
        """
        Load all data from the underlying storage (file, database, etc.)
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
        Load a single long term memory data. 

        Args:
            memory_id (str): the id of the long term memory. 
        
        Returns:
            Dict[str, Any]: the data that can be used to create an LongTermMemory instance. 
        """
        pass

    @abstractmethod
    def save_memory(self, memory_data: Dict[str, Any], **kwargs):
        """
        Save or update a single memory. 

        Args:
            memory_data (Dict[str, Any]): the long term memory's data. 
        """
        pass 

    @abstractmethod
    def load_agent(self, agent_name: str, **kwargs) -> Dict[str, Any]:
        # TODO
        """
        Load a single agent's data.

        Args: 
            agent_name (str): use agent name (unique identifier) to retrieve the data. 
        
        Returns:
            Dict[str, Any]: the data that can be used to create an Agent instance. 
        """
        pass 

    @abstractmethod
    def remove_agent(self, agent_name: str, **kwargs):
        # TODO 
        """
        Remove an agent from storage if the agent exists. 

        Args:
            agent_name (str) the name of the agent to be deleted.
        """
        pass

    @abstractmethod
    def save_agent(self, agent_data: Dict[str, Any], **kwargs):
        """
        Save or update a single agent's data.

        Args:
            agent_data (Dict[str, Any]): the agent's data. 
        """
        pass 

    
    @abstractmethod
    def load_workflow(self, workflow_id: str, **kwargs) -> Dict[str, Any]:
        """
        Load a single workflow's data. 

        Args: 
            workflow_id (str): the id of the workflow. 
        
        Returns:
            Dict[str, Any]: the data that can be used to create a WorkFlow instance.
        """
        pass 

    @abstractmethod
    def save_workflow(self, workflow_data: Dict[str, Any], **kwargs):
        """
        Save or update a workflow's data.

        Args:
            workflow_data (Dict[str, Any]): the workflow's data.
        """
        pass 


