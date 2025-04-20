from abc import ABC, abstractmethod
from typing import Dict, Any

from ..core.module import BaseModule


class StorageHandler(BaseModule, ABC):
    """An interface for all storage handlers.
    
    StorageHandler defines an abstraction of storage used for reading and writing data 
    (such as memory, agents, workflow, etc.). It can be implemented in various ways 
    such as file storage, database storage, cloud storage, etc.
    
    All storage handlers must inherit from this class and implement all abstract methods.
    This ensures consistent data access patterns across different storage implementations.
    
    Attributes:
        No explicit attributes defined at this level. Subclasses may define their own.
    """

    @abstractmethod
    def load(self, *args, **kwargs):
        """Load all data from the underlying storage.
        
        This method loads all available data from the storage system and
        should prepare it for use within the application.
        
        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
            
        Returns:
            Implementation-dependent, typically the loaded data in a usable format.
        """
        pass 

    @abstractmethod
    def save(self, *args, **kwargs):
        """Save all data to the underlying storage at once.
        
        This method persists all data to the configured storage system.
        
        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
            
        Returns:
            Implementation-dependent, typically a success indicator or error details.
        """
        pass 

    @abstractmethod
    def load_memory(self, memory_id: str, **kwargs) -> Dict[str, Any]:
        """Load a single long term memory data entry.

        Retrieves memory data identified by memory_id from the storage system.

        Args:
            memory_id: The unique identifier of the long term memory.
            **kwargs: Additional parameters for the loading process.
        
        Returns:
            A dictionary containing data that can be used to recreate a LongTermMemory instance.
        """
        pass

    @abstractmethod
    def save_memory(self, memory_data: Dict[str, Any], **kwargs):
        """Save or update a single memory.
        
        Persists memory data to the storage system. If the memory already exists,
        it should be updated with the new data.

        Args:
            memory_data: The dictionary containing the long term memory's data.
            **kwargs: Additional parameters for the saving process.
            
        Returns:
            Implementation-dependent, typically a success indicator or error details.
        """
        pass 

    @abstractmethod
    def load_agent(self, agent_name: str, **kwargs) -> Dict[str, Any]:
        """Load a single agent's data.

        Retrieves agent data identified by agent_name from the storage system.

        Args: 
            agent_name: The unique identifier (name) of the agent.
            **kwargs: Additional parameters for the loading process.
        
        Returns:
            A dictionary containing data that can be used to recreate an Agent instance.
        """
        pass 

    @abstractmethod
    def remove_agent(self, agent_name: str, **kwargs):
        """Remove an agent from storage if the agent exists.
        
        Deletes the agent data from the storage system.

        Args:
            agent_name: The name (unique identifier) of the agent to be deleted.
            **kwargs: Additional parameters for the removal process.
            
        Returns:
            Implementation-dependent, typically a success indicator or error details.
        """
        pass

    @abstractmethod
    def save_agent(self, agent_data: Dict[str, Any], **kwargs):
        """Save or update a single agent's data.
        
        Persists agent data to the storage system. If the agent already exists,
        it should be updated with the new data.

        Args:
            agent_data: The dictionary containing the agent's data.
            **kwargs: Additional parameters for the saving process.
            
        Returns:
            Implementation-dependent, typically a success indicator or error details.
        """
        pass 

    @abstractmethod
    def load_workflow(self, workflow_id: str, **kwargs) -> Dict[str, Any]:
        """Load a single workflow's data.
        
        Retrieves workflow data identified by workflow_id from the storage system.

        Args: 
            workflow_id: The unique identifier of the workflow.
            **kwargs: Additional parameters for the loading process.
        
        Returns:
            A dictionary containing data that can be used to recreate a WorkFlow instance.
        """
        pass 

    @abstractmethod
    def save_workflow(self, workflow_data: Dict[str, Any], **kwargs):
        """Save or update a workflow's data.
        
        Persists workflow data to the storage system. If the workflow already exists,
        it should be updated with the new data.

        Args:
            workflow_data: The dictionary containing the workflow's data.
            **kwargs: Additional parameters for the saving process.
            
        Returns:
            Implementation-dependent, typically a success indicator or error details.
        """
        pass


