from abc import ABC, abstractmethod
from typing import Optional, Literal


# Abstract base class for database store implementations
class DBStoreBase(ABC):
    """
    Abstract base class defining the interface for database storage operations.
    Subclasses must implement methods for inserting, deleting, updating, and retrieving metadata.
    """

    @abstractmethod
    def insert(self, metadata, store_type: Optional[Literal["memory", "agent", "workflow", "history"]], table=None):
        """
        Insert metadata into a specified table.
        """
        pass

    @abstractmethod
    def insert_memory(self, metadata, store_type: Optional[Literal["memory", "agent", "workflow", "history"]], table=None):
        """
        Insert memory metadata into a specified table.
        """
        pass

    @abstractmethod
    def insert_agent(self, metadata, store_type: Optional[Literal["memory", "agent", "workflow", "history"]], table=None):
        """
        Insert agent metadata into a specified table.
        """
        pass

    @abstractmethod
    def insert_workflow(self, metadata, store_type: Optional[Literal["memory", "agent", "workflow", "history"]], table=None):
        """
        Insert workflow metadata into a specified table.
        """
        pass

    @abstractmethod
    def insert_history(self, metadata, store_type: Optional[Literal["memory", "agent", "workflow", "history"]], table=None):
        """
        Insert history metadata into a specified table.
        """
        pass

    @abstractmethod
    def delete(self, metadata_id, store_type: Optional[Literal["memory", "agent", "workflow", "history"]], table=None):
        """
        Delete metadata by its ID from a specified table.
        """
        pass

    @abstractmethod
    def update(self, metadata_id, new_metadata=None, 
               store_type: Optional[Literal["memory", "agent", "workflow", "history"]]=None, table=None):
        """
        Update metadata by its ID in a specified table.
        """
        pass

    @abstractmethod
    def get_by_id(self, metadata_id, store_type: Optional[Literal["memory", "agent", "workflow", "history"]], table=None):
        """
        Retrieve metadata by its ID from a specified table.
        """
        pass

    @abstractmethod
    def col_info(self):
        """
        Retrieve information about the database collections (tables).
        """
        pass