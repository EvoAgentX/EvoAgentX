import json
import threading
from functools import wraps
from typing import Dict, Literal, Callable, Optional, List

import sqlite3

from .base import DBStoreBase
from evoagentx.storages.schema import TableType, MemoryStore, AgentStore, WorkflowStore, HistoryStore, IndexStore


# Helper function to generate SQL for creating a table
def _create_table(table: str, column: List[str]) -> str:
    """
    Generates SQL to create a table with the specified columns.
    The first column is set as the PRIMARY KEY.

    Attributes:
        table (str): The name of the table to create.
        column (List[str]): List of column names.

    Returns:
        str: SQL statement to create the table.
    """
    if not column:
        raise ValueError("Column list cannot be empty")

    # Quote column names to handle reserved keywords and add commas
    column_defs = [f'"{column[0]}" TEXT PRIMARY KEY'] + [f'"{col}" TEXT' for col in column[1:]]
    table_column = ", ".join(column_defs)
    table_sql = f"""CREATE TABLE IF NOT EXISTS {table} (
        {table_column}
    )"""

    return table_sql

# Helper function to generate SQL for inserting metadata
def _insert_meta(table: str, colum: List[str]) -> str:
    """
    Generates SQL to insert metadata into a table.

    Attributes:
        table (str): The name of the table.
        colum (List[str]): List of column names.

    Returns:
        str: SQL statement for inserting data.
    """
    value_ = ", ".join(["?"] * len(colum))
    insert_string = f"""
    INSERT INTO {table} ({", ".join([f'"{c}"' for c in colum])})
    VALUES ({value_})"""

    return insert_string 

# Decorator to validate metadata and ensure table exists
def check_db_format(func: Callable) -> Callable:
    """
    Decorator to validate metadata format and create tables if they don't exist.
    Ensures the metadata matches the expected Pydantic model for the store type.

    Attributes:
        func (Callable): The function to decorate.

    Returns:
        Callable: The wrapped function with validation and table creation logic.
    """
    @wraps(func)
    def worker(self, metadata, *args, **kwargs):
        # Extract table and store type from kwargs
        table = kwargs.get("table", None)
        store_type = kwargs.get("store_type")   # memory, workflow, agent, history, index

        # Use default table name if none provided
        if table is None:
            table = store_type

        # Validate metadata based on store type and convert to Pydantic model
        if store_type == TableType.store_memory:
            column = list(MemoryStore.model_fields.keys())
            metadata = MemoryStore.model_validate(metadata)

        elif store_type == TableType.store_agent:
            column = list(AgentStore.model_fields.keys())
            metadata = AgentStore.model_validate(metadata)

        elif store_type == TableType.store_workflow:
            column = list(WorkflowStore.model_fields.keys())
            metadata = WorkflowStore.model_validate(metadata)

        elif store_type == TableType.store_history:
            column = list(HistoryStore.model_fields.keys())
            metadata = HistoryStore.model_validate(metadata, strict=False)
        elif store_type == TableType.store_indexing:
            column = list(IndexStore.model_fields.keys())
            metadata = IndexStore.model_validate(metadata, strict=False)
        else:
            raise ValueError("The value of store type is not valid.")
        
        # Create table if it doesn't exist
        table_column = _create_table(table, column)

        with self._lock:
            with self.connection:
                self.connection.execute(table_column)
                self.connection.commit()

        kwargs["metadata"] = metadata
        return func(self, *args, **kwargs)
    
    return worker

# SQLite implementation of the DBStoreBase interface
class SQLite(DBStoreBase):
    """
    SQLite implementation of the DBStoreBase interface.
    Provides methods for inserting, deleting, updating, and retrieving metadata in a SQLite database.
    Uses thread-safe operations with locking.
    """
    def __init__(self, path, *args, **kwargs) -> None:
        """
        Initialize the SQLite database connection.

        Attributes:
            path (str): Path to the SQLite database file.

        """
        self.connection = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.Lock()
    
    @check_db_format
    def insert_memory(self, metadata: MemoryStore, store_type: Optional[Literal["memory", "agent", "workflow", "history", "indexing"]], 
                      table: Optional[str]=None, *args, **kwargs):
        """
        Insert memory metadata into the specified table.

        Attributes:
            metadata (MemoryStore): The memory metadata to insert.
            store_type (str): The type of store (e.g., 'memory').
            table (Optional[str]): The table name; defaults to 'memory' if None.
        """
        with self._lock:
            with self.connection:
                if table is None:
                    table = TableType.store_memory

                insert_string = _insert_meta(table, list(MemoryStore.model_fields.keys()))
                self.connection.execute(
                    insert_string,
                    tuple([json.dumps(meta) if not isinstance(meta, str) else meta \
                           for meta in metadata.model_dump().values()])
                )
                self.connection.commit()

    @check_db_format
    def insert_agent(self, metadata: AgentStore, store_type: Optional[Literal["memory", "agent", "workflow", "history", "indexing"]], 
                     table: Optional[str]=None, *args, **kwargs):
        """
        Insert agent metadata into the specified table.

        Attributes:
            metadata (AgentStore): The agent metadata to insert.
            store_type (str): The type of store (e.g., 'agent').
            table (Optional[str]): The table name; defaults to 'agent' if None.

        """
        with self._lock:
            with self.connection:
                if table is None:
                    table = TableType.store_agent

                insert_string = _insert_meta(table, list(AgentStore.model_fields.keys()))
                self.connection.execute(
                    insert_string,
                    tuple([json.dumps(meta) if not isinstance(meta, str) else meta \
                           for meta in metadata.model_dump().values()])
                )
                self.connection.commit()

    @check_db_format
    def insert_workflow(self, metadata: WorkflowStore, store_type: Optional[Literal["memory", "agent", "workflow", "history", "indexing"]], 
                        table: Optional[str]=None, *args, **kwargs):
        """
        Insert workflow metadata into the specified table.

        Attributes:
            metadata (WorkflowStore): The workflow metadata to insert.
            store_type (str): The type of store (e.g., 'workflow').
            table (Optional[str]): The table name; defaults to 'workflow' if None.

        """
        with self._lock:
            with self.connection:
                if table is None:
                    table = TableType.store_workflow

                insert_string = _insert_meta(table, list(WorkflowStore.model_fields.keys()))
                self.connection.execute(
                    insert_string,
                    tuple([json.dumps(meta) if not isinstance(meta, str) else meta \
                           for meta in metadata.model_dump().values()])
                )
                self.connection.commit()

    @check_db_format
    def insert_history(self, metadata: HistoryStore, store_type: Optional[Literal["memory", "agent", "workflow", "history", "indexing"]], 
                       table: Optional[str]=None, *args, **kwargs):
        """
        Insert history metadata into the specified table.

        Attributes:
            metadata (HistoryStore): The history metadata to insert.
            store_type (str): The type of store (e.g., 'history').
            table (Optional[str]): The table name; defaults to 'history' if None.

        """
        with self._lock:
            with self.connection:
                if table is None:
                    table = TableType.store_history

                insert_string = _insert_meta(table, list(HistoryStore.model_fields.keys()))
                self.connection.execute(
                    insert_string,
                    tuple([json.dumps(meta) if not isinstance(meta, str) else meta \
                           for meta in metadata.model_dump().values()])
                )
                self.connection.commit()

    @check_db_format
    def insert_index(self, metadata: IndexStore, store_type: Optional[Literal["memory", "agent", "workflow", "history", "indexing"]], 
                     table: Optional[str]=None, *args, **kwargs):
        """
        Insert index metadata into the specified table.

        Attributes:
            metadata (IndexStore): The index metadata to insert.
            store_type (str): The type of store (e.g., 'index').
            table (Optional[str]): The table name; defaults to 'index' if None.
        """
        with self._lock:
            with self.connection:
                if table is None:
                    table = TableType.store_indexing

                insert_string = _insert_meta(table, list(IndexStore.model_fields.keys()))
                self.connection.execute(
                    insert_string,
                    tuple([json.dumps(meta) if not isinstance(meta, str) else meta 
                           for meta in metadata.model_dump().values()])
                )
                self.connection.commit()

    def insert(self, metadata: Dict, store_type: Optional[Literal["memory", "agent", "workflow", "history", "indexing"]], 
               table: Optional[str]=None, *args, **kwargs):
        """
        Generic insert method that delegates to specific insert methods based on store_type.

        Attributes:
            metadata (Dict): The metadata to insert.
            store_type (str): The type of store (e.g., 'memory', 'agent').
            table (Optional[str]): The table name; defaults to store_type's default if None.

        """
        if store_type == TableType.store_memory:
            self.insert_memory(metadata, store_type=store_type, table=table, *args, **kwargs)
        elif store_type == TableType.store_agent:
            self.insert_agent(metadata, store_type=store_type, table=table, *args, **kwargs)
        elif store_type == TableType.store_workflow:
            self.insert_workflow(metadata, store_type=store_type, table=table, *args, **kwargs)
        elif store_type == TableType.store_history:
            self.insert_history(metadata, store_type=store_type, table=table, *args, **kwargs)
        elif store_type == TableType.store_indexing:
            self.insert_index(metadata, store_type=store_type, table=table, *args, **kwargs)
        else:
            raise ValueError("Invalid store_type provided.")

    def delete(self, metadata_id: str, store_type: Optional[Literal["memory", "agent", "workflow", "history", "indexing"]], table: Optional[str]=None, *args, **kwargs):
        """
        Delete metadata by its ID from the specified table.

        Attributes:
            metadata_id (str): The ID of the metadata to delete.
            store_type (str): The type of store (e.g., 'memory').
            table (Optional[str]): The table name; defaults to store_type's default if None.


        Returns:
            bool: True if deletion was successful, False if no record was found.
        """
        with self._lock:
            with self.connection:
                if table is None:
                    table = getattr(TableType, store_type)
                try:
                    cursor = self.connection.cursor()
                    delete_query = f"DELETE FROM {table} WHERE {self._get_id_column(store_type)} = ?"
                    cursor.execute(delete_query, (metadata_id,))
                    self.connection.commit()
                    return cursor.rowcount > 0
                except sqlite3.OperationalError:
                    # Logger
                    return False

    def update(self, metadata_id: str, new_metadata: Dict=None, store_type: Optional[Literal["memory", "agent", "workflow", "history", "indexing"]]=None, 
               table: Optional[str]=None, *args, **kwargs):
        """
        Update metadata by its ID in the specified table.

        Attributes:
            metadata_id (str): The ID of the metadata to update.
            new_metadata (Dict): The new metadata to apply.
            store_type (str): The type of store (e.g., 'memory').
            table (Optional[str]): The table name; defaults to store_type's default if None.


        Returns:
            bool: True if update was successful, False if no record was found.
        """
        with self._lock:
            with self.connection:
                if table is None:
                    table = store_type
                
                # Validate new_metadata with the appropriate Pydantic model
                if store_type == TableType.store_memory:
                    columns = list(MemoryStore.model_fields.keys())
                    new_metadata = MemoryStore.model_validate(new_metadata)
                elif store_type == TableType.store_agent:
                    columns = list(AgentStore.model_fields.keys())
                    new_metadata = AgentStore.model_validate(new_metadata)
                elif store_type == TableType.store_workflow:
                    columns = list(WorkflowStore.model_fields.keys())
                    new_metadata = WorkflowStore.model_validate(new_metadata)
                elif store_type == TableType.store_history:
                    columns = list(HistoryStore.model_fields.keys())
                    new_metadata = HistoryStore.model_validate(new_metadata)
                elif store_type == TableType.store_indexing:
                    columns = list(IndexStore.model_fields.keys())
                    new_metadata = IndexStore.model_validate(new_metadata)
                else:
                    raise ValueError("Invalid store_type provided.")
                
                # Generate SET clause for SQL update
                set_clause = ", ".join([f'"{col}" = ?' for col in columns[1:]])  # Exclude primary key
                update_query = f'UPDATE {table} SET {set_clause} WHERE "{columns[0]}" = ?'
                
                values = list([json.dumps(v) if not isinstance(v, str) else v \
                               for v in new_metadata.model_dump().values()])[1:] + [metadata_id]
                
                cursor = self.connection.cursor()
                cursor.execute(update_query, values)
                self.connection.commit()
                return cursor.rowcount > 0
    
    def get_by_id(self, metadata_id: str, store_type: Optional[Literal["memory", "agent", "workflow", "history", "indexing"]], 
                  table: Optional[str]=None, *args, **kwargs):
        """
        Retrieve metadata by its ID from the specified table.

        Attributes:
            metadata_id (str): The ID of the metadata to retrieve.
            store_type (str): The type of store (e.g., 'store_memory').
            table (Optional[str]): The table name; defaults to store_type's default if None.


        Returns:
            Dict: The retrieved metadata as a dictionary, or None if not found.
        """
        with self._lock:
            with self.connection:
                if table is None:
                    table = store_type
                
                # Determine columns based on store_type
                if store_type == TableType.store_memory:
                    columns = list(MemoryStore.model_fields.keys())
                elif store_type == TableType.store_agent:
                    columns = list(AgentStore.model_fields.keys())
                elif store_type == TableType.store_workflow:
                    columns = list(WorkflowStore.model_fields.keys())
                elif store_type == TableType.store_history:
                    columns = list(HistoryStore.model_fields.keys())
                elif store_type == TableType.store_indexing:
                    columns = list(IndexStore.model_fields.keys())
                else:
                    raise ValueError("Invalid store_type provided.")
                try:
                    cursor = self.connection.cursor()
                    select_query = f"SELECT * FROM {table} WHERE {columns[0]} = ?"
                    cursor.execute(select_query, (metadata_id,))
                    result = cursor.fetchone()
                
                    if result:
                        return dict(zip(columns, result))
                    return None
                except sqlite3.OperationalError:
                    return None

    def col_info(self):
        """
        Retrieve information about all tables in the database.

        Returns:
            List[Dict]: A list of dictionaries containing table names and their column information,
                        where columns is a dictionary mapping column names to their data types.
        """
        with self._lock:
            with self.connection:
                cursor = self.connection.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                tables = cursor.fetchall()
                
                table_info = []
                for table in tables:
                    table_name = table[0]
                    cursor.execute(f"PRAGMA table_info({table_name})")
                    columns = cursor.fetchall()
                    table_info.append({
                        "table_name": table_name,
                        "columns": {col[1]: col[2] for col in columns}  # Map column name to data type
                    })
                
                return table_info

    def _get_id_column(self, store_type: Optional[Literal["memory", "agent", "workflow", "history", "indexing"]]) -> str:
        """
        Helper method to get the primary key column name for a store type.

        Attributes:
            store_type (str): The type of store (e.g., 'memory').

        Returns:
            str: The name of the primary key column.

        Raises:
            ValueError: If store_type is invalid.
        """
        if store_type == TableType.store_memory:
            return list(MemoryStore.model_fields.keys())[0]
        elif store_type == TableType.store_agent:
            return list(AgentStore.model_fields.keys())[0]
        elif store_type == TableType.store_workflow:
            return list(WorkflowStore.model_fields.keys())[0]
        elif store_type == TableType.store_history:
            return list(HistoryStore.model_fields.keys())[0]
        elif store_type == TableType.store_indexing:
            return list(IndexStore.model_fields.keys())[0]      
        else:
            raise ValueError("Invalid store_type provided.")