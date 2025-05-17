from typing import Optional

from pydantic import Field

from ..core.base_config import BaseConfig


class DBConfig(BaseConfig):
    """
    Defines settings for connecting to a database, such as SQLite or PostgreSQL.
    """
    db_name: str = Field(default="sqlite", description="Name of the database provider (e.g., 'sqlite', 'posgre_sql')")
    path: Optional[str] = Field(default="", description="File path for file-based databases (e.g., SQLite)")
    ip: Optional[str] = Field(default="", description="IP address for network-based databases")
    port: Optional[str] = Field(default="", description="Port for network-based databases")


class VectorStoreConfig(BaseConfig):
    """
    Placeholder for settings related to vector databases (e.g., Qdrant, Chroma).
    """
    pass


class GraphStoreConfig(BaseConfig):
    """
    Placeholder for settings related to graph databases.
    """
    pass


class FileStoreConfig(BaseConfig):
    """
    Placeholder for settings related to file-based storage.
    """
    pass


class StoreConfig(BaseConfig):
    """
    Aggregates database, vector, file, and graph store configurations.
    """
    dbConfig: DBConfig = Field(..., description="Configuration for the database store")
    vectorConfig: Optional[VectorStoreConfig] = Field(None, description="Configuration for the vector store")
    fileConfig: Optional[FileStoreConfig] = Field(None, description="Optional configuration for the file store")
    graphConfig: Optional[GraphStoreConfig] = Field(None, description="Optional configuration for the graph store")