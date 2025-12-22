from abc import ABC, abstractmethod
from typing import Dict, Any, List, Union
from enum import Enum



class DatabaseType(Enum):
    """Enumeration of supported database types"""
    MONGODB = "mongodb"
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"
    SQLITE = "sqlite"
    REDIS = "redis"
    ELASTICSEARCH = "elasticsearch"
    NEO4J = "neo4j"
    VECTOR = "vector"  # For vector databases like Pinecone, Weaviate, etc.


class QueryType(Enum):
    """Enumeration of query types"""
    SELECT = "select"
    INSERT = "insert"
    UPDATE = "update"
    DELETE = "delete"
    CREATE = "create"
    DROP = "drop"
    ALTER = "alter"
    INDEX = "index"
    AGGREGATE = "aggregate"  # For MongoDB aggregation pipelines
    SEARCH = "search"  # For full-text search
    VECTOR_SEARCH = "vector_search"  # For vector similarity search


class DatabaseConnection:
    """Base class for database connection management"""
    
    def __init__(self, connection_string: str, **kwargs):
        self.connection_string = connection_string
        self.connection_params = kwargs
        self._connection = None
        self._is_connected = False
    
    @property
    def is_connected(self) -> bool:
        return self._is_connected
    
    @abstractmethod
    def connect(self) -> bool:
        """Establish connection to the database"""
        pass
    
    @abstractmethod
    def disconnect(self) -> bool:
        """Close connection to the database"""
        pass
    
    @abstractmethod
    def test_connection(self) -> bool:
        """Test if the connection is working"""
        pass


class DatabaseBase(ABC):
    """
    Abstract base class for database operations.
    Provides a common interface for different database types.
    """
    
    def __init__(self, 
                 connection_string: str = None,
                 database_name: str = None,
                 **kwargs):
        """
        Initialize the database base.
        
        Args:
            connection_string: Database connection string
            database_name: Name of the database to use
            **kwargs: Additional connection parameters
        """
        self.connection_string = connection_string
        self.database_name = database_name
        self.connection_params = kwargs
        self.db_type = self._get_database_type()
        self.connection = None
        self._is_initialized = False
        
        # Initialize connection if connection string is provided
        if connection_string:
            self.connect()
    
    @abstractmethod
    def _get_database_type(self) -> DatabaseType:
        """Return the database type"""
        pass
    
    @abstractmethod
    def connect(self) -> bool:
        """
        Establish connection to the database.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        pass
    
    @abstractmethod
    def disconnect(self) -> bool:
        """
        Close connection to the database.
        
        Returns:
            bool: True if disconnection successful, False otherwise
        """
        pass
    
    @abstractmethod
    def test_connection(self) -> bool:
        """
        Test if the database connection is working.
        
        Returns:
            bool: True if connection is working, False otherwise
        """
        pass
    
    @abstractmethod
    def execute_query(self, 
                     query: Union[str, Dict, List], 
                     query_type: QueryType = None,
                     **kwargs) -> Dict[str, Any]:
        """
        Execute a query on the database.
        
        Args:
            query: The query to execute (string for SQL, dict/list for NoSQL)
            query_type: Type of query being executed
            **kwargs: Additional query parameters
            
        Returns:
            Dict containing query results and metadata
        """
        pass
    
    @abstractmethod
    def get_database_info(self) -> Dict[str, Any]:
        """
        Get information about the database.
        
        Returns:
            Dict containing database information
        """
        pass
    
    @abstractmethod
    def list_collections(self) -> List[str]:
        """
        List all collections/tables in the database.
        
        Returns:
            List of collection/table names
        """
        pass
    
    @abstractmethod
    def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """
        Get information about a specific collection/table.
        
        Args:
            collection_name: Name of the collection/table
            
        Returns:
            Dict containing collection/table information
        """
        pass
    
    @abstractmethod
    def get_schema(self, collection_name: str = None) -> Dict[str, Any]:
        """
        Get the schema/structure of the database or a specific collection.
        
        Args:
            collection_name: Name of the collection/table (optional)
            
        Returns:
            Dict containing schema information
        """
        pass
    
    def validate_query(self, query: Union[str, Dict, List]) -> Dict[str, Any]:
        """
        Validate a query before execution.
        
        Args:
            query: The query to validate
            
        Returns:
            Dict containing validation results
        """
        try:
            # Basic validation - child classes can override for specific validation
            if isinstance(query, str):
                if not query.strip():
                    return {"valid": False, "error": "Query cannot be empty"}
            elif isinstance(query, (dict, list)):
                if not query:
                    return {"valid": False, "error": "Query cannot be empty"}
            else:
                return {"valid": False, "error": f"Unsupported query type: {type(query)}"}
            
            return {"valid": True, "error": None}
            
        except Exception as e:
            return {"valid": False, "error": str(e)}
    
    def format_query_result(self, 
                           data: Any, 
                           query_type: QueryType,
                           execution_time: float = None,
                           **kwargs) -> Dict[str, Any]:
        """
        Format query results into a standard structure.
        
        Args:
            data: Raw query results
            query_type: Type of query that was executed
            execution_time: Time taken to execute the query
            **kwargs: Additional metadata
            
        Returns:
            Dict containing formatted results
        """
        return {
            "success": True,
            "data": data,
            "query_type": query_type.value if query_type else None,
            "execution_time": execution_time,
            "row_count": len(data) if isinstance(data, (list, tuple)) else 1,
            "metadata": kwargs
        }
    
    def format_error_result(self, 
                           error: str, 
                           query_type: QueryType = None,
                           **kwargs) -> Dict[str, Any]:
        """
        Format error results into a standard structure.
        
        Args:
            error: Error message
            query_type: Type of query that failed
            **kwargs: Additional error metadata
            
        Returns:
            Dict containing formatted error results
        """
        return {
            "success": False,
            "error": error,
            "query_type": query_type.value if query_type else None,
            "data": None,
            "execution_time": None,
            "row_count": 0,
            "metadata": kwargs
        }
    
    def get_supported_query_types(self) -> List[QueryType]:
        """
        Get list of supported query types for this database.
        
        Returns:
            List of supported QueryType enums
        """
        return [
            QueryType.SELECT,
            QueryType.INSERT,
            QueryType.UPDATE,
            QueryType.DELETE,
            QueryType.CREATE,
            QueryType.DROP
        ]
    
    def get_capabilities(self) -> Dict[str, Any]:
        """
        Get database capabilities and features.
        
        Returns:
            Dict containing database capabilities
        """
        return {
            "database_type": self.db_type.value,
            "supports_sql": False,  # Override in SQL databases
            "supports_aggregation": False,  # Override in document databases
            "supports_full_text_search": False,  # Override in search databases
            "supports_vector_search": False,  # Override in vector databases
            "supports_transactions": False,  # Override if supported
            "supports_indexing": True,
            "supported_query_types": [qt.value for qt in self.get_supported_query_types()],
            "connection_info": {
                "is_connected": self.connection is not None,
                "database_name": self.database_name
            }
        }
    
    def __enter__(self):
        """Context manager entry"""
        if not self.connection:
            self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.disconnect()
    
    def __del__(self):
        """Cleanup on deletion"""
        try:
            self.disconnect()
        except Exception:
            pass 