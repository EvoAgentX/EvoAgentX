import time
import json
from typing import Dict, Any, List, Union, Optional
from pathlib import Path
from bson import json_util
from pymongo import MongoClient

from .database_base import DatabaseBase, DatabaseType, QueryType, DatabaseConnection
from .tool import Tool, Toolkit
from ..core.logging import logger


class MongoDBConnection(DatabaseConnection):
    """MongoDB-specific connection management"""
    
    def __init__(self, connection_string: str, **kwargs):
        super().__init__(connection_string, **kwargs)
        self.client = None
        self.database = None
    
    def connect(self) -> bool:
        """Establish connection to MongoDB"""
        try:
            # Parse connection string and extract database name
            if "mongodb://" in self.connection_string or "mongodb+srv://" in self.connection_string:
                # Full connection string with database
                self.client = MongoClient(self.connection_string, **self.connection_params)
            else:
                # Just host/port, need database name
                self.client = MongoClient(self.connection_string, **self.connection_params)
            
            # Test the connection
            self.client.admin.command('ping')
            self._is_connected = True
            logger.info("Successfully connected to MongoDB")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {str(e)}")
            self._is_connected = False
            return False
    
    def disconnect(self) -> bool:
        """Close MongoDB connection"""
        try:
            if self.client:
                self.client.close()
                self.client = None
                self.database = None
                self._is_connected = False
                logger.info("Disconnected from MongoDB")
            return True
        except Exception as e:
            logger.error(f"Error disconnecting from MongoDB: {str(e)}")
            return False
    
    def test_connection(self) -> bool:
        """Test MongoDB connection"""
        try:
            if self.client:
                self.client.admin.command('ping')
                return True
            return False
        except Exception:
            return False
    
    def get_database(self, database_name: str):
        """Get database instance"""
        if self.client and database_name:
            return self.client[database_name]
        return None


class MongoDBDatabase(DatabaseBase):
    """
    MongoDB database implementation with automatic initialization.
    Handles remote connections, existing local databases, and new local database creation.
    """
    
    def __init__(self, 
                 connection_string: str = None,
                 database_name: str = None,
                 local_path: str = None,
                 auto_save: bool = True,
                 read_only: bool = False,
                 **kwargs):
        """
        Initialize MongoDB database with automatic detection and setup.
        
        Args:
            connection_string: MongoDB connection string (for remote)
            database_name: Name of the database
            local_path: Path for local file-based database
            auto_save: Automatically save changes to local files
            read_only: If True, only read operations are allowed (no insert, update, delete)
            **kwargs: Additional connection parameters
        """
        # Prepare initialization parameters
        init_params = {
            'connection_string': connection_string,
            'database_name': database_name
        }
        
        # Initialize parent class first
        super().__init__(**init_params, **kwargs)
        
        # Now set instance variables after parent initialization
        self.local_path = Path(local_path) if local_path else None
        self.auto_save = auto_save
        self.read_only = read_only
        self.connection_params = kwargs
        
        # Initialize database-specific attributes
        self.is_local_database = False
        self.client = None
        self.database = None
        
        # Determine initialization mode and set up database
        if self._is_remote_connection():
            self._init_remote_database()
        elif self._is_existing_local_database():
            self._init_existing_local_database()
        else:
            self._init_new_local_database()
    
    def _is_remote_connection(self) -> bool:
        """Check if this is a remote MongoDB connection"""
        return (self.connection_string and 
                (self.connection_string.startswith(('mongodb://', 'mongodb+srv://')) or
                 'localhost' in self.connection_string or 
                 '127.0.0.1' in self.connection_string))
    
    def _is_existing_local_database(self) -> bool:
        """Check if there's an existing local database"""
        if not self.local_path:
            return False
        
        if not self.local_path.exists():
            return False
        
        # Check for database files
        json_files = list(self.local_path.glob("*.json"))
        db_info_file = self.local_path / "db_info.json"
        
        return len(json_files) > 0 or db_info_file.exists()
    
    def _init_remote_database(self):
        """Initialize remote MongoDB connection"""
        try:
            self.client = MongoClient(self.connection_string, **self.connection_params)
            self.client.admin.command('ping')  # Test connection
            
            if self.database_name:
                self.database = self.client[self.database_name]
            
            self._is_initialized = True
            self.is_local_database = False
            logger.info(f"Connected to remote MongoDB: {self.database_name}")
            
        except Exception as e:
            logger.error(f"Failed to connect to remote MongoDB: {str(e)}")
            self._is_initialized = False
            raise
    
    def _init_existing_local_database(self):
        """Initialize existing local database"""
        try:
            # Set up local MongoDB connection
            self.connection_string = "mongodb://localhost:27017"
            self.client = MongoClient(self.connection_string, **self.connection_params)
            
            # Use path name as database name if not specified
            if not self.database_name:
                self.database_name = self.local_path.name
            
            self.database = self.client[self.database_name]
            
            # Load existing collections from JSON files
            self._load_local_collections()
            
            self._is_initialized = True
            self.is_local_database = True
            logger.info(f"Loaded existing local database from: {self.local_path}")
            
        except Exception as e:
            logger.error(f"Failed to load existing local database: {str(e)}")
            self._is_initialized = False
            raise
    
    def _init_new_local_database(self):
        """Initialize new local database"""
        try:
            # Set default path if not specified
            if not self.local_path:
                self.local_path = Path("./mongodb_local")
            
            # Create directory
            self.local_path.mkdir(parents=True, exist_ok=True)
            
            # Set up local MongoDB connection
            self.connection_string = "mongodb://localhost:27017"
            self.client = MongoClient(self.connection_string, **self.connection_params)
            
            # Use path name as database name if not specified
            if not self.database_name:
                self.database_name = self.local_path.name
            
            self.database = self.client[self.database_name]
            
            # Create database info file
            self._create_db_info_file()
            
            self._is_initialized = True
            self.is_local_database = True
            logger.info(f"Created new local database at: {self.local_path}")
            
        except Exception as e:
            logger.error(f"Failed to create new local database: {str(e)}")
            self._is_initialized = False
            raise
    
    def _load_local_collections(self):
        """Load collections from local JSON files"""
        if not self.local_path or not self.local_path.exists():
            return
        
        json_files = [f for f in self.local_path.glob("*.json") if f.name != "db_info.json"]
        
        for json_file in json_files:
            collection_name = json_file.stem
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if isinstance(data, dict):
                    documents = [data]
                elif isinstance(data, list):
                    documents = data
                else:
                    continue
                
                if documents:
                    # Clean MongoDB-specific fields that can't be inserted directly
                    cleaned_documents = []
                    for doc in documents:
                        cleaned_doc = self._clean_document_for_insert(doc)
                        cleaned_documents.append(cleaned_doc)
                    
                    collection = self.database[collection_name]
                    collection.drop()  # Clear existing data
                    if cleaned_documents:
                        collection.insert_many(cleaned_documents)
                        logger.info(f"Loaded {len(cleaned_documents)} documents into '{collection_name}'")
                    
            except Exception as e:
                logger.warning(f"Failed to load collection from {json_file}: {str(e)}")
    
    def _clean_document_for_insert(self, doc: Dict) -> Dict:
        """Clean document by removing problematic MongoDB-specific fields"""
        if isinstance(doc, dict):
            cleaned = {}
            for key, value in doc.items():
                if key == '_id' and isinstance(value, dict) and '$oid' in value:
                    # Skip MongoDB ObjectId fields when loading from JSON
                    continue
                elif isinstance(value, dict):
                    cleaned[key] = self._clean_document_for_insert(value)
                elif isinstance(value, list):
                    cleaned[key] = [self._clean_document_for_insert(item) if isinstance(item, dict) else item for item in value]
                else:
                    cleaned[key] = value
            return cleaned
        return doc
    
    def _create_db_info_file(self):
        """Create database info file for new local database"""
        try:
            db_info = {
                "database_name": self.database_name,
                "created_at": time.time(),
                "local_path": str(self.local_path.absolute()),
                "auto_save": self.auto_save,
                "version": "1.0"
            }
            
            info_file = self.local_path / "db_info.json"
            with open(info_file, 'w', encoding='utf-8') as f:
                json.dump(db_info, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.warning(f"Failed to create db info file: {str(e)}")
    
    def _save_collection_to_file(self, collection_name: str):
        """Save collection to local JSON file"""
        if not self.is_local_database or not self.local_path:
            return
        
        try:
            collection = self.database[collection_name]
            documents = list(collection.find())
            
            # Convert ObjectId to string for JSON serialization
            for doc in documents:
                if '_id' in doc:
                    doc['_id'] = str(doc['_id'])
            
            file_path = self.local_path / f"{collection_name}.json"
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(documents, f, indent=2, ensure_ascii=False, default=str)
                
            logger.debug(f"Saved collection '{collection_name}' to {file_path}")
            
        except Exception as e:
            logger.warning(f"Failed to save collection '{collection_name}': {str(e)}")
    
    def _auto_save_if_needed(self, collection_name: str):
        """Auto-save collection if local database and auto_save is enabled"""
        if self.is_local_database and self.auto_save:
            self._save_collection_to_file(collection_name)

    def _get_database_type(self) -> DatabaseType:
        return DatabaseType.MONGODB
    
    def connect(self) -> bool:
        """Connection is already established in __init__"""
        return self._is_initialized
    
    def disconnect(self) -> bool:
        """Close MongoDB connection"""
        try:
            if self.client:
                self.client.close()
                self.client = None
                self.database = None
                self._is_initialized = False
                logger.info("Disconnected from MongoDB")
            return True
        except Exception as e:
            logger.error(f"Error disconnecting: {str(e)}")
            return False
    
    def test_connection(self) -> bool:
        """Test MongoDB connection"""
        try:
            if self.client:
                self.client.admin.command('ping')
                return True
            return False
        except Exception:
            return False
    
    def execute_query(self, 
                     query: Union[str, Dict, List], 
                     query_type: QueryType = None,
                     collection_name: str = None,
                     **kwargs) -> Dict[str, Any]:
        """Execute a query on MongoDB with automatic result handling"""
        if not self._is_initialized or self.database is None:
            return self.format_error_result("Database not connected")
        
        if not collection_name:
            return self.format_error_result("Collection name is required")
        
        start_time = time.time()
        
        try:
            collection = self.database[collection_name]
            
            # Determine query type if not provided
            if not query_type:
                query_type = self._infer_query_type(query)
            
            # Check read-only mode for write operations
            if self.read_only and query_type in [QueryType.INSERT, QueryType.UPDATE, QueryType.DELETE, QueryType.CREATE, QueryType.DROP]:
                return self.format_error_result(
                    f"Write operation '{query_type.value}' is not allowed in read-only mode. "
                    "Only SELECT and AGGREGATE operations are permitted.",
                    query_type,
                    execution_time=time.time() - start_time
                )
            
            # Execute based on query type
            if query_type == QueryType.SELECT:
                result = self._execute_find(collection, query, **kwargs)
            elif query_type == QueryType.INSERT:
                result = self._execute_insert(collection, query, **kwargs)
                self._auto_save_if_needed(collection_name)
            elif query_type == QueryType.UPDATE:
                result = self._execute_update(collection, query, **kwargs)
                self._auto_save_if_needed(collection_name)
            elif query_type == QueryType.DELETE:
                result = self._execute_delete(collection, query, **kwargs)
                self._auto_save_if_needed(collection_name)
            elif query_type == QueryType.AGGREGATE:
                result = self._execute_aggregate(collection, query, **kwargs)
            else:
                return self.format_error_result(f"Unsupported query type: {query_type}")
            
            execution_time = time.time() - start_time
            if isinstance(result, dict):
                result["execution_time"] = execution_time
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Error executing MongoDB query: {str(e)}")
            return self.format_error_result(str(e), query_type, execution_time=execution_time)
    
    def _infer_query_type(self, query: Union[str, Dict, List]) -> QueryType:
        """Infer query type from the query structure"""
        if isinstance(query, list):
            return QueryType.AGGREGATE
        elif isinstance(query, dict):
            # In read-only mode, only allow SELECT and AGGREGATE
            if self.read_only:
                if "insert" in query or "insertOne" in query or "insertMany" in query:
                    return QueryType.SELECT  # Treat as find query
                elif "update" in query or "updateOne" in query or "updateMany" in query:
                    return QueryType.SELECT  # Treat as find query
                elif "delete" in query or "deleteOne" in query or "deleteMany" in query:
                    return QueryType.SELECT  # Treat as find query
                elif "create" in query or "createCollection" in query:
                    return QueryType.SELECT  # Treat as find query
                elif "drop" in query or "dropCollection" in query:
                    return QueryType.SELECT  # Treat as find query
                else:
                    return QueryType.SELECT
            else:
                # Full functionality for write mode
                if "insert" in query or "insertOne" in query or "insertMany" in query:
                    return QueryType.INSERT
                elif "update" in query or "updateOne" in query or "updateMany" in query:
                    return QueryType.UPDATE
                elif "delete" in query or "deleteOne" in query or "deleteMany" in query:
                    return QueryType.DELETE
                elif "create" in query or "createCollection" in query:
                    return QueryType.CREATE
                elif "drop" in query or "dropCollection" in query:
                    return QueryType.DROP
                else:
                    return QueryType.SELECT
        elif isinstance(query, str):
            query_lower = query.lower().strip()
            if self.read_only:
                # In read-only mode, treat all string queries as SELECT
                return QueryType.SELECT
            else:
                # Full functionality for write mode
                if query_lower.startswith(("insert", "create")):
                    return QueryType.INSERT
                elif query_lower.startswith("update"):
                    return QueryType.UPDATE
                elif query_lower.startswith("delete"):
                    return QueryType.DELETE
                elif query_lower.startswith("drop"):
                    return QueryType.DROP
                else:
                    return QueryType.SELECT
        
        return QueryType.SELECT
    
    def _execute_find(self, collection, query: Dict, **kwargs) -> Dict[str, Any]:
        """Execute find query"""
        try:
            # Handle string queries (simple field queries)
            if isinstance(query, str):
                # Convert simple string queries to MongoDB format
                if "=" in query:
                    field, value = query.split("=", 1)
                    query = {field.strip(): value.strip()}
                else:
                    query = {}
            
            # Get query parameters
            filter_query = query.get("filter", query)
            projection = query.get("projection", {})
            sort = query.get("sort", None)
            limit = query.get("limit", kwargs.get("limit", 0))
            skip = query.get("skip", kwargs.get("skip", 0))
            
            # Execute find
            cursor = collection.find(filter_query, projection)
            
            if sort:
                cursor = cursor.sort(sort)
            if skip:
                cursor = cursor.skip(skip)
            if limit:
                cursor = cursor.limit(limit)
            
            # Convert cursor to list and handle ObjectId serialization
            results = []
            for doc in cursor:
                # Convert ObjectId to string for JSON serialization
                doc = json.loads(json_util.dumps(doc))
                results.append(doc)
            
            return self.format_query_result(
                results, 
                QueryType.SELECT,
                collection_name=collection.name,
                filter_applied=filter_query
            )
            
        except Exception as e:
            return self.format_error_result(str(e), QueryType.SELECT)
    
    def _execute_insert(self, collection, query: Union[Dict, List], **kwargs) -> Dict[str, Any]:
        """Execute insert operation"""
        try:
            if isinstance(query, dict):
                # Single document insert
                if "document" in query:
                    document = query["document"]
                else:
                    document = query
                
                result = collection.insert_one(document)
                return self.format_query_result(
                    {"inserted_id": str(result.inserted_id)},
                    QueryType.INSERT,
                    collection_name=collection.name
                )
            elif isinstance(query, list):
                # Multiple documents insert
                if all(isinstance(item, dict) for item in query):
                    documents = query
                else:
                    documents = [{"documents": query}]
                
                result = collection.insert_many(documents)
                return self.format_query_result(
                    {"inserted_ids": [str(id) for id in result.inserted_ids]},
                    QueryType.INSERT,
                    collection_name=collection.name
                )
            else:
                return self.format_error_result("Invalid insert query format", QueryType.INSERT)
                
        except Exception as e:
            return self.format_error_result(str(e), QueryType.INSERT)
    
    def _execute_update(self, collection, query: Dict, **kwargs) -> Dict[str, Any]:
        """Execute update operation"""
        try:
            filter_query = query.get("filter", {})
            update_query = query.get("update", {})
            upsert = query.get("upsert", False)
            multi = query.get("multi", False)
            
            if multi:
                result = collection.update_many(filter_query, update_query, upsert=upsert)
            else:
                result = collection.update_one(filter_query, update_query, upsert=upsert)
            
            return self.format_query_result(
                {
                    "matched_count": result.matched_count,
                    "modified_count": result.modified_count,
                    "upserted_id": str(result.upserted_id) if result.upserted_id else None
                },
                QueryType.UPDATE,
                collection_name=collection.name
            )
            
        except Exception as e:
            return self.format_error_result(str(e), QueryType.UPDATE)
    
    def _execute_delete(self, collection, query: Dict, **kwargs) -> Dict[str, Any]:
        """Execute delete operation"""
        try:
            filter_query = query.get("filter", query)
            multi = query.get("multi", False)
            
            if multi:
                result = collection.delete_many(filter_query)
            else:
                result = collection.delete_one(filter_query)
            
            return self.format_query_result(
                {"deleted_count": result.deleted_count},
                QueryType.DELETE,
                collection_name=collection.name
            )
            
        except Exception as e:
            return self.format_error_result(str(e), QueryType.DELETE)
    
    def _execute_aggregate(self, collection, pipeline: List, **kwargs) -> Dict[str, Any]:
        """Execute aggregation pipeline"""
        try:
            cursor = collection.aggregate(pipeline)
            results = []
            
            for doc in cursor:
                # Convert ObjectId to string for JSON serialization
                doc = json.loads(json_util.dumps(doc))
                results.append(doc)
            
            return self.format_query_result(
                results,
                QueryType.AGGREGATE,
                collection_name=collection.name,
                pipeline_stages=len(pipeline)
            )
            
        except Exception as e:
            return self.format_error_result(str(e), QueryType.AGGREGATE)
    

    
    def get_database_info(self) -> Dict[str, Any]:
        """Get MongoDB database information"""
        try:
            if not self._is_initialized or self.database is None:
                return self.format_error_result("Database not connected")
            
            # Get database stats
            stats = self.database.command("dbStats")
            
            # Get server info
            server_info = self.client.server_info()
            
            info = {
                "database_name": self.database_name,
                "collections": stats.get("collections", 0),
                "data_size": stats.get("dataSize", 0),
                "storage_size": stats.get("storageSize", 0),
                "indexes": stats.get("indexes", 0),
                "index_size": stats.get("indexSize", 0),
                "server_version": server_info.get("version", "Unknown"),
                "server_type": server_info.get("type", "Unknown"),
                "connection_string": self.connection_string,
                "is_connected": self._is_initialized
            }
            
            return self.format_query_result(info, QueryType.SELECT)
            
        except Exception as e:
            return self.format_error_result(str(e))
    
    def list_collections(self) -> List[str]:
        """List all collections in the database"""
        try:
            if not self._is_initialized or self.database is None:
                return []
            
            return self.database.list_collection_names()
            
        except Exception as e:
            logger.error(f"Error listing collections: {str(e)}")
            return []
    
    def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """Get information about a specific collection"""
        try:
            if not self._is_initialized or not self.database:
                return self.format_error_result("Database not connected")
            
            collection = self.database[collection_name]
            
            # Get collection stats
            stats = self.database.command("collStats", collection_name)
            
            # Get collection indexes
            indexes = list(collection.list_indexes())
            
            # Get sample documents for schema inference
            sample_docs = list(collection.find().limit(5))
            
            info = {
                "collection_name": collection_name,
                "document_count": stats.get("count", 0),
                "data_size": stats.get("size", 0),
                "storage_size": stats.get("storageSize", 0),
                "index_count": stats.get("nindexes", 0),
                "indexes": [{"name": idx["name"], "keys": idx["key"]} for idx in indexes],
                "sample_documents": sample_docs[:2]  # Limit sample size
            }
            
            return self.format_query_result(info, QueryType.SELECT)
            
        except Exception as e:
            return self.format_error_result(str(e))
    
    def get_schema(self, collection_name: str = None) -> Dict[str, Any]:
        """Get schema information for database or specific collection"""
        try:
            if not self._is_initialized or not self.database:
                return self.format_error_result("Database not connected")
            
            if collection_name:
                # Get schema for specific collection
                collection = self.database[collection_name]
                sample_docs = list(collection.find().limit(100))
                
                if not sample_docs:
                    return self.format_query_result(
                        {"collection_name": collection_name, "schema": {}, "message": "No documents found"},
                        QueryType.SELECT
                    )
                
                # Infer schema from sample documents
                schema = self._infer_schema_from_documents(sample_docs)
                
                return self.format_query_result(
                    {
                        "collection_name": collection_name,
                        "schema": schema,
                        "sample_count": len(sample_docs)
                    },
                    QueryType.SELECT
                )
            else:
                # Get schema for all collections
                collections = self.list_collections()
                schemas = {}
                
                for coll_name in collections[:10]:  # Limit to first 10 collections
                    coll_schema = self.get_schema(coll_name)
                    if coll_schema.get("success"):
                        schemas[coll_name] = coll_schema.get("data", {}).get("schema", {})
                
                return self.format_query_result(
                    {"database_name": self.database_name, "schemas": schemas},
                    QueryType.SELECT
                )
                
        except Exception as e:
            return self.format_error_result(str(e))
    
    def _infer_schema_from_documents(self, documents: List[Dict]) -> Dict[str, Any]:
        """Infer schema from a list of documents"""
        if not documents:
            return {}
        
        schema = {}
        
        for doc in documents:
            self._update_schema_from_document(schema, doc)
        
        return schema
    
    def _update_schema_from_document(self, schema: Dict, document: Dict, path: str = ""):
        """Recursively update schema from a document"""
        for key, value in document.items():
            current_path = f"{path}.{key}" if path else key
            
            if isinstance(value, dict):
                if current_path not in schema:
                    schema[current_path] = {"type": "object", "fields": {}}
                self._update_schema_from_document(schema[current_path]["fields"], value, current_path)
            elif isinstance(value, list):
                if current_path not in schema:
                    schema[current_path] = {"type": "array", "element_types": set()}
                # Analyze first few elements to determine array element types
                for item in value[:3]:
                    if isinstance(item, dict):
                        schema[current_path]["element_types"].add("object")
                    else:
                        schema[current_path]["element_types"].add(type(item).__name__)
                schema[current_path]["element_types"] = list(schema[current_path]["element_types"])
            else:
                if current_path not in schema:
                    schema[current_path] = {"type": type(value).__name__}
                elif schema[current_path]["type"] != type(value).__name__:
                    # Type conflict - mark as mixed
                    schema[current_path]["type"] = "mixed"
    
    def get_supported_query_types(self) -> List[QueryType]:
        """Get MongoDB-specific supported query types"""
        if self.read_only:
            return [
                QueryType.SELECT,
                QueryType.AGGREGATE
            ]
        else:
            return [
                QueryType.SELECT,
                QueryType.INSERT,
                QueryType.UPDATE,
                QueryType.DELETE,
                QueryType.CREATE,
                QueryType.DROP,
                QueryType.AGGREGATE,
                QueryType.INDEX
            ]
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get MongoDB-specific capabilities"""
        base_capabilities = super().get_capabilities()
        base_capabilities.update({
            "supports_aggregation": True,
            "supports_full_text_search": True,
            "supports_geospatial_queries": True,
            "supports_change_streams": True,
            "supports_transactions": True,
            "supports_indexing": True,
            "document_oriented": True,
            "schema_flexible": True,
            "read_only": self.read_only,
            "write_operations_allowed": not self.read_only
        })
        return base_capabilities 


class MongoDBExecuteQueryTool(Tool):
    name: str = "mongodb_execute_query"
    description: str = "Execute MongoDB queries including find and aggregation pipelines (read-only operations)"
    inputs: Dict[str, Dict[str, str]] = {
        "query": {
            "type": "string",
            "description": "MongoDB query (JSON string for find, array for aggregation pipeline)"
        },
        "query_type": {
            "type": "string",
            "description": "Type of query (select, aggregate) - auto-detected if not provided"
        },
        "collection_name": {
            "type": "string",
            "description": "Collection name (required for all operations)"
        }
    }
    required: Optional[List[str]] = ["query", "collection_name"]

    def __init__(self, database: MongoDBDatabase = None):
        super().__init__()
        self.database = database

    def __call__(self, query: str, query_type: str = None, collection_name: str = None) -> Dict[str, Any]:
        """Execute a MongoDB query"""
        try:
            if not self.database:
                return {"success": False, "error": "MongoDB database not initialized", "data": None}
            
            # Parse query if it's a JSON string
            parsed_query = self._parse_query(query)
            
            # Convert query_type string to enum if provided
            query_type_enum = None
            if query_type:
                try:
                    query_type_enum = QueryType(query_type.lower())
                except ValueError:
                    return {"success": False, "error": f"Invalid query type: {query_type}", "data": None}
            

            
            # Execute the query
            result = self.database.execute_query(
                query=parsed_query,
                query_type=query_type_enum,
                collection_name=collection_name
            )
            
            if result["success"]:
                logger.info(f"Successfully executed MongoDB query on collection {collection_name}")
            else:
                logger.error(f"Failed to execute MongoDB query: {result.get('error', 'Unknown error')}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in mongodb_execute_query tool: {str(e)}")
            return {"success": False, "error": str(e), "data": None}
    
    def _parse_query(self, query: str) -> Union[str, Dict, List]:
        """Parse query string into appropriate format"""
        try:
            import json
            return json.loads(query)
        except (json.JSONDecodeError, ValueError):
            return query


class MongoDBFindTool(Tool):
    name: str = "mongodb_find"
    description: str = "Find documents in a MongoDB collection with filtering, projection, sorting, and pagination"
    inputs: Dict[str, Dict[str, str]] = {
        "collection_name": {
            "type": "string",
            "description": "Collection name to query"
        },
        "filter": {
            "type": "string",
            "description": "MongoDB filter query (JSON string, e.g., '{\"age\": {\"$gt\": 18}}')"
        },
        "projection": {
            "type": "string",
            "description": "Fields to include/exclude (JSON string, e.g., '{\"name\": 1, \"_id\": 0}')"
        },
        "sort": {
            "type": "string",
            "description": "Sort criteria (JSON string, e.g., '{\"age\": -1}')"
        },
        "limit": {
            "type": "integer",
            "description": "Maximum number of documents to return"
        },
        "skip": {
            "type": "integer",
            "description": "Number of documents to skip"
        }
    }
    required: Optional[List[str]] = ["collection_name"]

    def __init__(self, database: MongoDBDatabase = None):
        super().__init__()
        self.database = database

    def __call__(self, collection_name: str, filter: str = "{}", projection: str = "{}", 
                 sort: str = None, limit: int = 0, skip: int = 0) -> Dict[str, Any]:
        """Find documents in MongoDB collection"""
        try:
            if not self.database:
                return {"success": False, "error": "MongoDB database not initialized", "data": None}
            
            # Parse JSON strings
            import json
            filter_dict = json.loads(filter) if filter else {}
            projection_dict = json.loads(projection) if projection else {}
            sort_dict = json.loads(sort) if sort else None
            
            # Build query
            query = {
                "filter": filter_dict,
                "projection": projection_dict,
                "limit": limit,
                "skip": skip
            }
            
            if sort_dict:
                query["sort"] = sort_dict
            
            # Execute query
            result = self.database.execute_query(
                query=query,
                query_type=QueryType.SELECT,
                collection_name=collection_name
            )
            
            if result["success"]:
                logger.info(f"Successfully found documents in collection {collection_name}")
            else:
                logger.error(f"Failed to find documents: {result.get('error', 'Unknown error')}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in mongodb_find tool: {str(e)}")
            return {"success": False, "error": str(e), "data": None}


class MongoDBUpdateTool(Tool):
    name: str = "mongodb_update"
    description: str = "Update documents in a MongoDB collection"
    inputs: Dict[str, Dict[str, str]] = {
        "collection_name": {
            "type": "string",
            "description": "Collection name to update"
        },
        "filter": {
            "type": "string",
            "description": "Filter to match documents to update (JSON string)"
        },
        "update": {
            "type": "string",
            "description": "Update operations (JSON string, e.g., '{\"$set\": {\"status\": \"active\"}}')"
        },
        "upsert": {
            "type": "boolean",
            "description": "Create document if it doesn't exist"
        },
        "multi": {
            "type": "boolean",
            "description": "Update multiple documents (default: false)"
        }
    }
    required: Optional[List[str]] = ["collection_name", "filter", "update"]

    def __init__(self, database: MongoDBDatabase = None):
        super().__init__()
        self.database = database

    def __call__(self, collection_name: str, filter: str, update: str, 
                 upsert: bool = False, multi: bool = False) -> Dict[str, Any]:
        """Update documents in MongoDB collection"""
        try:
            if not self.database:
                return {"success": False, "error": "MongoDB database not initialized", "data": None}
            
            # Parse JSON strings
            import json
            filter_dict = json.loads(filter)
            update_dict = json.loads(update)
            
            # Build query
            query = {
                "filter": filter_dict,
                "update": update_dict,
                "upsert": upsert,
                "multi": multi
            }
            
            # Execute update
            result = self.database.execute_query(
                query=query,
                query_type=QueryType.UPDATE,
                collection_name=collection_name
            )
            
            if result["success"]:
                logger.info(f"Successfully updated documents in collection {collection_name}")
            else:
                logger.error(f"Failed to update documents: {result.get('error', 'Unknown error')}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in mongodb_update tool: {str(e)}")
            return {"success": False, "error": str(e), "data": None}


class MongoDBDeleteTool(Tool):
    name: str = "mongodb_delete"
    description: str = "Delete documents from a MongoDB collection"
    inputs: Dict[str, Dict[str, str]] = {
        "collection_name": {
            "type": "string",
            "description": "Collection name to delete from"
        },
        "filter": {
            "type": "string",
            "description": "Filter to match documents to delete (JSON string)"
        },
        "multi": {
            "type": "boolean",
            "description": "Delete multiple documents (default: false)"
        }
    }
    required: Optional[List[str]] = ["collection_name", "filter"]

    def __init__(self, database: MongoDBDatabase = None):
        super().__init__()
        self.database = database

    def __call__(self, collection_name: str, filter: str, multi: bool = False) -> Dict[str, Any]:
        """Delete documents from MongoDB collection"""
        try:
            if not self.database:
                return {"success": False, "error": "MongoDB database not initialized", "data": None}
            
            # Parse filter
            import json
            filter_dict = json.loads(filter)
            
            # Build query
            query = {
                "filter": filter_dict,
                "multi": multi
            }
            
            # Execute delete
            result = self.database.execute_query(
                query=query,
                query_type=QueryType.DELETE,
                collection_name=collection_name
            )
            
            if result["success"]:
                logger.info(f"Successfully deleted documents from collection {collection_name}")
            else:
                logger.error(f"Failed to delete documents: {result.get('error', 'Unknown error')}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in mongodb_delete tool: {str(e)}")
            return {"success": False, "error": str(e), "data": None}


class MongoDBInfoTool(Tool):
    name: str = "mongodb_info"
    description: str = "Get MongoDB database and collection information"
    inputs: Dict[str, Dict[str, str]] = {
        "info_type": {
            "type": "string",
            "description": "Type of information (database, collections, collection, schema, capabilities)"
        },
        "collection_name": {
            "type": "string",
            "description": "Collection name for collection-specific info (optional)"
        }
    }
    required: Optional[List[str]] = []

    def __init__(self, database: MongoDBDatabase = None):
        super().__init__()
        self.database = database

    def __call__(self, info_type: str = "database", collection_name: str = None) -> Dict[str, Any]:
        """Get MongoDB information"""
        try:
            if not self.database:
                return {"success": False, "error": "MongoDB database not initialized", "data": None}
            
            info_type = info_type.lower()
            
            if info_type == "database":
                result = self.database.get_database_info()
            elif info_type == "collections":
                collections = self.database.list_collections()
                result = {"success": True, "data": collections, "collection_count": len(collections)}
            elif info_type == "collection" and collection_name:
                result = self.database.get_collection_info(collection_name)
            elif info_type == "schema":
                result = self.database.get_schema(collection_name)
            elif info_type == "capabilities":
                result = {"success": True, "data": self.database.get_capabilities()}
            else:
                return {"success": False, "error": f"Invalid info type: {info_type}", "data": None}
            
            if result["success"]:
                logger.info(f"Successfully retrieved {info_type} information")
            else:
                logger.error(f"Failed to retrieve {info_type} information: {result.get('error', 'Unknown error')}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in mongodb_info tool: {str(e)}")
            return {"success": False, "error": str(e), "data": None}


class MongoDBToolkit(Toolkit):
    """
    MongoDB-specific toolkit with simplified design.
    Automatically handles remote, local file-based, or new database creation.
    """
    
    def __init__(self, 
                 name: str = "MongoDBToolkit",
                 connection_string: str = None,
                 database_name: str = None,
                 local_path: str = None,
                 auto_save: bool = True,
                 read_only: bool = False,
                 **kwargs):
        """
        Initialize the MongoDB toolkit.
        
        Args:
            name: Name of the toolkit
            connection_string: MongoDB connection string (for remote/existing)
            database_name: Name of the database to use
            local_path: Path for local file-based database
            auto_save: Automatically save changes to local files
            read_only: If True, only read operations are allowed (no insert, update, delete)
            **kwargs: Additional connection parameters
        """
        # Initialize database with automatic detection
        database = MongoDBDatabase(
            connection_string=connection_string,
            database_name=database_name,
            local_path=local_path,
            auto_save=auto_save,
            read_only=read_only,
            **kwargs
        )
        
        # Initialize tools based on read-only mode
        if read_only:
            # Only include read-only tools
            tools = [
                MongoDBExecuteQueryTool(database=database),
                MongoDBFindTool(database=database),
                MongoDBInfoTool(database=database)
            ]
        else:
            # Include all tools for write mode
            tools = [
                MongoDBExecuteQueryTool(database=database),
                MongoDBFindTool(database=database),
                MongoDBUpdateTool(database=database),
                MongoDBDeleteTool(database=database),
                MongoDBInfoTool(database=database)
            ]
        
        # Initialize parent with tools
        super().__init__(name=name, tools=tools)
        
        # Store configuration after parent initialization
        self.database = database
        self.connection_string = connection_string
        self.database_name = database_name
        self.local_path = local_path
        self.auto_save = auto_save
        
        # Set up automatic cleanup
        import atexit
        atexit.register(self._cleanup)
    
    def _cleanup(self):
        """Cleanup function called when program exits"""
        try:
            if self.database.is_local_database and self.database.auto_save:
                logger.info("Auto-saving local database before exit...")
                collections = self.database.list_collections()
                for collection_name in collections:
                    self.database._save_collection_to_file(collection_name)
            
            if self.database:
                self.database.disconnect()
                logger.info("Disconnected from MongoDB database")
                
        except Exception as e:
            logger.warning(f"Error during cleanup: {str(e)}")
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get MongoDB-specific capabilities"""
        if self.database:
            capabilities = self.database.get_capabilities()
            capabilities.update({
                "is_local_database": self.database.is_local_database,
                "local_path": str(self.database.local_path) if self.database.local_path else None,
                "auto_save": self.database.auto_save,
                "read_only": self.database.read_only
            })
            return capabilities
        return {"error": "MongoDB database not initialized"}
    
    def connect(self) -> bool:
        """Connect to MongoDB"""
        return self.database.connect() if self.database else False
    
    def disconnect(self) -> bool:
        """Disconnect from MongoDB"""
        return self.database.disconnect() if self.database else False
    
    def test_connection(self) -> bool:
        """Test MongoDB connection"""
        return self.database.test_connection() if self.database else False
    
    def get_database(self) -> MongoDBDatabase:
        """Get the underlying MongoDB database instance"""
        return self.database
    
    def get_local_info(self) -> Dict[str, Any]:
        """Get information about local database setup"""
        return {
            "is_local_database": self.database.is_local_database,
            "local_path": str(self.database.local_path) if self.database.local_path else None,
            "auto_save": self.database.auto_save,
            "read_only": self.database.read_only,
            "database_name": self.database_name,
            "connection_string": self.connection_string
        } if self.database else {"error": "Database not initialized"}