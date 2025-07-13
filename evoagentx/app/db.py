"""
Database connection and models for EvoAgentX.
"""
import logging
from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any, Protocol
from abc import ABC, abstractmethod
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import ASCENDING, TEXT, ReturnDocument
from pydantic_core import core_schema
# Import ObjectId from bson for standard MongoDB compatibility
from bson import ObjectId
from pydantic import GetCoreSchemaHandler
from pydantic import Field, BaseModel
from evoagentx.app.config import settings
from supabase import create_client, Client
import asyncpg
import json
import uuid

# Setup logger
logger = logging.getLogger(__name__)

# Custom PyObjectId for MongoDB ObjectId compatibility with Pydantic
class PyObjectId(ObjectId):
    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler: GetCoreSchemaHandler):
        return core_schema.no_info_after_validator_function(cls.validate, core_schema.str_schema())

    @classmethod
    def validate(cls, v):
        # Try to create ObjectId, which now handles UUIDs
        try:
            return ObjectId(v)
        except:
            raise ValueError("Invalid ObjectId")

# Base model with ObjectId handling
class MongoBaseModel(BaseModel):
    id: Optional[PyObjectId] = Field(alias="_id", default=None)

    model_config = {
        "protected_namespaces": (),
        "populate_by_name": True,  # Replace `allow_population_by_field_name`
        "arbitrary_types_allowed": True,  # Keep custom types like ObjectId
        "json_encoders": {
            ObjectId: str  # Ensure ObjectId is serialized as a string
        }
    }

# Status Enums
class AgentStatus(str, Enum):
    CREATED = "created"
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"

class WorkflowStatus(str, Enum):
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class ExecutionStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"

# Database Models
class Agent(MongoBaseModel):
    id: str = Field(..., alias="_id")
    name: str
    description: Optional[str] = None
    config: Dict[str, Any]
    state: Dict[str, Any] = Field(default_factory=dict)
    runtime_params: Dict[str, Any] = Field(default_factory=dict)
    status: AgentStatus = AgentStatus.CREATED
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: Optional[str] = None
    tags: List[str] = Field(default_factory=list)

class Workflow(MongoBaseModel):
    id: str = Field(..., alias="_id")
    name: str
    description: Optional[str] = None
    definition: Dict[str, Any]
    agent_ids: List[str] = Field(default_factory=list)
    status: WorkflowStatus = WorkflowStatus.CREATED
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    version: int = 1

class ExecutionLog(MongoBaseModel):
    workflow_id: str
    execution_id: str
    step_id: Optional[str] = None
    agent_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    level: str = "INFO"
    message: str
    details: Dict[str, Any] = Field(default_factory=dict)

class WorkflowExecution(MongoBaseModel):
    workflow_id: str
    status: ExecutionStatus = ExecutionStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    input_params: Dict[str, Any] = Field(default_factory=dict)
    results: Dict[str, Any] = Field(default_factory=dict)
    created_by: Optional[str] = None
    step_results: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    current_step: Optional[str] = None
    error_message: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

# Abstract Database Interface
class Database(ABC):
    """Abstract base class for database implementations."""
    
    @abstractmethod
    async def connect(self) -> None:
        """Connect to the database."""
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from the database."""
        pass
    
    @abstractmethod
    async def ping(self) -> bool:
        """Check if database connection is alive."""
        pass
    
    @abstractmethod
    async def search(self, collection: str, query: Dict[str, Any], **kwargs) -> List[Dict[str, Any]]:
        """Search for documents in a collection."""
        pass
    
    @abstractmethod
    async def write(self, collection: str, data: Dict[str, Any]) -> str:
        """Write a document to a collection."""
        pass
    
    @abstractmethod
    async def update(self, collection: str, filter_query: Dict[str, Any], update_data: Dict[str, Any], **kwargs) -> bool:
        """Update documents in a collection."""
        pass
    
    @abstractmethod
    async def delete(self, collection: str, query: Dict[str, Any]) -> int:
        """Delete documents from a collection."""
        pass
    
    @abstractmethod
    async def count(self, collection: str, query: Dict[str, Any] = None) -> int:
        """Count documents in a collection."""
        pass
    
    @abstractmethod
    async def find_one(self, collection: str, query: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Find a single document in a collection."""
        pass

# MongoDB Implementation
class MongoDatabase(Database):
    """MongoDB implementation of the Database interface."""
    
    def __init__(self, url: str, db_name: str, table_names: Optional[Dict[str, str]] = None):
        self.url = url
        self.db_name = db_name
        self.client: Optional[AsyncIOMotorClient] = None
        self.db = None
        
        # Default collection names (can be customized)
        default_table_names = {
            "agents": "agents",
            "workflows": "workflows", 
            "executions": "workflow_executions",
            "logs": "execution_logs",
            "users": "users"
        }
        
        # Merge custom table names with defaults
        self.table_names = {**default_table_names, **(table_names or {})}
    
    # Collections (will be set in connect method)
        self.agents = None
        self.workflows = None
        self.executions = None
        self.logs = None
        self.users = None
    
    def _resolve_table_name(self, collection: str) -> str:
        """
        Resolve the actual collection name from the logical collection name.
        
        Args:
            collection: Logical collection name (e.g., 'agents', 'workflows')
            
        Returns:
            Actual collection name in the database
        """
        # If it's a logical name, return the mapped collection name
        if collection in self.table_names:
            return self.table_names[collection]
        
        # Otherwise, return the collection name as-is (for direct collection access)
        return collection
    
    async def connect(self) -> None:
        """Connect to MongoDB"""
        logger.info(f"Connecting to MongoDB at {self.url}...")
        self.client = AsyncIOMotorClient(self.url)
        self.db = self.client[self.db_name]
        
        # Set up collections using configured table names
        self.agents = self.db[self.table_names["agents"]]
        self.workflows = self.db[self.table_names["workflows"]]
        self.executions = self.db[self.table_names["executions"]]
        self.logs = self.db[self.table_names["logs"]]
        self.users = self.db[self.table_names["users"]]
        
        # Create indexes
        await self._create_indexes()
        
        logger.info("Connected to MongoDB successfully")
    
    async def disconnect(self) -> None:
        """Disconnect from MongoDB"""
        if self.client:
            self.client.close()
            logger.info("Disconnected from MongoDB")
    
    async def ping(self) -> bool:
        """Check if MongoDB connection is alive."""
        try:
            await self.db.command('ping')
            return True
        except Exception as e:
            logger.error(f"Database ping failed: {e}")
            return False
    
    async def search(self, collection: str, query: Dict[str, Any], **kwargs) -> List[Dict[str, Any]]:
        """Search for documents in a MongoDB collection."""
        # Try to get predefined collection object first
        collection_obj = getattr(self, collection, None)
        if collection_obj is None:
            # Resolve the actual collection name and get collection object
            actual_collection_name = self._resolve_table_name(collection)
            collection_obj = self.db[actual_collection_name]
        
        limit = kwargs.get('limit', 100)
        skip = kwargs.get('skip', 0)
        sort = kwargs.get('sort', [])
        
        cursor = collection_obj.find(query)
        
        if sort:
            cursor = cursor.sort(sort)
        
        cursor = cursor.skip(skip).limit(limit)
        
        return await cursor.to_list(length=limit)
    
    async def write(self, collection: str, data: Dict[str, Any]) -> str:
        """Write a document to a MongoDB collection."""
        # Try to get predefined collection object first
        collection_obj = getattr(self, collection, None)
        if collection_obj is None:
            # Resolve the actual collection name and get collection object
            actual_collection_name = self._resolve_table_name(collection)
            collection_obj = self.db[actual_collection_name]
        result = await collection_obj.insert_one(data)
        return str(result.inserted_id)
    
    async def update(self, collection: str, filter_query: Dict[str, Any], update_data: Dict[str, Any], **kwargs) -> bool:
        """Update documents in a MongoDB collection."""
        collection_obj = getattr(self, collection, None)
        if collection_obj is None:
            collection_obj = self.db[collection]
        
        upsert = kwargs.get('upsert', False)
        many = kwargs.get('many', False)
        
        if many:
            result = await collection_obj.update_many(filter_query, update_data, upsert=upsert)
        else:
            result = await collection_obj.update_one(filter_query, update_data, upsert=upsert)
        
        return result.modified_count > 0 or result.upserted_id is not None
    
    async def delete(self, collection: str, query: Dict[str, Any]) -> int:
        """Delete documents from a MongoDB collection."""
        collection_obj = getattr(self, collection, None)
        if collection_obj is None:
            collection_obj = self.db[collection]
        result = await collection_obj.delete_many(query)
        return result.deleted_count
    
    async def count(self, collection: str, query: Dict[str, Any] = None) -> int:
        """Count documents in a MongoDB collection."""
        collection_obj = getattr(self, collection, None)
        if collection_obj is None:
            collection_obj = self.db[collection]
        query = query or {}
        return await collection_obj.count_documents(query)
    
    async def find_one(self, collection: str, query: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Find a single document in a MongoDB collection."""
        collection_obj = getattr(self, collection, None)
        if collection_obj is None:
            collection_obj = self.db[collection]
        return await collection_obj.find_one(query)
    
    async def find_one_and_update(self, collection: str, filter_query: Dict[str, Any], update_data: Dict[str, Any], **kwargs) -> Optional[Dict[str, Any]]:
        """Find and update a single document in a MongoDB collection."""
        collection_obj = getattr(self, collection, None)
        if collection_obj is None:
            collection_obj = self.db[collection]
        
        # Default to returning the updated document
        if 'return_document' in kwargs and kwargs['return_document'] is True:
            kwargs['return_document'] = ReturnDocument.AFTER
        elif 'return_document' not in kwargs:
            kwargs['return_document'] = ReturnDocument.AFTER
            
        return await collection_obj.find_one_and_update(filter_query, update_data, **kwargs)
    
    async def create_index(self, collection: str, index_spec, **kwargs):
        """Create an index on a MongoDB collection."""
        collection_obj = getattr(self, collection, None)
        if collection_obj is None:
            collection_obj = self.db[collection]
        return await collection_obj.create_index(index_spec, **kwargs)
    
    async def _create_indexes(self):
        """Create indexes for collections"""
        # Agent indexes
        await self.agents.create_index([("name", ASCENDING)], unique=True)
        await self.agents.create_index([("name", TEXT), ("description", TEXT)])
        await self.agents.create_index([("created_at", ASCENDING)])
        await self.agents.create_index([("tags", ASCENDING)])
        
        # Workflow indexes
        await self.workflows.create_index([("name", ASCENDING)])
        await self.workflows.create_index([("name", TEXT), ("description", TEXT)])
        await self.workflows.create_index([("created_at", ASCENDING)])
        await self.workflows.create_index([("agent_ids", ASCENDING)])
        await self.workflows.create_index([("tags", ASCENDING)])
        
        # Execution indexes
        await self.executions.create_index([("workflow_id", ASCENDING)])
        await self.executions.create_index([("created_at", ASCENDING)])
        await self.executions.create_index([("status", ASCENDING)])
        
        # Log indexes
        await self.logs.create_index([("execution_id", ASCENDING)])
        await self.logs.create_index([("timestamp", ASCENDING)])
        await self.logs.create_index([("workflow_id", ASCENDING), ("execution_id", ASCENDING)])

# NoSQL/In-Memory Database Implementation
class NoDatabase(Database):
    """No-database implementation for testing or simple use cases."""
    
    def __init__(self):
        self.connected = False
        self.collections: Dict[str, List[Dict[str, Any]]] = {}
        self._id_counter = 0
    
    async def connect(self) -> None:
        """Connect to the no-database (just set flag)."""
        self.connected = True
        logger.info("No-database connected (in-memory storage)")
    
    async def disconnect(self) -> None:
        """Disconnect from the no-database."""
        self.connected = False
        self.collections.clear()
        logger.info("No-database disconnected")
    
    async def ping(self) -> bool:
        """Check if no-database connection is alive."""
        return self.connected
    
    async def search(self, collection: str, query: Dict[str, Any], **kwargs) -> List[Dict[str, Any]]:
        """Search for documents in the in-memory collection."""
        if collection not in self.collections:
            return []
        
        results = []
        for doc in self.collections[collection]:
            if self._matches_query(doc, query):
                results.append(doc.copy())
        
        # Apply sorting
        sort = kwargs.get('sort', [])
        if sort:
            def sort_key(doc):
                keys = []
                for sort_field, sort_direction in sort:
                    field_key = sort_field if sort_field != "_id" else "_id"
                    value = doc.get(field_key, "")
                    # Convert to string for consistent sorting
                    if isinstance(value, (int, float)):
                        keys.append(value * sort_direction)
                    else:
                        keys.append(str(value) if sort_direction == 1 else str(value))
                return keys
            
            results.sort(key=sort_key, reverse=any(direction == -1 for _, direction in sort))
        
        # Apply limit and skip
        skip = kwargs.get('skip', 0)
        limit = kwargs.get('limit', 100)
        
        return results[skip:skip + limit]
    
    async def write(self, collection: str, data: Dict[str, Any]) -> str:
        """Write a document to the in-memory collection."""
        if collection not in self.collections:
            self.collections[collection] = []
        
        # Generate ID if not present
        if '_id' not in data:
            # Generate a valid ObjectId string instead of simple counter
            data['_id'] = str(ObjectId())
        
        self.collections[collection].append(data.copy())
        return str(data['_id'])
    
    async def update(self, collection: str, filter_query: Dict[str, Any], update_data: Dict[str, Any], **kwargs) -> bool:
        """Update documents in the in-memory collection."""
        if collection not in self.collections:
            return False
        
        updated = False
        for doc in self.collections[collection]:
            if self._matches_query(doc, filter_query):
                # Simple update (assumes $set operation)
                if '$set' in update_data:
                    doc.update(update_data['$set'])
                else:
                    doc.update(update_data)
                updated = True
                if not kwargs.get('many', False):
                    break
        
        return updated
    
    async def delete(self, collection: str, query: Dict[str, Any]) -> int:
        """Delete documents from the in-memory collection."""
        if collection not in self.collections:
            return 0
        
        original_count = len(self.collections[collection])
        self.collections[collection] = [
            doc for doc in self.collections[collection]
            if not self._matches_query(doc, query)
        ]
        
        return original_count - len(self.collections[collection])
    
    async def count(self, collection: str, query: Dict[str, Any] = None) -> int:
        """Count documents in the in-memory collection."""
        if collection not in self.collections:
            return 0
        
        if query is None:
            return len(self.collections[collection])
        
        return len([doc for doc in self.collections[collection] if self._matches_query(doc, query)])
    
    async def find_one(self, collection: str, query: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Find a single document in the in-memory collection."""
        if collection not in self.collections:
            return None
        
        for doc in self.collections[collection]:
            if self._matches_query(doc, query):
                return doc.copy()
        
        return None
    
    async def find_one_and_update(self, collection: str, filter_query: Dict[str, Any], update_data: Dict[str, Any], **kwargs) -> Optional[Dict[str, Any]]:
        """Find and update a single document in the in-memory collection."""
        if collection not in self.collections:
            return None
        
        for doc in self.collections[collection]:
            if self._matches_query(doc, filter_query):
                # Apply the update
                if '$set' in update_data:
                    doc.update(update_data['$set'])
                else:
                    doc.update(update_data)
                return doc.copy()
        
        return None

    def _matches_query(self, doc: Dict[str, Any], query: Dict[str, Any]) -> bool:
        """Simple query matching for in-memory storage."""
        for key, value in query.items():
            if key not in doc:
                return False
            
            # Handle special MongoDB operators
            if isinstance(value, dict):
                if '$ne' in value:
                    # Handle $ne (not equal) operator
                    if key == "_id":
                        if str(doc[key]) == str(value['$ne']):
                            return False
                    elif doc[key] == value['$ne']:
                        return False
                    continue
                # Add more operators as needed
            
            # Handle ObjectId queries
            if isinstance(value, ObjectId):
                # Convert ObjectId to string for comparison
                if str(value) != str(doc[key]):
                    return False
            elif key == "_id":
                # Handle _id queries - always compare as strings
                if str(value) != str(doc[key]):
                    return False
            elif doc[key] != value:
                return False
        return True

# PostgreSQL Database Implementation
class PostgreSQLDatabase(Database):
    """PostgreSQL implementation using asyncpg."""
    
    def __init__(self, url: str, table_names: Optional[Dict[str, str]] = None):
        self.url = url
        self.connection_pool: Optional[asyncpg.Pool] = None
        
        # Default table names (can be customized)
        default_table_names = {
            "agents": "agents",
            "workflows": "workflows", 
            "executions": "workflow_executions",
            "logs": "execution_logs",
            "users": "users"
        }
        
        # Merge custom table names with defaults
        self.table_names = {**default_table_names, **(table_names or {})}
    
    def _resolve_table_name(self, collection: str) -> str:
        """
        Resolve the actual table name from the logical collection name.
        
        Args:
            collection: Logical collection name (e.g., 'agents', 'workflows')
            
        Returns:
            Actual table name in the database
        """
        # If it's a logical name, return the mapped table name
        if collection in self.table_names:
            return self.table_names[collection]
        
        # Otherwise, return the collection name as-is (for direct table access)
        return collection
    
    def _is_uuid_string(self, value: str) -> bool:
        """Check if a string is a valid UUID format."""
        if not isinstance(value, str):
            return False
        
        import re
        uuid_pattern = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', re.IGNORECASE)
        return bool(uuid_pattern.match(value))
    
    def _normalize_id_for_query(self, id_value):
        """
        Normalize ID values for database queries.
        Converts ObjectId to string, handles UUIDs, etc.
        """
        if isinstance(id_value, ObjectId):
            return str(id_value)
        elif isinstance(id_value, str):
            if ObjectId.is_valid(id_value):
                # It's a valid ObjectId string, convert to ObjectId then back to string
                return str(ObjectId(id_value))
            elif self._is_uuid_string(id_value):
                # It's a UUID string, use as is
                return id_value
            else:
                # It's some other string, use as is
                return id_value
        else:
            # Convert to string for safety
            return str(id_value)
    
    def _normalize_query(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize query dictionary to handle ObjectId/UUID conversions.
        This ensures compatibility with services that may pass ObjectId instances.
        """
        if not isinstance(query, dict):
            return query
        
        normalized = {}
        for key, value in query.items():
            if key == "_id":
                # Handle ID field specially
                if isinstance(value, dict):
                    # Handle operators like $ne, $in, etc.
                    normalized[key] = {}
                    for op, op_value in value.items():
                        if op in ['$ne']:
                            normalized[key][op] = self._normalize_id_for_query(op_value)
                        elif op in ['$in']:
                            normalized[key][op] = [self._normalize_id_for_query(v) for v in op_value]
                        else:
                            normalized[key][op] = op_value
                else:
                    # Direct ID value
                    normalized[key] = self._normalize_id_for_query(value)
            else:
                # Non-ID fields, copy as is
                normalized[key] = value
        
        return normalized
    
    async def connect(self) -> None:
        """Connect to PostgreSQL"""
        try:
            logger.info(f"Connecting to PostgreSQL...")
            
            # Create connection pool with timeout settings
            self.connection_pool = await asyncpg.create_pool(
                self.url,
                min_size=1,
                max_size=20,
                command_timeout=30,  # 30 second timeout for commands
                server_settings={
                    'application_name': 'evoagentx',
                }
            )
            
            # Test the connection
            async with self.connection_pool.acquire() as conn:
                await conn.execute('SELECT 1')
            
            # Create tables if they don't exist
            await self._create_tables()
            logger.info("Tables created successfully")
            
            # Create indexes
            await self._create_indexes()
            logger.info("Indexes created successfully")
            
            logger.info("Connected to PostgreSQL successfully")
            
        except Exception as e:
            logger.error(f"Error connecting to PostgreSQL: {e}")
            if self.connection_pool:
                await self.connection_pool.close()
                self.connection_pool = None
            raise Exception(f"Failed to connect to PostgreSQL: {str(e)}")
    
    async def disconnect(self) -> None:
        """Disconnect from PostgreSQL"""
        try:
            if self.connection_pool:
                await self.connection_pool.close()
                self.connection_pool = None
                logger.info("Disconnected from PostgreSQL")
        except Exception as e:
            logger.error(f"Error disconnecting from PostgreSQL: {e}")
            self.connection_pool = None  # Reset anyway
    
    async def ping(self) -> bool:
        """Check if PostgreSQL connection is alive."""
        try:
            if not self.connection_pool:
                return False
            
            async with self.connection_pool.acquire() as conn:
                await conn.execute('SELECT 1')
                return True
        except Exception as e:
            logger.error(f"Database ping failed: {e}")
            return False
    
    async def search(self, collection: str, query: Dict[str, Any], **kwargs) -> List[Dict[str, Any]]:
        """Search for documents in a PostgreSQL table."""
        try:
            if not self.connection_pool:
                raise Exception("Database connection not established")
                
            table_name = self._resolve_table_name(collection)
            limit = int(kwargs.get('limit', 100))
            skip = int(kwargs.get('skip', 0))
            sort = kwargs.get('sort', [])
            
            # Normalize query to handle ObjectId/UUID conversion
            normalized_query = self._normalize_query(query)
            
            # Build WHERE clause
            where_clause, params = self._build_where_clause(normalized_query)
            
            # Build ORDER BY clause
            order_clause = ""
            if sort:
                order_parts = []
                for sort_field, sort_direction in sort:
                    direction = "ASC" if sort_direction == 1 else "DESC"
                    if sort_field == "_id":
                        order_parts.append(f"id {direction}")
                    else:
                        order_parts.append(f"data->'{sort_field}' {direction}")
                order_clause = f"ORDER BY {', '.join(order_parts)}"
            
            # Build final query with correct parameter numbering
            limit_param = len(params) + 1
            offset_param = len(params) + 2
            
            sql = f"""
                SELECT id, data FROM {table_name}
                {where_clause}
                {order_clause}
                LIMIT ${limit_param} OFFSET ${offset_param}
            """
            
            # Add limit and offset params
            params.extend([limit, skip])
            
            async with self.connection_pool.acquire() as conn:
                rows = await conn.fetch(sql, *params)
                return [self._format_document(row) for row in rows]
        except Exception as e:
            logger.error(f"Error searching in {collection}: {str(e)}")
            raise Exception(f"Database search failed: {str(e)}")
    
    async def write(self, collection: str, data: Dict[str, Any]) -> str:
        """Write a document to a PostgreSQL table."""
        try:
            if not self.connection_pool:
                raise Exception("Database connection not established")
                
            table_name = self._resolve_table_name(collection)
            
            # Generate ID if not present
            if '_id' not in data:
                data['_id'] = str(ObjectId())
            
            doc_id = data['_id']
            
            # Convert ObjectId to string for PostgreSQL
            if isinstance(doc_id, ObjectId):
                doc_id = str(doc_id)
            
            # Prepare data for storage (exclude _id from JSON data)
            json_data = {k: v for k, v in data.items() if k != '_id'}
            
            sql = f"""
                INSERT INTO {table_name} (id, data)
                VALUES ($1, $2)
                ON CONFLICT (id) DO UPDATE SET
                    data = EXCLUDED.data
            """
            
            async with self.connection_pool.acquire() as conn:
                await conn.execute(sql, doc_id, json.dumps(json_data, default=str))
                
            return doc_id
        except Exception as e:
            logger.error(f"Error writing to {collection}: {str(e)}")
            raise Exception(f"Database write failed: {str(e)}")
    
    async def update(self, collection: str, filter_query: Dict[str, Any], update_data: Dict[str, Any], **kwargs) -> bool:
        """Update documents in a PostgreSQL table."""
        try:
            if not self.connection_pool:
                raise Exception("Database connection not established")
                
            table_name = self._resolve_table_name(collection)
            upsert = kwargs.get('upsert', False)
            many = kwargs.get('many', False)
            
            # Handle MongoDB-style update operations
            if '$set' in update_data:
                set_data = update_data['$set']
            else:
                set_data = update_data
            
            # Normalize query to handle ObjectId/UUID conversion
            normalized_filter = self._normalize_query(filter_query)
            
            # Build WHERE clause with parameter starting from 2 (since $1 is for the update data)
            where_clause, where_params = self._build_where_clause(normalized_filter, start_param=2)
            
            # Build update query
            if many:
                # Update multiple documents
                sql = f"""
                    UPDATE {table_name} 
                    SET data = data || $1::jsonb
                    {where_clause}
                """
                
                async with self.connection_pool.acquire() as conn:
                    result = await conn.execute(sql, json.dumps(set_data, default=str), *where_params)
                    return result.split()[-1] != "0"  # Check if any rows were updated
            else:
                # Update single document
                sql = f"""
                    UPDATE {table_name} 
                    SET data = data || $1::jsonb
                    {where_clause}
                """
                
                async with self.connection_pool.acquire() as conn:
                    result = await conn.execute(sql, json.dumps(set_data, default=str), *where_params)
                    
                    if result.split()[-1] == "0" and upsert:
                        # No rows updated, insert new document if upsert is True
                        new_data = {**filter_query, **set_data}
                        await self.write(collection, new_data)
                        return True
                    
                    return result.split()[-1] != "0"
        except Exception as e:
            logger.error(f"Error updating {collection}: {str(e)}")
            raise Exception(f"Database update failed: {str(e)}")
    
    async def delete(self, collection: str, query: Dict[str, Any]) -> int:
        """Delete documents from a PostgreSQL table."""
        try:
            if not self.connection_pool:
                raise Exception("Database connection not established")
                
            table_name = self._resolve_table_name(collection)
            
            # Normalize query to handle ObjectId/UUID conversion
            normalized_query = self._normalize_query(query)
            
            # Build WHERE clause
            where_clause, params = self._build_where_clause(normalized_query)
            
            sql = f"""
                DELETE FROM {table_name}
                {where_clause}
            """
            
            async with self.connection_pool.acquire() as conn:
                result = await conn.execute(sql, *params)
                return int(result.split()[-1])
        except Exception as e:
            logger.error(f"Error deleting from {collection}: {str(e)}")
            raise Exception(f"Database delete failed: {str(e)}")
    
    async def count(self, collection: str, query: Dict[str, Any] = None) -> int:
        """Count documents in a PostgreSQL table."""
        try:
            if not self.connection_pool:
                raise Exception("Database connection not established")
                
            table_name = self._resolve_table_name(collection)
            
            if query is None:
                query = {}
            
            # Normalize query to handle ObjectId/UUID conversion
            normalized_query = self._normalize_query(query)
            
            # Build WHERE clause
            where_clause, params = self._build_where_clause(normalized_query)
            
            sql = f"""
                SELECT COUNT(*) FROM {table_name}
                {where_clause}
            """
            
            async with self.connection_pool.acquire() as conn:
                result = await conn.fetchval(sql, *params)
                return result
        except Exception as e:
            logger.error(f"Error counting in {collection}: {str(e)}")
            raise Exception(f"Database count failed: {str(e)}")
    
    async def find_one(self, collection: str, query: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Find a single document in a PostgreSQL table."""
        try:
            if not self.connection_pool:
                raise Exception("Database connection not established")
                
            table_name = self._resolve_table_name(collection)
            
            # Normalize query to handle ObjectId/UUID conversion
            normalized_query = self._normalize_query(query)
            
            # Build WHERE clause
            where_clause, params = self._build_where_clause(normalized_query)
            
            sql = f"""
                SELECT id, data FROM {table_name}
                {where_clause}
                LIMIT 1
            """
            
            async with self.connection_pool.acquire() as conn:
                row = await conn.fetchrow(sql, *params)
                
                if row:
                    return self._format_document(row)
                return None
        except Exception as e:
            logger.error(f"Error finding one in {collection}: {str(e)}")
            raise Exception(f"Database find_one failed: {str(e)}")
    
    async def find_one_and_update(self, collection: str, filter_query: Dict[str, Any], update_data: Dict[str, Any], **kwargs) -> Optional[Dict[str, Any]]:
        """Find and update a single document in a PostgreSQL table."""
        try:
            if not self.connection_pool:
                raise Exception("Database connection not established")
                
            table_name = self._resolve_table_name(collection)
            
            # Handle MongoDB-style update operations
            if '$set' in update_data:
                set_data = update_data['$set']
            else:
                set_data = update_data
            
            # Normalize query to handle ObjectId/UUID conversion
            normalized_filter = self._normalize_query(filter_query)
            
            # Build WHERE clause with parameter starting from 2 (since $1 is for the update data)
            where_clause, where_params = self._build_where_clause(normalized_filter, start_param=2)
            
            sql = f"""
                UPDATE {table_name} 
                SET data = data || $1::jsonb
                {where_clause}
                RETURNING id, data
            """
            
            async with self.connection_pool.acquire() as conn:
                row = await conn.fetchrow(sql, json.dumps(set_data, default=str), *where_params)
                
                if row:
                    return self._format_document(row)
                return None
        except Exception as e:
            logger.error(f"Error in find_one_and_update for {collection}: {str(e)}")
            raise Exception(f"Database find_one_and_update failed: {str(e)}")
    
    async def create_index(self, collection: str, index_spec, **kwargs):
        """Create an index on a PostgreSQL table."""
        try:
            if not self.connection_pool:
                raise Exception("Database connection not established")
                
            table_name = self._resolve_table_name(collection)
            unique = kwargs.get('unique', False)
            
            async with self.connection_pool.acquire() as conn:
                # First check if the table exists
                table_exists = await conn.fetchval("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = $1
                    )
                """, table_name)
                
                if not table_exists:
                    logger.warning(f"Table {table_name} does not exist, skipping index creation")
                    return
                
                # Handle different index specifications
                if isinstance(index_spec, str):
                    # Simple field index like "email"
                    field_name = index_spec
                    index_name = f"idx_{table_name}_{field_name}{'_unique' if unique else ''}"
                    
                    if unique:
                        sql = f"""
                            CREATE UNIQUE INDEX IF NOT EXISTS {index_name} 
                            ON {table_name} ((data->>'{field_name}'))
                        """
                    else:
                        sql = f"""
                            CREATE INDEX IF NOT EXISTS {index_name} 
                            ON {table_name} ((data->>'{field_name}'))
                        """
                elif isinstance(index_spec, list):
                    # Compound index like [("field1", 1), ("field2", -1)]
                    fields = []
                    for field_spec in index_spec:
                        if isinstance(field_spec, tuple):
                            field_name, direction = field_spec
                            direction_str = "ASC" if direction == 1 else "DESC"
                            fields.append(f"(data->>'{field_name}') {direction_str}")
                        else:
                            fields.append(f"(data->>'{field_spec}')")
                    
                    index_name = f"idx_{table_name}_{hash(str(index_spec))}"
                    sql = f"""
                        CREATE INDEX IF NOT EXISTS {index_name} 
                        ON {table_name} ({', '.join(fields)})
                    """
                else:
                    # Default to GIN index for complex queries
                    index_name = f"idx_{table_name}_{hash(str(index_spec))}"
                    sql = f"""
                        CREATE INDEX IF NOT EXISTS {index_name} 
                        ON {table_name} USING GIN (data)
                    """
                
                await conn.execute(sql)
                logger.info(f"Index {index_name} created for table {table_name}")
                
        except Exception as e:
            logger.error(f"Error creating index for {collection}: {str(e)}")
            raise Exception(f"Database create_index failed: {str(e)}")
    
    async def _create_tables(self):
        """Create tables for each collection if they don't exist."""
        
        async with self.connection_pool.acquire() as conn:
            for logical_name, table_name in self.table_names.items():
                try:
                    sql = f"""
                        CREATE TABLE IF NOT EXISTS {table_name} (
                            id TEXT PRIMARY KEY,
                            data JSONB NOT NULL,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        )
                    """
                    await conn.execute(sql)
                    logger.info(f"Table {table_name} created/verified")
                    
                    # Create a trigger to update the updated_at timestamp
                    trigger_sql = f"""
                        CREATE OR REPLACE FUNCTION update_updated_at_column()
                        RETURNS TRIGGER AS $$
                        BEGIN
                            NEW.updated_at = CURRENT_TIMESTAMP;
                            RETURN NEW;
                        END;
                        $$ LANGUAGE plpgsql;
                        
                        DROP TRIGGER IF EXISTS update_{table_name}_updated_at ON {table_name};
                        CREATE TRIGGER update_{table_name}_updated_at
                            BEFORE UPDATE ON {table_name}
                            FOR EACH ROW
                            EXECUTE FUNCTION update_updated_at_column();
                    """
                    await conn.execute(trigger_sql)
                    logger.info(f"Trigger for {table_name} created/updated")
                    
                except Exception as e:
                    logger.error(f"Error creating table {table_name}: {e}")
                    raise
    
    async def _create_indexes(self):
        """Create indexes for better performance."""
        
        async with self.connection_pool.acquire() as conn:
            # Create indexes for each table
            for logical_name, table_name in self.table_names.items():
                try:
                    # General GIN index on data for JSON queries
                    await conn.execute(f"""
                        CREATE INDEX IF NOT EXISTS idx_{table_name}_data_gin 
                        ON {table_name} USING GIN (data)
                    """)
                    logger.info(f"GIN index created for {table_name}")
                except Exception as e:
                    logger.error(f"Error creating GIN index for {table_name}: {e}")
                    # Don't raise here - continue with other indexes
                
                # Specific indexes based on collection type
                try:
                    if logical_name == "agents":
                        await conn.execute(f"""
                            CREATE UNIQUE INDEX IF NOT EXISTS idx_{table_name}_name 
                            ON {table_name} ((data->>'name'))
                        """)
                        await conn.execute(f"""
                            CREATE INDEX IF NOT EXISTS idx_{table_name}_created_at 
                            ON {table_name} ((data->>'created_at'))
                        """)
                        await conn.execute(f"""
                            CREATE INDEX IF NOT EXISTS idx_{table_name}_tags 
                            ON {table_name} USING GIN ((data->'tags'))
                        """)
                        logger.info(f"Agent-specific indexes created for {table_name}")
                    
                    elif logical_name == "workflows":
                        await conn.execute(f"""
                            CREATE INDEX IF NOT EXISTS idx_{table_name}_name 
                            ON {table_name} ((data->>'name'))
                        """)
                        await conn.execute(f"""
                            CREATE INDEX IF NOT EXISTS idx_{table_name}_created_at 
                            ON {table_name} ((data->>'created_at'))
                        """)
                        await conn.execute(f"""
                            CREATE INDEX IF NOT EXISTS idx_{table_name}_agent_ids 
                            ON {table_name} USING GIN ((data->'agent_ids'))
                        """)
                        logger.info(f"Workflow-specific indexes created for {table_name}")
                    
                    elif logical_name == "executions":
                        await conn.execute(f"""
                            CREATE INDEX IF NOT EXISTS idx_{table_name}_workflow_id 
                            ON {table_name} ((data->>'workflow_id'))
                        """)
                        await conn.execute(f"""
                            CREATE INDEX IF NOT EXISTS idx_{table_name}_status 
                            ON {table_name} ((data->>'status'))
                        """)
                        await conn.execute(f"""
                            CREATE INDEX IF NOT EXISTS idx_{table_name}_created_at 
                            ON {table_name} ((data->>'created_at'))
                        """)
                        logger.info(f"Execution-specific indexes created for {table_name}")
                    
                    elif logical_name == "logs":
                        await conn.execute(f"""
                            CREATE INDEX IF NOT EXISTS idx_{table_name}_execution_id 
                            ON {table_name} ((data->>'execution_id'))
                        """)
                        await conn.execute(f"""
                            CREATE INDEX IF NOT EXISTS idx_{table_name}_workflow_id 
                            ON {table_name} ((data->>'workflow_id'))
                        """)
                        await conn.execute(f"""
                            CREATE INDEX IF NOT EXISTS idx_{table_name}_timestamp 
                            ON {table_name} ((data->>'timestamp'))
                        """)
                        logger.info(f"Log-specific indexes created for {table_name}")
                        
                except Exception as e:
                    logger.warning(f"Error creating specific indexes for {table_name}: {e}")
                    # Don't raise here - specific indexes are optional
    
    def _build_where_clause(self, query: Dict[str, Any], start_param: int = 1) -> tuple[str, List[Any]]:
        """Build WHERE clause from MongoDB-style query."""
        if not query:
            return "", []
        
        conditions = []
        params = []
        param_count = start_param - 1
        
        for key, value in query.items():
            param_count += 1
            
            if key == "_id":
                conditions.append(f"id = ${param_count}")
                # Normalize ID for PostgreSQL query
                params.append(self._normalize_id_for_query(value))
            elif isinstance(value, dict):
                # Handle special operators
                if '$ne' in value:
                    if key == "_id":
                        conditions.append(f"id != ${param_count}")
                        # Normalize ID for PostgreSQL query
                        params.append(self._normalize_id_for_query(value['$ne']))
                    else:
                        conditions.append(f"data->>'{key}' != ${param_count}")
                        params.append(str(value['$ne']))
                elif '$in' in value:
                    if key == "_id":
                        # For ID fields, use regular IN clause
                        placeholders = ','.join([f"${i}" for i in range(param_count, param_count + len(value['$in']))])
                        conditions.append(f"id IN ({placeholders})")
                        params.extend([self._normalize_id_for_query(v) for v in value['$in']])
                        param_count += len(value['$in']) - 1
                    else:
                        # For JSONB fields, use proper array handling
                        conditions.append(f"data->>'{key}' = ANY(${param_count})")
                        params.append(value['$in'])
                elif '$regex' in value:
                    conditions.append(f"data->>'{key}' ~ ${param_count}")
                    params.append(value['$regex'])
                elif '$gte' in value:
                    conditions.append(f"(data->>'{key}')::timestamp >= ${param_count}::timestamp")
                    params.append(value['$gte'].isoformat() if hasattr(value['$gte'], 'isoformat') else str(value['$gte']))
                elif '$lte' in value:
                    conditions.append(f"(data->>'{key}')::timestamp <= ${param_count}::timestamp")
                    params.append(value['$lte'].isoformat() if hasattr(value['$lte'], 'isoformat') else str(value['$lte']))
                elif '$gt' in value:
                    conditions.append(f"(data->>'{key}')::timestamp > ${param_count}::timestamp")
                    params.append(value['$gt'].isoformat() if hasattr(value['$gt'], 'isoformat') else str(value['$gt']))
                elif '$lt' in value:
                    conditions.append(f"(data->>'{key}')::timestamp < ${param_count}::timestamp")
                    params.append(value['$lt'].isoformat() if hasattr(value['$lt'], 'isoformat') else str(value['$lt']))
                else:
                    # Handle other operators as needed
                    conditions.append(f"data->>'{key}' = ${param_count}")
                    params.append(json.dumps(value))
            elif isinstance(value, list):
                # Handle array contains - check if JSONB array contains all values
                conditions.append(f"data->'{key}' ?& ${param_count}")
                params.append(value)
            else:
                # Check if this is a single value that should be searched in an array
                # For fields like 'tags', we want to check if the value is contained in the array
                if key == 'tags':
                    conditions.append(f"data->'{key}' ? ${param_count}")
                    params.append(str(value))
                else:
                    conditions.append(f"data->>'{key}' = ${param_count}")
                    params.append(str(value))
        
        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        return where_clause, params
    
    def _format_document(self, row) -> Dict[str, Any]:
        """Format a database row into a document."""
        doc = json.loads(row['data'])
        doc['_id'] = row['id']
        return doc


# Database Factory
def create_database(db_type: str = "", **kwargs) -> Database:
    """
    Factory function to create database instances.
    
    Supported database types:
    - "mongodb": MongoDB database with motor async driver
    - "supabase": Supabase database with PostgreSQL backend  
    - "postgresql": PostgreSQL database with asyncpg driver
    - default: In-memory NoDatabase for testing
    
    Common parameters:
    - table_names: Dict[str, str] - Custom table name mapping (optional)
    
    MongoDB specific parameters:
    - url: str - MongoDB connection URL
    - db_name: str - Database name
    
    PostgreSQL specific parameters:
    - url: str - PostgreSQL connection URL
    
    Supabase specific parameters:
    - url: str - Supabase project URL  
    - key: str - Supabase API key
    """
    if db_type.lower() == "mongodb":
        url = kwargs.get('url', settings.MONGODB_URL)
        db_name = kwargs.get('db_name', settings.MONGODB_DB_NAME)
        table_names = kwargs.get('table_names', None)
        return MongoDatabase(url, db_name, table_names)
    elif db_type.lower() == "postgresql":
        url = kwargs.get('url', settings.POSTGRESQL_URL)
        table_names = kwargs.get('table_names', None)
        return PostgreSQLDatabase(url, table_names)
    else:
        return NoDatabase()

# Global database instance
database: Database = create_database("postgresql")