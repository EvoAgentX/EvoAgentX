"""
Database abstraction for EvoAgentX server.
Simple and clean interface for database operations.
"""
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any
from abc import ABC, abstractmethod
import asyncio
import os
import uuid
# Setup logger
logger = logging.getLogger(__name__)
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), 'app.env'), override=True)

# Abstract Database Interface
class Database(ABC):
    """Abstract base class for database implementations."""
    
    def __init__(self, table_names: Optional[Dict[str, str]] = None):
        """
        Initialize database with optional table name mapping.
        
        Args:
            table_names: Dict mapping logical table names to actual table names
                        e.g., {"users": "user_accounts", "workflows": "wf_instances"}
        """
        self.table_names = table_names or {}
    
    def _get_table_name(self, table: str) -> str:
        """Get actual table name from logical table name."""
        return self.table_names.get(table, table)
    
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
    
    # Core CRUD operations
    @abstractmethod
    async def find_one(self, table: str, filter: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """Find a single document."""
        pass
    
    @abstractmethod
    async def find_many(self, table: str, filter: Dict[str, Any] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Find multiple documents."""
        pass
    
    @abstractmethod
    async def insert(self, table: str, data: Dict[str, Any]) -> str:
        """Insert a single document and return its ID."""
        pass
    
    @abstractmethod
    async def update(self, table: str, filter: Dict[str, Any], data: Dict[str, Any]) -> bool:
        """Update documents matching filter."""
        pass
    
    @abstractmethod
    async def delete(self, table: str, filter: Dict[str, Any]) -> int:
        """Delete documents matching filter and return count."""
        pass
    
    @abstractmethod
    async def count(self, table: str, filter: Dict[str, Any] = None) -> int:
        """Count documents matching filter."""
        pass

# In-Memory Database Implementation
class InMemoryDatabase(Database):
    """Simple in-memory database implementation."""
    
    def __init__(self, table_names: Optional[Dict[str, str]] = None):
        super().__init__(table_names)
        self.connected = False
        self.tables: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self._id_counter = 0
        
        logger.info("Initialized InMemoryDatabase")
    
    def _ensure_table(self, table: str) -> None:
        """Ensure table exists."""
        actual_table = self._get_table_name(table)
        if actual_table not in self.tables:
            self.tables[actual_table] = {}
    
    def _matches_filter(self, document: Dict[str, Any], filter: Dict[str, Any]) -> bool:
        """Check if document matches filter conditions."""
        if not filter:
            return True
            
        for key, value in filter.items():
            if key not in document:
                return False
            if document[key] != value:
                return False
        return True
    
    async def connect(self) -> None:
        """Connect to the in-memory database."""
        self.connected = True
        logger.info("InMemoryDatabase connected")
    
    async def disconnect(self) -> None:
        """Disconnect and clear all data."""
        self.connected = False
        self.tables.clear()
        logger.info("InMemoryDatabase disconnected")
    
    async def ping(self) -> bool:
        """Check if database is connected."""
        return self.connected
    
    async def find_one(self, table: str, filter: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """Find a single document."""
        actual_table = self._get_table_name(table)
        self._ensure_table(table)
        filter = filter or {}
        
        for doc_id, document in self.tables[actual_table].items():
            if self._matches_filter(document, filter):
                return document.copy()
        return None
    
    async def find_many(self, table: str, filter: Dict[str, Any] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Find multiple documents."""
        actual_table = self._get_table_name(table)
        self._ensure_table(table)
        filter = filter or {}
        
        results = []
        for doc_id, document in self.tables[actual_table].items():
            if self._matches_filter(document, filter):
                results.append(document.copy())
                if len(results) >= limit:
                    break
        
        return results
    
    async def insert(self, table: str, data: Dict[str, Any]) -> str:
        """Insert a single document."""
        actual_table = self._get_table_name(table)
        self._ensure_table(table)
        
        # Generate ID if not present
        if "_id" not in data:
            self._id_counter += 1
            doc_id = str(self._id_counter)
            data["_id"] = doc_id
        else:
            doc_id = str(data["_id"])
        
        # Add timestamps
        now = datetime.now().isoformat()
        if "created_at" not in data:
            data["created_at"] = now
        if "updated_at" not in data:
            data["updated_at"] = now
        
        self.tables[actual_table][doc_id] = data.copy()
        logger.debug(f"Inserted document with ID {doc_id} into {actual_table}")
        return doc_id
    
    async def update(self, table: str, filter: Dict[str, Any], data: Dict[str, Any]) -> bool:
        """Update documents matching filter."""
        actual_table = self._get_table_name(table)
        self._ensure_table(table)
        
        updated = False
        for doc_id, document in self.tables[actual_table].items():
            if self._matches_filter(document, filter):
                # Update the document
                document.update(data)
                document["updated_at"] = datetime.now().isoformat()
                updated = True
                logger.debug(f"Updated document {doc_id} in {actual_table}")
        
        return updated
    
    async def delete(self, table: str, filter: Dict[str, Any]) -> int:
        """Delete documents matching filter."""
        actual_table = self._get_table_name(table)
        self._ensure_table(table)
        
        to_delete = []
        for doc_id, document in self.tables[actual_table].items():
            if self._matches_filter(document, filter):
                to_delete.append(doc_id)
        
        for doc_id in to_delete:
            del self.tables[actual_table][doc_id]
        
        logger.debug(f"Deleted {len(to_delete)} documents from {actual_table}")
        return len(to_delete)
    
    async def count(self, table: str, filter: Dict[str, Any] = None) -> int:
        """Count documents matching filter."""
        actual_table = self._get_table_name(table)
        self._ensure_table(table)
        filter = filter or {}
        
        count = 0
        for doc_id, document in self.tables[actual_table].items():
            if self._matches_filter(document, filter):
                count += 1
        return count

# Supabase Database Implementation
class SupabaseDatabase(Database):
    """Supabase database implementation using supabase-py client."""
    
    def __init__(self, url: str = None, key: str = None, table_names: Optional[Dict[str, str]] = None):
        """
        Initialize Supabase database connection.
        
        Args:
            url: Supabase project URL (defaults to SUPABASE_URL env var)
            key: Supabase API key (defaults to SUPABASE_KEY env var)
            table_names: Dict mapping logical table names to actual table names
        """
        super().__init__(table_names)
        self.url = url or os.environ.get("SUPABASE_URL")
        self.key = key or os.environ.get("SUPABASE_KEY")
        self.client = None
        self.connected = False
        
        if not self.url or not self.key:
            raise ValueError("Supabase URL and KEY must be provided either as parameters or environment variables")
        
        logger.info("Initialized SupabaseDatabase")
    
    async def connect(self) -> None:
        """Connect to Supabase."""
        try:
            from supabase import create_client, Client
            self.client: Client = create_client(self.url, self.key)
            self.connected = True
            logger.info("SupabaseDatabase connected successfully")
        except ImportError:
            raise ImportError("supabase package is required. Install with: pip install supabase")
        except Exception as e:
            logger.error(f"Failed to connect to Supabase: {e}")
            raise
    
    async def disconnect(self) -> None:
        """Disconnect from Supabase."""
        self.client = None
        self.connected = False
        logger.info("SupabaseDatabase disconnected")
    
    async def ping(self) -> bool:
        """Check if Supabase connection is alive."""
        if not self.client or not self.connected:
            return False
        
        try:
            # Try a simple query to test connection - using a common system table
            # We'll catch the error if table doesn't exist, which is fine for ping
            response = self.client.table("users").select("*").limit(1).execute()
            return True
        except Exception as e:
            logger.debug(f"Ping failed: {e}")
            # Return True if connected but table doesn't exist (which is normal)
            return self.connected
    
    def _build_filter_query(self, query, filter: Dict[str, Any]):
        """Build filter conditions for Supabase query."""
        if not filter:
            return query
        
        for key, value in filter.items():
            if isinstance(value, dict):
                # Handle complex filters like {"$gt": 10}, {"$in": [1,2,3]}
                for op, op_value in value.items():
                    if op == "$eq":
                        query = query.eq(key, op_value)
                    elif op == "$ne":
                        query = query.neq(key, op_value)
                    elif op == "$gt":
                        query = query.gt(key, op_value)
                    elif op == "$gte":
                        query = query.gte(key, op_value)
                    elif op == "$lt":
                        query = query.lt(key, op_value)
                    elif op == "$lte":
                        query = query.lte(key, op_value)
                    elif op == "$in":
                        query = query.in_(key, op_value)
                    elif op == "$like":
                        query = query.like(key, op_value)
                    elif op == "$ilike":
                        query = query.ilike(key, op_value)
            else:
                # Simple equality filter
                query = query.eq(key, value)
        
        return query
    
    async def find_one(self, table: str, filter: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """Find a single document."""
        if not self.client:
            raise RuntimeError("Database not connected")
        
        actual_table = self._get_table_name(table)
        
        try:
            query = self.client.table(actual_table).select("*")
            query = self._build_filter_query(query, filter)
            print(f"Find one:\nQuery: {query}\nFilter: {filter}")
            response = query.limit(1).execute()
            print(f"Find one:\nResponse: {response}\nData: {response.data}")
            
            if response.data:
                return response.data[0]
            return None
        except Exception as e:
            logger.error(f"Error finding document in {actual_table}: {e}")
            raise
    
    async def find_many(self, table: str, filter: Dict[str, Any] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Find multiple documents."""
        if not self.client:
            raise RuntimeError("Database not connected")
        
        actual_table = self._get_table_name(table)
        
        try:
            query = self.client.table(actual_table).select("*")
            query = self._build_filter_query(query, filter)
            response = query.limit(limit).execute()
            
            return response.data or []
        except Exception as e:
            logger.error(f"Error finding documents in {actual_table}: {e}")
            raise
    
    async def insert(self, table: str, data: Dict[str, Any]) -> str:
        """Insert a single document and return its ID."""
        if not self.client:
            raise RuntimeError("Database not connected")
        
        actual_table = self._get_table_name(table)
        
        try:
            # Prepare data for insertion
            insert_data = data.copy()
            
            # Generate ID if not present
            if "_id" not in insert_data and "id" not in insert_data:
                insert_data["_id"] = str(uuid.uuid4())
            
            # Add timestamps
            now = datetime.now().isoformat()
            if "created_at" not in insert_data:
                insert_data["created_at"] = now
            if "updated_at" not in insert_data:
                insert_data["updated_at"] = now
            
            response = self.client.table(actual_table).insert(insert_data).execute()
            
            if response.data:
                # Return the ID of the inserted document
                inserted_doc = response.data[0]
                doc_id = inserted_doc.get("_id") or inserted_doc.get("id") or str(inserted_doc.get("uuid", ""))
                logger.debug(f"Inserted document with ID {doc_id} into {actual_table}")
                return str(doc_id)
            else:
                raise RuntimeError("Insert operation failed - no data returned")
                
        except Exception as e:
            logger.error(f"Error inserting document into {actual_table}: {e}")
            raise
    
    async def update(self, table: str, filter: Dict[str, Any], data: Dict[str, Any]) -> bool:
        """Update documents matching filter."""
        if not self.client:
            raise RuntimeError("Database not connected")
        
        actual_table = self._get_table_name(table)
        
        try:
            # Prepare update data
            update_data = data.copy()
            update_data["updated_at"] = datetime.now().isoformat()
            
            query = self.client.table(actual_table).update(update_data)
            query = self._build_filter_query(query, filter)
            response = query.execute()
            
            # Check if any rows were updated
            updated = response.data is not None and len(response.data) > 0
            if updated:
                logger.debug(f"Updated {len(response.data)} documents in {actual_table}")
            
            return updated
        except Exception as e:
            logger.error(f"Error updating documents in {actual_table}: {e}")
            raise
    
    async def delete(self, table: str, filter: Dict[str, Any]) -> int:
        """Delete documents matching filter and return count."""
        if not self.client:
            raise RuntimeError("Database not connected")
        
        actual_table = self._get_table_name(table)
        
        try:
            query = self.client.table(actual_table).delete()
            query = self._build_filter_query(query, filter)
            response = query.execute()
            
            # Count deleted documents
            deleted_count = len(response.data) if response.data else 0
            logger.debug(f"Deleted {deleted_count} documents from {actual_table}")
            return deleted_count
        except Exception as e:
            logger.error(f"Error deleting documents from {actual_table}: {e}")
            raise
    
    async def count(self, table: str, filter: Dict[str, Any] = None) -> int:
        """Count documents matching filter."""
        if not self.client:
            raise RuntimeError("Database not connected")
        
        actual_table = self._get_table_name(table)
        
        try:
            query = self.client.table(actual_table).select("*", count="exact")
            query = self._build_filter_query(query, filter)
            response = query.execute()
            
            return response.count or 0
        except Exception as e:
            logger.error(f"Error counting documents in {actual_table}: {e}")
            raise

# Database Factory


async def initialize_database(db_type: str = "memory", **kwargs) -> None:
    """Initialize the global database instance."""
    global database
    
    database = create_database(db_type, **kwargs)
    await database.connect()
    
    logger.info(f"Database initialized with type: {db_type}")

async def seed_database(database: Database) -> None:
    """Insert initial seed data into the database."""
    if not database:
        raise RuntimeError("Database not initialized")
    
    try:
        # Insert some sample users
        await database.insert("users", {
            "user_id": "admin-001",
            "name": "Admin User", 
            "email": "admin@example.com",
            "role": "admin",
            "status": "active"
        })
        
        await database.insert("users", {
            "user_id": "test-user-123",
            "name": "Test User",
            "email": "test@example.com", 
            "role": "user",
            "status": "active"
        })
        
        # Insert test_supa.py specific user (from TEST_CONFIG)
        await database.insert("users", {
            "user_id": "417b4875-e095-46d9-a46d-802dfef99d74",
            "name": "Test Supa User",
            "email": "test-supa@example.com", 
            "role": "user",
            "status": "active"
        })
        
        # Insert some sample requirements
        await database.insert("requirements", {
            "id": "req-gaokao-2024",
            "requirement_id": "req-gaokao-2024",
            "title": "Gaokao Score Estimation System",
            "description": "Build a system to estimate Gaokao scores based on subject performance",
            "category": "education",
            "status": "active",
            "goal": "Create a web application that can estimate total Gaokao scores based on individual subject scores in Math, English, and Physics"
        })
        
        await database.insert("requirements", {
            "id": "req-stock-analysis",
            "requirement_id": "req-stock-analysis",
            "title": "Stock Analysis Dashboard", 
            "description": "Create a dashboard for stock price analysis and trends",
            "category": "finance",
            "status": "active",
            "goal": "Build a comprehensive stock analysis workflow with price tracking and trend analysis"
        })
        
        # Insert test_supa.py specific requirement (from TEST_CONFIG)
        await database.insert("requirements", {
            "id": "04233f59-4670-452f-b823-c9d5560542bf",
            "requirement_id": "04233f59-4670-452f-b823-c9d5560542bf",
            "title": "Test Supa Requirement",
            "description": "Test requirement for Supabase workflow lifecycle testing",
            "category": "testing",
            "status": "active",
            "goal": "Analyze data and provide insights for workflow lifecycle testing"
        })
        
        # Insert some sample workflow templates
        await database.insert("workflow_templates", {
            "template_id": "template-web-app",
            "name": "Web Application Template",
            "description": "Standard template for web application workflows",
            "category": "web",
            "phases": ["setup", "design", "implementation", "testing", "deployment"]
        })
        
        logger.info("✅ Seed data inserted successfully")
        logger.info("✅ Test-specific data for test_supa.py included")
        
    except Exception as e:
        logger.error(f"❌ Error inserting seed data: {e}")

async def close_database() -> None:
    """Close the global database connection."""
    global database
    if database:
        await database.disconnect()
        logger.info("Database connection closed") 
        
def create_database(db_type: str = "memory", **kwargs) -> Database:
    """
    Create database instance.
    
    Supported types:
    - "memory": In-memory database
    - "supabase": Supabase database
    - "mongodb": MongoDB (to be implemented)
    - "postgresql": PostgreSQL (to be implemented)
    """
    db_type = db_type.lower()
    table_names = kwargs.get('table_names', None)
    
    if db_type in ["memory", "inmemory"]:
        return InMemoryDatabase(table_names=table_names)
    elif db_type == "supabase":
        url = kwargs.get('url')
        key = kwargs.get('key')
        return SupabaseDatabase(url=url, key=key, table_names=table_names)
    else:
        logger.warning(f"Unknown database type {db_type}, using in-memory")
        return InMemoryDatabase(table_names=table_names)

global requirement_database
requirement_database = create_database("supabase", url=os.environ.get("SUPABASE_URL_REQUIREMENT"), key=os.environ.get("SUPABASE_KEY_REQUIREMENT"))
asyncio.run(requirement_database.connect())

# # Global database instance
# asyncio.run(initialize_database("supabase", url=os.environ.get("SUPABASE_URL"), key=os.environ.get("SUPABASE_KEY"), table_names={"workflows": "create_x_workflows", "requirements": "create_x_project_requirements"}))

# In-memory database for testing
database = create_database("memory", table_names={"workflows": "create_x_workflows", "requirements": "create_x_project_requirements"})
print("🌱 Seeding database with initial data...")
asyncio.run(seed_database(database))
