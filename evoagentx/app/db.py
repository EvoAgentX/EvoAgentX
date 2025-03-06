"""
Database connection and models for EvoAgentX.
"""
import asyncio
import logging
from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any, Union
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import ASCENDING, TEXT
from pydantic_core import core_schema
from bson import ObjectId
from pydantic import GetCoreSchemaHandler
from pydantic import Field, BaseModel
from evoagentx.app.config import settings

# Setup logger
logger = logging.getLogger(__name__)

# Custom PyObjectId for MongoDB ObjectId compatibility with Pydantic
class PyObjectId(ObjectId):
    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler: GetCoreSchemaHandler):
        return core_schema.no_info_after_validator_function(cls.validate, core_schema.str_schema())

    @classmethod
    def validate(cls, v):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid ObjectId")
        return ObjectId(v)

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

# Database client
class Database:
    client: AsyncIOMotorClient = None
    db = None
    
    # Collections
    agents = None
    workflows = None
    executions = None
    logs = None
    
    @classmethod
    async def connect(cls):
        """Connect to MongoDB"""
        logger.info(f"Connecting to MongoDB at {settings.MONGODB_URL}...")
        cls.client = AsyncIOMotorClient(settings.MONGODB_URL)
        cls.db = cls.client[settings.MONGODB_DB_NAME]
        
        # Set up collections
        cls.agents = cls.db.agents
        cls.workflows = cls.db.workflows
        cls.executions = cls.db.workflow_executions
        cls.logs = cls.db.execution_logs
        
        # Create indexes
        await cls._create_indexes()
        
        logger.info("Connected to MongoDB successfully")
    
    @classmethod
    async def disconnect(cls):
        """Disconnect from MongoDB"""
        if cls.client:
            cls.client.close()
            logger.info("Disconnected from MongoDB")
    
    @classmethod
    async def _create_indexes(cls):
        """Create indexes for collections"""
        # Agent indexes
        await cls.agents.create_index([("name", ASCENDING)], unique=True)
        await cls.agents.create_index([("name", TEXT), ("description", TEXT)])
        await cls.agents.create_index([("created_at", ASCENDING)])
        await cls.agents.create_index([("tags", ASCENDING)])
        
        # Workflow indexes
        await cls.workflows.create_index([("name", ASCENDING)])
        await cls.workflows.create_index([("name", TEXT), ("description", TEXT)])
        await cls.workflows.create_index([("created_at", ASCENDING)])
        await cls.workflows.create_index([("agent_ids", ASCENDING)])
        await cls.workflows.create_index([("tags", ASCENDING)])
        
        # Execution indexes
        await cls.executions.create_index([("workflow_id", ASCENDING)])
        await cls.executions.create_index([("created_at", ASCENDING)])
        await cls.executions.create_index([("status", ASCENDING)])
        
        # Log indexes
        await cls.logs.create_index([("execution_id", ASCENDING)])
        await cls.logs.create_index([("timestamp", ASCENDING)])
        await cls.logs.create_index([("workflow_id", ASCENDING), ("execution_id", ASCENDING)])