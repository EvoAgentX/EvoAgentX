"""
Database connection and models for EvoAgentX.
"""
# import asyncio
import logging
from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any # , Union
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
    """Custom ObjectId class with Pydantic compatibility.
    
    This class extends MongoDB's ObjectId to work seamlessly with Pydantic models,
    providing validation and serialization capability for ObjectIds.
    """
    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler: GetCoreSchemaHandler):
        """Define Pydantic core schema for validation.
        
        Args:
            source_type: The source type being validated
            handler: Pydantic's core schema handler
            
        Returns:
            A Pydantic core schema for the ObjectId
        """
        return core_schema.no_info_after_validator_function(cls.validate, core_schema.str_schema())

    @classmethod
    def validate(cls, v):
        """Validate if a value is a valid ObjectId.
        
        Args:
            v: The value to validate
            
        Returns:
            ObjectId: The validated ObjectId
            
        Raises:
            ValueError: If the value is not a valid ObjectId
        """
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid ObjectId")
        return ObjectId(v)

# Base model with ObjectId handling
class MongoBaseModel(BaseModel):
    """Base model for MongoDB documents with ObjectId support.
    
    Provides common functionality for all database models, including
    ObjectId handling and proper JSON serialization.
    
    Attributes:
        id: Optional ObjectId for the document, aliased as "_id"
    """
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
    """Status enum for agents.
    
    Defines possible states for an agent in the system.
    
    Attributes:
        CREATED: Initial state when an agent is first defined
        ACTIVE: Agent is active and available for use
        INACTIVE: Agent is temporarily disabled
        ERROR: Agent is in an error state
    """
    CREATED = "created"
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"

class WorkflowStatus(str, Enum):
    """Status enum for workflows.
    
    Defines possible states for a workflow in the system.
    
    Attributes:
        CREATED: Initial state when a workflow is first defined
        RUNNING: Workflow is currently in progress
        COMPLETED: Workflow has completed successfully
        FAILED: Workflow has failed to complete
        CANCELLED: Workflow was cancelled during execution
    """
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class ExecutionStatus(str, Enum):
    """Status enum for workflow executions.
    
    Defines possible states for a workflow execution instance.
    
    Attributes:
        PENDING: Execution is scheduled but not yet started
        RUNNING: Execution is currently in progress
        COMPLETED: Execution has completed successfully
        FAILED: Execution has failed with an error
        TIMEOUT: Execution exceeded its time limit
        CANCELLED: Execution was manually cancelled
    """
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"

# Database Models
class Agent(MongoBaseModel):
    """Agent model representing a configurable AI agent in the system.
    
    Stores configuration and state for a language model agent that can
    be used in workflows.
    
    Attributes:
        id: Unique identifier for the agent
        name: Display name of the agent
        description: Optional description of the agent's purpose
        config: Configuration parameters for the agent
        state: Current state of the agent
        runtime_params: Parameters used during execution
        status: Current status of the agent
        created_at: Timestamp when the agent was created
        updated_at: Timestamp when the agent was last updated
        created_by: User ID who created the agent
        tags: List of tags for categorization and filtering
    """
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
    """Workflow model representing a sequence of agent operations.
    
    Defines a process consisting of multiple steps that can be executed
    with specified inputs to produce outputs.
    
    Attributes:
        id: Unique identifier for the workflow
        name: Display name of the workflow
        description: Optional description of the workflow's purpose
        definition: Full workflow definition including steps and logic
        agent_ids: List of agent IDs used in this workflow
        status: Current status of the workflow
        created_at: Timestamp when the workflow was created
        updated_at: Timestamp when the workflow was last updated
        created_by: User ID who created the workflow
        tags: List of tags for categorization and filtering
        version: Version number of the workflow definition
    """
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
    """Log entry for a workflow execution.
    
    Records events that occur during workflow execution for debugging
    and monitoring purposes.
    
    Attributes:
        workflow_id: ID of the workflow being executed
        execution_id: ID of the specific execution instance
        step_id: Optional ID of the workflow step this log relates to
        agent_id: Optional ID of the agent this log relates to
        timestamp: Time when this log entry was created
        level: Log level (INFO, WARNING, ERROR, etc.)
        message: Log message text
        details: Additional structured details about the event
    """
    workflow_id: str
    execution_id: str
    step_id: Optional[str] = None
    agent_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    level: str = "INFO"
    message: str
    details: Dict[str, Any] = Field(default_factory=dict)

class WorkflowExecution(MongoBaseModel):
    """Workflow execution instance.
    
    Represents a specific execution of a workflow, tracking its progress,
    inputs, and results.
    
    Attributes:
        workflow_id: ID of the workflow being executed
        status: Current status of the execution
        start_time: When the execution started
        end_time: When the execution completed
        input_params: Input parameters provided for this execution
        results: Results produced by the execution
        created_by: User ID who initiated the execution
        step_results: Results from individual workflow steps
        current_step: ID of the step currently being executed
        error_message: Error message if execution failed
        created_at: Timestamp when the execution was created
    """
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
    """MongoDB database client and collection management.
    
    Provides static methods for connecting to MongoDB and accessing
    collections. Handles connection lifecycle and index creation.
    """
    client: AsyncIOMotorClient = None
    db = None
    
    # Collections
    agents = None
    workflows = None
    executions = None
    logs = None
    
    @classmethod
    async def connect(cls):
        """Connect to MongoDB and initialize collections.
        
        Establishes a connection to the MongoDB server using the URL from settings,
        sets up collection references, and creates necessary indexes.
        
        Returns:
            None
        """
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
        """Disconnect from MongoDB.
        
        Closes the MongoDB client connection.
        
        Returns:
            None
        """
        if cls.client:
            cls.client.close()
            logger.info("Disconnected from MongoDB")
    
    @classmethod
    async def _create_indexes(cls):
        """Create indexes for all collections.
        
        Sets up database indexes to optimize queries across all collections.
        Includes text indexes for search functionality and regular indexes
        for common query patterns.
        
        Returns:
            None
        """
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