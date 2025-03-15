"""
Business logic for agents, workflows, and executions.
"""
import logging
import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional, Union, Tuple
from bson import ObjectId

from evoagentx.app.db import (
    Database, Agent, Workflow, WorkflowExecution, ExecutionLog,
    AgentStatus, WorkflowStatus, ExecutionStatus
)
from evoagentx.app.schemas import (
    AgentCreate, AgentUpdate, WorkflowCreate, WorkflowUpdate, 
    ExecutionCreate, PaginationParams, SearchParams
)

logger = logging.getLogger(__name__)

# Agent Service
class AgentService:
    @staticmethod
    async def create_agent(agent_data: AgentCreate, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Create a new agent."""
        agent_dict = agent_data.dict()
        agent_dict["created_by"] = user_id
        agent_dict["created_at"] = datetime.utcnow()
        agent_dict["updated_at"] = agent_dict["created_at"]
        agent_dict["status"] = AgentStatus.CREATED
        
        # Validate agent exists with the same name
        existing_agent = await Database.agents.find_one({"name": agent_dict["name"]})
        if existing_agent:
            raise ValueError(f"Agent with name '{agent_dict['name']}' already exists")
        
        result = await Database.agents.insert_one(agent_dict)
        agent_dict["_id"] = result.inserted_id
        
        logger.info(f"Created agent {agent_dict['name']} with ID {result.inserted_id}")
        
        return agent_dict
    
    @staticmethod
    async def get_agent(agent_id: str) -> Optional[Dict[str, Any]]:
        """Get an agent by ID."""
        if not ObjectId.is_valid(agent_id):
            raise ValueError(f"Invalid agent ID: {agent_id}")
            
        agent = await Database.agents.find_one({"_id": ObjectId(agent_id)})
        return agent
    
    @staticmethod
    async def get_agent_by_name(name: str) -> Optional[Dict[str, Any]]:
        """Get an agent by name."""
        return await Database.agents.find_one({"name": name})
    
    @staticmethod
    async def update_agent(agent_id: str, agent_data: AgentUpdate) -> Optional[Dict[str, Any]]:
        """Update an agent."""
        if not ObjectId.is_valid(agent_id):
            raise ValueError(f"Invalid agent ID: {agent_id}")
            
        agent = await Database.agents.find_one({"_id": ObjectId(agent_id)})
        if not agent:
            return None
        
        update_data = agent_data.dict(exclude_unset=True)
        update_data["updated_at"] = datetime.utcnow()
        
        if "name" in update_data:
            # Check if the new name already exists
            existing = await Database.agents.find_one({
                "name": update_data["name"],
                "_id": {"$ne": ObjectId(agent_id)}
            })
            if existing:
                raise ValueError(f"Agent with name '{update_data['name']}' already exists")
        
        await Database.agents.update_one(
            {"_id": ObjectId(agent_id)},
            {"$set": update_data}
        )
        
        updated_agent = await Database.agents.find_one({"_id": ObjectId(agent_id)})
        logger.info(f"Updated agent {agent_id}")
        
        return updated_agent
    
    @staticmethod
    async def delete_agent(agent_id: str) -> bool:
        """Delete an agent."""
        if not ObjectId.is_valid(agent_id):
            raise ValueError(f"Invalid agent ID: {agent_id}")
            
        # Check if agent is used in any workflows
        workflow_count = await Database.workflows.count_documents({"agent_ids": agent_id})
        if workflow_count > 0:
            raise ValueError(f"Cannot delete agent {agent_id} as it is used in {workflow_count} workflows")
        
        result = await Database.agents.delete_one({"_id": ObjectId(agent_id)})
        if result.deleted_count:
            logger.info(f"Deleted agent {agent_id}")
            return True
        return False
    
    @staticmethod
    async def list_agents(
        params: PaginationParams, 
        search: Optional[SearchParams] = None
    ) -> Tuple[List[Dict[str, Any]], int]:
        """List agents with pagination and search."""
        query = {}
        
        if search:
            if search.query:
                query["$text"] = {"$search": search.query}
            
            if search.tags:
                query["tags"] = {"$all": search.tags}
            
            if search.status:
                query["status"] = search.status
            
            if search.start_date and search.end_date:
                query["created_at"] = {
                    "$gte": search.start_date,
                    "$lte": search.end_date
                }
            elif search.start_date:
                query["created_at"] = {"$gte": search.start_date}
            elif search.end_date:
                query["created_at"] = {"$lte": search.end_date}
        
        total = await Database.agents.count_documents(query)
        
        cursor = Database.agents.find(query)\
            .sort("created_at", -1)\
            .skip(params.skip)\
            .limit(params.limit)
        
        agents = await cursor.to_list(length=params.limit)
        return agents, total

# Workflow Service
class WorkflowService:
    @staticmethod
    async def create_workflow(workflow_data: WorkflowCreate, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Create a new workflow."""
        workflow_dict = workflow_data.dict()
        workflow_dict["created_by"] = user_id
        workflow_dict["created_at"] = datetime.utcnow()
        workflow_dict["updated_at"] = workflow_dict["created_at"]
        workflow_dict["status"] = WorkflowStatus.CREATED
        workflow_dict["version"] = 1
        
        # Extract agent IDs from the workflow definition
        agent_ids = set()
        
        # Extract agent IDs from steps
        steps = workflow_dict["definition"].get("steps", [])
        for step in steps:
            if "agent_id" in step:
                agent_id = step["agent_id"]
                # Validate agent exists
                agent = await AgentService.get_agent(agent_id)
                if not agent:
                    raise ValueError(f"Agent with ID {agent_id} does not exist")
                agent_ids.add(agent_id)
        
        workflow_dict["agent_ids"] = list(agent_ids)
        
        # Check for existing workflow with the same name
        existing = await Database.workflows.find_one({"name": workflow_dict["name"]})
        if existing:
            raise ValueError(f"Workflow with name '{workflow_dict['name']}' already exists")
        
        result = await Database.workflows.insert_one(workflow_dict)
        workflow_dict["_id"] = result.inserted_id
        
        logger.info(f"Created workflow {workflow_dict['name']} with ID {result.inserted_id}")
        
        return workflow_dict
    
    @staticmethod
    async def get_workflow(workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get a workflow by ID."""
        if not ObjectId.is_valid(workflow_id):
            raise ValueError(f"Invalid workflow ID: {workflow_id}")
        workflow = await Database.workflows.find_one({"_id": ObjectId(workflow_id)})
        return workflow
    
    @staticmethod
    async def get_workflow_by_name(name: str) -> Optional[Dict[str, Any]]:
        """Get a workflow by name."""
        return await Database.workflows.find_one({"name": name})
    
    @staticmethod
    async def update_workflow(workflow_id: str, workflow_data: WorkflowUpdate) -> Optional[Dict[str, Any]]:
        """Update a workflow."""
        if not ObjectId.is_valid(workflow_id):
            raise ValueError(f"Invalid workflow ID: {workflow_id}")
            
        workflow = await Database.workflows.find_one({"_id": ObjectId(workflow_id)})
        if not workflow:
            return None
        
        update_data = workflow_data.dict(exclude_unset=True)
        update_data["updated_at"] = datetime.utcnow()
        
        # Update version if definition changes
        if "definition" in update_data:
            update_data["version"] = workflow.get("version", 1) + 1
            
            # Extract agent IDs from the updated workflow definition
            agent_ids = set()
            steps = update_data["definition"].get("steps", [])
            for step in steps:
                if "agent_id" in step:
                    agent_id = step["agent_id"]
                    # Validate agent exists
                    agent = await AgentService.get_agent(agent_id)
                    if not agent:
                        raise ValueError(f"Agent with ID {agent_id} does not exist")
                    agent_ids.add(agent_id)
            
            update_data["agent_ids"] = list(agent_ids)
        
        # Check for name conflict if name is being updated
        if "name" in update_data:
            existing = await Database.workflows.find_one({
                "name": update_data["name"],
                "_id": {"$ne": ObjectId(workflow_id)}
            })
            if existing:
                raise ValueError(f"Workflow with name '{update_data['name']}' already exists")
        
        await Database.workflows.update_one(
            {"_id": ObjectId(workflow_id)},
            {"$set": update_data}
        )
        
        updated_workflow = await Database.workflows.find_one({"_id": ObjectId(workflow_id)})
        logger.info(f"Updated workflow {workflow_id}")
        
        return updated_workflow
    
    @staticmethod
    async def delete_workflow(workflow_id: str) -> bool:
        """Delete a workflow."""
        if not ObjectId.is_valid(workflow_id):
            raise ValueError(f"Invalid workflow ID: {workflow_id}")
        
        # Check if workflow has any ongoing or recent executions
        recent_executions = await Database.executions.count_documents({
            "workflow_id": workflow_id,
            "status": {"$in": [
                ExecutionStatus.PENDING, 
                ExecutionStatus.RUNNING
            ]}
        })
        
        if recent_executions > 0:
            raise ValueError(f"Cannot delete workflow {workflow_id} with {recent_executions} active executions")

        result = await Database.workflows.delete_one({"_id": ObjectId(workflow_id)})
        if result.deleted_count:
            # Delete associated execution logs
            await Database.logs.delete_many({"workflow_id": workflow_id})
            await Database.executions.delete_many({"workflow_id": workflow_id})
            
            logger.info(f"Deleted workflow {workflow_id}")
            return True
        return False
    
    @staticmethod
    async def list_workflows(
        params: PaginationParams, 
        search: Optional[SearchParams] = None
    ) -> Tuple[List[Dict[str, Any]], int]:
        """List workflows with pagination and search."""
        query = {}
        
        if search:
            if search.query:
                query["$text"] = {"$search": search.query}
            
            if search.tags:
                query["tags"] = {"$all": search.tags}
            
            if search.status:
                query["status"] = search.status
            
            if search.start_date and search.end_date:
                query["created_at"] = {
                    "$gte": search.start_date,
                    "$lte": search.end_date
                }
            elif search.start_date:
                query["created_at"] = {"$gte": search.start_date}
            elif search.end_date:
                query["created_at"] = {"$lte": search.end_date}
        
        total = await Database.workflows.count_documents(query)
        
        cursor = Database.workflows.find(query)\
            .sort("created_at", -1)\
            .skip(params.skip)\
            .limit(params.limit)
        
        workflows = await cursor.to_list(length=params.limit)
        return workflows, total

# Workflow Execution Service
class WorkflowExecutionService:
    @staticmethod
    async def create_execution(execution_data: ExecutionCreate, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Create a new workflow execution."""
        # Validate workflow exists
        workflow = await WorkflowService.get_workflow(execution_data.workflow_id)
        if not workflow:
            raise ValueError(f"Workflow {execution_data.workflow_id} not found")
        
        # Prepare execution document
        execution_dict = {
            "workflow_id": execution_data.workflow_id,
            "status": ExecutionStatus.PENDING,
            "start_time": datetime.utcnow(),
            "input_params": execution_data.input_params,
            "created_by": user_id,
            "created_at": datetime.utcnow(),
            "step_results": {},
            "current_step": None,
            "results": {},
            "error_message": None
        }
        
        # Insert execution record
        result = await Database.executions.insert_one(execution_dict)
        execution_dict["_id"] = result.inserted_id
        
        logger.info(f"Created workflow execution {result.inserted_id}")
        
        # Optional: Queue execution for async processing
        # This would typically use a task queue like Celery
        # await execute_workflow_async.delay(execution_dict)
        
        return execution_dict
    
    @staticmethod
    async def get_execution(execution_id: str) -> Optional[Dict[str, Any]]:
        """Get a workflow execution by ID."""
        if not ObjectId.is_valid(execution_id):
            raise ValueError(f"Invalid execution ID: {execution_id}")
            
        execution = await Database.executions.find_one({"_id": ObjectId(execution_id)})
        return execution
    
    @staticmethod
    async def update_execution_status(execution_id: str, status: ExecutionStatus, error_message: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Update execution status."""
        if not ObjectId.is_valid(execution_id):
            raise ValueError(f"Invalid execution ID: {execution_id}")
        
        update_data = {
            "status": status,
            "updated_at": datetime.utcnow()
        }
        
        if status in [ExecutionStatus.COMPLETED, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED]:
            update_data["end_time"] = datetime.utcnow()
        
        if error_message:
            update_data["error_message"] = error_message
        
        result = await Database.executions.find_one_and_update(
            {"_id": ObjectId(execution_id)},
            {"$set": update_data},
            return_document=True
        )
        
        return result
    
    @staticmethod
    async def list_executions(
        workflow_id: Optional[str] = None,
        params: PaginationParams = PaginationParams(), 
        search: Optional[SearchParams] = None
    ) -> Tuple[List[Dict[str, Any]], int]:
        """List workflow executions with pagination and search."""
        query = {}
        
        if workflow_id:
            query["workflow_id"] = workflow_id
        
        if search:
            if search.status:
                query["status"] = search.status
            
            if search.start_date and search.end_date:
                query["created_at"] = {
                    "$gte": search.start_date,
                    "$lte": search.end_date
                }
            elif search.start_date:
                query["created_at"] = {"$gte": search.start_date}
            elif search.end_date:
                query["created_at"] = {"$lte": search.end_date}
        
        total = await Database.executions.count_documents(query)
        
        cursor = Database.executions.find(query)\
            .sort("created_at", -1)\
            .skip(params.skip)\
            .limit(params.limit)
        
        executions = await cursor.to_list(length=params.limit)
        return executions, total
    
    @staticmethod
    async def log_execution_event(
        workflow_id: str, 
        execution_id: str, 
        message: str,
        step_id: Optional[str] = None, 
        agent_id: Optional[str] = None, 
        level: str = "INFO", 
        details: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Log an event in a workflow execution."""
        log_entry = {
            "workflow_id": workflow_id,
            "execution_id": execution_id,
            "step_id": step_id,
            "agent_id": agent_id,
            "timestamp": datetime.utcnow(),
            "level": level,
            "message": message,
            "details": details or {}
        }
        
        result = await Database.logs.insert_one(log_entry)
        log_entry["_id"] = result.inserted_id
        
        return log_entry
    
    @staticmethod
    async def get_execution_logs(
        execution_id: str, 
        params: PaginationParams = PaginationParams()
    ) -> Tuple[List[Dict[str, Any]], int]:
        """Retrieve logs for a specific execution."""
        query = {"execution_id": execution_id}
        
        total = await Database.logs.count_documents(query)
        
        cursor = Database.logs.find(query)\
            .sort("timestamp", 1)\
            .skip(params.skip)\
            .limit(params.limit)
        
        logs = await cursor.to_list(length=params.limit)
        return logs, total