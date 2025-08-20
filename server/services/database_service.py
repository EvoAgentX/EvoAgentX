"""
Database service for EvoAgentX server.
Handles database operations and business logic.
"""

import json
import os
from datetime import datetime
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

from ..database.db import database, requirement_database

load_dotenv(os.path.join(os.path.dirname(__file__), '../config/app.env'))


class DatabaseService:
    """Service class for database operations and business logic"""
    
    @staticmethod
    async def get_workflow(workflow_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a workflow by ID"""
        try:
            workflow = await database.find_one("workflows", {"id": workflow_id})
            return workflow
        except Exception as e:
            raise Exception(f"Failed to retrieve workflow: {str(e)}")
    
    @staticmethod
    async def update_workflow(workflow_id: str, updates: Dict[str, Any]) -> bool:
        """Update a workflow with new data"""
        try:
            updates["updated_at"] = datetime.now()
            result = await database.update(
                "workflows", 
                {"id": workflow_id}, 
                updates
            )
            return result
        except Exception as e:
            raise Exception(f"Failed to update workflow: {str(e)}")
    
    @staticmethod
    async def update_workflow_status(workflow_id: str, status: str, **kwargs) -> bool:
        """Update workflow status and other fields"""
        try:
            updates = {"status": status, "updated_at": datetime.now(), **kwargs}
            result = await database.update(
                "workflows", 
                {"id": workflow_id}, 
                updates
            )
            return result
        except Exception as e:
            raise Exception(f"Failed to update workflow status: {str(e)}")
    
    @staticmethod
    async def list_workflows(
        skip: int = 0, 
        limit: int = 100, 
        status: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """List workflows with optional filtering"""
        try:
            query = {}
            if status:
                query["status"] = status
            
            workflows = await database.find_many(
                "workflows", 
                query, 
                limit=limit
            )
            # Apply skip manually since find_many doesn't support skip parameter
            if skip > 0:
                workflows = workflows[skip:]
            return workflows
        except Exception as e:
            raise Exception(f"Failed to list workflows: {str(e)}")
    
    @staticmethod
    async def delete_workflow(workflow_id: str) -> bool:
        """Delete a workflow by ID"""
        try:
            result = await database.delete("workflows", {"id": workflow_id})
            return result
        except Exception as e:
            raise Exception(f"Failed to delete workflow: {str(e)}")
    
    @staticmethod
    async def store_workflow_graph(workflow_id: str, workflow_graph: Dict[str, Any]) -> bool:
        """Store workflow graph in the database"""
        try:
            result = await database.update(
                "workflows",
                {"id": workflow_id},
                {
                    "workflow_graph": workflow_graph,
                    "status": "generated",
                    "updated_at": datetime.now()
                }
            )
            return result
        except Exception as e:
            raise Exception(f"Failed to store workflow graph: {str(e)}")
    
    @staticmethod
    async def store_execution_result(workflow_id: str, execution_result: Dict[str, Any]) -> bool:
        """Store workflow execution results"""
        try:
            result = await database.update(
                "workflows",
                {"id": workflow_id},
                {
                    "execution_result": execution_result,
                    "status": "completed",
                    "updated_at": datetime.now()
                }
            )
            return result
        except Exception as e:
            raise Exception(f"Failed to store execution result: {str(e)}")
    
    @staticmethod
    async def get_requirement_database() -> Dict[str, Any]:
        """Get the requirement database for workflow generation"""
        try:
            requirements = await requirement_database.find("requirements", {})
            return requirements
        except Exception as e:
            raise Exception(f"Failed to get requirement database: {str(e)}")
    
    @staticmethod
    async def add_requirement(requirement: Dict[str, Any]) -> bool:
        """Add a new requirement to the requirement database"""
        try:
            requirement["created_at"] = datetime.now()
            result = await requirement_database.insert("requirements", requirement)
            return result
        except Exception as e:
            raise Exception(f"Failed to add requirement: {str(e)}")
    
    @staticmethod
    async def search_requirements(query: str) -> List[Dict[str, Any]]:
        """Search requirements by query"""
        try:
            # Simple text search - can be enhanced with vector search later
            requirements = await requirement_database.find(
                "requirements", 
                {"$text": {"$search": query}}
            )
            return requirements
        except Exception as e:
            raise Exception(f"Failed to search requirements: {str(e)}")
