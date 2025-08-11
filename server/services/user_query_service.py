"""
User query service for EvoAgentX server.
Handles user query processing and routing.
"""

import json
from typing import Dict, Any, List, Optional
from ..components.user_query_router.user_query_router import UserQueryRouter
from ..database.db import database


class UserQueryService:
    """Service class for processing user queries"""
    
    def __init__(self):
        self.query_router = UserQueryRouter()
    
    async def process_user_query(
        self, 
        query: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process a user query and return appropriate response
        
        Args:
            query: The user's query string
            context: Optional context information
            
        Returns:
            Dict containing the processed response
        """
        try:
            # Route the query to appropriate handler
            routed_response = await self.query_router.route_query(query, context)
            
            # Process the response
            processed_response = await self._process_routed_response(routed_response, context)
            
            return {
                "success": True,
                "query": query,
                "response": processed_response,
                "routing_info": routed_response.get("routing_info", {})
            }
            
        except Exception as e:
            return {
                "success": False,
                "query": query,
                "error": str(e),
                "response": None
            }
    
    async def _process_routed_response(
        self, 
        routed_response: Dict[str, Any], 
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process the routed response based on routing type
        
        Args:
            routed_response: Response from query router
            context: Optional context information
            
        Returns:
            Processed response
        """
        routing_type = routed_response.get("routing_type")
        
        if routing_type == "workflow_setup":
            return await self._handle_workflow_setup(routed_response, context)
        elif routing_type == "workflow_generation":
            return await self._handle_workflow_generation(routed_response, context)
        elif routing_type == "workflow_execution":
            return await self._handle_workflow_execution(routed_response, context)
        elif routing_type == "information_query":
            return await self._handle_information_query(routed_response, context)
        else:
            return {
                "type": "general_response",
                "content": routed_response.get("response", "Query processed successfully")
            }
    
    async def _handle_workflow_setup(
        self, 
        routed_response: Dict[str, Any], 
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Handle workflow setup queries"""
        return {
            "type": "workflow_setup",
            "action": "setup_required",
            "details": routed_response.get("details", {}),
            "next_steps": ["Provide project details", "Specify requirements"]
        }
    
    async def _handle_workflow_generation(
        self, 
        routed_response: Dict[str, Any], 
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Handle workflow generation queries"""
        return {
            "type": "workflow_generation",
            "action": "generation_required",
            "details": routed_response.get("details", {}),
            "next_steps": ["Review generated workflow", "Modify if needed"]
        }
    
    async def _handle_workflow_execution(
        self, 
        routed_response: Dict[str, Any], 
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Handle workflow execution queries"""
        return {
            "type": "workflow_execution",
            "action": "execution_required",
            "details": routed_response.get("details", {}),
            "next_steps": ["Provide execution inputs", "Monitor progress"]
        }
    
    async def _handle_information_query(
        self, 
        routed_response: Dict[str, Any], 
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Handle information queries"""
        return {
            "type": "information_query",
            "content": routed_response.get("response", "Information retrieved"),
            "sources": routed_response.get("sources", []),
            "confidence": routed_response.get("confidence", 0.8)
        }
    
    async def get_query_history(
        self, 
        user_id: Optional[str] = None, 
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Get query history for a user
        
        Args:
            user_id: Optional user ID to filter queries
            limit: Maximum number of queries to return
            
        Returns:
            List of query history entries
        """
        try:
            query = {}
            if user_id:
                query["user_id"] = user_id
            
            # This would need to be implemented in the database layer
            # For now, return empty list
            return []
            
        except Exception as e:
            return []
    
    async def store_query(
        self, 
        query: str, 
        response: Dict[str, Any], 
        user_id: Optional[str] = None
    ) -> bool:
        """
        Store a query and its response
        
        Args:
            query: The user query
            response: The processed response
            user_id: Optional user ID
            
        Returns:
            True if stored successfully
        """
        try:
            # This would need to be implemented in the database layer
            # For now, return True
            return True
            
        except Exception as e:
            return False

