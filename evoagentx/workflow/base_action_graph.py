from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseActionGraph(ABC):
    """Base interface for all action graphs"""
    
    @abstractmethod
    async def execute_async(self, prompt: str) -> Dict[str, Any]:
        """Execute the action graph asynchronously
        
        Args:
            prompt: The input prompt for the action graph
            
        Returns:
            Dict containing the execution results
        """
        pass
    
    @abstractmethod
    def validate_response(self, response: Dict[str, Any]) -> bool:
        """Validate the response from the action graph
        
        Args:
            response: The response to validate
            
        Returns:
            bool indicating if the response is valid
        """
        pass 