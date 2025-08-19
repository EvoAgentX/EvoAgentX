"""
Message Storage Service for Socket Management.
Stores all socket messages for debugging, replay, and analysis.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

class MessageStore:
    """
    In-memory message store for socket communications.
    Stores messages per project with automatic cleanup.
    """
    
    def __init__(self, max_messages_per_project: int = 1000, cleanup_after_hours: int = 24):
        """
        Initialize message store.
        
        Args:
            max_messages_per_project: Maximum messages to store per project
            cleanup_after_hours: Hours after which to cleanup old messages
        """
        self.max_messages_per_project = max_messages_per_project
        self.cleanup_after_hours = cleanup_after_hours
        
        # Store messages per project: {project_short_id: deque([message, ...])}
        self.project_messages: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_messages_per_project))
        
        # Store message metadata: {project_short_id: {"first_message": datetime, "last_message": datetime, "total_count": int}}
        self.project_metadata: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "first_message": None,
            "last_message": None, 
            "total_count": 0,
            "message_types": defaultdict(int)
        })
        
        logger.info(f"MessageStore initialized (max_messages: {max_messages_per_project}, cleanup_after: {cleanup_after_hours}h)")
    
    def store_outgoing_message(self, project_short_id: str, message: Dict[str, Any], direction: str = "outgoing"):
        """
        Store an outgoing message (server -> client).
        
        Args:
            project_short_id: Project identifier
            message: Message data
            direction: Message direction ("outgoing" or "incoming")
        """
        self._store_message(project_short_id, message, direction)
    
    def store_incoming_message(self, project_short_id: str, message: Dict[str, Any], direction: str = "incoming"):
        """
        Store an incoming message (client -> server).
        
        Args:
            project_short_id: Project identifier  
            message: Message data
            direction: Message direction ("outgoing" or "incoming")
        """
        self._store_message(project_short_id, message, direction)
    
    def _store_message(self, project_short_id: str, message: Dict[str, Any], direction: str):
        """Internal method to store a message."""
        try:
            # Create message record
            message_record = {
                "timestamp": datetime.now().isoformat(),
                "direction": direction,
                "message": message.copy(),
                "message_type": message.get("type", "unknown"),
                "size_bytes": len(json.dumps(message, default=str))
            }
            
            # Store message
            self.project_messages[project_short_id].append(message_record)
            
            # Update metadata
            metadata = self.project_metadata[project_short_id]
            metadata["total_count"] += 1
            metadata["last_message"] = datetime.now()
            if metadata["first_message"] is None:
                metadata["first_message"] = datetime.now()
            
            # Track message types
            msg_type = message.get("type", "unknown")
            metadata["message_types"][msg_type] += 1
            
        except Exception as e:
            logger.error(f"Failed to store message for {project_short_id}: {e}")
    
    def get_project_messages(self, project_short_id: str, limit: Optional[int] = None, 
                           message_type: Optional[str] = None, since: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """
        Get messages for a project with optional filtering.
        
        Args:
            project_short_id: Project identifier
            limit: Maximum number of messages to return (most recent first)
            message_type: Filter by message type
            since: Only return messages after this datetime
            
        Returns:
            List of message records
        """
        if project_short_id not in self.project_messages:
            return []
        
        messages = list(self.project_messages[project_short_id])
        
        # Filter by message type
        if message_type:
            messages = [msg for msg in messages if msg.get("message", {}).get("type") == message_type]
        
        # Filter by time
        if since:
            messages = [msg for msg in messages if datetime.fromisoformat(msg["timestamp"]) >= since]
        
        # Sort by timestamp (most recent first)
        messages.sort(key=lambda x: x["timestamp"], reverse=True)
        
        # Apply limit
        if limit:
            messages = messages[:limit]
        
        return messages
    
    def get_project_stats(self, project_short_id: str) -> Dict[str, Any]:
        """Get statistics for a project's messages."""
        if project_short_id not in self.project_metadata:
            return {"error": "Project not found"}
        
        metadata = self.project_metadata[project_short_id]
        messages = self.project_messages[project_short_id]
        
        # Calculate size statistics
        total_size = sum(msg.get("size_bytes", 0) for msg in messages)
        avg_size = total_size / len(messages) if messages else 0
        
        # Calculate message rate (messages per minute)
        if metadata["first_message"] and metadata["last_message"]:
            duration = (metadata["last_message"] - metadata["first_message"]).total_seconds()
            msg_rate = (metadata["total_count"] / duration * 60) if duration > 0 else 0
        else:
            msg_rate = 0
        
        return {
            "project_short_id": project_short_id,
            "total_messages": len(messages),
            "total_messages_ever": metadata["total_count"],
            "first_message": metadata["first_message"].isoformat() if metadata["first_message"] else None,
            "last_message": metadata["last_message"].isoformat() if metadata["last_message"] else None,
            "message_types": dict(metadata["message_types"]),
            "total_size_bytes": total_size,
            "average_message_size": avg_size,
            "messages_per_minute": msg_rate
        }
    
    def get_all_projects_stats(self) -> Dict[str, Any]:
        """Get statistics for all projects."""
        stats = {
            "total_projects": len(self.project_messages),
            "active_projects": [],
            "total_messages": 0,
            "projects": {}
        }
        
        for project_short_id in self.project_messages.keys():
            project_stats = self.get_project_stats(project_short_id)
            stats["projects"][project_short_id] = project_stats
            stats["total_messages"] += project_stats["total_messages"]
            
            # Consider active if messaged in last hour
            if project_stats["last_message"]:
                last_msg = datetime.fromisoformat(project_stats["last_message"])
                if (datetime.now() - last_msg).total_seconds() < 3600:
                    stats["active_projects"].append(project_short_id)
        
        return stats
    
    def cleanup_old_messages(self):
        """Clean up old messages based on cleanup_after_hours setting."""
        cutoff_time = datetime.now() - timedelta(hours=self.cleanup_after_hours)
        cleaned_projects = []
        
        for project_short_id in list(self.project_messages.keys()):
            messages = self.project_messages[project_short_id]
            
            # Remove old messages
            original_count = len(messages)
            # Convert deque to list, filter, then back to deque
            recent_messages = [
                msg for msg in messages 
                if datetime.fromisoformat(msg["timestamp"]) >= cutoff_time
            ]
            
            if len(recent_messages) < original_count:
                # Replace deque with filtered messages
                self.project_messages[project_short_id] = deque(recent_messages, maxlen=self.max_messages_per_project)
                cleaned_projects.append(project_short_id)
                
            # Remove project entirely if no recent messages
            if not recent_messages:
                del self.project_messages[project_short_id]
                del self.project_metadata[project_short_id]
        
        if cleaned_projects:
            logger.info(f"Cleaned up old messages for {len(cleaned_projects)} projects")
    
    def export_project_messages(self, project_short_id: str, format: str = "json") -> str:
        """
        Export project messages in specified format.
        
        Args:
            project_short_id: Project identifier
            format: Export format ("json", "csv", "txt")
            
        Returns:
            Formatted string of messages
        """
        messages = self.get_project_messages(project_short_id)
        
        if format == "json":
            return json.dumps(messages, indent=2, default=str)
        
        elif format == "csv":
            if not messages:
                return "timestamp,direction,message_type,content,size_bytes\n"
            
            csv_lines = ["timestamp,direction,message_type,content,size_bytes"]
            for msg in messages:
                csv_lines.append(f"{msg['timestamp']},{msg['direction']},{msg['message_type']},{msg['message'].get('data', {}).get('content', '')},{msg['size_bytes']}")
            return "\n".join(csv_lines)
        
        elif format == "txt":
            if not messages:
                return "No messages found.\n"
            
            txt_lines = [f"Messages for project: {project_short_id}\n"]
            for msg in messages:
                txt_lines.append(f"[{msg['timestamp']}] {msg['direction'].upper()}: {msg['message_type']}")
                txt_lines.append(f"  Content: {msg['message'].get('data', {}).get('content', 'N/A')}")
                txt_lines.append("")
            return "\n".join(txt_lines)
        
        else:
            raise ValueError(f"Unsupported format: {format}")

# Global message store instance
message_store = MessageStore()
