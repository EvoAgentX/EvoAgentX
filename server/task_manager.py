from typing import Dict, Any
from datetime import datetime

# In-memory storage for demo purposes
# In production, use a proper database/cache system
tasks = {}
stream_tasks = {}

# Client session storage
client_sessions = {}
client_updates = {}

def store_task_result(task_id: str, response: Any):
    """Store a completed task result"""
    tasks[task_id] = response

def get_task(task_id: str) -> Dict[str, Any]:
    """Get a task by ID"""
    return tasks.get(task_id)

def create_stream_task(task_id: str, config: Dict[str, Any]):
    """Create a new streaming task"""
    stream_tasks[task_id] = {
        "updates": [],
        "completed": False,
        "config": config
    }

def update_stream_task(task_id: str, update: Dict[str, Any]):
    """Add an update to a streaming task"""
    if task_id in stream_tasks:
        stream_tasks[task_id]["updates"].append(update)

def complete_stream_task(task_id: str):
    """Mark a streaming task as completed"""
    if task_id in stream_tasks:
        stream_tasks[task_id]["completed"] = True

def get_stream_task(task_id: str) -> Dict[str, Any]:
    """Get a streaming task by ID"""
    return stream_tasks.get(task_id)

def get_stream_task_updates(task_id: str, start_index: int = 0) -> list:
    """Get updates for a streaming task starting from an index"""
    if task_id in stream_tasks:
        return stream_tasks[task_id]["updates"][start_index:]
    return []

def is_stream_task_completed(task_id: str) -> bool:
    """Check if a streaming task is completed"""
    return stream_tasks.get(task_id, {}).get("completed", False)

# Client session management functions
def create_client_session(client_id: str):
    """Create a new client session"""
    client_sessions[client_id] = {
        "created_at": datetime.now(),
        "last_activity": datetime.now(),
        "active_tasks": [],
        "is_active": True
    }
    client_updates[client_id] = []

def send_to_client(client_id: str, update: Dict[str, Any]):
    """Send update to specific client"""
    if client_id in client_updates:
        client_updates[client_id].append({
            **update,
            "timestamp": datetime.now().isoformat()
        })
        # Update last activity
        if client_id in client_sessions:
            client_sessions[client_id]["last_activity"] = datetime.now()

def get_client_session(client_id: str) -> Dict[str, Any]:
    """Get a client session by ID"""
    return client_sessions.get(client_id)

def get_client_updates(client_id: str, start_index: int = 0) -> list:
    """Get updates for a client starting from an index"""
    return client_updates.get(client_id, [])[start_index:]

def is_client_session_active(client_id: str) -> bool:
    """Check if a client session is active"""
    session = client_sessions.get(client_id)
    return session and session.get("is_active", False)

def add_task_to_client(client_id: str, task_id: str):
    """Associate a task with a client session"""
    if client_id in client_sessions:
        client_sessions[client_id]["active_tasks"].append(task_id)

def remove_task_from_client(client_id: str, task_id: str):
    """Remove a task from a client session"""
    if client_id in client_sessions and task_id in client_sessions[client_id]["active_tasks"]:
        client_sessions[client_id]["active_tasks"].remove(task_id)

def close_client_session(client_id: str):
    """Close a client session"""
    if client_id in client_sessions:
        client_sessions[client_id]["is_active"] = False

def cleanup_inactive_sessions(timeout_minutes: int = 60):
    """Clean up inactive client sessions"""
    current_time = datetime.now()
    inactive_clients = []
    
    for client_id, session in client_sessions.items():
        time_diff = (current_time - session["last_activity"]).total_seconds() / 60
        if time_diff > timeout_minutes:
            inactive_clients.append(client_id)
    
    for client_id in inactive_clients:
        if client_id in client_sessions:
            del client_sessions[client_id]
        if client_id in client_updates:
            del client_updates[client_id]
    
    return len(inactive_clients) 