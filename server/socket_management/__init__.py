"""
Socket Management Service for EvoAgentX
Provides real-time socket connections for project monitoring using the existing message format.
"""

from .socket_service import SocketService, socket_service
from .message_handler import MessageHandler
from .process_monitor import ProcessMonitor
from .message_store import MessageStore, message_store
from .protocols import MessageType

__all__ = [
    'SocketService',
    'socket_service',  # Global instance
    'MessageHandler', 
    'ProcessMonitor',
    'MessageType',
    'MessageStore',
    'message_store'  # Global instance
]

# Version info
__version__ = '1.0.0'
