"""
Sequential execution decorator and utilities for controlling parallelism in EvoAgentX.

This module provides decorators and utilities to force sequential execution of methods
that would otherwise run in parallel, with support for both synchronous and asynchronous
execution contexts.
"""

import asyncio
import threading
import functools
import weakref
import inspect
from typing import Any, Callable, Dict, Optional, TypeVar, Awaitable
import logging

logger = logging.getLogger(__name__)

# Type variables for better type hints
F = TypeVar('F', bound=Callable[..., Any])
AF = TypeVar('AF', bound=Callable[..., Awaitable[Any]])

class SequentialExecutor:
    """
    Manages sequential execution locks for methods and functions.
    
    This class provides both instance-level and global-level locking mechanisms
    to ensure sequential execution of decorated methods.
    """
    
    def __init__(self):
        # Global locks for functions (not bound to instances)
        self._global_locks: Dict[str, threading.Lock] = {}
        self._global_async_locks: Dict[str, asyncio.Lock] = {}
        
        # Instance-specific locks (using weak references to avoid memory leaks)
        self._instance_locks: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
        self._instance_async_locks: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
        
        # Lock for managing the locks themselves
        self._lock_manager_lock = threading.Lock()
    
    def get_sync_lock(self, key: str, instance: Optional[Any] = None) -> threading.Lock:
        """Get or create a synchronous lock for the given key and instance."""
        with self._lock_manager_lock:
            if instance is not None:
                # Instance-specific lock
                if instance not in self._instance_locks:
                    self._instance_locks[instance] = {}
                
                if key not in self._instance_locks[instance]:
                    self._instance_locks[instance][key] = threading.Lock()
                
                return self._instance_locks[instance][key]
            else:
                # Global lock
                if key not in self._global_locks:
                    self._global_locks[key] = threading.Lock()
                
                return self._global_locks[key]
    
    def get_async_lock(self, key: str, instance: Optional[Any] = None) -> asyncio.Lock:
        """Get or create an asynchronous lock for the given key and instance."""
        if instance is not None:
            # Instance-specific lock
            if instance not in self._instance_async_locks:
                self._instance_async_locks[instance] = {}
            
            if key not in self._instance_async_locks[instance]:
                self._instance_async_locks[instance][key] = asyncio.Lock()
            
            return self._instance_async_locks[instance][key]
        else:
            # Global lock
            if key not in self._global_async_locks:
                self._global_async_locks[key] = asyncio.Lock()
            
            return self._global_async_locks[key]

# Global executor instance
_executor = SequentialExecutor()

def sequential(
    scope: str = "instance", 
    key: Optional[str] = None,
    timeout: Optional[float] = None
) -> Callable[[F], F]:
    """
    Decorator to enforce sequential execution of a method or function.
    
    This decorator ensures that multiple calls to the decorated method/function
    are executed sequentially rather than in parallel, even when called from
    multiple threads or async tasks.
    
    Args:
        scope: Execution scope - "instance", "class", or "global"
            - "instance": Lock per instance (default for methods)
            - "class": Lock per class (shared across all instances)
            - "global": Global lock (shared across all classes and instances)
        key: Custom key for the lock. If None, uses function name
        timeout: Maximum time to wait for lock acquisition (None = no timeout)
    
    Returns:
        Decorated function that executes sequentially
        
    Examples:
        # Instance-level sequential execution (default for methods)
        class DatabaseTool:
            @sequential()
            def update_record(self, record_id, data):
                # This will be sequential per instance
                pass
        
        # Class-level sequential execution
        class FileProcessor:
            @sequential(scope="class")
            def process_file(self, filename):
                # This will be sequential across all instances of FileProcessor
                pass
        
        # Global sequential execution
        @sequential(scope="global", key="critical_section")
        def critical_operation():
            # This will be sequential globally across the entire application
            pass
        
        # Custom key for grouping
        class APIClient:
            @sequential(key="api_calls")
            def make_request(self, endpoint):
                # All methods with key="api_calls" will share the same lock
                pass
            
            @sequential(key="api_calls")  
            def upload_data(self, data):
                # This shares the same lock as make_request
                pass
    """
    
    def decorator(func: F) -> F:
        # Determine the lock key
        lock_key = key if key is not None else func.__name__
        
        # Store metadata on the function
        func._sequential_scope = scope
        func._sequential_key = lock_key
        func._sequential_timeout = timeout
        func._is_sequential = True
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Determine the instance for locking
            instance = None
            if scope == "instance" and args and hasattr(args[0], '__dict__'):
                instance = args[0]
            elif scope == "class" and args and hasattr(args[0], '__class__'):
                instance = args[0].__class__
            # For global scope, instance remains None
            
            # Get the appropriate lock
            lock = _executor.get_sync_lock(lock_key, instance)
            
            # Execute with lock
            if timeout is not None:
                acquired = lock.acquire(timeout=timeout)
                if not acquired:
                    raise TimeoutError(f"Failed to acquire lock for {func.__name__} within {timeout} seconds")
                try:
                    return func(*args, **kwargs)
                finally:
                    lock.release()
            else:
                with lock:
                    return func(*args, **kwargs)
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Determine the instance for locking
            instance = None
            if scope == "instance" and args and hasattr(args[0], '__dict__'):
                instance = args[0]
            elif scope == "class" and args and hasattr(args[0], '__class__'):
                instance = args[0].__class__
            # For global scope, instance remains None
            
            # Get the appropriate async lock
            async_lock = _executor.get_async_lock(lock_key, instance)
            
            # Execute with async lock
            if timeout is not None:
                try:
                    await asyncio.wait_for(async_lock.acquire(), timeout=timeout)
                except asyncio.TimeoutError:
                    raise TimeoutError(f"Failed to acquire async lock for {func.__name__} within {timeout} seconds")
                try:
                    if inspect.iscoroutinefunction(func):
                        return await func(*args, **kwargs)
                    else:
                        # Run sync function in executor
                        loop = asyncio.get_running_loop()
                        return await loop.run_in_executor(None, func, *args, **kwargs)
                finally:
                    async_lock.release()
            else:
                async with async_lock:
                    if inspect.iscoroutinefunction(func):
                        return await func(*args, **kwargs)
                    else:
                        # Run sync function in executor
                        loop = asyncio.get_running_loop()
                        return await loop.run_in_executor(None, func, *args, **kwargs)
        
        # Create a smart wrapper that chooses sync or async based on context
        @functools.wraps(func)
        def smart_wrapper(*args, **kwargs):
            try:
                # Check if we're in an async context
                asyncio.get_running_loop()
                # If we get here, we're in an async context
                return async_wrapper(*args, **kwargs)
            except RuntimeError:
                # No running loop, use sync wrapper
                return sync_wrapper(*args, **kwargs)
        
        # Attach both wrappers for direct access if needed
        smart_wrapper._sync_wrapper = sync_wrapper
        smart_wrapper._async_wrapper = async_wrapper
        smart_wrapper._original_func = func
        
        return smart_wrapper
    
    return decorator

def is_sequential(func: Callable) -> bool:
    """Check if a function/method is decorated with @sequential."""
    return getattr(func, '_is_sequential', False)

def get_sequential_info(func: Callable) -> Optional[Dict[str, Any]]:
    """Get sequential execution information from a decorated function."""
    if not is_sequential(func):
        return None
    
    return {
        'scope': getattr(func, '_sequential_scope', 'instance'),
        'key': getattr(func, '_sequential_key', func.__name__),
        'timeout': getattr(func, '_sequential_timeout', None)
    }

# Convenience decorators for common use cases
# Note: The single `sequential` decorator is the canonical API.