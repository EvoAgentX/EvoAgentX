import threading
from functools import wraps
from contextlib import nullcontext


def atomic(lock=None):
    """
    threading safe decorator, it can be used to decorate a function or receive a lock:
    1. directly decorate a function: @atomic 
    2. receive a lock: @atomic(lock=shared_lock)
    """
    lock = lock or threading.Lock()
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with lock:
                return func(*args, **kwargs)
        return wrapper
    
    return decorator if not callable(lock) else decorator(lock)


def atomic_method(func):
    """
    threading safe decorator for class methods. 
    If there are self._lock in the instance, it will use the lock. Otherwise, use nullcontext for execution.
    """
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        context = getattr(self, "_lock", nullcontext())
        with context:
            return func(self, *args, **kwargs)
    return wrapper


