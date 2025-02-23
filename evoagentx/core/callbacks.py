import threading
import contextvars
from contextlib import contextmanager

class Callback:

    """
    a base class for callbacks 
    """

    def on_error(self, exception, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        try:
            result = self.run(*args, **kwargs)
        except Exception as e:
            self.on_error(e, *args, kwargs)
            raise e 
        return result
    
    def run(self, *args, **kwargs):
        raise NotImplementedError(f"run is not implemented for {type(self).__name__}!")


class CallbackManager:

    def __init__(self):
        self.local_data = threading.local()
        self.local_data.callbacks = {}

    def set_callback(self, callback_type: str, callback: Callback):
        self.local_data.callbacks[callback_type] = callback

    def get_callback(self, callback_type: str):
        return self.local_data.callbacks.get(callback_type, None)
    
    def has_callback(self, callback_type: str):
        return callback_type in self.local_data.callbacks

    def clear_callback(self, callback_type: str):
        if callback_type in self.local_data.callbacks:
            del self.local_data.callbacks[callback_type]

    def clear_all(self):
        self.local_data.callbacks.clear()

callback_manager = CallbackManager()


class DeferredExceptionHandler(Callback):

    def __init__(self):
        self.exceptions = [] 
    
    def add(self, exception):
        self.exceptions.append(exception)
    

@contextmanager
def exception_buffer():
    if not callback_manager.has_callback("exception_buffer"):
        exception_handler = DeferredExceptionHandler()
        callback_manager.set_callback("exception_buffer", exception_handler)
    else:
        exception_handler = callback_manager.get_callback("exception_buffer")
    try:
        yield exception_handler
    finally:
        callback_manager.clear_callback("exception_buffer")
    

suppress_cost_logs = contextvars.ContextVar("suppress_cost_logs", default=False)

@contextmanager
def suppress_cost_logging():
    """Thread-safe context manager: only suppresses cost-related logs without affecting other info-level logs"""
    token = suppress_cost_logs.set(True)  # Set the value in the current thread/task
    try:
        yield
    finally:
        suppress_cost_logs.reset(token)  # Restore the previous value
