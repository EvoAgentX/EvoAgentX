import sys 
# import stopit
from overdue import timeout_set_to 
import threading
import contextvars
from typing import Union
from contextlib import contextmanager
from .logging import logger, get_log_file

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
        # self.local_data.callbacks = {}
    
    def _ensure_callbacks(self):
        if not hasattr(self.local_data, "callbacks"):
            self.local_data.callbacks = {}

    def set_callback(self, callback_type: str, callback: Callback):
        self._ensure_callbacks()
        self.local_data.callbacks[callback_type] = callback

    def get_callback(self, callback_type: str):
        self._ensure_callbacks()
        return self.local_data.callbacks.get(callback_type, None)
    
    def has_callback(self, callback_type: str):
        self._ensure_callbacks()
        return callback_type in self.local_data.callbacks

    def clear_callback(self, callback_type: str):
        self._ensure_callbacks()
        if callback_type in self.local_data.callbacks:
            del self.local_data.callbacks[callback_type]

    def clear_all(self):
        self._ensure_callbacks()
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


silence_nesting = contextvars.ContextVar("silence_nesting", default=0)

@contextmanager
def suppress_logger_info():
    token = None
    try:
        current_level = silence_nesting.get()
        token = silence_nesting.set(current_level + 1)
        
        if current_level == 0:
            logger.remove()
            logger.add(sys.stdout, level="WARNING")
            log_file = get_log_file()
            if log_file is not None:
                logger.add(
                    log_file,
                    encoding="utf-8",
                    level="WARNING", 
                    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
                )
        yield
    finally:
        new_level = silence_nesting.get() - 1
        silence_nesting.set(new_level)
        
        if new_level == 0:
            logger.remove()
            logger.add(sys.stdout, level="INFO")
            log_file = get_log_file()
            if log_file is not None:
                logger.add(
                    log_file,
                    encoding="utf-8",
                    level="INFO", 
                    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
                )
        if token:
            silence_nesting.reset(token)


class TimeoutException(Exception):
    pass

class TimeoutContext:
    """
    A reliable cross-platform timeout context manager using stopit
    
    Usage:
        with TimeoutContext(seconds=5):
            # code that may timeout
            do_something()
    """
    def __init__(self, seconds: Union[int, float]):
        self.seconds = float(seconds)
        # self._context: Optional[stopit.SignalTimeout] = None
        self._cm = None
        self._result = None
        
    def __enter__(self):
        # self._context = stopit.ThreadingTimeout(self.seconds)
        # self._context.__enter__()
        self._cm = timeout_set_to(self.seconds)
        self._result = self._cm.__enter__()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        # timeout_occurred = self._context.__exit__(exc_type, exc_val, exc_tb)
        # if timeout_occurred:
            # raise TimeoutException("Operation timed out")
        self._cm.__exit__(exc_type, exc_val, exc_tb)
        if self._result.triggered:
            raise TimeoutException("Operation timed out")
        return False

@contextmanager
def timeout(seconds: float):
    with TimeoutContext(seconds):
        yield
