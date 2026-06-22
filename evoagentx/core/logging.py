import os
import sys
import io
import contextvars
from loguru import logger

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# 清空默认 handler，避免重复输出
logger.remove()

# When set, records below WARNING are dropped for the *current* thread /
# asyncio-task only (contextvars isolate both). Suppression is implemented as a
# per-context filter on permanently-installed sinks rather than by swapping the
# global sinks, so concurrent runs that each want quiet output never race on the
# shared logger configuration. See `suppress_logger_info` in core/callbacks.py.
suppress_info = contextvars.ContextVar("suppress_info", default=False)

_WARNING_NO = logger.level("WARNING").no


def _suppression_filter(record) -> bool:
    """Drop sub-WARNING records while suppression is active in this context."""
    if suppress_info.get():
        return record["level"].no >= _WARNING_NO
    return True


# 控制台输出
logger.add(sys.stdout, level="INFO", filter=_suppression_filter)
# file_handler_id = None
save_logging_file = None

def save_logger(path: str):
    """
    Save the logging to a file.
    
    Args:
        path: The path to save the logging file
    """
    global save_logging_file
    save_logging_file = path

    parent_folder = os.path.dirname(path)
    os.makedirs(parent_folder, exist_ok=True)
    logger.add(
        path,
        encoding="utf-8",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
        filter=_suppression_filter,
    )

def get_log_file():
    """
    Get the path to the logging file.
    
    Returns:
        str: The path to the logging file
    """
    return save_logging_file 

__all__ = ["logger", "save_logger", "get_log_file", "suppress_info"]

