import os
from loguru import logger

# logger.remove()
# logger.add(sys.stdout, level="INFO")
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
    logger.add(path, encoding="utf-8", format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")

def get_log_file():
    """
    Get the path to the logging file.
    
    Returns:
        str: The path to the logging file
    """
    return save_logging_file 

__all__ = ["logger", "save_logger", "get_log_file"]

