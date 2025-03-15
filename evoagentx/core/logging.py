import os
from loguru import logger

logger.level("WARNING", color="<yellow>")

def save_logger(path: str):

    parent_folder = os.path.dirname(path)
    os.makedirs(parent_folder, exist_ok=True)
    logger.add(path, encoding="utf-8", format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")

__all__ = ["logger", "save_logger"]

