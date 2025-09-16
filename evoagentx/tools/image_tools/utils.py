import os
import time
from typing import Optional
from ..storage_handler import FileStorageHandler


def ensure_dir(path: str, storage_handler: Optional[FileStorageHandler] = None):
    """Ensure directory exists using storage handler or fallback to direct I/O"""
    if path:
        if storage_handler:
            storage_handler.create_directory(path)
        else:
            os.makedirs(path, exist_ok=True)


def unique_filename(prefix: str, ext: str = "png", storage_handler: Optional[FileStorageHandler] = None) -> str:
    """Generate unique filename using storage handler or fallback to timestamp"""
    if storage_handler:
        base_filename = f"{prefix}.{ext}"
        filename = base_filename
        counter = 1
        
        # Check if file exists and generate unique name
        while storage_handler.exists(filename):
            filename = f"{prefix}_{counter}.{ext}"
            counter += 1
            
        return filename
    else:
        # Fallback to timestamp-based naming
        ts = int(time.time())
        return f"{prefix}_{ts}.{ext}"


def ensure_dir_legacy(path: str):
    """Legacy function for backward compatibility - uses direct file I/O"""
    if path:
        os.makedirs(path, exist_ok=True)


def unique_filename_legacy(prefix: str, ext: str = "png") -> str:
    """Legacy function for backward compatibility - uses timestamp"""
    ts = int(time.time())
    return f"{prefix}_{ts}.{ext}"



