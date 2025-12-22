import base64
from typing import Optional
from ...storage_handler import FileStorageHandler, LocalStorageHandler


def file_to_base64(path: str, storage_handler: Optional[FileStorageHandler] = None) -> str:
    """Convert file to base64 using storage handler"""
    if storage_handler is None:
        storage_handler = LocalStorageHandler()
    
    result = storage_handler.read(path)
    if result["success"]:
        if isinstance(result["content"], bytes):
            return base64.b64encode(result["content"]).decode('utf-8')
        else:
            # If content is not bytes, convert to bytes first
            return base64.b64encode(str(result["content"]).encode('utf-8')).decode('utf-8')
    else:
        raise FileNotFoundError(f"Could not read file {path}: {result.get('error', 'Unknown error')}")


def file_to_base64_legacy(path: str) -> str:
    """Legacy function for backward compatibility - uses direct file I/O"""
    with open(path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')



