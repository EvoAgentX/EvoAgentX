import base64
from typing import Optional
from ...storage_handler import FileStorageHandler, LocalStorageHandler


def file_to_base64(path: str, storage_handler: Optional[FileStorageHandler] = None) -> str:
    """Convert file to base64 using storage handler"""
    if storage_handler is None:
        storage_handler = LocalStorageHandler()
    
    # for image files, directly read raw bytes data, instead of using automatic format recognition
    # because read() method returns PIL Image object instead of bytes for image files
    try:
        # convert relative path to full path
        full_path = storage_handler.translate_in(path)
        # use _read_raw to read raw binary content
        content_bytes = storage_handler._read_raw(full_path)
        return base64.b64encode(content_bytes).decode('utf-8')
    except Exception as e:
        raise FileNotFoundError(f"Could not read file {path}: {str(e)}")


def file_to_base64_legacy(path: str) -> str:
    """Legacy function for backward compatibility - uses direct file I/O"""
    with open(path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')



