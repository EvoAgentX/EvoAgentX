import os
import json
import base64
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime

from .storage_base import StorageBase
from .tool import Tool, Toolkit
from ..core.logging import logger





class SupabaseSaveTool(Tool):
    name: str = "supabase_save"
    description: str = "Save content to a file in Supabase remote storage with automatic format detection"
    inputs: Dict[str, Dict[str, str]] = {
        "file_path": {
            "type": "string",
            "description": "Path to the file to save in Supabase"
        },
        "content": {
            "type": "string",
            "description": "Content to save to the file (can be JSON string for structured data)"
        },
        "encoding": {
            "type": "string",
            "description": "Text encoding for text files (default: utf-8)"
        },
        "indent": {
            "type": "integer",
            "description": "Indentation for JSON files (default: 2)"
        },
        "sheet_name": {
            "type": "string",
            "description": "Sheet name for Excel files (default: Sheet1)"
        },
        "root_tag": {
            "type": "string",
            "description": "Root tag for XML files (default: root)"
        }
    }
    required: Optional[List[str]] = ["file_path", "content"]

    def __init__(self, storage_handler: SupabaseStorageHandler = None):
        super().__init__()
        self.storage_handler = storage_handler

    def __call__(self, file_path: str, content: str, encoding: str = "utf-8", indent: int = 2, 
                 sheet_name: str = "Sheet1", root_tag: str = "root") -> Dict[str, Any]:
        """
        Save content to a file in Supabase remote storage.
        
        Args:
            file_path: Path to the file to save
            content: Content to save to the file
            encoding: Text encoding for text files
            indent: Indentation for JSON files
            sheet_name: Sheet name for Excel files
            root_tag: Root tag for XML files
            
        Returns:
            Dictionary containing the save operation result
        """
        try:
            # Parse content based on file type
            file_extension = self.storage_handler.get_file_type(file_path)
            parsed_content = content
            
            # Try to parse JSON content for appropriate file types
            if file_extension in ['.json', '.yaml', '.yml', '.xml']:
                try:
                    parsed_content = json.loads(content)
                except json.JSONDecodeError:
                    # If not valid JSON, use as string
                    parsed_content = content
            
            # Handle CSV content
            elif file_extension == '.csv':
                try:
                    parsed_content = json.loads(content)
                    if not isinstance(parsed_content, list):
                        return {"success": False, "error": "CSV content must be a list of dictionaries"}
                except json.JSONDecodeError:
                    return {"success": False, "error": "CSV content must be valid JSON array"}
            
            # Handle Excel content
            elif file_extension == '.xlsx':
                try:
                    parsed_content = json.loads(content)
                    if not isinstance(parsed_content, list):
                        return {"success": False, "error": "Excel content must be a list of lists"}
                except json.JSONDecodeError:
                    return {"success": False, "error": "Excel content must be valid JSON array"}
            
            kwargs = {
                "encoding": encoding,
                "indent": indent,
                "sheet_name": sheet_name,
                "root_tag": root_tag
            }
            
            result = self.storage_handler.save(file_path, parsed_content, **kwargs)
            return result
            
        except Exception as e:
            logger.error(f"Error in SupabaseSaveTool: {str(e)}")
            return {"success": False, "error": str(e), "file_path": file_path}


class SupabaseReadTool(Tool):
    name: str = "supabase_read"
    description: str = "Read content from a file in Supabase remote storage with automatic format detection"
    inputs: Dict[str, Dict[str, str]] = {
        "file_path": {
            "type": "string",
            "description": "Path to the file to read from Supabase"
        },
        "encoding": {
            "type": "string",
            "description": "Text encoding for text files (default: utf-8)"
        },
        "sheet_name": {
            "type": "string",
            "description": "Sheet name for Excel files (optional)"
        },
        "head": {
            "type": "integer",
            "description": "Number of characters to return from the beginning of the file (default: 0 means return everything)"
        }
    }
    required: Optional[List[str]] = ["file_path"]

    def __init__(self, storage_handler: SupabaseStorageHandler = None):
        super().__init__()
        self.storage_handler = storage_handler

    def __call__(self, file_path: str, encoding: str = "utf-8", sheet_name: str = None, head: int = 0) -> Dict[str, Any]:
        """
        Read content from a file in Supabase remote storage.
        
        Args:
            file_path: Path to the file to read
            encoding: Text encoding for text files
            sheet_name: Sheet name for Excel files
            head: Number of characters to return from the beginning
            
        Returns:
            Dictionary containing the read operation result
        """
        try:
            kwargs = {
                "encoding": encoding,
                "sheet_name": sheet_name,
                "head": head
            }
            
            result = self.storage_handler.read(file_path, **kwargs)
            return result
            
        except Exception as e:
            logger.error(f"Error in SupabaseReadTool: {str(e)}")
            return {"success": False, "error": str(e), "file_path": file_path}


class SupabaseDeleteTool(Tool):
    name: str = "supabase_delete"
    description: str = "Delete a file from Supabase remote storage"
    inputs: Dict[str, Dict[str, str]] = {
        "path": {
            "type": "string",
            "description": "Path to the file to delete from Supabase"
        }
    }
    required: Optional[List[str]] = ["path"]

    def __init__(self, storage_handler: SupabaseStorageHandler = None):
        super().__init__()
        self.storage_handler = storage_handler

    def __call__(self, path: str) -> Dict[str, Any]:
        """
        Delete a file from Supabase remote storage.
        
        Args:
            path: Path to the file to delete
            
        Returns:
            Dictionary containing the delete operation result
        """
        try:
            result = self.storage_handler.delete(path)
            return result
            
        except Exception as e:
            logger.error(f"Error in SupabaseDeleteTool: {str(e)}")
            return {"success": False, "error": str(e), "path": path}


class SupabaseListTool(Tool):
    name: str = "supabase_list"
    description: str = "List files and directories in Supabase remote storage"
    inputs: Dict[str, Dict[str, str]] = {
        "path": {
            "type": "string",
            "description": "Path to list files from in Supabase (default: root directory)"
        },
        "max_depth": {
            "type": "integer",
            "description": "Maximum depth to traverse (default: 3)"
        },
        "include_hidden": {
            "type": "boolean",
            "description": "Include hidden files and directories (default: false)"
        }
    }
    required: Optional[List[str]] = []

    def __init__(self, storage_handler: SupabaseStorageHandler = None):
        super().__init__()
        self.storage_handler = storage_handler

    def __call__(self, path: str = None, max_depth: int = 3, include_hidden: bool = False) -> Dict[str, Any]:
        """
        List files and directories in Supabase remote storage.
        
        Args:
            path: Path to list files from
            max_depth: Maximum depth to traverse
            include_hidden: Include hidden files and directories
            
        Returns:
            Dictionary containing the list operation result
        """
        try:
            result = self.storage_handler.list(path, max_depth=max_depth, include_hidden=include_hidden)
            return result
            
        except Exception as e:
            logger.error(f"Error in SupabaseListTool: {str(e)}")
            return {"success": False, "error": str(e), "path": path}


class StorageSupabaseToolkit(Toolkit):
    """
    Supabase remote storage toolkit.
    Provides tools for reading, writing, deleting, and listing files in Supabase remote storage.
    Requires SUPABASE_URL and SUPABASE_ANON_KEY environment variables to be set.
    """
    
    def __init__(self, name: str = "StorageSupabaseToolkit", bucket_name: str = "default", base_path: str = "/"):
        """
        Initialize the Supabase storage toolkit.
        
        Args:
            name: Name of the toolkit
            bucket_name: Supabase storage bucket name
            base_path: Base path for storage operations
        """
        # Create the shared storage handler instance
        storage_handler = SupabaseStorageHandler(
            bucket_name=bucket_name,
            base_path=base_path
        )
        
        # Create tools with the storage handler
        tools = [
            SupabaseSaveTool(storage_handler=storage_handler),
            SupabaseReadTool(storage_handler=storage_handler),
            SupabaseDeleteTool(storage_handler=storage_handler),
            SupabaseListTool(storage_handler=storage_handler)
        ]
        
        super().__init__(name=name, tools=tools)
        self.storage_handler = storage_handler
