from .tool import Tool, Toolkit
from .storage_handler import FileStorageHandler, LocalStorageHandler
from typing import Dict, Any, List, Optional
from ..core.logging import logger




class SaveTool(Tool):
    name: str = "save"
    description: str = "Save content to a file with automatic format detection and support for various file types including documents, data files, images, videos, and sound files"
    inputs: Dict[str, Dict[str, str]] = {
        "file_path": {
            "type": "string",
            "description": "Path to the file to save"
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

    def __init__(self, storage_handler: FileStorageHandler = None):
        super().__init__()
        self.storage_handler = storage_handler or LocalStorageHandler()

    def __call__(self, file_path: str, content: str, encoding: str = "utf-8", indent: int = 2, 
                 sheet_name: str = "Sheet1", root_tag: str = "root") -> Dict[str, Any]:
        """
        Save content to a file with automatic format detection.
        
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
                    import json
                    parsed_content = json.loads(content)
                except json.JSONDecodeError:
                    # If not valid JSON, use as string
                    parsed_content = content
            
            # Handle CSV content
            elif file_extension == '.csv':
                try:
                    import json
                    parsed_content = json.loads(content)
                    if not isinstance(parsed_content, list):
                        return {"success": False, "error": "CSV content must be a list of dictionaries"}
                except json.JSONDecodeError:
                    return {"success": False, "error": "CSV content must be valid JSON array"}
            
            # Handle Excel content
            elif file_extension == '.xlsx':
                try:
                    import json
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
            logger.error(f"Error in SaveTool: {str(e)}")
            return {"success": False, "error": str(e), "file_path": file_path}


class ReadTool(Tool):
    name: str = "read"
    description: str = "Read content from a file with automatic format detection and support for various file types including documents, data files, images, videos, and sound files"
    inputs: Dict[str, Dict[str, str]] = {
        "file_path": {
            "type": "string",
            "description": "Path to the file to read"
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

    def __init__(self, storage_handler: FileStorageHandler = None):
        super().__init__()
        self.storage_handler = storage_handler or LocalStorageHandler()

    def __call__(self, file_path: str, encoding: str = "utf-8", sheet_name: str = None, head: int = 0) -> Dict[str, Any]:
        """
        Read content from a file with automatic format detection.
        
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
            logger.error(f"Error in ReadTool: {str(e)}")
            return {"success": False, "error": str(e), "file_path": file_path}


class AppendTool(Tool):
    name: str = "append"
    description: str = "Append content to a file (only for supported formats: txt, json, csv, yaml, pickle, xlsx)"
    inputs: Dict[str, Dict[str, str]] = {
        "file_path": {
            "type": "string",
            "description": "Path to the file to append to"
        },
        "content": {
            "type": "string",
            "description": "Content to append to the file (can be JSON string for structured data)"
        },
        "encoding": {
            "type": "string",
            "description": "Text encoding for text files (default: utf-8)"
        },
        "sheet_name": {
            "type": "string",
            "description": "Sheet name for Excel files (optional)"
        }
    }
    required: Optional[List[str]] = ["file_path", "content"]

    def __init__(self, storage_handler: FileStorageHandler = None):
        super().__init__()
        self.storage_handler = storage_handler or LocalStorageHandler()

    def __call__(self, file_path: str, content: str, encoding: str = "utf-8", sheet_name: str = None) -> Dict[str, Any]:
        """
        Append content to a file with automatic format detection.
        
        Args:
            file_path: Path to the file to append to
            content: Content to append to the file
            encoding: Text encoding for text files
            sheet_name: Sheet name for Excel files
            
        Returns:
            Dictionary containing the append operation result
        """
        try:
            # Parse content based on file type
            file_extension = self.storage_handler.get_file_type(file_path)
            parsed_content = content
            
            # Try to parse JSON content for appropriate file types
            if file_extension in ['.json', '.yaml', '.yml']:
                try:
                    import json
                    parsed_content = json.loads(content)
                except json.JSONDecodeError:
                    # If not valid JSON, use as string
                    parsed_content = content
            
            # Handle CSV content
            elif file_extension == '.csv':
                try:
                    import json
                    parsed_content = json.loads(content)
                    if not isinstance(parsed_content, list):
                        return {"success": False, "error": "CSV content must be a list of dictionaries"}
                except json.JSONDecodeError:
                    return {"success": False, "error": "CSV content must be valid JSON array"}
            
            # Handle Excel content
            elif file_extension == '.xlsx':
                try:
                    import json
                    parsed_content = json.loads(content)
                    if not isinstance(parsed_content, list):
                        return {"success": False, "error": "Excel content must be a list of lists"}
                except json.JSONDecodeError:
                    return {"success": False, "error": "Excel content must be valid JSON array"}
            
            kwargs = {
                "encoding": encoding,
                "sheet_name": sheet_name
            }
            
            result = self.storage_handler.append(file_path, parsed_content, **kwargs)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in AppendTool: {str(e)}")
            return {"success": False, "error": str(e), "file_path": file_path}


class DeleteTool(Tool):
    name: str = "delete"
    description: str = "Delete a file or directory"
    inputs: Dict[str, Dict[str, str]] = {
        "path": {
            "type": "string",
            "description": "Path to the file or directory to delete"
        }
    }
    required: Optional[List[str]] = ["path"]

    def __init__(self, storage_handler: FileStorageHandler = None):
        super().__init__()
        self.storage_handler = storage_handler or LocalStorageHandler()

    def __call__(self, path: str) -> Dict[str, Any]:
        """
        Delete a file or directory.
        
        Args:
            path: Path to the file or directory to delete
            
        Returns:
            Dictionary containing the delete operation result
        """
        try:
            result = self.storage_handler.delete(path)
            return result
            
        except Exception as e:
            logger.error(f"Error in DeleteTool: {str(e)}")
            return {"success": False, "error": str(e), "path": path}


class MoveTool(Tool):
    name: str = "move"
    description: str = "Move or rename a file or directory"
    inputs: Dict[str, Dict[str, str]] = {
        "source": {
            "type": "string",
            "description": "Source path of the file or directory to move"
        },
        "destination": {
            "type": "string",
            "description": "Destination path where to move the file or directory"
        }
    }
    required: Optional[List[str]] = ["source", "destination"]

    def __init__(self, storage_handler: FileStorageHandler = None):
        super().__init__()
        self.storage_handler = storage_handler or LocalStorageHandler()

    def __call__(self, source: str, destination: str) -> Dict[str, Any]:
        """
        Move or rename a file or directory.
        
        Args:
            source: Source path of the file or directory to move
            destination: Destination path where to move the file or directory
            
        Returns:
            Dictionary containing the move operation result
        """
        try:
            result = self.storage_handler.move(source, destination)
            return result
            
        except Exception as e:
            logger.error(f"Error in MoveTool: {str(e)}")
            return {"success": False, "error": str(e), "source": source, "destination": destination}


class CopyTool(Tool):
    name: str = "copy"
    description: str = "Copy a file"
    inputs: Dict[str, Dict[str, str]] = {
        "source": {
            "type": "string",
            "description": "Source path of the file to copy"
        },
        "destination": {
            "type": "string",
            "description": "Destination path where to copy the file"
        }
    }
    required: Optional[List[str]] = ["source", "destination"]

    def __init__(self, storage_handler: FileStorageHandler = None):
        super().__init__()
        self.storage_handler = storage_handler or LocalStorageHandler()

    def __call__(self, source: str, destination: str) -> Dict[str, Any]:
        """
        Copy a file.
        
        Args:
            source: Source path of the file to copy
            destination: Destination path where to copy the file
            
        Returns:
            Dictionary containing the copy operation result
        """
        try:
            result = self.storage_handler.copy(source, destination)
            return result
            
        except Exception as e:
            logger.error(f"Error in CopyTool: {str(e)}")
            return {"success": False, "error": str(e), "source": source, "destination": destination}


class CreateDirectoryTool(Tool):
    name: str = "create_directory"
    description: str = "Create a directory"
    inputs: Dict[str, Dict[str, str]] = {
        "path": {
            "type": "string",
            "description": "Path of the directory to create"
        }
    }
    required: Optional[List[str]] = ["path"]

    def __init__(self, storage_handler: FileStorageHandler = None):
        super().__init__()
        self.storage_handler = storage_handler or LocalStorageHandler()

    def __call__(self, path: str) -> Dict[str, Any]:
        """
        Create a directory.
        
        Args:
            path: Path of the directory to create
            
        Returns:
            Dictionary containing the create directory operation result
        """
        try:
            result = self.storage_handler.create_directory(path)
            return result
            
        except Exception as e:
            logger.error(f"Error in CreateDirectoryTool: {str(e)}")
            return {"success": False, "error": str(e), "path": path}


class ListFileTool(Tool):
    name: str = "list_files"
    description: str = "List files and directories in a path with structured information"
    inputs: Dict[str, Dict[str, str]] = {
        "path": {
            "type": "string",
            "description": "Path to list files from (default: current working directory)"
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

    def __init__(self, storage_handler: FileStorageHandler = None):
        super().__init__()
        self.storage_handler = storage_handler or LocalStorageHandler()

    def __call__(self, path: str = None, max_depth: int = 3, include_hidden: bool = False) -> Dict[str, Any]:
        """
        List files and directories in a path.
        
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
            logger.error(f"Error in ListFileTool: {str(e)}")
            return {"success": False, "error": str(e), "path": path}


class ExistsTool(Tool):
    name: str = "exists"
    description: str = "Check if a file or directory exists"
    inputs: Dict[str, Dict[str, str]] = {
        "path": {
            "type": "string",
            "description": "Path to check for existence"
        }
    }
    required: Optional[List[str]] = ["path"]

    def __init__(self, storage_handler: FileStorageHandler = None):
        super().__init__()
        self.storage_handler = storage_handler or LocalStorageHandler()

    def __call__(self, path: str) -> Dict[str, Any]:
        """
        Check if a file or directory exists.
        
        Args:
            path: Path to check for existence
            
        Returns:
            Dictionary containing the existence check result
        """
        try:
            exists = self.storage_handler.exists(path)
            return {
                "success": True,
                "path": path,
                "exists": exists
            }
            
        except Exception as e:
            logger.error(f"Error in ExistsTool: {str(e)}")
            return {"success": False, "error": str(e), "path": path}


class ListSupportedFormatsTool(Tool):
    name: str = "list_supported_formats"
    description: str = "List all supported file formats and their capabilities"
    inputs: Dict[str, Dict[str, str]] = {}
    required: Optional[List[str]] = []

    def __init__(self, storage_handler: FileStorageHandler = None):
        super().__init__()
        self.storage_handler = storage_handler or LocalStorageHandler()

    def __call__(self) -> Dict[str, Any]:
        """
        List all supported file formats and their capabilities.
        
        Returns:
            Dictionary containing supported formats information
        """
        try:
            result = self.storage_handler.get_supported_formats()
            return result
            
        except Exception as e:
            logger.error(f"Error in ListSupportedFormatsTool: {str(e)}")
            return {"success": False, "error": str(e)}


class StorageToolkit(Toolkit):
    """
    Comprehensive storage toolkit with local filesystem operations.
    Provides tools for reading, writing, appending, deleting, moving, copying files,
    creating directories, and listing files with support for various file formats.
    """
    
    def __init__(self, name: str = "StorageToolkit", base_path: str = "./workplace/storage", storage_handler: FileStorageHandler = None):
        """
        Initialize the storage toolkit.
        
        Args:
            name: Name of the toolkit
            base_path: Base directory for storage operations (default: ./workplace/storage)
            storage_handler: Storage handler instance (defaults to LocalStorageHandler)
        """
        # Create the shared storage handler instance
        if storage_handler is None:
            storage_handler = LocalStorageHandler(base_path=base_path)
        
        # Create tools with the storage handler
        tools = [
            SaveTool(storage_handler=storage_handler),
            ReadTool(storage_handler=storage_handler),
            AppendTool(storage_handler=storage_handler),
            DeleteTool(storage_handler=storage_handler),
            MoveTool(storage_handler=storage_handler),
            CopyTool(storage_handler=storage_handler),
            CreateDirectoryTool(storage_handler=storage_handler),
            ListFileTool(storage_handler=storage_handler),
            ExistsTool(storage_handler=storage_handler),
            ListSupportedFormatsTool(storage_handler=storage_handler)
        ]
        
        super().__init__(name=name, tools=tools)
        self.storage_handler = storage_handler 