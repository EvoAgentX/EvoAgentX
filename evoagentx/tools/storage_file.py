import json
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from ..core.logging import logger
from .storage_handler import FileStorageHandler, LocalStorageHandler
from .tool import Tool, Toolkit, ToolMetadata, ToolResult

load_dotenv()



# Mapping table to normalize common file type synonyms to canonical extensions
FILE_TYPE_ALIASES: Dict[str, str] = {
    # Text
    "text": "txt",
    "txt": "txt",
    # Markdown
    "md": "md",
    "markdown": "md",
    # Structured data
    "json": "json",
    "yaml": "yaml",
    "yml": "yaml",
    "xml": "xml",
    # Spreadsheets
    "excel": "xlsx",
    "xlsx": "xlsx",
    # Serialization
    "pickle": "pickle",
    "pkl": "pickle",
    # Images
    "jpg": "jpg",
    "jpeg": "jpeg",
    "png": "png",
    "gif": "gif",
    "bmp": "bmp",
    "tiff": "tiff",
    "tif": "tiff",
    # Documents
    "pdf": "pdf",
}


def resolve_extension(file_type: str) -> str:
    """Resolve a provided file_type (with or without leading dot) to a canonical extension with dot.
    Accepts common synonyms via FILE_TYPE_ALIASES.
    """
    ft = (file_type or "").strip().lower()
    if ft.startswith('.'):
        ft = ft[1:]
    canonical = FILE_TYPE_ALIASES.get(ft, ft)
    return f".{canonical}"



class SaveTool(Tool):
    name: str = "save"
    description: str = "Saves content to a local file. Returns the file URL/path. Supports `json`, `yaml`, `csv`, `pdf`, `md` and `txt` file types."
    inputs: Dict[str, Dict[str, Any]] = {
        "file_type": {
            "type": "string",
            "description": "File type/extension",
            "enum": ["json", "yaml", "csv", "pdf", "md", "txt"]
        },
        "content": {
            "type": "string",
            "description": "Content to save to the file"
        }
    }
    required: Optional[List[str]] = ["file_type", "content"]

    def __init__(self, storage_handler: FileStorageHandler = None):
        super().__init__()
        self.storage_handler = storage_handler or LocalStorageHandler()


    def __call__(self, file_type: str, content: str, file_name: str = "saved_file", encoding: str = "utf-8", indent: int = 2, 
                 sheet_name: str = "Sheet1", root_tag: str = "root") -> ToolResult:
        """
        Save content to a file with automatic format detection.
        
        Args:
            file_type: File type/extension (e.g., json, txt, csv, yaml, xml, xlsx)
            file_name: File name without extension
            content: Content to save to the file (string for text, dict/list for JSON, list for CSV/Excel)
            encoding: Text encoding for text files
            indent: Indentation for JSON files
            sheet_name: Sheet name for Excel files
            root_tag: Root tag for XML files
            
        Returns:
            Dictionary containing the save operation result
        """
        metadata = ToolMetadata(
            tool_name=self.name,
            args={
                "file_type": file_type, 
                "content": content
            }
        )

        try:
            # Compose file path from type and name using normalization table
            ext = resolve_extension(file_type)
            file_path = f"{file_name}{ext}"
            # Ensure unique name by appending a numeric suffix if needed
            
            # renamed = False
            if self.storage_handler.exists(file_path):
                counter = 1
                while self.storage_handler.exists(f"{file_name}_{counter}{ext}"):
                    counter += 1
                file_path = f"{file_name}_{counter}{ext}"
                # renamed = True
            
            # Parse content based on file type
            file_extension = self.storage_handler.get_file_type(file_path)
            parsed_content = content
            
            # Handle JSON content - convert Python objects to Python objects for StorageToolkit
            if file_extension in ['.json', '.yaml', '.yml', '.xml']:
                # If content is already a string, try to parse it as JSON
                if isinstance(content, str):
                    try:
                        parsed_content = json.loads(content)
                    except json.JSONDecodeError:
                        # If not valid JSON, use as string
                        parsed_content = content
                else:
                    # If content is a Python object, pass it directly to StorageToolkit
                    # StorageToolkit will handle the JSON serialization
                    parsed_content = content
            
            # Handle CSV content
            elif file_extension == '.csv':
                # If content is already a list, use it directly
                if isinstance(content, list):
                    parsed_content = content
                else:
                    # Try to parse as JSON first (for structured data)
                    try:
                        parsed_content = json.loads(content)
                        if not isinstance(parsed_content, list):
                            # If JSON parsing succeeded but it's not a list, treat as raw CSV
                            parsed_content = content
                    except json.JSONDecodeError:
                        # If JSON parsing fails, treat as raw CSV string
                        parsed_content = content
            
            # Handle Excel content
            elif file_extension == '.xlsx':
                # If content is already a list, use it directly
                if isinstance(content, list):
                    parsed_content = content
                else:
                    # If content is a string, try to parse it as JSON
                    try:
                        parsed_content = json.loads(content)
                        if not isinstance(parsed_content, list):
                            output = {"error": "Excel content must be a list of lists"}
                            return ToolResult(result=output, metadata=metadata)
                    except json.JSONDecodeError:
                        output = {"error": "Excel content must be valid JSON array"}
                        return ToolResult(result=output, metadata=metadata)
            
            kwargs = {
                "encoding": encoding,
                "indent": indent,
                "sheet_name": sheet_name,
                "root_tag": root_tag
            }
            
            result = self.storage_handler.save(file_path=file_path, content=parsed_content, **kwargs)
            
            # Add warning if the filename was auto-suffixed due to collision
            # if renamed:
            #     result["warnning"] = (
            #         f"original name: {file_name}{ext}, is already in used, name changed to: {file_path}"
            #     )
            
            return ToolResult(result=result, metadata=metadata)
            
        except Exception as e:
            output = {"error": str(e)}
            logger.error(f"Error in `{self.name}`: {str(e)}")
            return ToolResult(result=output, metadata=metadata)


class ReadTool(Tool):
    name: str = "read"
    description: str = "Reads file content from a local file path or file URL. Supports `json`, `yaml`, `csv`, `pdf`, `md` and `txt` file types."
    inputs: Dict[str, Dict[str, str]] = {
        "file_path": {
            "type": "string",
            "description": "Local file path to read"
        },
        "url": {
            "type": "string",
            "description": "Optional URL of the file to read"
        }
    }
    required: Optional[List[str]] = [] 

    def __init__(self, storage_handler: FileStorageHandler = None):
        super().__init__()
        self.storage_handler = storage_handler or LocalStorageHandler()

    def __call__(self, file_path: str = None, url: str = None) -> ToolResult:
        """
        Read content from a file at the given path.
        
        Args:
            file_path: Local file path to read
            url: URL of the file to read
            
        Returns:
            Dictionary containing the read operation result
        """

        metadata = ToolMetadata(
            tool_name=self.name,
            args={"file_path": file_path, "url": url}
        )

        try:
            result = self.storage_handler.read(file_path=file_path, url=url)
        except Exception as e:
            logger.error(f"Error in ReadTool: {str(e)}")
            result = {"error": str(e)}
        
        return ToolResult(result=result, metadata=metadata)


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
                    parsed_content = json.loads(content)
                except json.JSONDecodeError:
                    # If not valid JSON, use as string
                    parsed_content = content
            
            # Handle CSV content
            elif file_extension == '.csv':
                try:
                    parsed_content = json.loads(content)
                    if not isinstance(parsed_content, list):
                        # If JSON parsing succeeded but it's not a list, treat as raw CSV
                        parsed_content = content
                except json.JSONDecodeError:
                    # If JSON parsing fails, treat as raw CSV string
                    parsed_content = content
            
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
            res = {
                "success": True,
                "path": path,
                "exists": exists
            }
            # If the underlying handler supports URL decoration, add it using the real path translation
            try:
                real_path = self.storage_handler.translate_in(path)
                if getattr(self.storage_handler, "return_file_url", False):
                    res["url"] = self.storage_handler._get_file_url(real_path)
            except Exception:
                pass
            return res
            
        except Exception as e:
            logger.error(f"Error in ExistsTool: {str(e)}")
            return {"success": False, "error": str(e), "path": path}

class URLTool(Tool):
    name: str = "get_file_url"
    description: str = "Generate URL for a file on the given path for wider access"
    inputs: Dict[str, Dict[str, str]] = {
        "path": {
            "type": "string",
            "description": "Path to the file"
        }
    }
    required: Optional[List[str]] = ["path"]

    def __init__(self, storage_handler: FileStorageHandler = None):
        super().__init__()
        self.storage_handler = storage_handler or LocalStorageHandler()

    def __call__(self, path: str) -> Dict[str, Any]:
        """
        Generate URL for a file path.
        
        Args:
            path: Path to the file
            
        Returns:
            Dictionary containing the URL result
        """
        try:
            exists = self.storage_handler.exists(path)
            res = {
                "success": True,
                "path": path,
                "exists": exists,
                "url": None
            }
            # If the underlying handler supports URL decoration, add it using the real path translation
            try:
                real_path = self.storage_handler.translate_in(path)
                if getattr(self.storage_handler, "return_file_url", False):
                    res["url"] = self.storage_handler._get_file_url(real_path)
            except Exception:
                pass
            return res
            
        except Exception as e:
            logger.error(f"Error in URLTool: {str(e)}")
            return {"success": False, "error": str(e), "path": path}



class StorageToolkit(Toolkit):
    """
    Comprehensive storage toolkit with local filesystem operations.
    Provides tools for reading, writing, appending, deleting, moving, copying files,
    creating directories, and listing files with support for various file formats.
    """
    
    def __init__(self, name: str = "StorageToolkit", base_path: str = "./workplace/storage", storage_handler: LocalStorageHandler = None, **kwargs):
        """
        Initialize the storage toolkit.
        
        Args:
            name: Name of the toolkit
            base_path: Base directory for storage operations (default: ./workplace/storage)
            storage_handler: Storage handler instance (defaults to LocalStorageHandler)
        """
        if storage_handler is None:
            storage_handler = LocalStorageHandler(base_path=base_path)
        
        # Create tools with the storage handler
        tools = [
            SaveTool(storage_handler=storage_handler),
            ReadTool(storage_handler=storage_handler),
            # URLTool(storage_handler=storage_handler),
        ]
        
        super().__init__(name=name, tools=tools, **kwargs)
        self.storage_handler = storage_handler
