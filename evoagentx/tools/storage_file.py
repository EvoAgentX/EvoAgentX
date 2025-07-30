from .tool import Tool, Toolkit
from .storage_base import StorageBase
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

    def __init__(self, storage_base: StorageBase = None):
        super().__init__()
        self.storage_base = storage_base or StorageBase()

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
            file_extension = self.storage_base.get_file_type(file_path)
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
            
            result = self.storage_base.save(file_path, parsed_content, **kwargs)
            
            if result["success"]:
                logger.info(f"Successfully saved file: {file_path}")
            else:
                logger.error(f"Failed to save file {file_path}: {result.get('error', 'Unknown error')}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in save tool: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "file_path": file_path
            }


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

    def __init__(self, storage_base: StorageBase = None):
        super().__init__()
        self.storage_base = storage_base or StorageBase()

    def __call__(self, file_path: str, encoding: str = "utf-8", sheet_name: str = None, head: int = 0) -> Dict[str, Any]:
        """
        Read content from a file with automatic format detection.
        
        Args:
            file_path: Path to the file to read
            encoding: Text encoding for text files
            sheet_name: Sheet name for Excel files
            head: Number of characters to return from the beginning (0 means return everything)
            
        Returns:
            Dictionary containing the read content and metadata
        """
        try:
            kwargs = {"encoding": encoding}
            if sheet_name:
                kwargs["sheet_name"] = sheet_name
            
            result = self.storage_base.read(file_path, **kwargs)
            
            if result["success"]:
                # Apply head limit if specified
                if head > 0:
                    content = result.get("content")
                    if isinstance(content, str):
                        # For string content, truncate by characters
                        original_length = len(content)
                        result["content"] = content[:head]
                        result["original_length"] = original_length
                        result["truncated_length"] = len(result["content"])
                        logger.info(f"Successfully read file: {file_path} (truncated to {head} characters)")
                    elif isinstance(content, list):
                        # For list content (like JSON arrays), truncate by number of items
                        original_length = len(content)
                        result["content"] = content[:head]
                        result["original_length"] = original_length
                        result["truncated_length"] = len(result["content"])
                        logger.info(f"Successfully read file: {file_path} (truncated to {head} items)")
                    else:
                        # For other data types, convert to string and truncate
                        content_str = str(content)
                        original_length = len(content_str)
                        result["content"] = content_str[:head]
                        result["original_length"] = original_length
                        result["truncated_length"] = len(result["content"])
                        logger.info(f"Successfully read file: {file_path} (truncated to {head} characters)")
                else:
                    logger.info(f"Successfully read file: {file_path}")
            else:
                logger.error(f"Failed to read file {file_path}: {result.get('error', 'Unknown error')}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in read tool: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "file_path": file_path
            }


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

    def __init__(self, storage_base: StorageBase = None):
        super().__init__()
        self.storage_base = storage_base or StorageBase()

    def __call__(self, file_path: str, content: str, encoding: str = "utf-8", sheet_name: str = None) -> Dict[str, Any]:
        """
        Append content to a file (only for supported formats).
        
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
            file_extension = self.storage_base.get_file_type(file_path)
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
            
            kwargs = {"encoding": encoding}
            if sheet_name:
                kwargs["sheet_name"] = sheet_name
            
            result = self.storage_base.append(file_path, parsed_content, **kwargs)
            
            if result["success"]:
                logger.info(f"Successfully appended to file: {file_path}")
            else:
                logger.error(f"Failed to append to file {file_path}: {result.get('error', 'Unknown error')}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in append tool: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "file_path": file_path
            }


class ListFileTool(Tool):
    name: str = "list_files"
    description: str = "List files and directories in a path with structured tree-like information"
    inputs: Dict[str, Dict[str, str]] = {
        "path": {
            "type": "string",
            "description": "Path to list files from (default: current workplace directory)"
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

    def __init__(self, storage_base: StorageBase = None):
        super().__init__()
        self.storage_base = storage_base or StorageBase()

    def __call__(self, path: str = None, max_depth: int = 3, include_hidden: bool = False) -> Dict[str, Any]:
        """
        List files and directories in a structured tree format.
        
        Args:
            path: Path to list files from
            max_depth: Maximum depth to traverse
            include_hidden: Include hidden files and directories
            
        Returns:
            Dictionary containing structured file tree information
        """
        try:
            if path is None:
                path = str(self.storage_base.base_path)
            
            result = self.storage_base.list_files(path, max_depth, include_hidden)
            
            if result["success"]:
                logger.info(f"Successfully listed files in: {path}")
            else:
                logger.error(f"Failed to list files in {path}: {result.get('error', 'Unknown error')}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in list_files tool: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "path": path
            }


class ListSupportedFormatsTool(Tool):
    name: str = "list_supported_formats"
    description: str = "List all supported file formats and their capabilities"
    inputs: Dict[str, Dict[str, str]] = {}
    required: Optional[List[str]] = []

    def __init__(self, storage_base: StorageBase = None):
        super().__init__()
        self.storage_base = storage_base or StorageBase()

    def __call__(self) -> Dict[str, Any]:
        """
        List all supported file formats and their capabilities.
        
        Returns:
            Dictionary containing supported formats and their capabilities
        """
        try:
            result = self.storage_base.get_supported_formats()
            
            if result["success"]:
                logger.info("Successfully retrieved supported formats")
            else:
                logger.error("Failed to get supported formats")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in list_supported_formats tool: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }






"""
TODO:
API key stored in Header (Authorization)
Fixed API key -- simple check (STR compare?)
Encode - Decode -- SHA265?


Further (plan / Optional):
- IP whitelist
- Expire?
- Pipline check -- Middle ware

"""





class StorageToolkit(Toolkit):
    """
    Comprehensive storage toolkit that provides save, read, and append functionality
    for various file types including documents, data files, images, videos, and sound files.
    Designed with scalability in mind for future database integration.
    """
    
    def __init__(self, name: str = "StorageToolkit", base_path: str = "."):
        """
        Initialize the StorageToolkit with a shared storage base instance.
        
        Args:
            name: Name of the toolkit
            base_path: Base directory for storage operations (default: current directory)
        """
        # Create the shared storage base instance
        storage_base = StorageBase(base_path=base_path)
        
        # Initialize tools with the shared storage base
        tools = [
            SaveTool(storage_base=storage_base),
            ReadTool(storage_base=storage_base),
            AppendTool(storage_base=storage_base),
            ListFileTool(storage_base=storage_base),
            ListSupportedFormatsTool(storage_base=storage_base)
        ]
        
        # Initialize parent with tools
        super().__init__(name=name, tools=tools)
        
        # Store storage_base as instance variable
        self.storage_base = storage_base 