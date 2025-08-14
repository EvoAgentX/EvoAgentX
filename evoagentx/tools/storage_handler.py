import os
import json
import shutil
from typing import Dict, Any, List
from pathlib import Path
from datetime import datetime

from .storage_base import StorageBase
from ..core.logging import logger


class FileStorageHandler(StorageBase):
    """
    Unified storage handler that provides a consistent interface for all storage operations.
    This class serves as the main entry point for storage operations, inheriting from StorageBase
    and providing the core CRUD operations (Create, Read, Update, Delete).
    """
    
    def __init__(self, base_path: str = ".", **kwargs):
        """
        Initialize the storage handler.
        
        Args:
            base_path (str): Base directory for storage operations (default: current directory)
            **kwargs: Additional keyword arguments for parent class initialization
        """
        super().__init__(base_path=base_path, **kwargs)
    
    def create_file(self, file_path: str, content: Any, **kwargs) -> Dict[str, Any]:
        """
        Create a new file with the specified content.
        
        Args:
            file_path (str): Path where the file should be created
            content (Any): Content to write to the file
            **kwargs: Additional arguments for file creation (encoding, format, etc.)
            
        Returns:
            Dict[str, Any]: Result of the operation with success status and details
        """
        return self.save(file_path, content, **kwargs)
    
    def read_file(self, file_path: str, **kwargs) -> Dict[str, Any]:
        """
        Read content from an existing file.
        
        Args:
            file_path (str): Path of the file to read
            **kwargs: Additional arguments for file reading (encoding, format, etc.)
            
        Returns:
            Dict[str, Any]: Result of the operation with file content and success status
        """
        return self.read(file_path, **kwargs)
    
    def update_file(self, file_path: str, content: Any, **kwargs) -> Dict[str, Any]:
        """
        Update an existing file with new content.
        
        Args:
            file_path (str): Path of the file to update
            content (Any): New content to write to the file
            **kwargs: Additional arguments for file update (encoding, format, etc.)
            
        Returns:
            Dict[str, Any]: Result of the operation with success status and details
        """
        # For update, we use the same save method as it handles both create and update
        return self.save(file_path, content, **kwargs)
    
    def delete_file(self, file_path: str) -> Dict[str, Any]:
        """
        Delete a file or directory.
        
        Args:
            file_path (str): Path of the file or directory to delete
            
        Returns:
            Dict[str, Any]: Result of the operation with success status and details
        """
        return self.delete(file_path)
    
    def list_files(self, path: str = None, max_depth: int = 3, include_hidden: bool = False) -> Dict[str, Any]:
        """
        List files and directories in the specified path.
        
        Args:
            path (str): Path to list (default: base_path)
            max_depth (int): Maximum depth for recursive listing
            include_hidden (bool): Whether to include hidden files
            
        Returns:
            Dict[str, Any]: Result of the operation with file list and success status
        """
        return self.list(path, max_depth, include_hidden)
    
    def file_exists(self, file_path: str) -> bool:
        """
        Check if a file exists.
        
        Args:
            file_path (str): Path of the file to check
            
        Returns:
            bool: True if file exists, False otherwise
        """
        return self.exists(file_path)
    
    def get_file_information(self, file_path: str) -> Dict[str, Any]:
        """
        Get comprehensive information about a file.
        
        Args:
            file_path (str): Path of the file to get information for
            
        Returns:
            Dict[str, Any]: File information including size, type, timestamps, etc.
        """
        return self.get_file_info(file_path)
    
    def create_directory(self, path: str) -> Dict[str, Any]:
        """
        Create a new directory.
        
        Args:
            path (str): Path where the directory should be created
            
        Returns:
            Dict[str, Any]: Result of the operation with success status and details
        """
        return super().create_directory(path)
    
    def copy_file(self, source: str, destination: str) -> Dict[str, Any]:
        """
        Copy a file from source to destination.
        
        Args:
            source (str): Source file path
            destination (str): Destination file path
            
        Returns:
            Dict[str, Any]: Result of the operation with success status and details
        """
        return self.copy(source, destination)
    
    def move_file(self, source: str, destination: str) -> Dict[str, Any]:
        """
        Move/rename a file from source to destination.
        
        Args:
            source (str): Source file path
            destination (str): Destination file path
            
        Returns:
            Dict[str, Any]: Result of the operation with success status and details
        """
        return self.move(source, destination)
    
    def append_to_file(self, file_path: str, content: Any, **kwargs) -> Dict[str, Any]:
        """
        Append content to an existing file.
        
        Args:
            file_path (str): Path of the file to append to
            content (Any): Content to append
            **kwargs: Additional arguments for appending (encoding, format, etc.)
            
        Returns:
            Dict[str, Any]: Result of the operation with success status and details
        """
        return self.append(file_path, content, **kwargs)


class LocalStorageHandler(FileStorageHandler):
    """
    Local filesystem storage implementation.
    Provides all file operations for local storage with default working directory.
    """
    
    def __init__(self, base_path: str = ".", **kwargs):
        """
        Initialize local storage handler.
        
        Args:
            base_path (str): Base directory for storage operations (default: current directory)
            **kwargs: Additional keyword arguments for parent class initialization
        """
        super().__init__(base_path=base_path, **kwargs)
    
    def _initialize_storage(self):
        """Initialize local storage - create base directory if it doesn't exist"""
        # Convert base_path to Path for local filesystem operations
        base_path = Path(self.base_path)
        # Ensure base directory exists
        base_path.mkdir(parents=True, exist_ok=True)
    
    def _resolve_path(self, file_path: str) -> str:
        """Resolve file path for local filesystem"""
        path = Path(file_path)
        if not path.is_absolute():
            # If it's a relative path, prepend the base path
            path = Path(self.base_path) / path
        return str(path)
    
    def _read_raw(self, path: str, **kwargs) -> bytes:
        """Read raw file content from local filesystem"""
        try:
            with open(path, 'rb') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error reading file {path}: {str(e)}")
            raise
    
    def _write_raw(self, path: str, content: bytes, **kwargs) -> bool:
        """Write raw file content to local filesystem"""
        try:
            # Ensure directory exists
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            with open(path, 'wb') as f:
                f.write(content)
            return True
        except Exception as e:
            logger.error(f"Error writing file {path}: {str(e)}")
            return False
    
    def _delete_raw(self, path: str) -> bool:
        """Delete file or directory from local filesystem"""
        try:
            path_obj = Path(path)
            if path_obj.is_file():
                path_obj.unlink()
            elif path_obj.is_dir():
                shutil.rmtree(path_obj)
            else:
                return False
            return True
        except Exception as e:
            logger.error(f"Error deleting {path}: {str(e)}")
            return False
    
    def _list_raw(self, path: str = None, max_depth: int = 3, include_hidden: bool = False) -> List[Dict[str, Any]]:
        """List files and directories in local filesystem"""
        try:
            if path is None:
                path = str(self.base_path)
            
            path_obj = Path(path)
            if not path_obj.exists() or not path_obj.is_dir():
                return []
            
            items = []
            
            def scan_directory(current_path: Path, current_depth: int):
                if current_depth > max_depth:
                    return
                
                try:
                    for item in current_path.iterdir():
                        # Skip hidden files if not included
                        if not include_hidden and item.name.startswith('.'):
                            continue
                        
                        try:
                            stat = item.stat()
                            item_info = {
                                "name": item.name,
                                "path": str(item),
                                "type": "directory" if item.is_dir() else "file",
                                "size_bytes": stat.st_size if item.is_file() else 0,
                                "size_mb": round(stat.st_size / (1024 * 1024), 2) if item.is_file() else 0,
                                "modified_time": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                                "extension": item.suffix.lower() if item.is_file() else "",
                                "is_hidden": item.name.startswith('.')
                            }
                            
                            items.append(item_info)
                            
                            # Recursively scan subdirectories
                            if item.is_dir() and current_depth < max_depth:
                                scan_directory(item, current_depth + 1)
                                
                        except (PermissionError, OSError):
                            # Skip files we can't access
                            continue
                            
                except (PermissionError, OSError) as e:
                    logger.warning(f"Error scanning directory {current_path}: {str(e)}")
            
            scan_directory(path_obj, 0)
            return items
            
        except Exception as e:
            logger.error(f"Error listing directory {path}: {str(e)}")
            return []
    
    def _exists_raw(self, path: str) -> bool:
        """Check if path exists in local filesystem"""
        return Path(path).exists()
    
    def _create_directory_raw(self, path: str) -> bool:
        """Create directory in local filesystem"""
        try:
            Path(path).mkdir(parents=True, exist_ok=True)
            return True
        except Exception as e:
            logger.error(f"Error creating directory {path}: {str(e)}")
            return False
    
    def get_file_info(self, file_path: str) -> Dict[str, Any]:
        """Get comprehensive information about a file"""
        try:
            resolved_path = self._resolve_path(file_path)
            path_obj = Path(resolved_path)
            
            if not path_obj.exists():
                return {"success": False, "error": f"File {file_path} does not exist"}
            
            stat = path_obj.stat()
            return {
                "success": True,
                "file_path": resolved_path,
                "file_name": path_obj.name,
                "file_extension": path_obj.suffix.lower(),
                "mime_type": self.get_mime_type(resolved_path),
                "size_bytes": stat.st_size,
                "size_mb": round(stat.st_size / (1024 * 1024), 2),
                "created_time": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                "modified_time": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "is_file": path_obj.is_file(),
                "is_directory": path_obj.is_dir(),
                "is_readable": os.access(path_obj, os.R_OK),
                "is_writable": os.access(path_obj, os.W_OK),
                "exists": True
            }
        except Exception as e:
            logger.error(f"Error getting file info for {file_path}: {str(e)}")
            return {"success": False, "error": str(e), "file_path": file_path}


class SupabaseStorageHandler(FileStorageHandler):
    """
    Supabase remote storage implementation.
    Provides file operations via Supabase Storage API with environment-based configuration.
    """
    
    def __init__(self, bucket_name: str = None, base_path: str = "/", **kwargs):
        """
        Initialize Supabase storage handler.
        
        Args:
            bucket_name: Supabase storage bucket name (default: from environment or "default")
            base_path: Base path for storage operations (default: "/")
            **kwargs: Additional keyword arguments for parent class initialization
        """
        super().__init__(base_path=base_path, **kwargs)
        
        # Get bucket name from environment or use default
        self.bucket_name = bucket_name or os.getenv("SUPABASE_BUCKET_STORAGE") or "default"
        self.supabase_url = os.getenv("SUPABASE_URL_STORAGE")
        self.supabase_key = os.getenv("SUPABASE_KEY_STORAGE")
        
        if not self.supabase_url or not self.supabase_key:
            raise ValueError(
                "Supabase configuration not found in environment variables. "
                "Please set SUPABASE_URL/SUPABASE_KEY environment variables."
            )
        
        # Initialize Supabase client
        try:
            from supabase import create_client, Client
            logger.info(f"Creating Supabase client with URL: {self.supabase_url[:30]}...")
            self.supabase: Client = create_client(self.supabase_url, self.supabase_key)
            logger.info(f"Successfully initialized Supabase client for bucket: {bucket_name}")
        except ImportError:
            raise ImportError(
                "Supabase Python client not installed. "
                "Please install it with: pip install supabase"
            )
        except Exception as e:
            logger.error(f"Failed to initialize Supabase client: {str(e)}")
            raise Exception(f"Failed to initialize Supabase client: {str(e)}")
        
        # Initialize storage after all attributes are set
        self._initialize_storage()
    
    def _initialize_storage(self):
        """Initialize remote storage - verify bucket exists and is accessible"""
        try:
            # Test bucket access by listing files (empty list is fine)
            logger.info(f"Testing bucket access for: {self.bucket_name}")
            self.supabase.storage.from_(self.bucket_name).list()
            logger.info(f"Successfully connected to Supabase bucket: {self.bucket_name}")
        except Exception as e:
            logger.warning(f"Could not verify bucket access: {str(e)}")
            # Don't raise error as bucket might be empty or have different permissions
    
    def _resolve_path(self, file_path: str) -> str:
        """Resolve file path for remote storage"""
        # For Supabase, we use the base_path as a prefix
        # Remove any leading slash from file_path
        clean_file_path = file_path.lstrip('/')
        
        # If base_path is just "/", don't add it
        if self.base_path == "/":
            return clean_file_path
        
        # Otherwise, combine base_path and file_path
        if not clean_file_path.startswith(self.base_path):
            return f"{self.base_path}/{clean_file_path}".replace('//', '/')
        else:
            return clean_file_path
    
    def _read_raw(self, path: str, **kwargs) -> bytes:
        """Read raw file content from Supabase Storage"""
        try:
            # Remove leading slash if present
            file_path = path.lstrip('/')
            
            # Download file from Supabase Storage
            response = self.supabase.storage.from_(self.bucket_name).download(file_path)
            
            if isinstance(response, bytes):
                return response
            else:
                # If response is not bytes, try to convert
                return bytes(response) if response else b""
                
        except Exception as e:
            logger.error(f"Error reading file {path} from Supabase: {str(e)}")
            raise
    
    def _write_raw(self, path: str, content: bytes, **kwargs) -> bool:
        """Write raw file content to Supabase Storage with smart insert/update logic"""
        try:
            # Remove leading slash if present
            file_path = path.lstrip('/')
            
            # Check if file already exists
            file_exists = self._exists_raw(file_path)
            
            if file_exists:
                # File exists, use update method
                logger.info(f"File {file_path} exists, using update method")
                response = self.supabase.storage.from_(self.bucket_name).update(
                    path=file_path,
                    file=content,
                    file_options={
                        "content-type": kwargs.get("content_type", "application/octet-stream"),
                        "upsert": "true"  # Ensure update works even if there are issues
                    }
                )
            else:
                # File doesn't exist, use upload method
                logger.info(f"File {file_path} doesn't exist, using upload method")
                response = self.supabase.storage.from_(self.bucket_name).upload(
                    path=file_path,
                    file=content,
                    file_options={"content-type": kwargs.get("content_type", "application/octet-stream")}
                )
            
            # Check if operation was successful
            if response and (not isinstance(response, dict) or response.get("error") is None):
                operation = "updated" if file_exists else "uploaded"
                logger.info(f"Successfully {operation} file to Supabase: {file_path}")
                return True
            else:
                logger.error(f"Operation failed: {response}")
                return False
                
        except Exception as e:
            logger.error(f"Error writing file {path} to Supabase: {str(e)}")
            return False
    
    def _delete_raw(self, path: str) -> bool:
        """Delete file from Supabase Storage"""
        try:
            # Remove leading slash if present
            file_path = path.lstrip('/')
            
            # Delete file from Supabase Storage
            response = self.supabase.storage.from_(self.bucket_name).remove([file_path])
            
            # Check if deletion was successful
            if response and (not isinstance(response, dict) or response.get("error") is None):
                logger.info(f"Successfully deleted file from Supabase: {file_path}")
                return True
            else:
                logger.error(f"Deletion failed: {response}")
                return False
                
        except Exception as e:
            logger.error(f"Error deleting {path} from Supabase: {str(e)}")
            return False
    
    def _list_raw(self, path: str = None, max_depth: int = 3, include_hidden: bool = False) -> List[Dict[str, Any]]:
        """List files in Supabase Storage"""
        try:
            # Remove leading slash if present
            list_path = (path or self.base_path).lstrip('/')
            
            # List files from Supabase Storage
            response = self.supabase.storage.from_(self.bucket_name).list(list_path)
            
            items = []
            if response and isinstance(response, list):
                for item in response:
                    # Skip hidden files if not included
                    if not include_hidden and item.get('name', '').startswith('.'):
                        continue
                    
                    # Calculate full path
                    full_path = f"{list_path}/{item['name']}" if list_path else item['name']
                    
                    items.append({
                        "name": item.get('name', ''),
                        "path": full_path,
                        "type": "directory" if item.get('metadata', {}).get('mimetype') == 'application/x-directory' else "file",
                        "size_bytes": item.get('metadata', {}).get('size', 0),
                        "size_mb": round(item.get('metadata', {}).get('size', 0) / (1024 * 1024), 2),
                        "modified_time": item.get('updated_at', ''),
                        "extension": Path(item.get('name', '')).suffix.lower(),
                        "is_hidden": item.get('name', '').startswith('.'),
                        "mime_type": item.get('metadata', {}).get('mimetype', '')
                    })
            
            return items
            
        except Exception as e:
            logger.error(f"Error listing directory {path} from Supabase: {str(e)}")
            return []
    
    def _exists_raw(self, path: str) -> bool:
        """Check if path exists in Supabase Storage"""
        try:
            # Remove leading slash if present
            file_path = path.lstrip('/')
            
            # Get the parent directory and filename
            parent_dir = os.path.dirname(file_path)
            file_name = os.path.basename(file_path)
            
            # If no parent directory, check root
            if not parent_dir:
                parent_dir = ""
            
            try:
                # List files in the parent directory
                response = self.supabase.storage.from_(self.bucket_name).list(parent_dir)
                
                if response and isinstance(response, list):
                    # Check if our filename exists in the directory
                    for item in response:
                        if item.get('name') == file_name:
                            return True
                
                return False
                
            except Exception as e:
                logger.warning(f"Error listing directory {parent_dir}: {str(e)}")
                return False
                
        except Exception as e:
            logger.warning(f"Error checking if file {path} exists: {str(e)}")
            return False
    
    def _create_directory_raw(self, path: str) -> bool:
        """Create directory in Supabase Storage"""
        try:
            # Remove leading slash if present
            dir_path = path.lstrip('/')
            
            # Create a placeholder file to establish the directory
            placeholder_content = b"# Directory placeholder"
            placeholder_path = f"{dir_path}/.placeholder"
            
            response = self.supabase.storage.from_(self.bucket_name).upload(
                path=placeholder_path,
                file=placeholder_content,
                file_options={"content-type": "text/plain"}
            )
            
            # Check if upload was successful
            if response and not isinstance(response, dict) or response.get("error") is None:
                return True
            else:
                logger.error(f"Directory creation failed: {response}")
                return False
                
        except Exception as e:
            logger.error(f"Error creating directory {path} in Supabase: {str(e)}")
            return False
    
    def _save_text(self, file_path: str, content: Any, encoding: str = 'utf-8', **kwargs) -> Dict[str, Any]:
        """Save text content to Supabase Storage"""
        try:
            # Convert content to bytes with specified encoding
            if isinstance(content, str):
                content_bytes = content.encode(encoding)
            else:
                content_bytes = str(content).encode(encoding)
            
            # Use the raw write method
            success = self._write_raw(file_path, content_bytes, **kwargs)
            
            if success:
                return {
                    "success": True,
                    "message": f"Text file saved to Supabase: {file_path}",
                    "file_path": file_path,
                    "content_length": len(content_bytes)
                }
            else:
                return {"success": False, "error": "Failed to upload to Supabase", "file_path": file_path}
                
        except Exception as e:
            logger.error(f"Error saving text file {file_path} to Supabase: {str(e)}")
            return {"success": False, "error": str(e), "file_path": file_path}
    
    def _read_text(self, file_path: str, encoding: str = 'utf-8', **kwargs) -> Dict[str, Any]:
        """Read text content from Supabase Storage"""
        try:
            # Use the raw read method
            content_bytes = self._read_raw(file_path, **kwargs)
            content = content_bytes.decode(encoding)
            
            return {
                "success": True,
                "content": content,
                "file_path": file_path,
                "content_length": len(content)
            }
        except Exception as e:
            logger.error(f"Error reading text file {file_path} from Supabase: {str(e)}")
            return {"success": False, "error": str(e), "file_path": file_path}
    
    def _save_json(self, file_path: str, content: Any, indent: int = 2, **kwargs) -> Dict[str, Any]:
        """Save JSON content to Supabase Storage"""
        try:
            # Convert content to JSON string
            if isinstance(content, str):
                # If it's already a string, try to parse it to validate JSON
                json.loads(content)
                json_content = content
            else:
                json_content = json.dumps(content, indent=indent, ensure_ascii=False)
            
            # Convert to bytes
            content_bytes = json_content.encode('utf-8')
            
            # Use the raw write method
            success = self._write_raw(file_path, content_bytes, **kwargs)
            
            if success:
                return {
                    "success": True,
                    "message": f"JSON file saved to Supabase: {file_path}",
                    "file_path": file_path
                }
            else:
                return {"success": False, "error": "Failed to upload to Supabase", "file_path": file_path}
                
        except Exception as e:
            logger.error(f"Error saving JSON file {file_path} to Supabase: {str(e)}")
            return {"success": False, "error": str(e), "file_path": file_path}
    
    def _read_json(self, file_path: str, **kwargs) -> Dict[str, Any]:
        """Read JSON content from Supabase Storage"""
        try:
            # Use the raw read method
            content_bytes = self._read_raw(file_path, **kwargs)
            content_str = content_bytes.decode('utf-8')
            
            # Parse JSON
            content = json.loads(content_str)
            
            return {
                "success": True,
                "content": content,
                "file_path": file_path
            }
        except Exception as e:
            logger.error(f"Error reading JSON file {file_path} from Supabase: {str(e)}")
            return {"success": False, "error": str(e), "file_path": file_path}
    
    def get_file_info(self, file_path: str) -> Dict[str, Any]:
        """Get comprehensive information about a file in Supabase Storage"""
        try:
            resolved_path = self._resolve_path(file_path)
            
            if not self._exists_raw(resolved_path):
                return {"success": False, "error": f"File {file_path} does not exist"}
            
            # Get file content to determine size
            content = self._read_raw(resolved_path)
            
            # Try to get additional metadata from list
            list_path = os.path.dirname(resolved_path.lstrip('/'))
            file_name = os.path.basename(resolved_path)
            
            metadata = {}
            try:
                files = self.supabase.storage.from_(self.bucket_name).list(list_path)
                for file_info in files:
                    if file_info.get('name') == file_name:
                        metadata = file_info.get('metadata', {})
                        break
            except Exception:
                pass
            
            return {
                "success": True,
                "file_path": resolved_path,
                "file_name": Path(resolved_path).name,
                "file_extension": Path(resolved_path).suffix.lower(),
                "mime_type": metadata.get('mimetype', self.get_mime_type(resolved_path)),
                "size_bytes": len(content),
                "size_mb": round(len(content) / (1024 * 1024), 2),
                "modified_time": metadata.get('updated_at', datetime.now().isoformat()),
                "is_file": True,
                "is_directory": False,
                "is_readable": True,
                "is_writable": True,
                "exists": True
            }
        except Exception as e:
            logger.error(f"Error getting file info for {file_path}: {str(e)}")
            return {"success": False, "error": str(e), "file_path": file_path}