import os
import re
import shutil
import uuid
import asyncio
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime
from abc import abstractmethod
from urllib.parse import quote, unquote, urlparse

from .storage_base import StorageBase
from .tool_utils import URL_CHAR
from ..core.logging import logger
from ..core.sequential import sequential
from ..core.decorators import async_atomic_method


class FileStorageHandler(StorageBase):
    """
    Reference implementation showing all available _raw_xxx methods.
    This class serves as a template for developers creating new storage handlers.
    Concrete handlers only need to implement the _raw_xxx methods they need.
    """
    
    def __init__(self, base_path: str = ".", **kwargs):
        """
        Initialize the storage handler.
        
        Args:
            base_path (str): Base directory for storage operations (default: current directory)
            **kwargs: Additional keyword arguments for parent class initialization
        """
        super().__init__(base_path=base_path, **kwargs)
    
    # ____________________ How to use it ____________________ #
    def create(self, file_path: str, content: Any, **kwargs) -> Dict[str, Any]:
        return super().save(file_path, content, **kwargs)
    
    def read(self, file_path: Optional[str] = None, url: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        return super().read(file_path=file_path, url=url, **kwargs)
    
    def list(self, path: str = None, max_depth: int = 3, include_hidden: bool = False) -> Dict[str, Any]:
        return super().list(path, max_depth, include_hidden)
    
    def delete(self, file_path: str, **kwargs) -> Dict[str, Any]:
        return super().delete(file_path, **kwargs)
    
    def move(self, source: str, destination: str, **kwargs) -> Dict[str, Any]:
        return super().move(source, destination, **kwargs)
    
    def copy(self, source: str, destination: str, **kwargs) -> Dict[str, Any]:
        return super().copy(source, destination, **kwargs)
    
    def create_directory(self, path: str, **kwargs) -> Dict[str, Any]:
        return super().create_directory(path, **kwargs)
    
    def _get_file_url(self, file_path: str, **kwargs) -> str:
        """Create URL for others access"""
        return super()._get_file_url(file_path, **kwargs)
    
    
    
    # ____________________ Required Methods ____________________ #
    @abstractmethod
    def _initialize_storage(self):
        """Initialize storage - must be implemented by subclasses"""
        pass
    
    @abstractmethod
    def _read_raw(self, path: str, **kwargs) -> bytes:
        """Read raw file content - must be implemented by subclasses"""
        pass
    
    @abstractmethod
    def _write_raw(self, path: str, content: bytes, **kwargs) -> bool:
        """Write raw file content - must be implemented by subclasses"""
        pass
    
    @abstractmethod
    def _delete_raw(self, path: str) -> bool:
        """Delete file or directory - must be implemented by subclasses"""
        pass
    
    @abstractmethod
    def _exists_raw(self, path: str) -> bool:
        """Check if path exists - must be implemented by subclasses"""
        pass
    
    @abstractmethod
    def _create_directory_raw(self, path: str) -> bool:
        """Create directory - must be implemented by subclasses"""
        pass
    
    @abstractmethod
    def _list_raw(self, path: str = None, **kwargs) -> List[Dict[str, Any]]:
        """List files and directories - must be implemented by subclasses"""
        pass
    
    @abstractmethod
    def force_cache_file_to_url(self, cache_file_path: str) -> str:
        """Convert absolute local cache file path to URL for others access - must be implemented by subclasses"""
        pass

    @abstractmethod
    def force_cache_url_to_file(self, cache_url: str, cache_dir: str) -> str:
        """Convert cache URL back to absolute local cache file path - must be implemented by subclasses"""
        pass 

    @abstractmethod
    def find_cache_urls(self, text: str, cache_dir: str) -> List[str]:
        """"Find all cache URLs in text that belong to cache_dir - must be implemented by subclasses"""
        pass

    # ____________________ Extra Mapping ____________________ #
    def create_file(self, file_path: str, content: Any, **kwargs) -> Dict[str, Any]:
        return self.save(file_path, content, **kwargs)
    
    def read_file(self, file_path: str = None, url: str = None, **kwargs) -> Dict[str, Any]:
        return self.read(file_path = file_path, url = url, **kwargs)
    
    def list_files(self, path: str = None, max_depth: int = 3, include_hidden: bool = False) -> Dict[str, Any]:
        return self.list(path, max_depth, include_hidden)
    
    def delete_file(self, file_path: str, **kwargs) -> Dict[str, Any]:
        return self.delete(file_path, **kwargs)
    
    def move_file(self, source: str, destination: str, **kwargs) -> Dict[str, Any]:
        return self.move(source, destination, **kwargs)
    
    def copy_file(self, source: str, destination: str, **kwargs) -> Dict[str, Any]:
        return self.copy(source, destination, **kwargs)
    
    def _normalize_path_to_posix(self, path: str) -> str:
        """
        Normalize a file path to POSIX style for cross-platform compatibility.

        This method:
        1. Normalizes the path (resolves .., ., etc.)
        2. Removes drive letters (Windows)
        3. Converts all separators to forward slashes
        4. Strips leading slashes

        Args:
            path: File path to normalize (supports Linux/Windows)

        Returns:
            str: Normalized POSIX-style path without leading slash
        """
        # Normalize path
        normalized = os.path.normpath(path)
        # Remove drive letter (Windows)
        _, normalized = os.path.splitdrive(normalized)
        # Convert to POSIX style: replace both separators to handle cross-platform
        # Use explicit backslash replacement to work on both Linux and Windows
        posix_path = normalized.replace("\\", "/").lstrip("/")
        return posix_path


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
        """Initialize local storage - ensure base directory exists"""
        try:
            # Ensure the base directory exists
            Path(self.base_path).mkdir(parents=True, exist_ok=True)
            logger.info(f"Local storage initialized with base path: {self.base_path}")
        except Exception as e:
            logger.error(f"Error initializing local storage: {str(e)}")
            raise
    
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
    
    def _get_file_url(self, file_path: str) -> str:
        """Create URL for others access"""
        try:
            # file_path provided to this method is already the resolved system path
            # (callers pass translate_in(file_path)). To avoid duplicating base_path,
            # simply convert to an absolute path and build the file URL.
            abs_path = os.path.abspath(file_path)
            return f"file://{abs_path}"
        except Exception:
            # Fallback: return a best-effort URL without altering the path
            return f"file://{file_path}"
        
    def force_cache_file_to_url(self, cache_file_path: str) -> str:
        """
        Convert absolute local path to file:// URL.

        Args:
            cache_file_path: Absolute local file path (e.g., "/tmp/file.mp4" or "C:\\tmp\\file.mp4")

        Returns:
            str: file:// URL with normalized POSIX path (or original input if invalid)

        Raises:
            ValueError: Only if cache_file_path is not an absolute path
        """
        if not cache_file_path:
            logger.warning("cache_file_path is empty, returning as-is")
            return cache_file_path

        # If already a file:// URL, return as-is
        if cache_file_path.startswith("file://"):
            return cache_file_path

        # Validate absolute path
        if not os.path.isabs(cache_file_path):
            raise ValueError(f"cache_file_path must be an absolute path, got: {cache_file_path}")

        # Generate normalized POSIX path
        normalized_path = self._normalize_path_to_posix(cache_file_path)

        # Ensure path starts with / for proper file:// URL format (file:///path)
        path_component = "/" + normalized_path if not normalized_path.startswith("/") else normalized_path

        encoded_path = quote(path_component, safe="/")
        return f"file://{encoded_path}"

    def force_cache_url_to_file(self, cache_url: str, cache_dir: str) -> str:
        """
        Convert file:// URL back to original local cache file path.

        Args:
            cache_url: file:// URL
            cache_dir: Absolute local cache directory (e.g., "/tmp/cache" or "C:\\tmp\\cache")
                      Required to reconstruct the original absolute path

        Returns:
            str: Original local cache file path (or original input if invalid)

        Raises:
            ValueError: Only if cache_dir is not an absolute path
        """
        if not cache_url:
            logger.warning("cache_url is empty, returning as-is")
            return cache_url

        # If not a file:// URL, return as-is
        if not cache_url.startswith("file://"):
            return cache_url

        # Validate cache_dir
        if not cache_dir:
            logger.warning("cache_dir is not provided, returning original URL")
            return cache_url

        # ONLY check that raises ValueError
        if not os.path.isabs(cache_dir):
            raise ValueError(f"cache_dir must be an absolute path, got: {cache_dir}")

        # Parse file:// URL and extract path with symmetric decoding.
        parsed = urlparse(cache_url)
        url_path = unquote(parsed.path).lstrip('/')

        # Normalize cache_dir to match URL format (POSIX without leading slash)
        cache_dir_normalized = self._normalize_path_to_posix(cache_dir)

        # Verify URL path starts with cache_dir
        if not url_path.startswith(cache_dir_normalized):
            logger.warning(
                f"URL path does not start with cache_dir. "
                f"Expected prefix: {cache_dir_normalized}, got: {url_path}. "
                f"Returning original URL."
            )
            return cache_url

        # Strict boundary check: ensure we're not matching partial directory names
        # The character after cache_dir_normalized must be '/' or end of string
        if len(url_path) > len(cache_dir_normalized):
            next_char = url_path[len(cache_dir_normalized)]
            if next_char != '/':
                logger.warning(
                    f"URL path contains cache_dir as substring but not as complete directory. "
                    f"Expected '{cache_dir_normalized}/' but found '{cache_dir_normalized}{next_char}'. "
                    f"Full URL path: {url_path}. Returning original URL."
                )
                return cache_url

        # Extract relative path after cache_dir
        relative_path = url_path[len(cache_dir_normalized):].lstrip("/")

        # Detect path separator style from cache_dir
        sep = "\\" if "\\" in cache_dir else "/"

        if relative_path:
            return cache_dir.rstrip("/\\") + sep + relative_path.replace("/", sep)
        else:
            return cache_dir

    def find_cache_urls(self, text: str, cache_dir: str) -> List[str]:
        """
        Find all file:// cache URLs in text that belong to cache_dir.

        Args:
            text: Text to search for cache URLs
            cache_dir: Absolute local cache directory

        Returns:
            List[str]: List of file:// URLs found in cache_dir
        """
        if not text:
            return []

        if not cache_dir:
            logger.warning("cache_dir is not provided, returning empty list")
            return []

        # Validate cache_dir is absolute
        if not os.path.isabs(cache_dir):
            logger.warning(f"cache_dir must be an absolute path, got: {cache_dir}")
            return []

        # Normalize cache_dir to POSIX format (for matching in URLs)
        cache_dir_normalized = self._normalize_path_to_posix(cache_dir)

        # Pattern for file:// URLs
        file_url_pattern = re.compile(
            r'file://' + rf'(?P<path>(?:(?!(?:https?|file)://){URL_CHAR})+(?<=[a-zA-Z0-9]))',
            re.IGNORECASE
        )

        results = []
        for match in file_url_pattern.finditer(text):
            url = match.group(0)
            try:
                # Extract and decode the path
                path_part = match.group('path')
                decoded_path = unquote(path_part).lstrip('/')

                # Check if path belongs to cache_dir
                if decoded_path.startswith(cache_dir_normalized):
                    # Boundary check: ensure complete directory match
                    if len(decoded_path) > len(cache_dir_normalized):
                        next_char = decoded_path[len(cache_dir_normalized)]
                        if next_char == '/':
                            results.append(url)
                    else:
                        # Exact match
                        results.append(url)
            except Exception as e:
                logger.debug(f"Error processing file URL {url}: {e}")
                continue

        return results
    


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
        # Call parent constructor first
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
        logger.info(f"Creating Supabase client with URL: {self.supabase_url[:30]}...")
        from supabase import create_client, Client
        self.supabase: Client = create_client(self.supabase_url, self.supabase_key)
        logger.info(f"Successfully initialized Supabase client for bucket: {bucket_name}")

        # Async Supabase client will be initialized lazily
        self.async_supabase = None
        self._async_supabase_initialized = False
        self._async_lock = asyncio.Lock()
        logger.info("Async Supabase client will be initialized on first async use")
        
        # Initialize storage after all attributes are set
        self._initialize_storage()
    
    
    def _initialize_storage(self):
        """Initialize remote storage - verify bucket exists and is accessible"""
        # Check if required attributes are available
        if not hasattr(self, 'bucket_name') or not hasattr(self, 'supabase'):
            # If attributes aren't set yet, skip initialization
            # This will be called again after attributes are set in __init__
            return
        
        try:
            # Test bucket access by listing files (empty list is fine)
            logger.info(f"Testing bucket access for: {self.bucket_name}")
            self.supabase.storage.from_(self.bucket_name).list()
            logger.info(f"Successfully connected to Supabase bucket: {self.bucket_name}")
        except Exception as e:
            logger.warning(f"Could not verify bucket access: {str(e)}")
            # Don't raise error as bucket might be empty or have different permissions
    
    def translate_in(self, file_path: str) -> str:
        """Resolve file path for remote storage"""
        # Use the translate_in method to combine base_path with file_path
        # For Supabase, we need to handle the special case where base_path is "/"
        if self.base_path == "/":
            # If base_path is "/", just clean the file_path
            translated_path = file_path.lstrip('/')
            return translated_path
        else:
            # Use the standard translate_in method
            translated_path = super().translate_in(file_path)
            return translated_path
    
    @sequential(scope="instance")
    def _read_raw(self, path: str, **kwargs) -> bytes:
        """Read raw file content from Supabase Storage"""
        try:
            # Ensures paths like '../raw_files/doc.pdf' are treated within bucket root
            file_path = os.path.normpath('/' + path.lstrip('/')).lstrip('/')
            
            # Download file from Supabase Storage
            acutal_path = os.path.normpath(file_path.lstrip('/'))
            response = self.supabase.storage.from_(self.bucket_name).download(acutal_path)
            
            if isinstance(response, bytes):
                return response
            else:
                # If response is not bytes, try to convert
                return bytes(response) if response else b""
                
        except Exception as e:
            logger.error(f"Error reading file {path} from Supabase: {str(e)}")
            raise
    
    @sequential(scope="instance")
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
                acutal_path = os.path.normpath(file_path.lstrip('/'))
                response = self.supabase.storage.from_(self.bucket_name).update(
                    path=acutal_path,
                    file=content,
                    file_options={
                        "content-type": kwargs.get("content_type", "application/octet-stream"),
                        "upsert": "true"  # Ensure update works even if there are issues
                    }
                )
            else:
                # File doesn't exist, use upload method
                logger.info(f"File {file_path} doesn't exist, using upload method")
                acutal_path = os.path.normpath(file_path.lstrip('/'))
                # change potential \\ to /
                acutal_path = acutal_path.replace("\\", "/")
                response = self.supabase.storage.from_(self.bucket_name).upload(
                    path=acutal_path,
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
    
    @sequential(scope="instance")
    def _delete_raw(self, path: str) -> bool:
        """Delete file from Supabase Storage"""
        try:
            # Remove leading slash if present
            file_path = path.lstrip('/')
            
            # Delete file from Supabase Storage
            acutal_path = os.path.normpath(path.lstrip('/'))
            response = self.supabase.storage.from_(self.bucket_name).remove([acutal_path])
            
            # Check if deletion was successful
            # Supabase remove() returns an empty list [] when successful
            if response is not None:
                if isinstance(response, list):
                    # Empty list means successful deletion
                    logger.info(f"Successfully deleted file from Supabase: {file_path}")
                    return True
                elif isinstance(response, dict) and response.get("error") is None:
                    # Some responses might be dict format
                    logger.info(f"Successfully deleted file from Supabase: {file_path}")
                    return True
                else:
                    logger.error(f"Deletion failed: {response}")
                    return False
            else:
                logger.error(f"Deletion failed: {response}")
                return False
                
        except Exception as e:
            logger.error(f"Error deleting {path} from Supabase: {str(e)}")
            return False
    
    @sequential(scope="instance")
    def _list_raw(self, path: str = None, max_depth: int = 3, include_hidden: bool = False) -> List[Dict[str, Any]]:
        """List files in Supabase Storage"""
        try:
            # Remove leading slash if present
            list_path = (path or self.base_path).lstrip('/')
            list_path = os.path.normpath(list_path.lstrip('/'))
            
            # List files from Supabase Storage
            acutal_path = os.path.normpath(list_path.lstrip('/'))
            response = self.supabase.storage.from_(self.bucket_name).list(acutal_path)
            
            
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
    
    @sequential(scope="instance")
    def _exists_raw(self, path: str) -> bool:
        """Check if path exists in Supabase Storage"""
        try:
            # Remove leading slash if present
            file_path = os.path.normpath(path.lstrip('/'))
            
            # Get the parent directory and filename
            parent_dir = os.path.dirname(file_path)
            file_name = os.path.basename(file_path)
            
            # If no parent directory, check root
            if not parent_dir or parent_dir == '.':
                parent_dir = ""
            
            
            try:
                # List files in the parent directory
                acutal_path = os.path.normpath(parent_dir.lstrip('/'))
                response = self.supabase.storage.from_(self.bucket_name).list(acutal_path)
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
    
    @sequential(scope="instance")
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
            if "already exists" in str(e):
                return True
            logger.error(f"Error creating directory {path} in Supabase: {str(e)}")
            return False
    
    @sequential(scope="instance")
    def _get_file_url(self, file_path: str) -> str:
        """Create URL for others access"""
        try:
            public_url: str = self.supabase.storage.from_(self.bucket_name).get_public_url(file_path)
            # Supabase SDK appends a bare '?' with no query params; strip it.
            return public_url.rstrip('?')
        except Exception as e:
            logger.warning(f"Failed to create signed URL for {file_path}: {e}")
            return file_path
    
    @sequential(scope="instance")
    def read(self, url: Optional[str] = None, file_path: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Read content from a file by path or URL."""
        return super().read(file_path=file_path, url=url, **kwargs)
    
    @sequential(scope="instance")
    def _direct_upload(self, file_path: str, upload_path: str, **kwargs) -> dict:
        """Directly upload a file to Supabase Storage, matching save behavior."""
        try:
            # Resolve path
            target_path = self.translate_in(upload_path)
            
            # Check existence and rename if necessary to avoid overwrite
            if self._exists_raw(target_path):
                # Add first 5 digits of UUID as suffix
                uid = uuid.uuid4().hex[:5]
                p = Path(target_path)
                target_path = str(p.parent / f"{p.stem}_{uid}{p.suffix}")
            
            # Normalize path for Supabase
            actual_path = os.path.normpath(target_path.lstrip('/')).replace("\\", "/")
            
            # Configure options
            file_opts = {"content-type": kwargs.get("content_type", "application/octet-stream")}

            with open(file_path, 'rb') as f:
                response = self.supabase.storage.from_(self.bucket_name).upload(
                    path=actual_path,
                    file=f,
                    file_options=file_opts
                )

            if response and (not isinstance(response, dict) or response.get("error") is None):
                return {
                    "success": True,
                    "url": self._get_file_url(target_path),
                    "file_path": self.translate_out(target_path),
                }
            
            logger.error(f"Direct upload failed: {response}")
            return {"success": False, "error": str(response)}
        except Exception as e:
            logger.error(f"Error directly uploading {file_path} to {upload_path}: {e}")
            return {"success": False, "error": str(e)}
    
    @sequential(scope="instance")
    def save(self, file_name: str = "saved_files", content: bytes = None, return_path: bool = False, file_path: Optional[str] = None, **kwargs) -> dict:
        """Save content to a file at the given path."""
        file_name = file_name if file_name.isascii() else self._uuid_name(file_name)
        if not content:
            logger.error("Content is None, cannot save file.")
            return {"success": False, "error": "Content is None"}
        try:
            if  file_path:
                response = super().save(file_path=file_path, content=content, **kwargs)
            else:
                response = super().save(file_path=file_name, content=content, **kwargs)
            if not response.get("success") or "url" not in response:
                return response
            output = {
                "success": True,
                "url": response["url"],
            }
            if return_path:
                output["file_path"] = response.get("file_path", None)
            return output
        except Exception as e:
            logger.error(f"Error saving file {file_name} to Supabase: {str(e)}")
            return {"success": False, "error": str(e)}
        
            
    def _uuid_name(self, name: str) -> str:
        ext = Path(name).suffix
        return f"{uuid.uuid4().hex}{ext}" if ext else uuid.uuid4().hex

    @async_atomic_method
    async def _ensure_async_client(self):
        """Ensure async Supabase client is initialized."""
        from supabase import acreate_client
        if not self._async_supabase_initialized:
            self.async_supabase = await acreate_client(self.supabase_url, self.supabase_key)
            self._async_supabase_initialized = True
            logger.info("Async Supabase client initialized successfully")
        return self.async_supabase

    async def async_direct_upload(self, file_path: str, upload_path: str, **kwargs) -> dict:
        """Asynchronously upload a file to Supabase Storage with retry logic for duplicate names.

        Args:
            file_path: Local file path to upload
            upload_path: Target path in Supabase storage
            **kwargs: Additional options like content_type

        Returns:
            Dict containing success status, URL, and file_path
        """
        max_retries = 3

        try:
            # Ensure async client is initialized
            await self._ensure_async_client()

            # Resolve target path
            target_path = self.translate_in(upload_path)

            # Read file content once
            with open(file_path, 'rb') as f:
                file_content = f.read()

            # Configure upload options
            file_opts = {"content-type": kwargs.get("content_type", "application/octet-stream")}

            # Retry loop
            for attempt in range(max_retries):
                try:
                    # Normalize path for Supabase
                    actual_path = os.path.normpath(target_path.lstrip('/')).replace("\\", "/")

                    # Attempt upload
                    response = await self.async_supabase.storage.from_(self.bucket_name).upload(
                        path=actual_path,
                        file=file_content,
                        file_options=file_opts
                    )

                    # Check if upload was successful
                    if response and (not isinstance(response, dict) or response.get("error") is None):
                        logger.info(f"Successfully uploaded file to Supabase: {actual_path} (attempt {attempt + 1})")
                        return {
                            "success": True,
                            "url": self._get_file_url(target_path),
                            "file_path": self.translate_out(target_path),
                        }

                    # Check if response indicates duplicate file error
                    error_msg = str(response.get("error") if isinstance(response, dict) else response).lower()
                    if "already exists" in error_msg or "duplicate" in error_msg or "409" in error_msg:
                        # Duplicate file, add UUID suffix and retry
                        if attempt < max_retries - 1:
                            uid = uuid.uuid4().hex[:5]
                            p = Path(target_path)
                            target_path = str(p.parent / f"{p.stem}_{uid}{p.suffix}")
                            logger.info(f"File already exists, retrying with UUID suffix: {target_path}")
                            continue

                    # Other error, return immediately
                    logger.error(f"Upload failed: {response}")
                    return {"success": False, "error": str(response)}

                except Exception as e:
                    error_str = str(e).lower()
                    # Check if error is due to duplicate file
                    if "already exists" in error_str or "duplicate" in error_str or "409" in error_str:
                        if attempt < max_retries - 1:
                            # Add UUID suffix and retry
                            uid = uuid.uuid4().hex[:5]
                            p = Path(target_path)
                            target_path = str(p.parent / f"{p.stem}_{uid}{p.suffix}")
                            logger.info(f"File already exists, retrying with UUID suffix: {target_path}")
                            continue
                        else:
                            logger.error(f"Max retries reached for duplicate file: {target_path}")
                            return {"success": False, "error": f"File already exists after {max_retries} attempts"}
                    else:
                        # Other error, raise immediately
                        logger.error(f"Upload failed with non-duplicate error: {e}")
                        raise

            # Max retries exhausted
            logger.error(f"Max retries ({max_retries}) exhausted for {upload_path}")
            return {"success": False, "error": f"Max retries ({max_retries}) exhausted"}

        except Exception as e:
            logger.error(f"Error asynchronously uploading {file_path} to {upload_path}: {e}")
            return {"success": False, "error": str(e)}

    async def async_save(self, file_name: str = "saved_files", content: bytes = None, return_path: bool = False, file_path: Optional[str] = None, **kwargs) -> dict:
        """Asynchronously save content to a file at the given path with retry logic for duplicate names.

        Unlike the synchronous save method, this does not pre-check for file existence.
        Instead, it directly attempts upload and retries with UUID suffix on duplicate errors.

        Args:
            file_name: Default file name if file_path is not provided
            content: File content as bytes
            return_path: Whether to include file_path in response
            file_path: Specific file path to save to (overrides file_name)
            **kwargs: Additional options like content_type

        Returns:
            Dict containing success status, URL, and optionally file_path
        """
        max_retries = 3

        try:
            # Validate content
            if not content:
                logger.error("Content is None, cannot save file.")
                return {"success": False, "error": "Content is None"}

            # Keep parity with sync save for non-bytes content types
            if not isinstance(content, bytes):
                return self.save(
                    file_name=file_name,
                    content=content,
                    return_path=return_path,
                    file_path=file_path,
                    **kwargs
                )

            # Handle file name (convert non-ASCII to UUID)
            file_name = file_name if file_name.isascii() else self._uuid_name(file_name)

            # Determine target path
            target_path_input = file_path if file_path else file_name

            # Ensure async client is initialized
            await self._ensure_async_client()

            # Resolve target path
            target_path = self.translate_in(target_path_input)

            # Configure upload options
            file_opts = {"content-type": kwargs.get("content_type", "application/octet-stream")}

            # Retry loop
            for attempt in range(max_retries):
                try:
                    # Normalize path for Supabase
                    actual_path = os.path.normpath(target_path.lstrip('/')).replace("\\", "/")

                    # Attempt upload
                    response = await self.async_supabase.storage.from_(self.bucket_name).upload(
                        path=actual_path,
                        file=content,
                        file_options=file_opts
                    )

                    # Check if upload was successful
                    if response and (not isinstance(response, dict) or response.get("error") is None):
                        logger.info(f"Successfully saved file to Supabase: {actual_path} (attempt {attempt + 1})")

                        output = {
                            "success": True,
                            "url": self._get_file_url(target_path),
                        }

                        if return_path:
                            output["file_path"] = self.translate_out(target_path)

                        return output

                    # Check if response indicates duplicate file error
                    error_msg = str(response.get("error") if isinstance(response, dict) else response).lower()
                    if "already exists" in error_msg or "duplicate" in error_msg or "409" in error_msg:
                        # Duplicate file, add UUID suffix and retry
                        if attempt < max_retries - 1:
                            uid = uuid.uuid4().hex[:5]
                            p = Path(target_path)
                            target_path = str(p.parent / f"{p.stem}_{uid}{p.suffix}")
                            logger.info(f"File already exists, retrying with UUID suffix: {target_path}")
                            continue

                    # Other error, return immediately
                    logger.error(f"Upload failed: {response}")
                    return {"success": False, "error": str(response)}

                except Exception as e:
                    error_str = str(e).lower()
                    # Check if error is due to duplicate file
                    if "already exists" in error_str or "duplicate" in error_str or "409" in error_str:
                        if attempt < max_retries - 1:
                            # Add UUID suffix and retry
                            uid = uuid.uuid4().hex[:5]
                            p = Path(target_path)
                            target_path = str(p.parent / f"{p.stem}_{uid}{p.suffix}")
                            logger.info(f"File already exists, retrying with UUID suffix: {target_path}")
                            continue
                        else:
                            logger.error(f"Max retries reached for duplicate file: {target_path}")
                            return {"success": False, "error": f"File already exists after {max_retries} attempts"}
                    else:
                        # Other error, raise immediately
                        logger.error(f"Upload failed with non-duplicate error: {e}")
                        raise

            # Max retries exhausted
            logger.error(f"Max retries ({max_retries}) exhausted for {target_path_input}")
            return {"success": False, "error": f"Max retries ({max_retries}) exhausted"}

        except Exception as e:
            logger.error(f"Error asynchronously saving file to Supabase: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def force_cache_file_to_url(self, cache_file_path: str) -> str:
        """
        Convert absolute local cache file path to Supabase URL.

        Args:
            cache_file_path: Absolute local file path (e.g., "/tmp/cache/file.mp4" or "C:\\tmp\\cache\\file.mp4")

        Returns:
            str: Supabase public URL (or original input if invalid)

        Raises:
            ValueError: Only if cache_file_path is not an absolute path
        """
        if not cache_file_path:
            logger.warning("cache_file_path is empty, returning as-is")
            return cache_file_path

        # If already a URL, return as-is
        if cache_file_path.startswith(("http://", "https://", "file://")):
            return cache_file_path

        # Validate absolute path (ONLY check that raises ValueError)
        if not os.path.isabs(cache_file_path):
            raise ValueError(f"cache_file_path must be an absolute path, got: {cache_file_path}")

        # Generate normalized POSIX path (removes drive letters, converts to forward slashes, strips leading slash)
        normalized_path = self._normalize_path_to_posix(cache_file_path)

        # Get storage path and public URL
        storage_path = self.translate_in(normalized_path)
        public_url = self._get_file_url(storage_path)

        return public_url

    def force_cache_url_to_file(self, cache_url: str, cache_dir: str) -> str:
        """
        Convert Supabase URL back to original local cache file path.

        Args:
            cache_url: Supabase URL
            cache_dir: Absolute local cache directory (e.g., "/tmp/cache" or "C:\\tmp\\cache")

        Returns:
            str: Original local cache file path (or original input if invalid)

        Raises:
            ValueError: Only if cache_dir is not an absolute path
        """
        if not cache_url:
            logger.warning("cache_url is empty, returning as-is")
            return cache_url

        # If not a URL, return as-is (assume it's already a file path)
        if not cache_url.startswith(("http://", "https://")):
            return cache_url

        # Validate cache_dir
        if not cache_dir:
            logger.warning("cache_dir is not provided, returning original URL")
            return cache_url

        # ONLY check that raises ValueError
        if not os.path.isabs(cache_dir):
            raise ValueError(f"cache_dir must be an absolute path, got: {cache_dir}")

        # Parse URL to get path
        parsed = urlparse(cache_url)
        url_path = unquote(parsed.path)

        # Normalize cache_dir to POSIX (without leading slash, to match URL path format)
        cache_dir_normalized = self._normalize_path_to_posix(cache_dir)

        # Find the normalized cache_dir in the URL path.
        idx = url_path.find(cache_dir_normalized)

        if idx < 0:
            logger.warning(
                f"URL path does not contain cache_dir. "
                f"Expected to find: {cache_dir_normalized}, in URL path: {url_path}. "
                f"Returning original URL."
            )
            return cache_url

        # Left boundary check: ensure match starts at path start or after a slash
        if idx > 0 and url_path[idx - 1] != '/':
            logger.warning(
                f"URL path contains cache_dir as substring but not as complete directory segment. "
                f"Expected left boundary before '{cache_dir_normalized}' to be '/', got: '{url_path[idx - 1]}'. "
                f"Full URL path: {url_path}. Returning original URL."
            )
            return cache_url

        # Extract the full path from cache_dir onwards
        full_path_in_url = url_path[idx:]

        # Extract the relative path after cache_dir (with strict boundary check)
        if full_path_in_url.startswith(cache_dir_normalized):
            # Strict boundary check: ensure we're not matching partial directory names
            # The character after cache_dir_normalized must be '/' or end of string
            if len(full_path_in_url) > len(cache_dir_normalized):
                next_char = full_path_in_url[len(cache_dir_normalized)]
                if next_char != '/':
                    logger.warning(
                        f"URL path contains cache_dir as substring but not as complete directory. "
                        f"Expected '{cache_dir_normalized}/' but found '{cache_dir_normalized}{next_char}'. "
                        f"Full URL path: {url_path}. Returning original URL."
                    )
                    return cache_url

            remaining_path = full_path_in_url[len(cache_dir_normalized):].lstrip("/")
        else:
            logger.warning(
                f"Unexpected URL path format. "
                f"Expected path to start with: {cache_dir_normalized}, got: {full_path_in_url}. "
                f"Returning original URL."
            )
            return cache_url

        # Detect path separator from cache_dir
        sep = "\\" if "\\" in cache_dir else "/"

        # Reconstruct absolute path
        if remaining_path:
            return cache_dir.rstrip("/\\") + sep + remaining_path.replace("/", sep)
        else:
            return cache_dir

    def find_cache_urls(self, text: str, cache_dir: str) -> List[str]:
        """
        Find all Supabase cache URLs in text that belong to cache_dir.

        Args:
            text: Text to search for cache URLs
            cache_dir: Absolute local cache directory (required, used to filter URLs)

        Returns:
            List[str]: List of Supabase URLs found in this bucket that belong to cache_dir
        """
        if not text:
            return []

        if not cache_dir:
            logger.warning("cache_dir is required, returning empty list")
            return []

        # Validate cache_dir is absolute
        if not os.path.isabs(cache_dir):
            logger.warning(f"cache_dir must be an absolute path, got: {cache_dir}")
            return []

        # Normalize cache_dir to POSIX format (for matching in URLs)
        cache_dir_normalized = self._normalize_path_to_posix(cache_dir)

        # Pattern for Supabase storage URLs in this bucket
        # Format: https://{host}.supabase.co/storage/v1/object/public/{bucket}/{path}
        supabase_url_pattern = re.compile(
            r'https://(?P<host>[a-z0-9]+)\.supabase\.co/'
            r'storage/v1/object/public/' + re.escape(self.bucket_name) + r'/'
            rf'(?P<path>(?:(?!https?://){URL_CHAR})+(?<=[a-zA-Z0-9]))',
            re.IGNORECASE
        )

        # Find and filter URLs that contain cache_dir_normalized
        results = []
        for match in supabase_url_pattern.finditer(text):
            url = match.group(0)
            try:
                # Extract and decode the path
                path_part = match.group('path')
                decoded_path = unquote(path_part).lstrip('/')

                # Find cache_dir_normalized anywhere in decoded_path (it may be
                # preceded by base_path segments added by translate_in)
                idx = decoded_path.find(cache_dir_normalized)
                if idx < 0:
                    continue
                # Left boundary check: must start at path start or after a '/'
                if idx > 0 and decoded_path[idx - 1] != '/':
                    continue
                # Right boundary check: must be followed by '/' or end of string
                end_idx = idx + len(cache_dir_normalized)
                if end_idx < len(decoded_path) and decoded_path[end_idx] != '/':
                    continue

                results.append(url)
            except Exception as e:
                logger.debug(f"Error processing URL {url}: {e}")
                continue

        return results



sample_project_id = "rp6jerj"

#### Change the save location if you want
# sample_execution_info = {"user_id": "eax_test_user"}
sample_execution_info = {"user_id": "eax_test_user", "execution_id": "unified_execution_location"}
sample_supabase_bucket = "howone"
base_path = "/projects/{project_short_id}/executions/{user_id}/{execution_id}"

def get_test_supabase_handler(project_short_id: str = sample_project_id, execution_info: dict = sample_execution_info):
    if "execution_id" not in execution_info:
        execution_info["execution_id"] = str(uuid.uuid4())
    if "user_id" not in execution_info:
        execution_info["user_id"] = str(uuid.uuid4())
    
    storage_handler = SupabaseStorageHandler(
        bucket_name=sample_supabase_bucket,
        base_path=base_path.format(project_short_id=project_short_id, user_id=execution_info["user_id"], execution_id=execution_info["execution_id"]),
        return_file_url = True
    )
    
    return storage_handler
