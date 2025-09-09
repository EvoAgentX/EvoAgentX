import json
import pickle
import csv
import yaml
import xml.etree.ElementTree as ET
import shutil
from typing import Dict, Any, List
from pathlib import Path
import mimetypes
import hashlib
from abc import ABC, abstractmethod

# For handling various file types
try:
    import pymupdf
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

try:
    from PIL import Image
    PILLOW_AVAILABLE = True
except ImportError:
    PILLOW_AVAILABLE = False

try:
    from openpyxl import Workbook, load_workbook
    EXCEL_AVAILABLE = True
except ImportError:
    EXCEL_AVAILABLE = False

from ..core.module import BaseModule
from ..core.logging import logger


class StorageBase(BaseModule, ABC):
    """
    Abstract base class for comprehensive storage operations supporting various file types.
    Provides unified interface for local and remote storage operations.
    """
    
    def __init__(self, base_path: str = ".", **kwargs):
        """
        Initialize the StorageBase with configuration options.
        
        Args:
            base_path (str): Base directory for storage operations (default: current directory)
            **kwargs: Additional keyword arguments for parent class initialization
        """
        super().__init__(**kwargs)
        self.base_path = base_path
        
        # File types that support append operations
        self.appendable_formats = {
            '.txt': self._append_text,
            '.json': self._append_json,
            '.csv': self._append_csv,
            '.yaml': self._append_yaml,
            '.yml': self._append_yaml,
            '.pickle': self._append_pickle,
            '.xlsx': self._append_excel
        }
        
        # Initialize storage-specific setup
        self._initialize_storage()
    
    # Abstract methods that must be implemented by subclasses
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
    def _list_raw(self, path: str = None, **kwargs) -> List[Dict[str, Any]]:
        """List files and directories - must be implemented by subclasses"""
        pass
    
    @abstractmethod
    def _exists_raw(self, path: str) -> bool:
        """Check if path exists - must be implemented by subclasses"""
        pass
    
    @abstractmethod
    def _create_directory_raw(self, path: str) -> bool:
        """Create directory - must be implemented by subclasses"""
        pass
    
    def _resolve_path(self, file_path: str) -> str:
        """Resolve file path, prepending base_path if it's a relative path"""
        # For remote storage, we might need different path resolution logic
        # This is a basic implementation that can be overridden by subclasses
        if not file_path.startswith('/') and not file_path.startswith('./'):
            # If it's a relative path, prepend the base path
            return f"{self.base_path}/{file_path}"
        return file_path
    
    def _initialize_storage(self):
        """
        Initialize storage-specific setup. Override in subclasses for storage-specific initialization.
        """
        pass
    
    def get_file_type(self, file_path: str) -> str:
        """Get the file extension from a file path"""
        return Path(file_path).suffix.lower()
    
    def get_mime_type(self, file_path: str) -> str:
        """Get the MIME type of a file"""
        return mimetypes.guess_type(file_path)[0] or 'application/octet-stream'
    
    def exists(self, path: str) -> bool:
        """Check if path exists"""
        resolved_path = self._resolve_path(path)
        return self._exists_raw(resolved_path)
    
    def get_file_info(self, file_path: str) -> Dict[str, Any]:
        """Get comprehensive information about a file"""
        try:
            resolved_path = self._resolve_path(file_path)
            if not self._exists_raw(resolved_path):
                return {"success": False, "error": f"File {file_path} does not exist"}
            
            # For now, return basic info - subclasses can override for more details
            return {
                "success": True,
                "file_path": resolved_path,
                "file_name": Path(resolved_path).name,
                "file_extension": Path(resolved_path).suffix.lower(),
                "mime_type": self.get_mime_type(resolved_path),
                "exists": True
            }
        except Exception as e:
            logger.error(f"Error getting file info for {file_path}: {str(e)}")
            return {"success": False, "error": str(e), "file_path": file_path}
    
    def calculate_file_hash(self, file_path: str, algorithm: str = 'md5') -> Dict[str, Any]:
        """Calculate hash of a file"""
        try:
            resolved_path = self._resolve_path(file_path)
            content = self._read_raw(resolved_path)
            hash_func = hashlib.new(algorithm)
            hash_func.update(content)
            
            return {
                "success": True,
                "file_path": resolved_path,
                "algorithm": algorithm,
                "hash": hash_func.hexdigest()
            }
        except Exception as e:
            logger.error(f"Error calculating hash for {file_path}: {str(e)}")
            return {"success": False, "error": str(e), "file_path": file_path}
    
    def create_directory(self, path: str) -> Dict[str, Any]:
        """Create directory"""
        try:
            resolved_path = self._resolve_path(path)
            success = self._create_directory_raw(resolved_path)
            if success:
                return {"success": True, "path": resolved_path, "message": "Directory created successfully"}
            else:
                return {"success": False, "error": "Failed to create directory", "path": resolved_path}
        except Exception as e:
            logger.error(f"Error creating directory {path}: {str(e)}")
            return {"success": False, "error": str(e), "path": path}
    
    def delete(self, path: str) -> Dict[str, Any]:
        """Delete file or directory"""
        try:
            resolved_path = self._resolve_path(path)
            success = self._delete_raw(resolved_path)
            if success:
                return {"success": True, "path": resolved_path, "message": "Deleted successfully"}
            else:
                return {"success": False, "error": "Failed to delete", "path": resolved_path}
        except Exception as e:
            logger.error(f"Error deleting {path}: {str(e)}")
            return {"success": False, "error": str(e), "path": path}
    
    def move(self, source: str, destination: str) -> Dict[str, Any]:
        """Move/rename file or directory"""
        try:
            resolved_source = self._resolve_path(source)
            resolved_destination = self._resolve_path(destination)
            
            # Read source content
            content = self._read_raw(resolved_source)
            
            # Write to destination
            success = self._write_raw(resolved_destination, content)
            if success:
                # Delete source
                self._delete_raw(resolved_source)
                return {"success": True, "source": resolved_source, "destination": resolved_destination, "message": "Moved successfully"}
            else:
                return {"success": False, "error": "Failed to write to destination", "source": resolved_source, "destination": resolved_destination}
        except Exception as e:
            logger.error(f"Error moving {source} to {destination}: {str(e)}")
            return {"success": False, "error": str(e), "source": source, "destination": destination}
    
    def copy(self, source: str, destination: str) -> Dict[str, Any]:
        """Copy file"""
        try:
            resolved_source = self._resolve_path(source)
            resolved_destination = self._resolve_path(destination)
            
            # Read source content
            content = self._read_raw(resolved_source)
            
            # Write to destination
            success = self._write_raw(resolved_destination, content)
            if success:
                return {"success": True, "source": resolved_source, "destination": resolved_destination, "message": "Copied successfully"}
            else:
                return {"success": False, "error": "Failed to write to destination", "source": resolved_source, "destination": resolved_destination}
        except Exception as e:
            logger.error(f"Error copying {source} to {destination}: {str(e)}")
            return {"success": False, "error": str(e), "source": source, "destination": destination}
    
    def list(self, path: str = None, max_depth: int = 3, include_hidden: bool = False) -> Dict[str, Any]:
        """List files and directories"""
        try:
            resolved_path = self._resolve_path(path) if path else str(self.base_path)
            items = self._list_raw(resolved_path, max_depth=max_depth, include_hidden=include_hidden)
            
            return {
                "success": True,
                "path": resolved_path,
                "items": items,
                "total_count": len(items)
            }
        except Exception as e:
            logger.error(f"Error listing {path}: {str(e)}")
            return {"success": False, "error": str(e), "path": path}
    
    def save(self, file_path: str, content: Any, **kwargs) -> Dict[str, Any]:
        """Save content to a file with automatic format detection"""
        try:
            # Resolve the file path
            resolved_path = self._resolve_path(file_path)
            file_extension = Path(resolved_path).suffix.lower()
            
            # Handle different file types
            if file_extension == '.json':
                return self._save_json(resolved_path, content, **kwargs)
            elif file_extension in ['.yaml', '.yml']:
                return self._save_yaml(resolved_path, content, **kwargs)
            elif file_extension == '.csv':
                return self._save_csv(resolved_path, content, **kwargs)
            elif file_extension == '.xlsx':
                return self._save_excel(resolved_path, content, **kwargs)
            elif file_extension == '.xml':
                return self._save_xml(resolved_path, content, **kwargs)
            elif file_extension == '.pickle':
                return self._save_pickle(resolved_path, content, **kwargs)
            elif file_extension == '.pdf':
                return self._save_pdf(resolved_path, content, **kwargs)
            elif file_extension in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']:
                return self._save_image(resolved_path, content, **kwargs)
            else:
                # Default to text
                return self._save_text(resolved_path, content, **kwargs)
                
        except Exception as e:
            logger.error(f"Error saving {file_path}: {str(e)}")
            return {"success": False, "error": str(e), "file_path": file_path}
    
    def read(self, file_path: str, **kwargs) -> Dict[str, Any]:
        """Read content from a file with automatic format detection"""
        try:
            resolved_path = self._resolve_path(file_path)
            file_extension = Path(resolved_path).suffix.lower()
            
            # Handle different file types
            if file_extension == '.json':
                return self._read_json(resolved_path, **kwargs)
            elif file_extension in ['.yaml', '.yml']:
                return self._read_yaml(resolved_path, **kwargs)
            elif file_extension == '.csv':
                return self._read_csv(resolved_path, **kwargs)
            elif file_extension == '.xlsx':
                return self._read_excel(resolved_path, **kwargs)
            elif file_extension == '.xml':
                return self._read_xml(resolved_path, **kwargs)
            elif file_extension == '.pickle':
                return self._read_pickle(resolved_path, **kwargs)
            elif file_extension == '.pdf':
                return self._read_pdf(resolved_path, **kwargs)
            elif file_extension in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']:
                return self._read_image(resolved_path, **kwargs)
            else:
                # Default to text
                return self._read_text(resolved_path, **kwargs)
                
        except Exception as e:
            logger.error(f"Error reading {file_path}: {str(e)}")
            return {"success": False, "error": str(e), "file_path": file_path}
    
    def append(self, file_path: str, content: Any, **kwargs) -> Dict[str, Any]:
        """Append content to a file (only for supported formats)"""
        try:
            resolved_path = self._resolve_path(file_path)
            file_extension = Path(resolved_path).suffix.lower()
            
            if file_extension in self.appendable_formats:
                return self.appendable_formats[file_extension](resolved_path, content, **kwargs)
            else:
                return {"success": False, "error": f"Append not supported for {file_extension} files"}
                
        except Exception as e:
            logger.error(f"Error appending to {file_path}: {str(e)}")
            return {"success": False, "error": str(e), "file_path": file_path}
    
    def get_supported_formats(self) -> Dict[str, Any]:
        """Get information about supported file formats"""
        return {
            "success": True,
            "supported_formats": {
                "text": [".txt", ".md", ".log"],
                "structured_data": [".json", ".yaml", ".yml", ".xml", ".csv"],
                "spreadsheets": [".xlsx", ".xls"],
                "binary": [".pickle", ".pkl"],
                "documents": [".pdf"],
                "images": [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff"],
                "appendable": list(self.appendable_formats.keys())
            },
            "features": {
                "read": True,
                "write": True,
                "append": True,
                "delete": True,
                "move": True,
                "copy": True,
                "list": True,
                "create_directory": True
            }
        }
    
    # Text file handlers
    def _save_text(self, file_path: str, content: Any, encoding: str = 'utf-8', **kwargs) -> Dict[str, Any]:
        """Save text content to a file"""
        try:
            with open(file_path, 'w', encoding=encoding) as f:
                f.write(str(content))
            
            return {
                "success": True,
                "message": f"File saved to {file_path}",
                "file_path": file_path,
                "content_length": len(str(content))
            }
        except Exception as e:
            logger.error(f"Error saving text file {file_path}: {str(e)}")
            return {"success": False, "error": str(e), "file_path": file_path}
    
    def _read_text(self, file_path: str, encoding: str = 'utf-8', **kwargs) -> Dict[str, Any]:
        """Read text content from a file"""
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                content = f.read()
            
            return {
                "success": True,
                "content": content,
                "file_path": file_path,
                "content_length": len(content)
            }
        except Exception as e:
            logger.error(f"Error reading text file {file_path}: {str(e)}")
            return {"success": False, "error": str(e), "file_path": file_path}
        
    def _append_text(self, file_path: str, content: str, encoding: str = 'utf-8', **kwargs) -> Dict[str, Any]:
        """Append text content to a file"""
        try:
            with open(file_path, 'a', encoding=encoding) as f:
                f.write(str(content))
            
            return {
                "success": True,
                "message": f"Content appended to file {file_path}",
                "file_path": file_path
            }
        except Exception as e:
            logger.error(f"Error appending to text file {file_path}: {str(e)}")
            return {"success": False, "error": str(e), "file_path": file_path}
    
    # JSON file handlers
    def _save_json(self, file_path: str, content: Any, indent: int = 2, **kwargs) -> Dict[str, Any]:
        """Save JSON content to a file"""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(content, f, indent=indent, ensure_ascii=False)
            
            return {
                "success": True,
                "message": f"JSON file saved to {file_path}",
                "file_path": file_path
            }
        except Exception as e:
            logger.error(f"Error saving JSON file {file_path}: {str(e)}")
            return {"success": False, "error": str(e), "file_path": file_path}
    
    def _read_json(self, file_path: str, **kwargs) -> Dict[str, Any]:
        """Read JSON content from a file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = json.load(f)
            
            return {
                "success": True,
                "content": content,
                "file_path": file_path
            }
        except Exception as e:
            logger.error(f"Error reading JSON file {file_path}: {str(e)}")
            return {"success": False, "error": str(e), "file_path": file_path}
    
    def _append_json(self, file_path: str, content: Any, **kwargs) -> Dict[str, Any]:
        """Append content to JSON file (for arrays)"""
        try:
            existing_content = []
            if Path(file_path).exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    existing_content = json.load(f)
            
            if isinstance(existing_content, list):
                if isinstance(content, list):
                    existing_content.extend(content)
                else:
                    existing_content.append(content)
            elif isinstance(existing_content, dict):
                # If existing content is a dict, merge with new content
                if isinstance(content, dict):
                    existing_content.update(content)
                else:
                    return {"success": False, "error": "Cannot append non-dict to JSON dict"}
            else:
                # Convert to list and append
                existing_content = [existing_content]
                if isinstance(content, list):
                    existing_content.extend(content)
                else:
                    existing_content.append(content)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(existing_content, f, indent=2, ensure_ascii=False)
            
            return {
                "success": True,
                "message": f"Content appended to JSON file {file_path}",
                "file_path": file_path
            }
        except Exception as e:
            logger.error(f"Error appending to JSON file {file_path}: {str(e)}")
            return {"success": False, "error": str(e), "file_path": file_path}
    
    # CSV file handlers
    def _save_csv(self, file_path: str, content: Any, **kwargs) -> Dict[str, Any]:
        """Save CSV content to a file - handles both raw CSV strings and structured data"""
        try:
            if not content:
                return {"success": False, "error": "No content to save"}
            
            with open(file_path, 'w', newline='', encoding='utf-8') as f:
                # If content is a string, write it directly as raw CSV
                if isinstance(content, str):
                    f.write(content)
                    # Count rows by counting newlines
                    rows = content.count('\n')
                    return {
                        "success": True,
                        "message": f"CSV file saved to {file_path}",
                        "file_path": file_path,
                        "rows": rows
                    }
                
                # If content is a list of dictionaries, use CSV writer
                elif isinstance(content, list) and content and isinstance(content[0], dict):
                    fieldnames = content[0].keys()
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(content)
                    
                    return {
                        "success": True,
                        "message": f"CSV file saved to {file_path}",
                        "file_path": file_path,
                        "rows": len(content)
                    }
                
                # If content is a list of lists, use CSV writer
                elif isinstance(content, list) and content and isinstance(content[0], list):
                    writer = csv.writer(f)
                    writer.writerows(content)
                    
                    return {
                        "success": True,
                        "message": f"CSV file saved to {file_path}",
                        "file_path": file_path,
                        "rows": len(content)
                    }
                
                else:
                    return {"success": False, "error": "CSV content must be a string, list of dictionaries, or list of lists"}
            
        except Exception as e:
            logger.error(f"Error saving CSV file {file_path}: {str(e)}")
            return {"success": False, "error": str(e), "file_path": file_path}
    
    def _read_csv(self, file_path: str, **kwargs) -> Dict[str, Any]:
        """Read CSV content from a file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                content = list(reader)
            
            return {
                "success": True,
                "content": content,
                "file_path": file_path,
                "rows": len(content)
            }
        except Exception as e:
            logger.error(f"Error reading CSV file {file_path}: {str(e)}")
            return {"success": False, "error": str(e), "file_path": file_path}
    
    def _append_csv(self, file_path: str, content: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        """Append content to CSV file"""
        try:
            with open(file_path, 'a', newline='', encoding='utf-8') as f:
                if not content:
                    return {"success": False, "error": "No content to append"}
                
                fieldnames = content[0].keys()
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writerows(content)
            
            return {
                "success": True,
                "message": f"Content appended to CSV file {file_path}",
                "file_path": file_path,
                "appended_rows": len(content)
            }
        except Exception as e:
            logger.error(f"Error appending to CSV file {file_path}: {str(e)}")
            return {"success": False, "error": str(e), "file_path": file_path}
    
    # YAML file handlers
    def _save_yaml(self, file_path: str, content: Any, **kwargs) -> Dict[str, Any]:
        """Save YAML content to a file"""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                yaml.dump(content, f, default_flow_style=False, allow_unicode=True)
            
            return {
                "success": True,
                "message": f"YAML file saved to {file_path}",
                "file_path": file_path
            }
        except Exception as e:
            logger.error(f"Error saving YAML file {file_path}: {str(e)}")
            return {"success": False, "error": str(e), "file_path": file_path}
    
    def _read_yaml(self, file_path: str, **kwargs) -> Dict[str, Any]:
        """Read YAML content from a file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = yaml.safe_load(f)
            
            return {
                "success": True,
                "content": content,
                "file_path": file_path
            }
        except Exception as e:
            logger.error(f"Error reading YAML file {file_path}: {str(e)}")
            return {"success": False, "error": str(e), "file_path": file_path}
    
    def _append_yaml(self, file_path: str, content: Any, **kwargs) -> Dict[str, Any]:
        """Append content to YAML file (for lists)"""
        try:
            existing_content = []
            if Path(file_path).exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    existing_content = yaml.safe_load(f) or []
            
            if isinstance(existing_content, list):
                if isinstance(content, list):
                    existing_content.extend(content)
                else:
                    existing_content.append(content)
            elif isinstance(existing_content, dict):
                # If existing content is a dict, merge with new content
                if isinstance(content, dict):
                    existing_content.update(content)
                else:
                    return {"success": False, "error": "Cannot append non-dict to YAML dict"}
            else:
                # Convert to list and append
                existing_content = [existing_content]
                if isinstance(content, list):
                    existing_content.extend(content)
                else:
                    existing_content.append(content)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                yaml.dump(existing_content, f, default_flow_style=False, allow_unicode=True)
            
            return {
                "success": True,
                "message": f"Content appended to YAML file {file_path}",
                "file_path": file_path
            }
        except Exception as e:
            logger.error(f"Error appending to YAML file {file_path}: {str(e)}")
            return {"success": False, "error": str(e), "file_path": file_path}
    
    # XML file handlers
    def _save_xml(self, file_path: str, content: Any, root_tag: str = "root", **kwargs) -> Dict[str, Any]:
        """Save XML content to a file"""
        try:
            # If content is already a string (raw XML), write it directly
            if isinstance(content, str):
                # Check if it's already valid XML
                try:
                    ET.fromstring(content)
                    # It's valid XML, write it directly
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    return {
                        "success": True,
                        "message": f"XML file saved to {file_path}",
                        "file_path": file_path
                    }
                except ET.ParseError:
                    # Not valid XML, treat as text content and wrap it
                    pass
            
            # If content is a dictionary, convert to XML
            if isinstance(content, dict):
                def dict_to_xml(data, root):
                    for key, value in data.items():
                        child = ET.SubElement(root, key)
                        if isinstance(value, dict):
                            dict_to_xml(value, child)
                        else:
                            child.text = str(value)
                
                root = ET.Element(root_tag)
                dict_to_xml(content, root)
                tree = ET.ElementTree(root)
                tree.write(file_path, encoding='utf-8', xml_declaration=True)
            else:
                # For other types, wrap in root element
                root = ET.Element(root_tag)
                root.text = str(content)
                tree = ET.ElementTree(root)
                tree.write(file_path, encoding='utf-8', xml_declaration=True)
            
            return {
                "success": True,
                "message": f"XML file saved to {file_path}",
                "file_path": file_path
            }
        except Exception as e:
            logger.error(f"Error saving XML file {file_path}: {str(e)}")
            return {"success": False, "error": str(e), "file_path": file_path}
    
    def _read_xml(self, file_path: str, **kwargs) -> Dict[str, Any]:
        """Read XML content from a file"""
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            
            def xml_to_dict(element):
                result = {}
                for child in element:
                    if len(child) == 0:
                        result[child.tag] = child.text
                    else:
                        result[child.tag] = xml_to_dict(child)
                return result
            
            content = xml_to_dict(root)
            
            return {
                "success": True,
                "content": content,
                "file_path": file_path
            }
        except Exception as e:
            logger.error(f"Error reading XML file {file_path}: {str(e)}")
            return {"success": False, "error": str(e), "file_path": file_path}
    
    # Excel file handlers
    def _save_excel(self, file_path: str, content: List[List[Any]], sheet_name: str = "Sheet1", **kwargs) -> Dict[str, Any]:
        """Save Excel content to a file"""
        if not EXCEL_AVAILABLE:
            return {"success": False, "error": "openpyxl library not available"}
        
        try:
            workbook = Workbook()
            worksheet = workbook.active
            worksheet.title = sheet_name
            
            for row in content:
                worksheet.append(row)
            
            workbook.save(file_path)
            
            return {
                "success": True,
                "message": f"Excel file saved to {file_path}",
                "file_path": file_path,
                "rows": len(content)
            }
        except Exception as e:
            logger.error(f"Error saving Excel file {file_path}: {str(e)}")
            return {"success": False, "error": str(e), "file_path": file_path}
    
    def _read_excel(self, file_path: str, sheet_name: str = None, **kwargs) -> Dict[str, Any]:
        """Read Excel content from a file"""
        if not EXCEL_AVAILABLE:
            return {"success": False, "error": "openpyxl library not available"}
        
        try:
            workbook = load_workbook(file_path, data_only=True)
            sheet_names = workbook.sheetnames
            
            if sheet_name is None:
                sheet_name = sheet_names[0]
            
            if sheet_name not in sheet_names:
                return {"success": False, "error": f"Sheet '{sheet_name}' not found"}
            
            worksheet = workbook[sheet_name]
            content = []
            
            for row in worksheet.iter_rows(values_only=True):
                if any(cell is not None for cell in row):
                    content.append(list(row))
            
            return {
                "success": True,
                "content": content,
                "file_path": file_path,
                "sheet_name": sheet_name,
                "rows": len(content)
            }
        except Exception as e:
            logger.error(f"Error reading Excel file {file_path}: {str(e)}")
            return {"success": False, "error": str(e), "file_path": file_path}
    
    def _append_excel(self, file_path: str, content: List[List[Any]], sheet_name: str = None, **kwargs) -> Dict[str, Any]:
        """Append content to Excel file"""
        if not EXCEL_AVAILABLE:
            return {"success": False, "error": "openpyxl library not available"}
        
        try:
            if not Path(file_path).exists():
                return self._save_excel(file_path, content, sheet_name or "Sheet1", **kwargs)
            
            workbook = load_workbook(file_path)
            sheet_names = workbook.sheetnames
            
            if sheet_name is None:
                sheet_name = sheet_names[0]
            
            if sheet_name not in sheet_names:
                return {"success": False, "error": f"Sheet '{sheet_name}' not found"}
            
            worksheet = workbook[sheet_name]
            
            for row in content:
                worksheet.append(row)
            
            workbook.save(file_path)
            
            return {
                "success": True,
                "message": f"Content appended to Excel file {file_path}",
                "file_path": file_path,
                "appended_rows": len(content)
            }
        except Exception as e:
            logger.error(f"Error appending to Excel file {file_path}: {str(e)}")
            return {"success": False, "error": str(e), "file_path": file_path}
    
    # Pickle file handlers
    def _save_pickle(self, file_path: str, content: Any, **kwargs) -> Dict[str, Any]:
        """Save pickle content to a file"""
        try:
            with open(file_path, 'wb') as f:
                pickle.dump(content, f)
            
            return {
                "success": True,
                "message": f"Pickle file saved to {file_path}",
                "file_path": file_path
            }
        except Exception as e:
            logger.error(f"Error saving pickle file {file_path}: {str(e)}")
            return {"success": False, "error": str(e), "file_path": file_path}
    
    def _read_pickle(self, file_path: str, **kwargs) -> Dict[str, Any]:
        """Read pickle content from a file"""
        try:
            with open(file_path, 'rb') as f:
                content = pickle.load(f)
            
            return {
                "success": True,
                "content": content,
                "file_path": file_path
            }
        except Exception as e:
            logger.error(f"Error reading pickle file {file_path}: {str(e)}")
            return {"success": False, "error": str(e), "file_path": file_path}
    
    def _append_pickle(self, file_path: str, content: Any, **kwargs) -> Dict[str, Any]:
        """Append content to pickle file (for lists)"""
        try:
            existing_content = []
            if Path(file_path).exists():
                with open(file_path, 'rb') as f:
                    existing_content = pickle.load(f)
            
            if isinstance(existing_content, list):
                if isinstance(content, list):
                    existing_content.extend(content)
                else:
                    existing_content.append(content)
            elif isinstance(existing_content, dict):
                # If existing content is a dict, try to merge intelligently
                if isinstance(content, dict):
                    existing_content.update(content)
                elif isinstance(content, list):
                    # If appending a list to a dict, add it as a new key
                    existing_content["appended_list"] = content
                else:
                    # If appending a single value to a dict, add it as a new key
                    existing_content["appended_value"] = content
            else:
                # Convert to list and append
                existing_content = [existing_content]
                if isinstance(content, list):
                    existing_content.extend(content)
                else:
                    existing_content.append(content)
            
            with open(file_path, 'wb') as f:
                pickle.dump(existing_content, f)
            
            return {
                "success": True,
                "message": f"Content appended to pickle file {file_path}",
                "file_path": file_path
            }
        except Exception as e:
            logger.error(f"Error appending to pickle file {file_path}: {str(e)}")
            return {"success": False, "error": str(e), "file_path": file_path}
    
    # PDF file handlers
    def _save_pdf(self, file_path: str, content: str, **kwargs) -> Dict[str, Any]:
        """Save content to a PDF file"""
        try:
            # Use reportlab to create a proper PDF with text content
            from reportlab.lib.pagesizes import letter
            from reportlab.lib.styles import getSampleStyleSheet
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
            
            # Create PDF document
            doc = SimpleDocTemplate(file_path, pagesize=letter)
            styles = getSampleStyleSheet()
            story = []
            
            # Split content into paragraphs
            paragraphs = content.split('\n')
            
            for para_text in paragraphs:
                if para_text.strip():  # Non-empty paragraph
                    para = Paragraph(para_text, styles['Normal'])
                    story.append(para)
                    story.append(Spacer(1, 12))
                else:
                    story.append(Spacer(1, 12))
            
            # Build the PDF
            doc.build(story)
            
            return {
                "success": True,
                "message": f"PDF file saved to {file_path}",
                "file_path": file_path
            }
                
        except ImportError:
            return {"success": False, "error": "reportlab library not available for PDF creation"}
        except Exception as e:
            logger.error(f"Error saving PDF file {file_path}: {str(e)}")
            return {"success": False, "error": str(e), "file_path": file_path}
    
    def _read_pdf(self, file_path: str, **kwargs) -> Dict[str, Any]:
        """Read content from a PDF file"""
        if not PDF_AVAILABLE:
            return {"success": False, "error": "unstructured library not available"}
        try:
            doc = pymupdf.open(file_path)
            all_text = []
            for page in doc:
                text = page.get_text()
                all_text.append(text)
            text = "\n\n".join(all_text)
            return {
                "success": True,
                "content": text,
                "file_path": file_path
            }
                
        except Exception as e:
            logger.error(f"Error reading PDF file {file_path}: {str(e)}")
            return {"success": False, "error": str(e), "file_path": file_path}
    
    # Image file handlers
    def _save_image(self, file_path: str, content: Any, **kwargs) -> Dict[str, Any]:
        """Save image content to a file"""
        if not PILLOW_AVAILABLE:
            return {"success": False, "error": "Pillow library not available"}
        
        try:
            # Check if content is already a PIL Image object
            if hasattr(content, 'save') and callable(getattr(content, 'save', None)):
                # Content is a PIL Image object
                # Ensure directory exists
                Path(file_path).parent.mkdir(parents=True, exist_ok=True)
                content.save(file_path)
                return {
                    "success": True,
                    "message": f"Image saved to {file_path}",
                    "file_path": file_path,
                    "format": content.format,
                    "size": content.size
                }
            elif isinstance(content, bytes):
                # Content is binary image data
                # Ensure directory exists
                Path(file_path).parent.mkdir(parents=True, exist_ok=True)
                with open(file_path, 'wb') as f:
                    f.write(content)
                return {
                    "success": True,
                    "message": f"Image saved to {file_path}",
                    "file_path": file_path
                }
            elif isinstance(content, str) and Path(content).exists():
                # Content is a file path to an existing image
                shutil.copy2(content, file_path)
                return {
                    "success": True,
                    "message": f"Image copied from {content} to {file_path}",
                    "file_path": file_path
                }
            else:
                return {"success": False, "error": "Content must be a PIL Image object, binary data, or valid file path"}
                
        except Exception as e:
            logger.error(f"Error saving image file {file_path}: {str(e)}")
            return {"success": False, "error": str(e), "file_path": file_path}
    
    def _read_image(self, file_path: str, **kwargs) -> Dict[str, Any]:
        """Read image and return PIL Image object"""
        if not PILLOW_AVAILABLE:
            return {"success": False, "error": "Pillow library not available"}
        
        try:
            with Image.open(file_path) as img:
                # Convert to RGB if necessary for consistency
                if img.mode in ('RGBA', 'LA', 'P'):
                    img = img.convert('RGB')
                
                metadata = {
                    "format": img.format,
                    "mode": img.mode,
                    "size": img.size,
                    "width": img.width,
                    "height": img.height
                }
                
                return {
                    "success": True,
                    "content": img,  # Return the PIL Image object
                    "metadata": metadata,
                    "file_path": file_path
                }
                
        except Exception as e:
            logger.error(f"Error reading image file {file_path}: {str(e)}")
            return {"success": False, "error": str(e), "file_path": file_path}
    
    # Placeholder for future database integration
    def _get_database_connection(self, db_type: str, connection_string: str) -> Any:
        """Placeholder for future database integration"""
        # This will be implemented when adding database support
        raise NotImplementedError("Database integration not yet implemented") 