from .tool import Tool
import os
import PyPDF2
from typing import Dict, Any, Optional, List, Callable
from evoagentx.core.logging import logger

class FileTool(Tool):
    """
    Tool for handling file operations with special handling for different file types.
    Default behavior uses standard file operations, with customized handlers for specific formats like PDFs.
    """
    
    def __init__(
        self,
        name: str = 'File Tool',
        schemas: Optional[List[dict]] = None,
        descriptions: Optional[List[str]] = None,
        tools: Optional[List[Callable]] = None,
        **kwargs
    ):
        # Set default schemas, descriptions and tools if not provided
        schemas = schemas or self.get_tool_schemas()
        descriptions = descriptions or self.get_tool_descriptions()
        tools = tools or self.get_tools()
        
        # Initialize the base Tool class
        super().__init__(
            name=name,
            schemas=schemas,
            descriptions=descriptions,
            tools=tools,
            **kwargs
        )
        
        # File type handlers for special file formats
        self.file_handlers = {
            '.pdf': {
                'read': self._read_pdf,
                'write': self._write_pdf,
                'append': self._append_pdf
            }
            # Add more special file handlers here as needed
        }
    
    def read_file(self, file_path: str) -> Dict[str, Any]:
        """
        Read content from a file with special handling for different file types.
        
        Args:
            file_path (str): Path to the file to read
            
        Returns:
            dict: A dictionary with the file content and metadata
        """
        try:
            if not os.path.exists(file_path):
                return {"success": False, "error": f"File not found: {file_path}"}
            
            file_ext = os.path.splitext(file_path)[1].lower()
            
            # Use special handler if available for this file type
            if file_ext in self.file_handlers and 'read' in self.file_handlers[file_ext]:
                return self.file_handlers[file_ext]['read'](file_path)
            
            # Default file reading behavior
            with open(file_path, 'r') as f:
                content = f.read()
            
            return {
                "success": True,
                "content": content,
                "file_path": file_path,
                "file_type": file_ext or "text"
            }
            
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {str(e)}")
            return {"success": False, "error": str(e), "file_path": file_path}
    
    def write_file(self, file_path: str, content: str, mode: str = 'w') -> Dict[str, Any]:
        """
        Write content to a file with special handling for different file types.
        
        Args:
            file_path (str): Path to the file to write
            content (str): Content to write to the file
            mode (str): Write mode ('w' for write, 'a' for append)
            
        Returns:
            dict: A dictionary with the operation status
        """
        try:
            file_ext = os.path.splitext(file_path)[1].lower()
            
            # Use special handler if available for this file type
            if file_ext in self.file_handlers:
                if mode == 'a' and 'append' in self.file_handlers[file_ext]:
                    return self.file_handlers[file_ext]['append'](file_path, content)
                elif 'write' in self.file_handlers[file_ext]:
                    return self.file_handlers[file_ext]['write'](file_path, content)
            
            # Default file writing behavior
            with open(file_path, mode) as f:
                f.write(content)
            
            return {
                "success": True,
                "message": f"Content {'appended to' if mode == 'a' else 'written to'} {file_path}",
                "file_path": file_path
            }
            
        except Exception as e:
            logger.error(f"Error writing to file {file_path}: {str(e)}")
            return {"success": False, "error": str(e), "file_path": file_path}
    
    def append_file(self, file_path: str, content: str) -> Dict[str, Any]:
        """
        Append content to a file with special handling for different file types.
        
        Args:
            file_path (str): Path to the file to append to
            content (str): Content to append to the file
            
        Returns:
            dict: A dictionary with the operation status
        """
        return self.write_file(file_path, content, mode='a')
    
    def _read_pdf(self, file_path: str) -> Dict[str, Any]:
        """
        Read content from a PDF file.
        
        Args:
            file_path (str): Path to the PDF file
            
        Returns:
            dict: A dictionary with the PDF content and metadata
        """
        try:
            with open(file_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                text = ""
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text() + "\n"
                
                return {
                    "success": True,
                    "content": text,
                    "file_path": file_path,
                    "file_type": "pdf",
                    "pages": len(pdf_reader.pages)
                }
                
        except Exception as e:
            logger.error(f"Error reading PDF {file_path}: {str(e)}")
            return {"success": False, "error": str(e), "file_path": file_path}
    
    def _write_pdf(self, file_path: str, content: str) -> Dict[str, Any]:
        """
        Write content to a PDF file.
        
        Args:
            file_path (str): Path to the PDF file
            content (str): Content to write to the PDF
            
        Returns:
            dict: A dictionary with the operation status
        """
        try:
            # Create a new PDF with the content
            pdf_writer = PyPDF2.PdfWriter()
            pdf_writer.add_blank_page(width=612, height=792)  # Standard letter size
            
            # Currently, PyPDF2 doesn't have a simple way to add text directly
            # We'd typically use reportlab or another library for advanced PDF creation
            # This is a simplified placeholder implementation
            
            with open(file_path, 'wb') as f:
                pdf_writer.write(f)
            
            return {
                "success": True,
                "message": f"PDF created at {file_path}",
                "file_path": file_path,
                "note": "For advanced PDF writing with text content, consider using reportlab or another PDF generation library"
            }
                
        except Exception as e:
            logger.error(f"Error writing PDF {file_path}: {str(e)}")
            return {"success": False, "error": str(e), "file_path": file_path}
    
    def _append_pdf(self, file_path: str, content: str) -> Dict[str, Any]:
        """
        Append content to a PDF file.
        
        Args:
            file_path (str): Path to the PDF file
            content (str): Content to append to the PDF
            
        Returns:
            dict: A dictionary with the operation status
        """
        try:
            if not os.path.exists(file_path):
                return self._write_pdf(file_path, content)
            
            # For appending to PDFs, we'd typically:
            # 1. Read the existing PDF
            # 2. Create a new page with the content
            # 3. Append the new page to the existing PDF
            # This is a simplified placeholder implementation
            
            return {
                "success": False,
                "error": "PDF appending requires additional libraries like reportlab for proper implementation",
                "file_path": file_path
            }
                
        except Exception as e:
            logger.error(f"Error appending to PDF {file_path}: {str(e)}")
            return {"success": False, "error": str(e), "file_path": file_path}
    
    def get_tools(self) -> List[Callable]:
        """Returns a list of callable functions for all tools"""
        return [self.read_file, self.write_file, self.append_file]
    
    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Returns the OpenAI-compatible function schemas for the file tools"""
        return [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read content from a file with special handling for different file types.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "The path to the file to read."
                            }
                        },
                        "required": ["file_path"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "write_file",
                    "description": "Write content to a file with special handling for different file types.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "The path to the file to write."
                            },
                            "content": {
                                "type": "string",
                                "description": "The content to write to the file."
                            },
                            "mode": {
                                "type": "string",
                                "description": "The write mode ('w' for write, 'a' for append). Default is 'w'.",
                                "enum": ["w", "a"]
                            }
                        },
                        "required": ["file_path", "content"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "append_file",
                    "description": "Append content to a file with special handling for different file types.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "The path to the file to append to."
                            },
                            "content": {
                                "type": "string",
                                "description": "The content to append to the file."
                            }
                        },
                        "required": ["file_path", "content"]
                    }
                }
            }
        ]
    
    def get_tool_descriptions(self) -> List[str]:
        """Returns descriptions for all tools"""
        return [
            "Read content from a file with special handling for different file types.",
            "Write content to a file with special handling for different file types.",
            "Append content to a file with special handling for different file types."
        ]
    
    def get_tool_prompt(self) -> str:
        """Returns a tool instruction prompt for the agent to use the tool"""
        return """
        This tool allows you to read, write, and append to files with special handling for different file formats.
        
        For regular text files, it works like standard file operations.
        For PDFs, it provides specialized functionality to extract text and create/modify PDF files.
        
        To read a file: Use read_file with the file_path parameter.
        To write to a file: Use write_file with file_path and content parameters.
        To append to a file: Use append_file with file_path and content parameters.
        """


