from .tool import Tool,Toolkit
import os
import PyPDF2
from typing import Dict, Any, List, Optional
from ..core.logging import logger
from ..core.module import BaseModule


class FileBase(BaseModule):
    """
    Base class containing shared file handling logic for different file types.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs) 
        # File type handlers for special file formats
        self.file_handlers = {
            '.pdf': {
                'read': self._read_pdf,
                'write': self._write_pdf,
                'append': self._append_pdf
            }
            # Add more special file handlers here as needed
        }
    
    def get_file_handlers(self):
        """Returns file type handlers for special file formats"""
        return self.file_handlers
    
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
        Write content to a PDF file using reportlab.
        
        Args:
            file_path (str): Path to the PDF file
            content (str): Content to write to the PDF
            
        Returns:
            dict: A dictionary with the operation status
        """
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
                    # Create paragraph with normal style
                    para = Paragraph(para_text, styles['Normal'])
                    story.append(para)
                    story.append(Spacer(1, 12))  # Add space between paragraphs
                else:
                    # Empty line - add space
                    story.append(Spacer(1, 12))
            
            # Build the PDF
            doc.build(story)
            
            return {
                "success": True,
                "message": f"PDF created at {file_path} with text content using reportlab",
                "file_path": file_path,
                "content_length": len(content),
                "paragraphs": len([p for p in paragraphs if p.strip()]),
                "library_used": "reportlab"
            }
                
        except ImportError:
            # Fallback: Create a basic PDF with PyPDF2 but inform user about limitation
            pdf_writer = PyPDF2.PdfWriter()
            pdf_writer.add_blank_page(width=612, height=792)  # Standard letter size
            
            with open(file_path, 'wb') as f:
                pdf_writer.write(f)
            
            return {
                "success": True,
                "message": f"Basic PDF created at {file_path} (blank page only - reportlab not available)",
                "file_path": file_path,
                "warning": "Text content not added - reportlab library not found",
                "note": "Install reportlab for full PDF text support: pip install reportlab",
                "library_used": "PyPDF2"
            }
            
        except Exception as e:
            logger.error(f"Error writing PDF {file_path}: {str(e)}")
            return {"success": False, "error": str(e), "file_path": file_path}
    
    def _append_pdf(self, file_path: str, content: str) -> Dict[str, Any]:
        """
        Append content to a PDF file by creating a new page.
        
        Args:
            file_path (str): Path to the PDF file
            content (str): Content to append to the PDF
            
        Returns:
            dict: A dictionary with the operation status
        """
        try:
            if not os.path.exists(file_path):
                return self._write_pdf(file_path, content)
            
            try:
                # Import required libraries
                from reportlab.lib.pagesizes import letter
                from reportlab.lib.styles import getSampleStyleSheet
                from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
                import tempfile
                
                # Create a temporary PDF with the new content
                with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
                    temp_pdf_path = temp_file.name
                
                # Create the new page with content using reportlab
                doc = SimpleDocTemplate(temp_pdf_path, pagesize=letter)
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
                        # Empty line - add space
                        story.append(Spacer(1, 12))
                
                # Build the temporary PDF
                doc.build(story)
                
                # Now merge the existing PDF with the new page
                pdf_writer = PyPDF2.PdfWriter()
                
                # Add all pages from the existing PDF
                with open(file_path, 'rb') as existing_file:
                    pdf_reader = PyPDF2.PdfReader(existing_file)
                    for page_num in range(len(pdf_reader.pages)):
                        pdf_writer.add_page(pdf_reader.pages[page_num])
                
                # Add the new page(s) from the temporary PDF
                with open(temp_pdf_path, 'rb') as temp_file:
                    temp_reader = PyPDF2.PdfReader(temp_file)
                    for page_num in range(len(temp_reader.pages)):
                        pdf_writer.add_page(temp_reader.pages[page_num])
                
                # Write the merged PDF back to the original file
                with open(file_path, 'wb') as output_file:
                    pdf_writer.write(output_file)
                
                # Clean up temporary file
                os.unlink(temp_pdf_path)
                
                return {
                    "success": True,
                    "message": f"Content appended as new page(s) to PDF at {file_path}",
                    "file_path": file_path,
                    "operation": "append_new_page",
                    "appended_content_length": len(content),
                    "paragraphs_added": len([p for p in paragraphs if p.strip()]),
                    "library_used": "reportlab + PyPDF2"
                }
                
            except ImportError:
                # Fallback: Basic PDF page append using only PyPDF2
                pdf_writer = PyPDF2.PdfWriter()
                
                # Add all pages from the existing PDF
                with open(file_path, 'rb') as existing_file:
                    pdf_reader = PyPDF2.PdfReader(existing_file)
                    for page_num in range(len(pdf_reader.pages)):
                        pdf_writer.add_page(pdf_reader.pages[page_num])
                
                # Add a blank page (since we can't add text without reportlab)
                pdf_writer.add_blank_page(width=612, height=792)
                
                # Write back to file
                with open(file_path, 'wb') as output_file:
                    pdf_writer.write(output_file)
                
                return {
                    "success": True,
                    "message": f"Blank page appended to PDF at {file_path} (reportlab not available for text)",
                    "file_path": file_path,
                    "operation": "append_blank_page",
                    "warning": "Text content not added - reportlab library not found",
                    "note": "Install reportlab for full PDF text support: pip install reportlab",
                    "library_used": "PyPDF2"
                }
                
        except Exception as e:
            logger.error(f"Error appending to PDF {file_path}: {str(e)}")
            return {"success": False, "error": str(e), "file_path": file_path}
    

class ReadFileTool(Tool):
    name: str = "read_file"
    description: str = "Read content from a file with special handling for different file types like PDFs"
    inputs: Dict[str, Dict[str, str]] = {
        "file_path": {
            "type": "string",
            "description": "Path to the file to read"
        }
    }
    required: Optional[List[str]] = ["file_path"]
    
    def __init__(self, file_base: FileBase = None):
        super().__init__()
        self.file_base = file_base
    
    def __call__(self, file_path: str) -> Dict[str, Any]:
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
            if self.file_base and file_ext in self.file_base.get_file_handlers():
                file_handlers = self.file_base.get_file_handlers()
                if 'read' in file_handlers[file_ext]:
                    return file_handlers[file_ext]['read'](file_path)
            
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


class WriteFileTool(Tool):
    name: str = "write_file"
    description: str = "Write content to a file with special handling for different file types like PDFs"
    inputs: Dict[str, Dict[str, str]] = {
        "file_path": {
            "type": "string",
            "description": "Path to the file to write"
        },
        "content": {
            "type": "string",
            "description": "Content to write to the file"
        }
    }
    required: Optional[List[str]] = ["file_path", "content"]
    
    def __init__(self, file_base: FileBase = None):
        super().__init__()
        self.file_base = file_base
    
    def __call__(self, file_path: str, content: str) -> Dict[str, Any]:
        """
        Write content to a file with special handling for different file types.
        
        Args:
            file_path (str): Path to the file to write
            content (str): Content to write to the file
            
        Returns:
            dict: A dictionary with the operation status
        """
        try:
            file_ext = os.path.splitext(file_path)[1].lower()
            
            # Create directory if it doesn't exist
            directory = os.path.dirname(file_path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            
            # Use special handler if available for this file type
            if self.file_base and file_ext in self.file_base.get_file_handlers():
                file_handlers = self.file_base.get_file_handlers()
                if 'write' in file_handlers[file_ext]:
                    return file_handlers[file_ext]['write'](file_path, content)
            
            # Default file writing behavior
            with open(file_path, 'w') as f:
                f.write(content)
            
            return {
                "success": True,
                "message": f"Content written to {file_path}",
                "file_path": file_path
            }
            
        except Exception as e:
            logger.error(f"Error writing to file {file_path}: {str(e)}")
            return {"success": False, "error": str(e), "file_path": file_path}


class AppendFileTool(Tool):
    name: str = "append_file"
    description: str = "Append content to a file with special handling for different file types like PDFs"
    inputs: Dict[str, Dict[str, str]] = {
        "file_path": {
            "type": "string",
            "description": "Path to the file to append to"
        },
        "content": {
            "type": "string",
            "description": "Content to append to the file"
        }
    }
    required: Optional[List[str]] = ["file_path", "content"]
    
    def __init__(self, file_base: FileBase = None):
        super().__init__()
        self.file_base = file_base
    
    def __call__(self, file_path: str, content: str) -> Dict[str, Any]:
        """
        Append content to a file with special handling for different file types.
        
        Args:
            file_path (str): Path to the file to append to
            content (str): Content to append to the file
            
        Returns:
            dict: A dictionary with the operation status
        """
        file_ext = os.path.splitext(file_path)[1].lower()
        
        # Create directory if it doesn't exist
        directory = os.path.dirname(file_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        
        # Use special handler if available for this file type
        if self.file_base and file_ext in self.file_base.get_file_handlers():
            file_handlers = self.file_base.get_file_handlers()
            if 'append' in file_handlers[file_ext]:
                return file_handlers[file_ext]['append'](file_path, content)
        
        # Default file appending behavior
        try:
            with open(file_path, 'a') as f:
                f.write(content)
            
            return {
                "success": True,
                "message": f"Content appended to {file_path}",
                "file_path": file_path
            }
            
        except Exception as e:
            logger.error(f"Error appending to file {file_path}: {str(e)}")
            return {"success": False, "error": str(e), "file_path": file_path}


class FileToolkit(Toolkit):
    def __init__(self, name: str = "FileToolkit"):
        # Create the shared file base instance
        file_base = FileBase()
        
        # Create tools with the shared file base
        tools = [
            ReadFileTool(file_base=file_base),
            WriteFileTool(file_base=file_base),
            AppendFileTool(file_base=file_base)
        ]
        
        # Initialize parent with tools
        super().__init__(name=name, tools=tools)
        
        # Store file_base as instance variable
        self.file_base = file_base
    


