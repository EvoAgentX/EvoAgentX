import os
from typing import Optional, List, Dict
from pydantic import Field

from ..models.base_model import BaseLLM, LLMOutputParser
from .action import Action, ActionInput, ActionOutput
from ..prompts.code_extraction import CODE_EXTRACTION


class CodeExtractionInput(ActionInput):
    """
    Input parameters for the CodeExtraction action.
    """
    code_string: str = Field(description="The string containing code blocks to extract")
    target_directory: str = Field(description="The directory path where extracted code files will be saved")
    project_name: Optional[str] = Field(default=None, description="Optional name for the project folder")


class CodeExtractionOutput(ActionOutput):
    """
    Output of the CodeExtraction action.
    """
    extracted_files: Dict[str, str] = Field(description="Map of filename to file path of saved files")
    main_file: Optional[str] = Field(default=None, description="Path to the main file if identified")
    error: Optional[str] = Field(default=None, description="Error message if any operation failed")


class CodeBlockInfo(LLMOutputParser):
    """
    Information about an extracted code block.
    """
    language: str = Field(description="Programming language of the code block")
    filename: str = Field(description="Suggested filename for the code block")
    content: str = Field(description="The actual code content")


class CodeBlockList(LLMOutputParser):
    """
    List of code blocks extracted from text.
    """
    code_blocks: List[CodeBlockInfo] = Field(description="List of code blocks")


class CodeExtraction(Action):
    """
    An action that extracts and organizes code blocks from text.
    
    This action uses an LLM to analyze text containing code blocks, extract them,
    suggest appropriate filenames, and save them to a specified directory. It can
    also identify which file is likely the main entry point based on heuristics.
    
    Attributes:
        Inherits all attributes from the Action base class
    """

    def __init__(self, **kwargs):
        
        name = kwargs.pop("name") if "name" in kwargs else CODE_EXTRACTION["name"]
        description = kwargs.pop("description") if "description" in kwargs else CODE_EXTRACTION["description"]
        prompt = kwargs.pop("prompt") if "prompt" in kwargs else CODE_EXTRACTION["prompt"]
        # inputs_format = kwargs.pop("inputs_format") if "inputs_format" in kwargs else CodeExtractionInput
        # outputs_format = kwargs.pop("outputs_format") if "outputs_format" in kwargs else CodeExtractionOutput
        inputs_format = kwargs.pop("inputs_format", None) or CodeExtractionInput
        outputs_format = kwargs.pop("outputs_format", None) or CodeExtractionOutput
        super().__init__(name=name, description=description, prompt=prompt, inputs_format=inputs_format, outputs_format=outputs_format, **kwargs)

    def identify_main_file(self, saved_files: Dict[str, str]) -> Optional[str]:
        """Identify the main file from the saved files based on content and file type.
        
        This method uses a combination of common filename conventions and content
        analysis to determine which file is likely the main entry point of a project.
        
        Args:
            saved_files: Dictionary mapping filenames to their full paths
            
        Returns:
            Path to the main file if found, None otherwise
            
        """
        # Priority lookup for common main files by language
        main_file_priorities = [
            # HTML files
            "index.html",
            # Python files
            "main.py", 
            "app.py",
            # JavaScript files
            "index.js",
            "main.js",
            "app.js",
            # Java files
            "Main.java",
            # C/C++ files
            "main.cpp", 
            "main.c",
            # Go files
            "main.go",
            # Other common entry points
            "index.php",
            "Program.cs"
        ]
        
        # First check priority list
        for main_file in main_file_priorities:
            if main_file in saved_files:
                return saved_files[main_file]
        
        # If no priority file found, use heuristics based on file extensions
        
        # If we have HTML files, use the first one
        html_files = {k: v for k, v in saved_files.items() if k.endswith('.html')}
        if html_files:
            return next(iter(html_files.values()))
        
        # Check for Python files with "__main__" section
        py_files = {k: v for k, v in saved_files.items() if k.endswith('.py')}
        if py_files:
            for filename, path in py_files.items():
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if "if __name__ == '__main__'" in content or 'if __name__ == "__main__"' in content:
                        return path
            # If no main found, return the first Python file
            if py_files:
                return next(iter(py_files.values()))
        
        # If we have Java files, look for one with a main method
        java_files = {k: v for k, v in saved_files.items() if k.endswith('.java')}
        if java_files:
            for filename, path in java_files.items():
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if "public static void main" in content:
                        return path
            # If no main found, return the first Java file
            if java_files:
                return next(iter(java_files.values()))
                
        # For JavaScript applications
        js_files = {k: v for k, v in saved_files.items() if k.endswith('.js')}
        if js_files:
            return next(iter(js_files.values()))
                
        # If all else fails, return the first file
        if saved_files:
            return next(iter(saved_files.values()))
                
        # No files found
        return None

    def save_code_blocks(self, code_blocks: List[Dict], target_directory: str) -> Dict[str, str]:
        """Save code blocks to files in the target directory.
        
        Creates the target directory if it doesn't exist and saves each code block
        to a file with an appropriate name, handling filename conflicts.
        
        Args:
            code_blocks: List of dictionaries containing code block information
            target_directory: Directory path where files should be saved
            
        Returns:
            Dictionary mapping filenames to their full paths
        """
        os.makedirs(target_directory, exist_ok=True)
        saved_files = {}
        
        for block in code_blocks:
            filename = block.get("filename", "unknown.txt")
            content = block.get("content", "")
            
            # Skip empty blocks
            if not content.strip():
                continue
            
            # Handle filename conflicts
            base_filename = filename
            counter = 1
            while filename in saved_files:
                name_parts = base_filename.split('.')
                if len(name_parts) > 1:
                    filename = f"{'.'.join(name_parts[:-1])}_{counter}.{name_parts[-1]}"
                else:
                    filename = f"{base_filename}_{counter}"
                counter += 1
                
            # Save to file
            file_path = os.path.join(target_directory, filename)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
                
            # Add to map
            saved_files[filename] = file_path
            
        return saved_files

    def execute(self, llm: Optional[BaseLLM] = None, inputs: Optional[dict] = None, sys_msg: Optional[str]=None, return_prompt: bool = False, **kwargs) -> CodeExtractionOutput:
        """Execute the CodeExtraction action.
        
        Extracts code blocks from the provided text using the specified LLM,
        saves them to the target directory, and identifies the main file.
        
        Args:
            llm: The LLM to use for code extraction
            inputs: Dictionary containing:
                - code_string: The string with code blocks to extract
                - target_directory: Where to save the files
                - project_name: Optional project folder name
            sys_msg: Optional system message override for the LLM
            return_prompt: Whether to return the prompt along with the result
            **kwargs: Additional keyword arguments
            
        Returns:
            CodeExtractionOutput with extracted file information
        """
        if not llm:
            error_msg = "CodeExtraction action requires an LLM."
            return CodeExtractionOutput(extracted_files={}, error=error_msg)
            
        if not inputs:
            error_msg = "CodeExtraction action received invalid `inputs`: None or empty."
            return CodeExtractionOutput(extracted_files={}, error=error_msg)
        
        code_string = inputs.get("code_string", "")
        target_directory = inputs.get("target_directory", "")
        project_name = inputs.get("project_name", None)
        
        if not code_string:
            error_msg = "No code string provided."
            return CodeExtractionOutput(extracted_files={}, error=error_msg)
            
        if not target_directory:
            error_msg = "No target directory provided."
            return CodeExtractionOutput(extracted_files={}, error=error_msg)
        
        # Create project folder if name is provided
        if project_name:
            project_dir = os.path.join(target_directory, project_name)
        else:
            project_dir = target_directory
            
        try:
            # Use LLM to extract code blocks and suggest filenames
            prompt_params = {"code_string": code_string}
            system_message = CODE_EXTRACTION["system_prompt"] if sys_msg is None else sys_msg

            llm_response: CodeBlockList = llm.generate(
                prompt=self.prompt.format(**prompt_params),
                system_message=system_message,
                parser=CodeBlockList,
                parse_mode="json"
            )
            code_blocks = llm_response.get_structured_data().get("code_blocks", [])

            # Save code blocks to files
            saved_files = self.save_code_blocks(code_blocks, project_dir)
            
            # Identify main file
            main_file = self.identify_main_file(saved_files)
            
            result = CodeExtractionOutput(
                extracted_files=saved_files,
                main_file=main_file
            )
            
            if return_prompt:
                return result, self.prompt.format(**prompt_params)
                
            return result
            
        except Exception as e:
            error_msg = f"Error extracting code: {str(e)}"
            return CodeExtractionOutput(extracted_files={}, error=error_msg)