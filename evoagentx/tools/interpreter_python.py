import ast
import contextlib
import io
import importlib
import sys
import os
import traceback
from typing import List, Set, Optional, Union, Dict
from .interpreter_base import BaseInterpreter
from .tool import Tool,Toolkit
from .storage_handler import FileStorageHandler
from pydantic import Field

# Constants
DEFAULT_ENCODING = 'utf-8'

class PythonInterpreter(BaseInterpreter):

    project_path:Optional[str] = Field(default=".", description="Path to the project directory")
    directory_names:Optional[List[str]] = Field(default_factory=list, description="List of directory names to check for imports")
    allowed_imports:Optional[Set[str]] = Field(default_factory=set, description="Set of allowed imports")

    def __init__(
        self, 
        name: str = 'PythonInterpreter',
        project_path:Optional[str] = ".",
        directory_names:Optional[List[str]] = [],
        allowed_imports:Optional[Set[str]] = None,
        storage_handler: FileStorageHandler = None,
        **kwargs
    ):
        """
        Initialize a Python interpreter for executing code in a controlled environment.
        
        Args:
            name (str): The name of the interpreter
            project_path (Optional[str]): Path to the project directory for module resolution
            directory_names (Optional[List[str]]): List of directory names to check for imports
            allowed_imports (Optional[Set[str]]): Set of allowed module imports to enforce security
            storage_handler (Optional[FileStorageHandler]): Storage handler for file operations
            **kwargs: Additional data to pass to the parent class
        """
        super().__init__(
            name=name, 
            project_path=project_path,
            directory_names=directory_names,
            allowed_imports=allowed_imports,
            **kwargs
        )
        self.allowed_imports = allowed_imports or set()
        self.namespace = {}
        self.visited_modules = {}
        
        # Initialize storage handler
        if storage_handler is None:
            from .storage_file import LocalStorageHandler
            self.storage_handler = LocalStorageHandler(base_path="./workplace/interpreter")
        else:
            self.storage_handler = storage_handler

    def _get_file_and_folder_names(self, target_path: str) -> List[str]:
        """Retrieves the names of files and folders (without extensions) in a given directory.
        Args:
            target_path (str): Path to the target directory.
        Returns:
            List[str]: List of file and folder names (excluding extensions).
        """
        names = []
        for item in os.listdir(target_path):
            name, _ = os.path.splitext(item)  # Extract filename without extension
            names.append(name)
        return names

    def _extract_definitions(self, module_name: str, path: str, potential_names: Optional[Set[str]] = None) -> List[str]:
        """Extracts function and class definitions from a module file while ensuring safety.
        Args:
            module_name (str): The name of the module.
            path (str): The file path of the module.
            potential_names (Optional[Set[str]]): The specific functions/classes to import (for ImportFrom).
        Returns:
            List[str]: A list of violations found during analysis. An empty list indicates no issues.
        """
        if path in self.namespace:  # Avoid re-importing if already processed
            return []
        
        try:
            # Attempt to dynamically load the module
            module_spec = importlib.util.spec_from_file_location(module_name, path)
            loaded_module = importlib.util.module_from_spec(module_spec)
            module_spec.loader.exec_module(loaded_module)

            # Register the module in self.namespace
            self.namespace[module_name] = loaded_module

        except Exception:
            exc_type, exc_value, exc_tb = sys.exc_info()
            return ["".join(traceback.format_exception(exc_type, exc_value, exc_tb))]

        
        # Read the module file to perform code analysis
        result = self.storage_handler.read(path)
        if result["success"]:
            code = result["content"]
        else:
            raise FileNotFoundError(f"Could not read file {path}: {result.get('error', 'Unknown error')}")

        # Perform safety check before adding functions/classes
        violations = self._analyze_code(code)
        if violations:
            return violations  # Stop execution if safety violations are detected

        # Parse the AST to extract function and class names
        tree = ast.parse(code)
        available_symbols = {}

        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                available_symbols[node.name] = node  # Store detected functions/classes

        # Dynamically load specific functions/classes if requested
        try:
            spec = importlib.util.spec_from_file_location(module_name, path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            if potential_names is None:
                # Import all detected functions/classes
                for name in available_symbols:
                    if hasattr(module, name):
                        self.namespace[name] = getattr(module, name)
            else:
                # Import only specified functions/classes
                for name in potential_names:
                    if name in available_symbols and hasattr(module, name):
                        self.namespace[name] = getattr(module, name)
                    else:
                        violations.append(f"Function or class '{name}' not found in {module_name}")

        except Exception:
            exc_type, exc_value, exc_tb = sys.exc_info()
            return ["".join(traceback.format_exception(exc_type, exc_value, exc_tb))]


        return violations

    def _check_project(self, module: Union[ast.Import, ast.ImportFrom]) -> List[str]:
        """Checks and imports a local project module while ensuring safety.

        Args:
            module (Union[ast.Import, ast.ImportFrom]): The AST import node representing the module.

        Returns:
            List[str]: A list of violations found during analysis.
        """
        
        if isinstance(module, ast.Import):
            module_name = module.name
            potential_names = None  # Full module import
        else:
            module_name = module.module
            potential_names = {name.name for name in module.names}  # Selective import

        # Construct the module file path based on project structure
        if len(module_name.split(".")) > 1:
            module_path = os.path.join(self.project_path, *module_name.split(".")) + ".py"
        else:
            module_path = os.path.join(self.project_path, module_name + ".py")

        # Attempt to safely extract functions/classes
        if os.path.exists(module_path):
            violations = self._extract_definitions(module_name, module_path, potential_names)
        else:
            return [f"Module not found: {module_name}"]

        if violations:
            return violations  # Stop execution if safety violations are detected

        # Dynamically load the module and update self.namespace
        try:
            module_spec = importlib.util.spec_from_file_location(module_name, module_path)
            loaded_module = importlib.util.module_from_spec(module_spec)
            module_spec.loader.exec_module(loaded_module)

            # Register the module in self.namespace
            self.namespace[module_name] = loaded_module
            
        except Exception:
            exc_type, exc_value, exc_tb = sys.exc_info()
            return ["".join(traceback.format_exception(exc_type, exc_value, exc_tb))]

        return violations

    def _execute_import(self, import_module: ast.Import) -> List[str]:
        """Processes an import statement, verifying permissions and adding modules to the namespace.

        Args:
            import_module (ast.Import): The AST node representing an import statement.

        Returns:
            List[str]: A list of violations found during import handling.
        """
        violations = []
        
        for module in import_module.names:
            # Check if the module is part of the project directory (local module)
            if module.name.split(".")[0] in self.directory_names:
                violations += self._check_project(module)
                continue

            # Check if the import is explicitly allowed
            if module.name not in self.allowed_imports:
                violations.append(f"Unauthorized import: {module.name}")
                return violations

            # Attempt to import the module
            try:
                alias = module.asname or module.name
                imported_module = importlib.import_module(module.name)
                self.namespace[alias] = imported_module
            except ImportError:
                exc_type, exc_value, exc_tb = sys.exc_info()
                violations.append("".join(traceback.format_exception(exc_type, exc_value, exc_tb)))

        return violations

    def _execute_import_from(self, import_from: ast.ImportFrom) -> List[str]:
        """Processes a 'from module import name' statement, ensuring safety and adding modules to the namespace.

        Args:
            import_from (ast.ImportFrom): The AST node representing an 'import from' statement.

        Returns:
            List[str]: A list of violations found during import handling.
        """
        # Ensure that relative imports (e.g., 'from . import') are not allowed
        if import_from.module is None:
            return ["'from . import' is not supported."]

        # Check if the module is a part of the project directory (local module)
        if import_from.module.split(".")[0] in self.directory_names:
            return self._check_project(import_from)

        # Ensure that the module is explicitly allowed
        if import_from.module not in self.allowed_imports:
            return [f"Unauthorized import: {import_from.module}"]

        try:
            # Attempt to import the specified components from the module
            for import_name in import_from.names:
                imported_module = importlib.import_module(import_from.module)
                alias = import_name.asname or import_name.name
                self.namespace[alias] = getattr(imported_module, import_name.name)
            return []
        except ImportError:
            exc_type, exc_value, exc_tb = sys.exc_info()
            return ["".join(traceback.format_exception(exc_type, exc_value, exc_tb))]

    def _analyze_code(self, code: str) -> List[str]:
        """Parses and analyzes the code for import violations before execution.

        Args:
            code (str): The raw Python code to analyze.

        Returns:
            List[str]: A list of violations detected in the code.
        """
        violations = []

        try:
            # Parse the provided code into an Abstract Syntax Tree (AST)
            tree = ast.parse(code)

            # Traverse the AST and check for import violations
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    violations += self._execute_import(node)
                elif isinstance(node, ast.ImportFrom):
                    violations += self._execute_import_from(node)
        except SyntaxError:
            exc_type, exc_value, exc_tb = sys.exc_info()
            violations.append("".join(traceback.format_exception(exc_type, exc_value, exc_tb)))

        return violations

    def execute(self, code: str, language: str = "python") -> str:
        """
        Analyzes and executes the provided Python code in a controlled environment.
        
        NOTE: This method only returns content printed to stdout during execution.
        It does not return any values from the code itself. To see results, use
        print statements in your code.
        
        WARNING: This method uses Python's exec() function internally, which executes
        code with full privileges. While safety checks are performed, there is still
        a security risk. Do not use with untrusted code.

        Args:
            code (str): The Python code to execute.
            language (str, optional): The programming language of the code. Defaults to "python".

        Returns:
            str: The output of the executed code (printed content only), or a list of violations if found.
        """
        # Verify language is python
        if language.lower() != "python":
            return f"Error: This interpreter only supports Python language. Received: {language}"
            
        self.visited_modules = {}
        self.namespace = {}

        # Change to the project directory and update sys.path for module resolution
        if not self.project_path:
            raise ValueError("Project path (project_path) is not set")
        
        if not os.path.exists(self.project_path):
            raise ValueError(f"Project path '{self.project_path}' does not exist")
            
        if not os.path.isdir(self.project_path):
            raise ValueError(f"Project path '{self.project_path}' is not a directory")
            
        os.chdir(self.project_path)
        sys.path.insert(0, self.project_path)

        if self.allowed_imports:
            violations = self._analyze_code(code)
            if violations:
                return"\n".join(violations)
                

        # Capture standard output during execution
        stdout_capture = io.StringIO()
        with contextlib.redirect_stdout(stdout_capture):
            try:
                # Execute the code with basic builtins
                exec(code, {"__builtins__": __builtins__})
            except Exception:
                exc_type, exc_value, exc_tb = sys.exc_info()
                error_msg = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
                return error_msg

        # Retrieve and return the captured output
        return stdout_capture.getvalue().strip()

    def execute_script(self, file_path: str, language: str = "python") -> str:
        """
        Reads Python code from a file and executes it using the `execute` method.
        
        NOTE: This method only returns content printed to stdout during execution.
        It does not return any values from the code itself. To see results, use
        print statements in your code.
        
        WARNING: This method uses Python's exec() function internally, which executes
        code with full privileges. While safety checks are performed, there is still
        a security risk. Do not use with untrusted code.

        Args:
            file_path (str): The path to the Python file to be executed.
            language (str, optional): The programming language of the code. Defaults to "python".

        Returns:
            str: The output of the executed code (printed content only), or an error message if the execution fails.
        """
        
        # Read file using storage handler
        result = self.storage_handler.read(file_path)
        if result["success"]:
            code = result["content"]
        else:
            return f"Error: Could not read file '{file_path}': {result.get('error', 'Unknown error')}"
            
        return self.execute(code, language)
    

class PythonExecuteTool(Tool):
    name: str = "python_execute"
    description: str = "Execute Python code in a controlled environment with safety checks"
    inputs: Dict[str, Dict[str, str]] = {
        "code": {
            "type": "string",
            "description": "The Python code to execute"
        },
        "language": {
            "type": "string",
            "description": "The programming language of the code (only 'python' is supported)"
        }
    }
    required: Optional[List[str]] = ["code"]
    
    def __init__(self, python_interpreter: PythonInterpreter = None):
        super().__init__()
        self.python_interpreter = python_interpreter
    
    def __call__(self, code: str, language: str = "python") -> str:
        """Execute Python code using the Python interpreter."""
        if not self.python_interpreter:
            raise RuntimeError("Python interpreter not initialized")
        
        try:
            return self.python_interpreter.execute(code, language)
        except Exception as e:
            return f"Error executing code: {str(e)}"


class PythonExecuteScriptTool(Tool):
    name: str = "python_execute_script"
    description: str = "Execute Python code from a file in a controlled environment with safety checks"
    inputs: Dict[str, Dict[str, str]] = {
        "file_path": {
            "type": "string",
            "description": "The path to the Python file to be executed"
        },
        "language": {
            "type": "string",
            "description": "The programming language of the code (only 'python' is supported)"
        }
    }
    required: Optional[List[str]] = ["file_path"]
    
    def __init__(self, python_interpreter: PythonInterpreter = None):
        super().__init__()
        self.python_interpreter = python_interpreter
    
    def __call__(self, file_path: str, language: str = "python") -> str:
        """Execute Python script file using the Python interpreter."""
        if not self.python_interpreter:
            raise RuntimeError("Python interpreter not initialized")
        
        try:
            return self.python_interpreter.execute_script(file_path, language)
        except Exception as e:
            return f"Error executing script: {str(e)}"


class PythonInterpreterToolkit(Toolkit):
    def __init__(
        self,
        name: str = "PythonInterpreterToolkit",
        project_path: Optional[str] = ".",
        directory_names: Optional[List[str]] = None,
        allowed_imports: Optional[Set[str]] = None,
        storage_handler: FileStorageHandler = None,
        **kwargs
    ):
        # Initialize storage handler if not provided
        if storage_handler is None:
            from .storage_file import LocalStorageHandler
            storage_handler = LocalStorageHandler(base_path="./workplace/python")
        
        # Create the shared Python interpreter instance with storage handler
        python_interpreter = PythonInterpreter(
            name="PythonInterpreter",
            project_path=project_path,
            directory_names=directory_names or [],
            allowed_imports=allowed_imports,
            storage_handler=storage_handler,
            **kwargs
        )
        
        # Create tools with the shared interpreter
        tools = [
            PythonExecuteTool(python_interpreter=python_interpreter),
            PythonExecuteScriptTool(python_interpreter=python_interpreter)
        ]
        
        # Initialize parent with tools
        super().__init__(name=name, tools=tools)
        
        # Store python_interpreter as instance variable
        self.python_interpreter = python_interpreter
    

    
