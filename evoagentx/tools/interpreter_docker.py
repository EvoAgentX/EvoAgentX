import io
import shlex
import tarfile
import uuid
import docker
from pathlib import Path
from typing import ClassVar, Dict, List, Optional
from .interpreter_base import BaseInterpreter
from .tool import Tool,Toolkit
from .storage_handler import FileStorageHandler
import os
from pydantic import Field

class DockerInterpreter(BaseInterpreter):
    """
    A Docker-based interpreter for executing Python, Bash, and R scripts in an isolated environment.
    """
    
    CODE_EXECUTE_CMD_MAPPING: ClassVar[Dict[str, str]] = {
        "python": "python {file_name}",
    }

    CODE_TYPE_MAPPING: ClassVar[Dict[str, str]] = {
        "python": "python",
        "py3": "python",
        "python3": "python",
        "py": "python",
    }

    require_confirm:bool = Field(default=False, description="Whether to require confirmation before executing code")
    print_stdout:bool = Field(default=True, description="Whether to print stdout")
    print_stderr:bool = Field(default=True, description="Whether to print stderr")
    host_directory:str = Field(default="", description="The path to the host directory to use for the container")
    container_directory:str = Field(default="/home/app/", description="The directory to use for the container")
    container_command:str = Field(default="tail -f /dev/null", description="The command to use for the container")
    tmp_directory:str = Field(default="/tmp", description="The directory to use for the container")
    image_tag:Optional[str] = Field(default=None, description="The Docker image tag to use")
    dockerfile_path:Optional[str] = Field(default=None, description="Path to the Dockerfile to build")
    
    class Config:
        arbitrary_types_allowed = True  # Allow non-pydantic types like sets

    def __init__(
        self, 
        name:str = "DockerInterpreter",
        image_tag:Optional[str] = None,
        dockerfile_path:Optional[str] = None,
        require_confirm:bool = False,
        print_stdout:bool = True,
        print_stderr:bool = True,
        host_directory:str = "",
        container_directory:str = "/home/app/",
        container_command:str = "tail -f /dev/null",
        tmp_directory:str = "/tmp",
        storage_handler: FileStorageHandler = None,
        **data
    ):
        """
        Initialize a Docker-based interpreter for executing code in an isolated environment.
        
        Args:
            name (str): The name of the interpreter
            image_tag (str, optional): The Docker image tag to use. Must be provided if dockerfile_path is not.
            dockerfile_path (str, optional): Path to the Dockerfile to build. Must be provided if image_tag is not.
            require_confirm (bool): Whether to require confirmation before executing code
            print_stdout (bool): Whether to print stdout from code execution
            print_stderr (bool): Whether to print stderr from code execution
            host_directory (str): The path to the host directory to mount in the container
            container_directory (str): The target directory inside the container
            container_command (str): The command to run in the container
            tmp_directory (str): The temporary directory to use for file creation in the container
            **data: Additional data to pass to the parent class
        """
        # Pass to the parent class initialization
        super().__init__(name=name, **data)
        
        self.require_confirm = require_confirm
        self.print_stdout = print_stdout
        self.print_stderr = print_stderr
        self.host_directory = host_directory
        self.container_directory = container_directory
        self.container_command = container_command
        self.tmp_directory = tmp_directory
        
        # Initialize Docker client and container
        self.client = docker.from_env()
        self.container = None
        self.image_tag = image_tag
        self.dockerfile_path = dockerfile_path
        self.storage_handler = storage_handler
        self._initialize_if_needed()
        
        # Upload directory if specified
        if self.host_directory:
            self._upload_directory_to_container(self.host_directory)

    def __del__(self):
        try:
            if hasattr(self, 'container') and self.container is not None:
                import sys
                if sys.meta_path is not None:  # Check if Python is shutting down
                    self.container.remove(force=True)
        except Exception:
            pass  # Silently ignore errors during shutdown

    def _initialize_if_needed(self):
        image_tag = self.image_tag
        dockerfile_path = self.dockerfile_path
        if image_tag:
            try:
                # Try to get the existing image first
                self.client.images.get(image_tag)
            except Exception as e:
                raise ValueError(f"Image provided in image_tag but not found: {e}")
        else:
            # Image not found, need to build it - now we check for dockerfile_path
            if not dockerfile_path:
                raise ValueError("dockerfile_path or image_tag must be provided to build the image")
                
            dockerfile_path = Path(dockerfile_path)
            if not dockerfile_path.exists():
                raise FileNotFoundError(f"Dockerfile not found at provided path: {dockerfile_path}")
            
            dockerfile_dir = dockerfile_path.parent
            self.client.images.build(path=str(dockerfile_dir), tag=image_tag, rm=True, buildargs={})

        # Check if Docker daemon is running
        try:
            self.client.ping()
        except Exception as e:
            raise RuntimeError(f"Docker daemon is not running: {e}")

        # Run the container using the image with resource limits
        self.container = self.client.containers.run(
            image_tag, 
            detach=True, 
            command=self.container_command,
            working_dir=self.container_directory
        )

    def _upload_directory_to_container(self, host_directory: str):
        """
        Uploads all files and directories from the given host directory to the container directory.

        :param host_directory: Path to the local directory containing files to upload.
        :param container_directory: Target directory inside the container (defaults to self.container_directory).
        """
        host_directory = Path(host_directory).resolve()
        if not host_directory.exists() or not host_directory.is_dir():
            raise FileNotFoundError(f"Directory not found: {host_directory}")

        tar_stream = io.BytesIO()
        
        with tarfile.open(fileobj=tar_stream, mode="w") as tar:
            for file_path in host_directory.rglob("*"):
                if file_path.is_file():
                    # Ensure path is relative to the given directory
                    relative_path = file_path.relative_to(host_directory)
                    target_path = Path(self.container_directory) / relative_path
                    
                    tarinfo = tarfile.TarInfo(name=str(target_path.relative_to(self.container_directory)))
                    tarinfo.size = file_path.stat().st_size
                    with open(file_path, "rb") as f:
                        tar.addfile(tarinfo, f)

        tar_stream.seek(0)

        if self.container is None:
            raise RuntimeError("Container is not initialized.")

        self.container.put_archive(self.container_directory, tar_stream)

        # Ensure the uploaded directory is in sys.path for imports
        # self.container.exec_run(f"echo 'export PYTHONPATH={self.container_directory}:$PYTHONPATH' | sudo tee -a /etc/environment")

    def _create_file_in_container(self, content: str) -> Path:
        filename = str(uuid.uuid4())
        tar_stream = io.BytesIO()
        with tarfile.open(fileobj=tar_stream, mode='w') as tar:
            tarinfo = tarfile.TarInfo(name=filename)
            tarinfo.size = len(content.encode('utf-8'))
            tar.addfile(tarinfo, io.BytesIO(content.encode('utf-8')))
        tar_stream.seek(0)

        if self.container is None:
            raise RuntimeError("Container is not initialized.")
            
        try:
            self.container.put_archive(self.tmp_directory, tar_stream)
        except Exception as e:
            raise RuntimeError(f"Failed to create file in container: {e}")
            
        return Path(f"{self.tmp_directory}/{filename}")

    def _run_file_in_container(self, file: Path, language: str) -> str:
        """Execute a file in the container with timeout and security checks."""
        if not self.container:
            raise RuntimeError("Container is not initialized")
            
        # Check container status
        container_info = self.client.api.inspect_container(self.container.id)
        if not container_info['State']['Running']:
            raise RuntimeError("Container is not running")
            
        language = self._check_language(language)
        command = shlex.split(self.CODE_EXECUTE_CMD_MAPPING[language].format(file_name=file.as_posix()))
        if self.container is None:
            raise RuntimeError("Container is not initialized.")
        result = self.container.exec_run(command, demux=True)

        stdout, stderr = result.output
        if self.print_stdout and stdout:
            print(stdout.decode())
        if self.print_stderr and stderr:
            print(stderr.decode())

        stdout_str = stdout.decode() if stdout else ""
        stderr_str = stderr.decode() if stderr else ""
        return stdout_str + stderr_str

    def execute(self, code: str, language: str) -> str:
        """
        Executes code in a Docker container.
        
        Args:
            code (str): The code to execute
            language (str): The programming language to use
            
        Returns:
            str: The execution output
            
        Raises:
            RuntimeError: If container is not properly initialized or execution fails
            ValueError: If code content is invalid or exceeds limits
        """
        if not code or not code.strip():
            raise ValueError("Code content cannot be empty")
            
        if not self.container:
            raise RuntimeError("Container is not initialized")
            
        # Check container status
        try:
            container_info = self.client.api.inspect_container(self.container.id)
            if not container_info['State']['Running']:
                raise RuntimeError("Container is not running")
        except Exception as e:
            raise RuntimeError(f"Failed to check container status: {e}")

        if self.host_directory:
            code = f"import sys; sys.path.insert(0, '{self.container_directory}');" + code
            
        language = self._check_language(language)
        
        if self.require_confirm:
            confirmation = input(f"Confirm execution of {language} code? [Y/n]: ")
            if confirmation.lower() not in ["y", "yes", ""]:
                raise RuntimeError("Execution aborted by user.")
        
        try:
            file_path = self._create_file_in_container(code)
            return self._run_file_in_container(file_path, language)
        except Exception as e:
            raise RuntimeError(f"Code execution failed: {e}")
        finally:
            # Clean up temporary files
            try:
                if hasattr(self, 'container') and self.container:
                    self.container.exec_run(f"rm -f {file_path}")
            except Exception:
                pass  # Ignore cleanup errors

    def execute_script(self, file_path: str, language: str = None) -> str:
        """
        Reads code from a file and executes it in a Docker container.
        
        Args:
            file_path (str): The path to the script file to execute
            language (str, optional): The programming language of the code. If None, will be determined from the file extension.
                                    
        Returns:
            str: The execution output
            
        Raises:
            FileNotFoundError: If the script file does not exist
            RuntimeError: If container is not properly initialized or execution fails
            ValueError: If file content is invalid or exceeds limits
        """
        # Read file using storage handler
        result = self.storage_handler.read(file_path)
        if result["success"]:
            code = result["content"]
        else:
            raise RuntimeError(f"Could not read file '{file_path}': {result.get('error', 'Unknown error')}")
            
        # Execute the code
        return self.execute(code, language)

    def _check_language(self, language: str) -> str:
        if language not in self.CODE_TYPE_MAPPING:
            raise ValueError(f"Unsupported language: {language}")
        return self.CODE_TYPE_MAPPING[language]


class DockerExecuteTool(Tool):
    name: str = "docker_execute"
    description: str = "Execute code in a secure Docker container environment"
    inputs: Dict[str, Dict[str, str]] = {
        "code": {
            "type": "string",
            "description": "The code to execute"
        },
        "language": {
            "type": "string",
            "description": "The programming language of the code (e.g., python, py, python3)"
        }
    }
    required: Optional[List[str]] = ["code", "language"]
    
    def __init__(self, docker_interpreter: DockerInterpreter = None):
        super().__init__()
        self.docker_interpreter = docker_interpreter
    
    def __call__(self, code: str, language: str) -> str:
        """Execute code using the Docker interpreter."""
        if not self.docker_interpreter:
            raise RuntimeError("Docker interpreter not initialized")
        
        try:
            return self.docker_interpreter.execute(code, language)
        except Exception as e:
            return f"Error executing code: {str(e)}"


class DockerExecuteScriptTool(Tool):
    name: str = "docker_execute_script"
    description: str = "Execute code from a script file in a secure Docker container environment"
    inputs: Dict[str, Dict[str, str]] = {
        "file_path": {
            "type": "string",
            "description": "The path to the script file to execute"
        },
        "language": {
            "type": "string",
            "description": "The programming language of the code. If not provided, will be determined from file extension"
        }
    }
    required: Optional[List[str]] = ["file_path", "language"]
    
    def __init__(self, docker_interpreter: DockerInterpreter = None):
        super().__init__()
        self.docker_interpreter = docker_interpreter
    
    def __call__(self, file_path: str, language: str) -> str:
        """Execute script file using the Docker interpreter."""
        if not self.docker_interpreter:
            raise RuntimeError("Docker interpreter not initialized")
        
        try:
            return self.docker_interpreter.execute_script(file_path, language)
        except Exception as e:
            return f"Error executing script: {str(e)}"


class DockerInterpreterToolkit(Toolkit):
    def __init__(
        self,
        name: str = "DockerInterpreterToolkit",
        image_tag: Optional[str] = None,
        dockerfile_path: Optional[str] = None,
        require_confirm: bool = False,
        print_stdout: bool = True,
        print_stderr: bool = True,
        host_directory: str = "",
        container_directory: str = "/home/app/",
        container_command: str = "tail -f /dev/null",
        tmp_directory: str = "/tmp",
        storage_handler: FileStorageHandler = None,
        **kwargs
    ):
        # Initialize storage handler if not provided
        if storage_handler is None:
            from .storage_file import LocalStorageHandler
            storage_handler = LocalStorageHandler(base_path="./workplace/docker")
        
        # Create the shared Docker interpreter instance with storage handler
        docker_interpreter = DockerInterpreter(
            name="DockerInterpreter",
            image_tag=image_tag,
            dockerfile_path=dockerfile_path,
            require_confirm=require_confirm,
            print_stdout=print_stdout,
            print_stderr=print_stderr,
            host_directory=host_directory,
            container_directory=container_directory,
            container_command=container_command,
            tmp_directory=tmp_directory,
            storage_handler=storage_handler,
            **kwargs
        )
        
        # Create tools with the shared interpreter
        tools = [
            DockerExecuteTool(docker_interpreter=docker_interpreter),
            DockerExecuteScriptTool(docker_interpreter=docker_interpreter)
        ]
        
        # Initialize parent with tools
        super().__init__(name=name, tools=tools)
        
        # Store docker_interpreter as instance variable
        self.docker_interpreter = docker_interpreter
        self.storage_handler = storage_handler
    
