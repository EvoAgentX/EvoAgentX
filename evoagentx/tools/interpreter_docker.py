import io
import shlex
import tarfile
import uuid
import docker
from pathlib import Path
from typing import ClassVar, Dict, Any, List
from .interpreter_base import BaseInterpreter
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
    container:docker.models.containers.Container = Field(default=None, description="The container to use for the interpreter")
    client:docker.client.DockerClient = Field(default=None, description="Docker client used for container operations")
    image_tag:str = Field(default=None, description="The image tag to use for the container")
    dockerfile_path:str = Field(default="", description="The path to the dockerfile to use for the container")
    host_directory:str = Field(default="", description="The path to the host directory to use for the container")
    container_directory:str = Field(default="/home/app/", description="The directory to use for the container")
    container_command:str = Field(default="tail -f /dev/null", description="The command to use for the container")
    tmp_directory:str = Field(default="/tmp", description="The directory to use for the container")
    
    class Config:
        arbitrary_types_allowed = True  # Allow non-pydantic types like sets

    def __init__(self, **data):
        # Initialize with default name and tool info
        name = data.get('name', 'DockerInterpreter')
        schemas = self.get_tool_schemas()
        descriptions = self.get_tool_descriptions()
        tools = self.get_tools()
        
        # Pass these to the parent class initialization
        super().__init__(
            name=name,
            schemas=schemas, 
            descriptions=descriptions,
            tools=tools,
            **data
        )
        self.dockerfile_path = data.get('dockerfile_path', "")
        
        # Initialize Docker client and container
        self.client = docker.from_env()
        self._initialize_if_needed()
        self.image_tag = data.get('image_tag', None)
        
        # Upload directory if specified
        if self.host_directory:
            self._upload_directory_to_container(self.host_directory)

    def __del__(self):
        if self.container is not None:
            self.container.remove(force=True)

    def _initialize_if_needed(self):
        if self.container is not None:
            return
        if self.image_tag:
            try:
                # Try to get the existing image first
                self.client.images.get(self.image_tag)
            except Exception as e:
                raise ValueError(f"Image provided in image_tag but not found: {e}")
        else:
            # Image not found, need to build it - now we check for dockerfile_path
            if not self.dockerfile_path:
                raise ValueError("dockerfile_path or image_tag must be provided to build the image")
                
            dockerfile_path = Path(self.dockerfile_path)
            if not dockerfile_path.exists():
                raise FileNotFoundError(f"Dockerfile not found at provided path: {dockerfile_path}")
            
            dockerfile_dir = dockerfile_path.parent
            self.client.images.build(path=str(dockerfile_dir), tag=self.image_tag, rm=True, buildargs={})

        # Run the container using the image (whether pre-existing or newly built)
        self.container = self.client.containers.run(
            self.image_tag, 
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
            tarinfo.size = len(content)
            tar.addfile(tarinfo, io.BytesIO(content.encode('utf-8')))
        tar_stream.seek(0)

        if self.container is None:
            raise RuntimeError("Container is not initialized.")
        self.container.put_archive(self.tmp_directory, tar_stream)
        return Path(f"{self.tmp_directory}/{filename}")

    def _run_file_in_container(self, file: Path, language: str) -> str:
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
        """
        if self.host_directory:
            code = f"import sys; sys.path.insert(0, '{self.container_directory}');" + code
        language = self._check_language(language)
        if self.require_confirm:
            confirmation = input(f"Confirm execution of {language} code? [Y/n]: ")
            if confirmation.lower() not in ["y", "yes", ""]:
                raise RuntimeError("Execution aborted by user.")
        
        
        file_path = self._create_file_in_container(code)
        return self._run_file_in_container(file_path, language)
        
    def execute_script(self, file_path: str, language: str = None) -> str:
        """
        Reads code from a file and executes it in a Docker container.
        
        Args:
            file_path (str): The path to the script file to execute
            language (str, optional): The programming language of the code. If None, 
                                      will be determined from the file extension.
                                      
        Returns:
            str: The execution output
        """
        # Check if file exists
        if not os.path.isfile(file_path):
            return f"Error: File '{file_path}' does not exist."
        # Read the file content
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
        except Exception as e:
            return f"Error reading file: {e}"
            
        # Execute the code
        return self.execute(code, language)

    def _check_language(self, language: str) -> str:
        if language not in self.CODE_TYPE_MAPPING:
            raise ValueError(f"Unsupported language: {language}")
        return self.CODE_TYPE_MAPPING[language]

    def get_tool_schemas(self) -> list[Dict[str, Any]]:
        """
        Returns the OpenAI-compatible function schema for the Docker interpreter.
        
        Returns:
            list[Dict[str, Any]]: Function schema in OpenAI format
        """
        schemas = [
            {
                "type": "function",
                "function": {
                    "name": "execute",
                    "description": "The Docker Interpreter Tool provides a secure and isolated environment for executing code inside a Docker container.",
                    "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "The code to execute"
                        },
                        "language": {
                            "type": "string",
                            "description": "The programming language of the code (e.g., python, py, python3)"
                        }
                    },
                        "required": ["code", "language"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "execute_script",
                    "description": "Execute code from a script file in a secure Docker container environment.",
                    "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "The path to the script file to execute"
                        },
                        "language": {
                            "type": "string",
                            "description": "The programming language of the code. If not provided, will be determined from file extension."
                        }
                    },
                        "required": ["file_path"]
                    }
                }
            }
        ]
        return schemas
        
    def get_tool_descriptions(self) -> List[str]:
        """
        Returns a brief description of the Docker interpreter tool.
        
        Returns:
            List[str]: Tool description
        """
        return [
            "Docker Interpreter Tool that provides a secure and isolated environment for executing code inside Docker containers.",
            "Docker Script Execution Tool that allows running code from script files in a containerized environment."
        ]
        
    def get_tools(self) -> List[callable]:
        """
        Returns a list of callable methods provided by this tool.
        
        Returns:
            List[callable]: List of callable methods
        """
        return [self.execute, self.execute_script]