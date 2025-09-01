import platform
import subprocess
from typing import Dict, Any, List, Optional

from .tool import Tool, Toolkit
from .storage_handler import FileStorageHandler
from ..core.logging import logger


class CMDBase:
    """
    Base class for command execution with permission checking and cross-platform support.
    """
    
    def __init__(self, default_shell: str = None, storage_handler: FileStorageHandler = None):
        """
        Initialize CMDBase with system detection and shell configuration.
        
        Args:
            default_shell: Override default shell detection
            storage_handler: Storage handler for file operations
        """
        self.system = platform.system().lower()
        self.default_shell = default_shell or self._detect_default_shell()
        self.permission_cache = {}  # Cache permission responses
        self.storage_handler = storage_handler
        
    def _detect_default_shell(self) -> str:
        """Detect the default shell for the current system."""
        if self.system == "windows":
            return "cmd"
        elif self.system == "darwin":  # macOS
            return "bash"
        else:  # Linux and others
            return "bash"
    
    def _is_dangerous_command(self, command: str) -> Dict[str, Any]:
        """
        Check if a command is potentially dangerous.
        
        Args:
            command: The command to check
            
        Returns:
            Dictionary with danger assessment
        """
        dangerous_patterns = [
            # System modification
            r"\brm\s+-rf\b",  # Recursive force delete
            r"\bdel\s+/[sq]\b",  # Windows force delete
            r"\bformat\b",  # Disk formatting
            r"\bdd\b",  # Disk operations
            r"\bshutdown\b",  # System shutdown
            r"\breboot\b",  # System reboot
            r"\binit\s+[06]\b",  # System halt/reboot
            
            # Network operations
            r"\bnetcat\b",  # Network operations
            r"\bnc\b",  # Netcat shorthand
            r"\bssh\b",  # SSH connections
            r"\bscp\b",  # SCP file transfer
            
            # Process management
            r"\bkill\s+-9\b",  # Force kill
            r"\btaskkill\s+/f\b",  # Windows force kill
            
            # Package management (system-wide)
            r"\bapt\s+install\b",  # Ubuntu/Debian
            r"\byum\s+install\b",  # RHEL/CentOS
            r"\bbrew\s+install\b",  # macOS
            r"\bchoco\s+install\b",  # Windows Chocolatey
            
            # User management
            r"\buseradd\b",  # Add user
            r"\buserdel\b",  # Delete user
            r"\bpasswd\b",  # Change password
            
            # File system operations
            r"\bmount\b",  # Mount operations
            r"\bumount\b",  # Unmount operations
            r"\bchmod\s+777\b",  # Dangerous permissions
            r"\bchown\s+root\b",  # Change ownership to root
        ]
        
        import re
        command_lower = command.lower()
        
        for pattern in dangerous_patterns:
            if re.search(pattern, command_lower):
                return {
                    "is_dangerous": True,
                    "reason": f"Command matches dangerous pattern: {pattern}",
                    "risk_level": "high"
                }
        
        # Check for sudo/administrator commands
        if command_lower.startswith(("sudo ", "runas ")):
            return {
                "is_dangerous": True,
                "reason": "Command requires elevated privileges",
                "risk_level": "high"
            }
        
        # Check for file operations in system directories
        system_dirs = ["/etc/", "/usr/", "/var/", "/bin/", "/sbin/", "C:\\Windows\\", "C:\\Program Files\\"]
        for sys_dir in system_dirs:
            if sys_dir in command:
                return {
                    "is_dangerous": True,
                    "reason": f"Command operates on system directory: {sys_dir}",
                    "risk_level": "medium"
                }
        
        return {"is_dangerous": False, "risk_level": "low"}
    
    def _request_permission(self, command: str, danger_assessment: Dict[str, Any]) -> bool:
        """
        Request permission from user to execute command.
        
        Args:
            command: The command to execute
            danger_assessment: Assessment of command danger
            
        Returns:
            True if permission granted, False otherwise
        """
        print(f"\n{'='*60}")
        print("🔒 PERMISSION REQUEST")
        print(f"{'='*60}")
        print(f"Command: {command}")
        print(f"System: {self.system}")
        print(f"Shell: {self.default_shell}")
        
        if danger_assessment["is_dangerous"]:
            print(f"⚠️  WARNING: {danger_assessment['reason']}")
            print(f"Risk Level: {danger_assessment['risk_level'].upper()}")
        else:
            print("✅ Command appears safe")
        
        print("\nDo you want to execute this command?")
        print("Options:")
        print("  y/Y - Yes, execute the command")
        print("  n/N - No, do not execute")
        print("  [reason] - No, with explanation")
        print("  [empty] - No, without explanation")
        
        try:
            response = input("\nYour response: ").strip().lower()
            
            if response in ['y', 'yes']:
                print("✅ Permission granted. Executing command...")
                return True
            elif response in ['n', 'no', '']:
                print("❌ Permission denied.")
                return False
            else:
                print(f"❌ Permission denied. Reason: {response}")
                return False
                
        except KeyboardInterrupt:
            print("\n❌ Permission request cancelled by user.")
            return False
    
    def execute_command(self, command: str, timeout: int = 30, cwd: str = None) -> Dict[str, Any]:
        """
        Execute a command with permission checking.
        
        Args:
            command: The command to execute
            timeout: Command timeout in seconds
            cwd: Working directory for command execution
            
        Returns:
            Dictionary with execution results
        """
        try:
            # Check if command is dangerous
            danger_assessment = self._is_dangerous_command(command)
            
            # Request permission
            if not self._request_permission(command, danger_assessment):
                return {
                    "success": False,
                    "error": "Permission denied by user",
                    "command": command,
                    "stdout": "",
                    "stderr": "",
                    "return_code": None
                }
            
            # Prepare command execution
            if self.system == "windows":
                # Windows command execution
                if self.default_shell == "cmd":
                    cmd_args = ["cmd", "/c", command]
                else:  # PowerShell
                    cmd_args = ["powershell", "-Command", command]
            else:
                # Unix-like systems
                cmd_args = [self.default_shell, "-c", command]
            
            # Execute command
            logger.info(f"Executing command: {command}")
            
            result = subprocess.run(
                cmd_args,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=cwd,
                shell=False  # We're already using shell commands
            )
            
            result_dict = {
                "success": result.returncode == 0,
                "command": command,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_code": result.returncode,
                "system": self.system,
                "shell": self.default_shell
            }
            
            # Add file operation context if storage handler is available
            if self.storage_handler:
                result_dict["storage_handler"] = type(self.storage_handler).__name__
                result_dict["storage_base_path"] = str(self.storage_handler.base_path)
            
            return result_dict
            
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": f"Command timed out after {timeout} seconds",
                "command": command,
                "stdout": "",
                "stderr": "",
                "return_code": None
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "command": command,
                "stdout": "",
                "stderr": "",
                "return_code": None
            }


class ExecuteCommandTool(Tool):
    name: str = "execute_command"
    description: str = "Execute a command line operation with permission checking and cross-platform support. Can handle all command line operations including directory creation, file listing, system info, and more."
    inputs: Dict[str, Dict[str, str]] = {
        "command": {
            "type": "string",
            "description": "The command to execute (e.g., 'ls -la', 'dir', 'mkdir test', 'pwd', 'whoami', 'date', etc.)"
        },
        "timeout": {
            "type": "integer",
            "description": "Command timeout in seconds (default: 30)"
        },
        "working_directory": {
            "type": "string",
            "description": "Working directory for command execution (optional)"
        }
    }
    required: Optional[List[str]] = ["command"]

    def __init__(self, cmd_base: CMDBase = None):
        super().__init__()
        self.cmd_base = cmd_base or CMDBase()

    def __call__(self, command: str, timeout: int = 30, working_directory: str = None) -> Dict[str, Any]:
        """
        Execute a command with permission checking.
        
        Args:
            command: The command to execute
            timeout: Command timeout in seconds
            working_directory: Working directory for command execution
            
        Returns:
            Dictionary containing the command execution result
        """
        try:
            result = self.cmd_base.execute_command(
                command=command,
                timeout=timeout,
                cwd=working_directory
            )
            
            if result["success"]:
                logger.info(f"Successfully executed command: {command}")
            else:
                logger.error(f"Failed to execute command {command}: {result.get('error', 'Unknown error')}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in execute_command tool: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "command": command
            }


class CMDToolkit(Toolkit):
    """
    Command line toolkit that provides safe command execution with permission checking
    and cross-platform support. Supports Linux, macOS, and Windows.
    """
    
    def __init__(self, name: str = "CMDToolkit", default_shell: str = None, storage_handler: FileStorageHandler = None):
        """
        Initialize the CMDToolkit with a shared command base instance.
        
        Args:
            name: Name of the toolkit
            default_shell: Override default shell detection
            storage_handler: Storage handler for file operations
        """
        # Initialize storage handler if not provided
        if storage_handler is None:
            from .storage_file import LocalStorageHandler
            storage_handler = LocalStorageHandler(base_path="./workplace/cmd")
        
        # Create the shared command base instance with storage handler
        cmd_base = CMDBase(default_shell=default_shell, storage_handler=storage_handler)
        
        # Initialize tools with the shared command base
        tools = [
            ExecuteCommandTool(cmd_base=cmd_base)
        ]
        
        # Initialize parent with tools
        super().__init__(name=name, tools=tools)
        self.cmd_base = cmd_base
        self.storage_handler = storage_handler 