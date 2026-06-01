import contextlib
import io
from pathlib import Path
from typing import Dict, List, Optional

from evoagentx.tools.alita_agent import GeneratedCodeTool
from evoagentx.tools.interpreter_python import PythonExecuteTool, PythonInterpreter
from evoagentx.tools.tool import Tool


class FakePythonExecutor(Tool):
    name: str = "fake_python_execute"
    description: str = "Execute Python code and return stdout."
    inputs: Dict[str, Dict[str, str]] = {
        "code": {"type": "string", "description": "Python source code."},
        "language": {"type": "string", "description": "Execution language."},
    }
    required: Optional[List[str]] = ["code"]

    def __call__(self, code: str, language: str = "python") -> str:
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            exec(code, {})
        return stdout.getvalue()


class AppendingPythonExecutor(FakePythonExecutor):
    name: str = "appending_python_execute"
    description: str = "Execute Python code and append output after execution."
    inputs: Dict[str, Dict[str, str]] = {
        "code": {"type": "string", "description": "Python source code."},
        "language": {"type": "string", "description": "Execution language."},
    }
    required: Optional[List[str]] = ["code"]

    def __call__(self, code: str, language: str = "python") -> str:
        return super().__call__(code, language) + "stderr: warning\n"


class MarkerCollisionPythonExecutor(FakePythonExecutor):
    name: str = "marker_collision_python_execute"
    description: str = "Execute Python code and append a colliding marker line."
    inputs: Dict[str, Dict[str, str]] = {
        "code": {"type": "string", "description": "Python source code."},
        "language": {"type": "string", "description": "Execution language."},
    }
    required: Optional[List[str]] = ["code"]

    def __call__(self, code: str, language: str = "python") -> str:
        marker_start = code.rfind(GeneratedCodeTool._RESULT_MARKER_PREFIX)
        marker_end = code.find("=", marker_start)
        assert marker_start != -1
        assert marker_end != -1
        marker = code[marker_start : marker_end + 1]
        return super().__call__(code, language) + f"{marker}not-json\n"


def test_generated_code_tool_returns_structured_result_without_logs():
    tool = GeneratedCodeTool(
        code_executor=FakePythonExecutor(),
        tool_name="clean_tool",
        description="Clean generated tool",
        code='result = {"echo": payload.get("text")}',
    )

    output = tool(payload={"text": "hello"})

    assert output["success"] is True
    assert output["result"] == {"echo": "hello"}
    assert output["raw_output"]
    assert "logs" not in output


def test_generated_code_tool_separates_stdout_logs_from_result():
    tool = GeneratedCodeTool(
        code_executor=FakePythonExecutor(),
        tool_name="noisy_tool",
        description="Noisy generated tool",
        code='print("debug: starting")\nresult = {"echo": payload.get("text")}',
    )

    output = tool(payload={"text": "hello"})

    assert output["success"] is True
    assert output["result"] == {"echo": "hello"}
    assert output["logs"] == "debug: starting"
    assert "debug: starting" in output["raw_output"]


def test_generated_code_tool_handles_output_after_marked_result():
    tool = GeneratedCodeTool(
        code_executor=AppendingPythonExecutor(),
        tool_name="stderr_tool",
        description="Generated tool with post-result output",
        code='print("debug: starting")\nresult = {"echo": payload.get("text")}',
    )

    output = tool(payload={"text": "hello"})

    assert output["success"] is True
    assert output["result"] == {"echo": "hello"}
    assert output["logs"] == "debug: starting\nstderr: warning"


def test_generated_code_tool_ignores_invalid_marker_collision_after_result():
    tool = GeneratedCodeTool(
        code_executor=MarkerCollisionPythonExecutor(),
        tool_name="collision_tool",
        description="Generated tool with a post-result marker collision",
        code='print("debug: starting")\nresult = {"echo": payload.get("text")}',
    )

    output = tool(payload={"text": "hello"})

    assert output["success"] is True
    assert output["result"] == {"echo": "hello"}
    assert output["logs"].startswith("debug: starting\n")
    assert "not-json" in output["logs"]


def test_generated_code_tool_preserves_legacy_full_json_output():
    output = GeneratedCodeTool._parse_execution_output('{"echo": "hello"}')

    assert output["success"] is True
    assert output["result"] == {"echo": "hello"}
    assert output["raw_output"] == '{"echo": "hello"}'


def test_generated_code_tool_does_not_guess_from_unmarked_json_log():
    raw_output = '{"debug": true}\nTraceback (most recent call last):\nboom'

    output = GeneratedCodeTool._parse_execution_output(raw_output)

    assert output["success"] is True
    assert output["result"] == raw_output


def test_generated_code_tool_separates_stdout_logs_with_python_interpreter():
    repo_root = Path(__file__).resolve().parents[3]
    python_executor = PythonExecuteTool(
        python_interpreter=PythonInterpreter(project_path=str(repo_root))
    )
    tool = GeneratedCodeTool(
        code_executor=python_executor,
        tool_name="real_python_tool",
        description="Generated tool using the real Python executor",
        code='print("debug: starting")\nresult = {"echo": payload.get("text")}',
    )

    output = tool(payload={"text": "hello"})

    assert output["success"] is True
    assert output["result"] == {"echo": "hello"}
    assert output["logs"] == "debug: starting"
