from __future__ import annotations

from pathlib import Path


def read_pyproject_text() -> str:
    repo_root = Path(__file__).resolve().parents[3]
    return (repo_root / "pyproject.toml").read_text(encoding="utf-8")


def parse_list_block(text: str, start_key: str) -> list[str]:
    start = text.find(f"{start_key} = [")
    assert start != -1
    block = text[start:].split("]", 1)[0]
    lines = block.splitlines()
    items = []
    for line in lines:
        stripped = line.strip().strip(",")
        if stripped.startswith('"') and stripped.endswith('"'):
            items.append(stripped.strip('"'))
    return items


def parse_optional_dependency_keys(text: str) -> set[str]:
    lines = text.splitlines()
    keys = set()
    in_optional = False
    for line in lines:
        if line.strip() == "[project.optional-dependencies]":
            in_optional = True
            continue
        if in_optional and line.strip().startswith("["):
            break
        if in_optional:
            stripped = line.strip()
            if stripped.endswith("["):
                key = stripped.split("=", 1)[0].strip()
                keys.add(key)
    return keys


def test_optional_dependency_groups_present():
    text = read_pyproject_text()
    optional_keys = parse_optional_dependency_keys(text)
    expected_groups = {
        "server",
        "rag",
        "tools",
        "multimodal",
        "optimizers",
        "benchmarks",
        "viz",
        "all",
    }
    assert expected_groups.issubset(optional_keys)


def test_core_dependencies_do_not_include_server_packages():
    text = read_pyproject_text()
    core_deps = parse_list_block(text, "dependencies")
    forbidden = {"fastapi", "uvicorn", "motor", "redis", "celery"}
    assert all(not any(dep.startswith(pkg) for pkg in forbidden) for dep in core_deps)
