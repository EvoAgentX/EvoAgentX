from __future__ import annotations

from pathlib import Path


def test_legacy_app_directory_removed():
    repo_root = Path(__file__).resolve().parents[3]
    legacy_app_dir = repo_root / "evoagentx" / "app"
    assert not any(legacy_app_dir.rglob("*"))
