from __future__ import annotations

from pathlib import Path


def test_app_env_credentials_removed():
    repo_root = Path(__file__).resolve().parents[3]
    legacy_app_env = repo_root / "evoagentx" / "app" / "app.env"
    assert not legacy_app_env.exists()


def test_app_env_example_exists_and_has_no_real_mongodb_uri():
    repo_root = Path(__file__).resolve().parents[3]
    env_example = repo_root / "evoagentx" / "app" / ".env.example"
    assert env_example.exists()

    content = env_example.read_text(encoding="utf-8")
    assert "mongodb+srv://" not in content
    assert "eax:eax@" not in content
