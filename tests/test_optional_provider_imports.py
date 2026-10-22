import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


@pytest.mark.parametrize("source", [
    """
    from evoagentx.benchmark import GSM8K
    from evoagentx.benchmark.gsm8k import GSM8K as Implementation
    assert GSM8K is Implementation
    assert "evoagentx.benchmark.livecodebench" not in sys.modules
    """,
    """
    from evoagentx.rag.rag_config import EmbeddingConfig
    from evoagentx.rag.embeddings import EmbeddingFactory, OpenAIEmbeddingWrapper
    from evoagentx.rag.embeddings.base import EmbeddingProvider
    wrapper = EmbeddingFactory().create(
        EmbeddingProvider.OPENAI,
        {"provider": "openai", "api_key": "test", "model_name": "text-embedding-ada-002"},
    )
    assert isinstance(wrapper, OpenAIEmbeddingWrapper)
    assert "evoagentx.rag.embeddings.huggingface_embedding" not in sys.modules
    """,
])
def test_unrelated_optional_providers_are_not_imported(source):
    guard = """
import importlib.abc
import sys

class BlockOptionalProviders(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"sentence_transformers", "datasets"}:
            raise AssertionError(f"Unexpected optional dependency: {fullname}")

sys.meta_path.insert(0, BlockOptionalProviders())
"""
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(guard) + textwrap.dedent(source)],
        cwd=Path(__file__).resolve().parents[1],
        env={**os.environ, "LITELLM_LOCAL_MODEL_COST_MAP": "True"},
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
