"""Check optional imports in fresh interpreters to avoid module-cache false positives."""

import os
import subprocess
import sys
import textwrap
from pathlib import Path


def run_isolated(source):
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(source)],
        cwd=Path(__file__).resolve().parents[3],
        env={**os.environ, "LITELLM_LOCAL_MODEL_COST_MAP": "True"},
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_agent_and_short_term_memory_do_not_import_rag():
    run_isolated("""
        import importlib.abc
        import sys

        class BlockRAG(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "evoagentx.rag" or fullname.startswith("evoagentx.rag."):
                    raise AssertionError(f"Unexpected RAG import: {fullname}")

        sys.meta_path.insert(0, BlockRAG())
        from evoagentx.memory import BaseMemory, ShortTermMemory
        from evoagentx.agents import Agent

        agent = Agent(
            name="test", description="test", is_human=True,
            long_term_memory=None, long_term_memory_manager=None,
        )
        assert isinstance(agent.short_term_memory, ShortTermMemory)
        restored = Agent.from_dict(agent.to_dict())
        assert restored.long_term_memory is None
        assert "evoagentx.memory.long_term_memory" not in sys.modules
        assert "evoagentx.memory.memory_manager" not in sys.modules
    """)


def test_rag_package_and_schema_do_not_import_engine():
    run_isolated("""
        import importlib.abc
        import sys

        class BlockEngine(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "evoagentx.rag.rag":
                    raise AssertionError("Unexpected RAG engine import")

        sys.meta_path.insert(0, BlockEngine())
        import evoagentx.rag as rag
        assert "evoagentx.rag.schema" not in sys.modules
        assert set(rag.__all__) <= set(dir(rag))
        from evoagentx.rag import Query
        from evoagentx.rag.schema import Query as SchemaQuery
        assert Query is SchemaQuery
        assert rag.Query is Query
        try:
            rag.missing_export
        except AttributeError:
            pass
        else:
            raise AssertionError("Unknown exports must raise AttributeError")
    """)


def test_optional_memory_fields_keep_concrete_type_validation():
    run_isolated("""
        import sys
        from types import ModuleType

        from pydantic import ValidationError
        from evoagentx.agents import Agent
        from evoagentx.core.module import BaseModule

        # Replace the optional RAG backend while exercising the real memory models.
        class RAGConfig(BaseModule):
            pass

        class RAGEngine(BaseModule):
            pass

        for module_name, name, value in [
            ("evoagentx.rag.rag_config", "RAGConfig", RAGConfig),
            ("evoagentx.rag.rag", "RAGEngine", RAGEngine),
        ]:
            module = ModuleType(module_name)
            setattr(module, name, value)
            sys.modules[module_name] = module

        from evoagentx.memory import LongTermMemory, MemoryManager
        from evoagentx.storages.base import StorageHandler

        # No storage or indexing work is needed to validate Agent's fields.
        LongTermMemory.init_module = lambda self: None
        memory_data = {
            "rag_config": RAGConfig(),
            "storage_handler": StorageHandler.model_construct(),
        }
        memory = LongTermMemory(**memory_data)
        manager = MemoryManager(memory=memory)
        agent_data = dict(name="test", description="test", is_human=True)
        agent = Agent(**agent_data, long_term_memory=memory,
                      long_term_memory_manager=manager)
        assert agent.long_term_memory is memory
        assert agent.long_term_memory_manager is manager

        for extra in ({}, {"class_name": "LongTermMemory"}):
            restored = Agent.from_dict({
                **agent_data, "long_term_memory": {**memory_data, **extra},
            })
            assert isinstance(restored.long_term_memory, LongTermMemory)

        for field in ("long_term_memory", "long_term_memory_manager"):
            try:
                Agent(**agent_data, **{field: object()})
            except ValidationError:
                pass
            else:
                raise AssertionError(f"Invalid {field} must be rejected")
    """)
