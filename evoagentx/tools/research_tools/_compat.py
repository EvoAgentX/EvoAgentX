"""
Compatibility shim for research_tools.

This module patches symbols that the research-tools sub-modules expect to find
in the wider EvoAgentX project but which do not exist there yet.  It MUST be
imported before any other sub-module of ``research_tools`` (the package
``__init__.py`` takes care of this).

**Patched locations:**

1. ``evoagentx.tools.tool``      → ``ToolMetadata``, ``ToolResult``
2. ``evoagentx.utils.utils``     → ``add_dict``, ``ContextualThreadPoolExecutor``
3. ``evoagentx.models.model_utils`` → ``track_cost``, ``CostTracker``

After this module executes, regular imports such as
``from ..tool import ToolMetadata`` will resolve normally.
"""

from __future__ import annotations

import contextvars
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict

# ---------------------------------------------------------------------------
# 1.  ToolMetadata / ToolResult  →  evoagentx.tools.tool
# ---------------------------------------------------------------------------
import evoagentx.tools.tool as _tool_mod  # always importable

if not hasattr(_tool_mod, "ToolMetadata"):
    try:
        from pydantic import BaseModel, Field as PydanticField
    except ImportError:  # pydantic not installed – provide a minimal stand-in
        from evoagentx.core.module import BaseModule as BaseModel  # type: ignore
        PydanticField = field  # type: ignore

    class ToolMetadata(BaseModel):
        """Per-invocation metadata (tool name, call args, cost breakdown)."""
        tool_name: str = ""
        args: Dict[str, Any] = PydanticField(default_factory=dict)
        cost_breakdown: Dict[str, float] = PydanticField(default_factory=dict)

        def add_cost_breakdown(self, cost_breakdown: Dict[str, float]) -> None:
            for key, value in cost_breakdown.items():
                self.cost_breakdown[key] = self.cost_breakdown.get(key, 0.0) + value

    class ToolResult(BaseModel):
        """Bundles a tool's output with its invocation metadata."""
        metadata: ToolMetadata = PydanticField(default_factory=ToolMetadata)
        result: Any = None

    # Inject into the module so ``from ..tool import ToolMetadata`` works
    _tool_mod.ToolMetadata = ToolMetadata  # type: ignore[attr-defined]
    _tool_mod.ToolResult = ToolResult  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# 2.  add_dict / ContextualThreadPoolExecutor  →  evoagentx.utils.utils
# ---------------------------------------------------------------------------
import evoagentx.utils.utils as _utils_mod

if not hasattr(_utils_mod, "add_dict"):
    def add_dict(base: Dict[str, float], update: Dict[str, float]) -> Dict[str, float]:
        """Merge two cost-breakdown dicts by summing values."""
        result = dict(base)
        for k, v in update.items():
            result[k] = result.get(k, 0.0) + v
        return result

    _utils_mod.add_dict = add_dict  # type: ignore[attr-defined]

if not hasattr(_utils_mod, "ContextualThreadPoolExecutor"):
    class ContextualThreadPoolExecutor(ThreadPoolExecutor):
        """ThreadPoolExecutor that copies *contextvars* into worker threads."""

        def submit(self, fn, /, *args, **kwargs):  # noqa: D102
            ctx = contextvars.copy_context()
            return super().submit(ctx.run, fn, *args, **kwargs)

    _utils_mod.ContextualThreadPoolExecutor = ContextualThreadPoolExecutor  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# 3.  track_cost / CostTracker  →  evoagentx.models.model_utils
# ---------------------------------------------------------------------------
import evoagentx.models.model_utils as _mu_mod

if not hasattr(_mu_mod, "track_cost"):
    @dataclass
    class CostTracker:
        """Holds the incremental LLM cost incurred inside a ``track_cost`` block."""
        cost_per_model: Dict[str, float] = field(default_factory=dict)
        total_llm_cost: float = 0.0

    @contextmanager
    def track_cost():
        """Capture incremental LLM cost by diffing ``cost_manager`` totals."""
        # ``cost_manager`` lives on the model_utils module; grab a snapshot
        cm = getattr(_mu_mod, "cost_manager", None)
        if cm is None:
            # No cost manager available – yield a no-op tracker
            tracker = CostTracker()
            yield tracker
            return

        snapshot: Dict[str, float] = dict(getattr(cm, "total_cost", {}))
        tracker = CostTracker()
        yield tracker
        current = getattr(cm, "total_cost", {})
        total_delta = 0.0
        for model, post_total in current.items():
            delta = post_total - snapshot.get(model, 0.0)
            if delta > 0:
                tracker.cost_per_model[model] = delta
                total_delta += delta
        tracker.total_llm_cost = total_delta

    _mu_mod.CostTracker = CostTracker  # type: ignore[attr-defined]
    _mu_mod.track_cost = track_cost  # type: ignore[attr-defined]
