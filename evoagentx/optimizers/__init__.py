from __future__ import annotations
from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .legacy.aflow_optimizer import AFlowOptimizer
    from .legacy.map_elites_optimizer import MapElitesOptimizer
    from .legacy.mipro_optimizer import MiproOptimizer, WorkFlowMiproOptimizer
    from .legacy.sew_optimizer import SEWOptimizer
    from .legacy.textgrad_optimizer import TextGradOptimizer

__all__ = [
    "SEWOptimizer",
    "AFlowOptimizer",
    "TextGradOptimizer",
    "MiproOptimizer",
    "WorkFlowMiproOptimizer",
    "MapElitesOptimizer",
]

_EXPORTS = {
    "SEWOptimizer": ".legacy.sew_optimizer",
    "AFlowOptimizer": ".legacy.aflow_optimizer",
    "TextGradOptimizer": ".legacy.textgrad_optimizer",
    "MiproOptimizer": ".legacy.mipro_optimizer",
    "WorkFlowMiproOptimizer": ".legacy.mipro_optimizer",
    "MapElitesOptimizer": ".legacy.map_elites_optimizer",
}


def __getattr__(name: str):
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    try:
        value = getattr(import_module(module_name, __name__), name)
    except ModuleNotFoundError as exc:
        if name in {"MiproOptimizer", "WorkFlowMiproOptimizer"} and exc.name == "optuna":
            raise ModuleNotFoundError(
                "Mipro optimizers require the optional dependency 'optuna'. "
                "Install it to use evoagentx.optimizers.MiproOptimizer or "
                "evoagentx.optimizers.WorkFlowMiproOptimizer."
            ) from exc
        raise

    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
