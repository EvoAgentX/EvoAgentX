from importlib import import_module

from .memory import BaseMemory, ShortTermMemory

__all__ = ["BaseMemory", "LongTermMemory", "MemoryManager", "ShortTermMemory"]

_EXPORTS = {
    "LongTermMemory": ".long_term_memory",
    "MemoryManager": ".memory_manager",
}


def __getattr__(name: str):
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
