"""Load benchmark implementations only when requested."""

from importlib import import_module

__all__ = [
    "GSM8K",
    "MATH",
    "MBPP",
    "NQ",
    "AFlowGSM8K",
    "AFlowHotPotQA",
    "AFlowHumanEval",
    "AFlowMBPP",
    "HotPotQA",
    "HumanEval",
    "LiveCodeBench"
]

_EXPORTS = {
    "NQ": ".nq",
    "HotPotQA": ".hotpotqa",
    "AFlowHotPotQA": ".hotpotqa",
    "GSM8K": ".gsm8k",
    "AFlowGSM8K": ".gsm8k",
    "MBPP": ".mbpp",
    "AFlowMBPP": ".mbpp",
    "MATH": ".math_benchmark",
    "HumanEval": ".humaneval",
    "AFlowHumanEval": ".humaneval",
    "LiveCodeBench": ".livecodebench",
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
