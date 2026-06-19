from typing import Any, Callable, List, Tuple, Optional
from .registry import ParamRegistry

# --------- EntryPoint decorator ---------
class EntryPoint:
    _entry_func: Optional[Callable] = None
    def __call__(self, func: Callable):
        EntryPoint._entry_func = func
        return func
    @classmethod
    def get_entry(cls) -> Optional[Callable]:
        return cls._entry_func

class OptimizeParam:
    """
    Class-based decorator for registering tunable optimization parameters.

    Supports:
    - Decorating functions with parameters and optional execution callbacks.
    - Functions without parameters can be registered for execution callbacks only.
    - Automatic deduplication and selective parameter registration.
    """

    # Internal storage: (function, list of parameter names, optional execution callback)
    _targets: List[Tuple[Callable, List[str], Optional[Callable]]] = []

    def __init__(self, *params: str, on_execute: Optional[Callable] = None):
        """
        :param params: parameter paths to register (optional)
        :param on_execute: optional callback triggered when the decorated function executes,
                           signature: callback(func: Callable, *args, **kwargs)
        """
        self.param_names = list(params)
        self.on_execute = on_execute

    def __call__(self, func: Callable):
        # Remove previously registered entries for the same function to ensure override
        self._targets = [t for t in self._targets if t[0] != func]

        def wrapped_func(*args, **kwargs):
            # Trigger execution callback if set
            if self.on_execute:
                self.on_execute(func, *args, **kwargs)
            return func(*args, **kwargs)

        # Store the wrapped function along with its parameters and callback
        self._targets.append((wrapped_func, self.param_names, self.on_execute))
        return wrapped_func

    @classmethod
    def register_all(cls, program_instance: Any, registry: ParamRegistry, verbose: bool = False):
        """
        Register all decorated functions' parameters on the given program instance.
        Functions without parameter paths are skipped for parameter registration.
        """
        seen = set(registry.names())  # Parameters already manually registered
        for _, param_names, _ in cls._targets:
            if not param_names:
                # Skip registration for functions with no parameter paths
                continue
            for name in param_names:
                if name in seen:
                    if verbose:
                        print(f"[OptParam] Skipped already registered: {name}")
                else:
                    seen.add(name)
                    registry.track(program_instance, name)
                    if verbose:
                        print(f"[OptParam] Registered from decorator: {name}")

    @classmethod
    def get_all(cls) -> List[Tuple[Callable, List[str], Optional[Callable]]]:
        """Return all decorated functions along with their parameters and callbacks."""
        return cls._targets

    @classmethod
    def get_decorated_functions(cls) -> List[Callable]:
        """Return all wrapped decorated functions."""
        return [t[0] for t in cls._targets]

    @classmethod
    def get_params_for_func(cls, func: Callable) -> List[str]:
        """Return the list of parameter paths registered for a specific function."""
        for f, params, _ in cls._targets:
            if f == func:
                return params
        return []


# 用函数封装装饰器实例
def optimize_param(*args, **kwargs):
    return OptimizeParam(*args, **kwargs)

__all__ = ["optimize_param"]
