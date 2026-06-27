import asyncio
import inspect
from collections.abc import Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Type, TypeVar


T = TypeVar("T")


def run_coroutine_sync(coro: Awaitable[T]) -> T:
    """
    Run an awaitable from synchronous code, including when the caller is already
    inside a running event loop.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


def is_method_overridden(instance: Any, base_cls: Type[Any], method_name: str) -> bool:
    """
    Return whether ``method_name`` is implemented on ``instance``'s class rather
    than inherited unchanged from ``base_cls``.
    """
    base_method = _normalize_method(getattr(base_cls, method_name, None))
    instance_dict = getattr(instance, "__dict__", {})

    if method_name in instance_dict:
        instance_method = _normalize_method(inspect.unwrap(instance_dict[method_name]))
        return instance_method is not None and instance_method is not base_method

    instance_method = _normalize_method(getattr(type(instance), method_name, None))
    return instance_method is not None and instance_method is not base_method


def _normalize_method(method: Any) -> Any:
    return getattr(method, "__func__", method)


async def call_maybe_async(func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """
    Await async callables directly and run sync callables in a worker thread.
    """
    if inspect.iscoroutinefunction(func):
        return await func(*args, **kwargs)

    result = await asyncio.to_thread(func, *args, **kwargs)
    if inspect.isawaitable(result):
        return await result
    return result
