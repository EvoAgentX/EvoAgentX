# Acknowledgement: Copied from OpenEvolve (https://github.com/codelion/openevolve/blob/main/openevolve/utils/async_utils.py) under Apache-2.0 license

import asyncio
import functools
from typing import Any, Callable, List


def run_in_executor(f: Callable) -> Callable:
    """
    Decorator to run a synchronous function in an executor

    Args:
        f: Function to decorate

    Returns:
        Decorated function that runs in an executor
    """

    @functools.wraps(f)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, functools.partial(f, *args, **kwargs))

    return wrapper


async def gather_with_concurrency(
    n: int, *tasks: asyncio.Future, return_exceptions: bool = False
) -> List[Any]:
    """
    Run tasks with a concurrency limit

    Args:
        n: Maximum number of tasks to run concurrently
        *tasks: Tasks to run
        return_exceptions: Whether to return exceptions instead of raising them

    Returns:
        List of task results
    """
    semaphore = asyncio.Semaphore(n)

    async def sem_task(task: asyncio.Future) -> Any:
        async with semaphore:
            return await task

    return await asyncio.gather(
        *(sem_task(task) for task in tasks), return_exceptions=return_exceptions
    )


class TaskPool:
    """
    A simple task pool for managing and limiting concurrent tasks
    """

    def __init__(self, max_concurrency: int = 10):
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self.tasks: List[asyncio.Task] = []

    async def run(self, coro: Callable, *args: Any, **kwargs: Any) -> Any:
        """
        Run a coroutine in the pool

        Args:
            coro: Coroutine function to run
            *args: Arguments to pass to the coroutine
            **kwargs: Keyword arguments to pass to the coroutine

        Returns:
            Result of the coroutine
        """
        async with self.semaphore:
            return await coro(*args, **kwargs)

    def create_task(self, coro: Callable, *args: Any, **kwargs: Any) -> asyncio.Task:
        """
        Create and track a task in the pool

        Args:
            coro: Coroutine function to run
            *args: Arguments to pass to the coroutine
            **kwargs: Keyword arguments to pass to the coroutine

        Returns:
            Task object
        """
        task = asyncio.create_task(self.run(coro, *args, **kwargs))
        self.tasks.append(task)
        task.add_done_callback(lambda t: self.tasks.remove(t))
        return task

    async def wait_all(self) -> None:
        """Wait for all tasks in the pool to complete"""
        if self.tasks:
            await asyncio.gather(*self.tasks)

    async def cancel_all(self) -> None:
        """Cancel all tasks in the pool"""
        for task in self.tasks:
            task.cancel()

        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)
