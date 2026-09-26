"""Execution-local cancellation and ownership of asynchronous cleanup."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import ParamSpec, TypeVar

T = TypeVar("T")
P = ParamSpec("P")


class CancellationState:
    """Latch a host checker that raises only when execution should stop."""

    def __init__(self, check_host: Callable[[], None]) -> None:
        self.check_host = check_host
        self.reason: BaseException | None = None

    def check(self) -> None:
        if self.reason is not None:
            raise self.reason
        try:
            self.check_host()
        except BaseException as exc:
            self.reason = exc
            raise

    async def wait(self, awaitable: Awaitable[T]) -> T:
        task = asyncio.ensure_future(awaitable)
        try:
            while True:
                self.check()
                done, _ = await asyncio.wait({task}, timeout=0.1)
                if done:
                    self.check()
                    return task.result()
        except BaseException as exc:
            if isinstance(exc, asyncio.CancelledError) and self.reason is None:
                self.reason = exc
            if not task.done():
                task.cancel()
            await drain(task)
            raise


async def drain(task: asyncio.Future[object]) -> None:
    """Finish owned cleanup despite repeated cancellation; preserve the caller's error."""
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            continue
        except BaseException:
            break
    if not task.cancelled():
        task.exception()


async def run_blocking(function: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
    """Join started thread work before returning or propagating cancellation."""
    return await finish_owned(asyncio.to_thread(function, *args, **kwargs))


async def finish_owned(awaitable: Awaitable[T]) -> T:
    """Do not abandon a commit or resource cleanup once it has begun."""
    task = asyncio.ensure_future(awaitable)
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        await drain(task)
        raise
