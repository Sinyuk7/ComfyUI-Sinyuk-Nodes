"""Independent short-lived account balance query and lifecycle-owned refreshes."""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from typing import TypeGuard

import aiohttp

from .client import cancellable
from .errors import GrsaiError, clean_message


def _is_string_mapping(value: object) -> TypeGuard[Mapping[str, object]]:
    return isinstance(value, Mapping) and all(isinstance(key, str) for key in value)


def parse_credits(data: object, token: str) -> float | int:
    if not _is_string_mapping(data):
        raise GrsaiError("Balance response must be a JSON object.")
    code = data.get("code")
    if type(code) not in (int, float) or code != 0:
        raise GrsaiError(clean_message(data.get("msg", "Balance business error."), (token,)))
    details = data.get("data")
    credits = details.get("credits") if _is_string_mapping(details) else None
    if type(credits) not in (int, float):
        raise GrsaiError("Balance credits must be a finite number.")
    numeric_credits = float(credits) if isinstance(credits, int) else credits
    try:
        finite = math.isfinite(numeric_credits)
    except OverflowError:
        finite = False
    if not finite:
        raise GrsaiError("Balance credits must be a finite number.")
    return credits


async def query_balance(
    base_url: str, token: str, check_cancel: Callable[[], None] = lambda: None
) -> float | int:
    async def query() -> float | int:
        async with (
            aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15, connect=15)) as session,
            session.post(
                base_url + "/client/openapi/getCredits",
                json={"token": token},
                allow_redirects=False,
            ) as response,
        ):
            if not 200 <= response.status < 300:
                raise GrsaiError(f"Balance HTTP {response.status}.")
            try:
                data = await response.json(content_type=None)
            except (ValueError, UnicodeError):
                raise GrsaiError("Balance response is not JSON.") from None
            return parse_credits(data, token)

    try:
        return await cancellable(query(), check_cancel)
    except (TimeoutError, aiohttp.ClientError):
        raise GrsaiError("Balance query failed or timed out.") from None


class BalanceManager:
    """Lives exclusively on the server loop, not the per-prompt execution loop."""

    def __init__(
        self,
        emit: Callable[[dict[str, object], str], None],
        interrupted: Callable[[], bool] = lambda: False,
    ) -> None:
        self.emit = emit
        self.interrupted = interrupted
        self.tasks: dict[tuple[str, str], asyncio.Task[None]] = {}
        self.pending: set[asyncio.Task[None]] = set()
        self.closed = False

    def start(
        self,
        node_id: str,
        client_id: str,
        base_url: str,
        token: str,
        sequence: float,
        ui_token: str | None = None,
    ) -> None:
        if self.closed:
            return
        scope = (client_id, node_id)
        previous = self.tasks.get(scope)
        if previous:
            previous.cancel()
        payload = {"node_id": node_id, "sequence": sequence, "ui_token": ui_token}
        if self.interrupted():
            self.emit({**payload, "state": "stale"}, client_id)
            return
        task = asyncio.create_task(self._refresh(payload, client_id, base_url, token))
        self.tasks[scope] = task
        self.pending.add(task)

        def done(completed: asyncio.Task[None]) -> None:
            self.pending.discard(completed)
            if self.tasks.get(scope) is completed:
                self.tasks.pop(scope, None)
            if not completed.cancelled():
                completed.exception()

        task.add_done_callback(done)

    async def _refresh(
        self, payload: dict[str, object], client_id: str, base_url: str, token: str
    ) -> None:
        def check():
            if self.interrupted():
                raise asyncio.CancelledError()

        self.emit({**payload, "state": "loading"}, client_id)
        try:
            credits = await query_balance(base_url, token, check)
        except asyncio.CancelledError:
            self.emit({**payload, "state": "stale"}, client_id)
            raise
        except Exception as exc:
            message = str(exc) if isinstance(exc, GrsaiError) else "Balance query failed."
            self.emit(
                {**payload, "state": "error", "message": clean_message(message, (token,))},
                client_id,
            )
        else:
            self.emit(
                {
                    **payload,
                    "state": "ready",
                    "credits": credits,
                    "queried_at": datetime.now(UTC).isoformat(),
                },
                client_id,
            )

    async def close(self, _app: object = None) -> None:
        self.closed = True
        tasks = list(self.pending)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self.tasks.clear()
        self.pending.clear()


def query_sequence() -> float:
    # Milliseconds fit exactly in JavaScript numbers; fractional milliseconds order runs.
    return time.time_ns() / 1_000_000
