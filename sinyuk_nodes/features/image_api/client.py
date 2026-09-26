"""Submit-once asynchronous generation, interruptible polling and downloads."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import re
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import AsyncExitStack, asynccontextmanager
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from time import monotonic
from typing import TypeGuard, TypeVar
from urllib.parse import urlsplit

import aiohttp
import torch
from sinyuk_nodes.common.cancellation import CancellationState, run_blocking

from .config import Config
from .diagnostics import log_event, new_run_id
from .errors import GrsaiError, clean_message
from .images import decode_image

logger = logging.getLogger(__name__)

RETRYABLE_HTTP_STATUSES = {429, 502, 503, 504}
T = TypeVar("T")
Progress = Callable[[str, int | float | None, str | None, dict[str, object]], Awaitable[None]]
Accepted = Callable[[str, str | None], Awaitable[None]]
Result = Callable[[torch.Tensor, int, int], Awaitable[None]]
Pressure = Callable[[float], Awaitable[None]]


def _is_string_mapping(value: object) -> TypeGuard[dict[str, object]]:
    # Callers pass objects decoded from JSON, whose object keys are always strings.
    return isinstance(value, dict)


def _is_object_list(value: object) -> TypeGuard[list[object]]:
    return isinstance(value, list)


def _response_object(value: object) -> Mapping[str, object]:
    if not _is_string_mapping(value):
        raise GrsaiError("Expected a JSON object from generation service.")
    return value


async def cancellable(
    awaitable: Awaitable[T], check_cancel: Callable[[], None] = lambda: None
) -> T:
    """Check the host signal while I/O is pending, then await cancellation cleanup."""
    return await CancellationState(check_cancel).wait(awaitable)


def retry_after(value: str | None) -> float | None:
    if not value:
        return None
    try:
        seconds = float(value)
    except ValueError:
        try:
            date = parsedate_to_datetime(value)
            if date.tzinfo is None:
                date = date.replace(tzinfo=UTC)
            seconds = (date - datetime.now(UTC)).total_seconds()
        except (ValueError, TypeError, OverflowError):
            return None
    return max(0.0, seconds) if math.isfinite(seconds) else None


class GrsaiClient:
    provider = "grsai"

    def __init__(
        self,
        config: Config,
        api_key: str,
        check_cancel: Callable[[], None] = lambda: None,
        progress: Progress | None = None,
        *,
        accepted: Accepted | None = None,
        result: Result | None = None,
        pressure: Pressure | None = None,
        sessions: tuple[aiohttp.ClientSession, aiohttp.ClientSession] | None = None,
        log_context: Mapping[str, str] | None = None,
    ) -> None:
        self.config = config
        self.api_key = api_key
        self.cancellation = CancellationState(check_cancel)
        self.check_cancel = self.cancellation.check
        self.phase = "preparing"
        self.exit_reason: str | None = None
        self.submission_state = "not_started"
        self.remote_cancel_result = "not_verified"
        self.attempt = 0
        self.progress = progress
        self.task_id = None
        self.submitted = False
        self.submission_unknown = False
        self.accepted = accepted
        self.result = result
        self.pressure = pressure
        self.sessions = sessions
        self.http_status = None
        self.remote_status = None
        self.generation_progress = None
        self._secrets: tuple[str, ...] = (api_key,)
        self.log_context = dict(log_context or {"run_id": new_run_id()})
        self.log_context.setdefault("provider", self.provider)

    def _error(self, message: object, index: int | None = None) -> GrsaiError:
        return GrsaiError(clean_message(message, self._secrets), self.task_id, index)

    async def _notify(
        self,
        stage: str,
        progress: int | float | None = None,
        details: dict[str, object] | None = None,
    ) -> None:
        if stage != "succeeded":
            self._set_phase({"running": "polling", "reconnecting": "polling"}.get(stage, stage))
        if self.progress:
            await self.progress(stage, progress, self.task_id, details or {})

    def _set_phase(self, phase: str) -> None:
        if phase != self.phase:
            self.phase = phase
            self.attempt = 0
            log_event("task.phase", **self.log_context, phase=phase, task_id=self.task_id)

    async def _wait(self, seconds: float) -> None:
        await asyncio.sleep(seconds)

    def _timeout(self, total: float) -> aiohttp.ClientTimeout:
        return aiohttp.ClientTimeout(
            total=total, connect=self.config.transport.connect_timeout_seconds
        )

    async def _json(
        self,
        session: aiohttp.ClientSession,
        method: str,
        path: str,
        timeout: float,
        *,
        json_body: Mapping[str, object] | None = None,
        params: Mapping[str, str] | None = None,
        data: aiohttp.FormData | None = None,
    ) -> tuple[int, object, float | None]:
        if json_body is not None and data is not None:
            raise ValueError("JSON and multipart bodies are mutually exclusive.")
        self.attempt += 1
        started = monotonic()
        if self.phase != "polling" or self.attempt == 1:
            log_event(
                "http.started",
                **self.log_context,
                phase=self.phase,
                attempt=self.attempt,
                task_id=self.task_id,
            )
        async with session.request(
            method,
            self.config.base_url + path,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=self._timeout(timeout),
            allow_redirects=False,
            json=json_body,
            data=data,
            params=params,
        ) as response:
            raw = await response.read()
            log_event(
                "http.response",
                level=logging.DEBUG
                if self.phase == "polling" and response.status < 400
                else logging.INFO,
                **self.log_context,
                phase=self.phase,
                attempt=self.attempt,
                task_id=self.task_id,
                http_status=response.status,
                elapsed_ms=round((monotonic() - started) * 1000),
            )
            try:
                decoded: object = json.loads(raw)
            except (ValueError, UnicodeError):
                decoded = None
            return response.status, decoded, retry_after(response.headers.get("Retry-After"))

    def _remember_id(self, data: object) -> None:
        if not _is_string_mapping(data) or "id" not in data:
            return
        task_id = data["id"]
        if (
            not isinstance(task_id, str)
            or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,200}", task_id)
            or self.api_key in task_id
        ):
            raise self._error("Invalid task ID in upstream response.")
        if self.task_id and self.task_id != task_id:
            raise self._error("Upstream returned a different task ID.")
        if not self.task_id:
            self.task_id = task_id
            logger.info("GRSAI task accepted task_id=%s", task_id)
            log_event(
                "task.accepted",
                run_id=self.log_context.get("run_id"),
                node_id=self.log_context.get("node_id"),
                task_id=task_id,
            )

    def _validate(self, status: int, data: object) -> str:
        self._remember_id(data)
        if not 200 <= status < 300:
            detail = (
                data.get("error", "Request rejected.")
                if _is_string_mapping(data)
                else "Non-JSON response."
            )
            raise self._error(f"HTTP {status}: {clean_message(detail, self._secrets)}")
        if not _is_string_mapping(data):
            raise self._error("Expected a JSON object from generation service.")
        state = data.get("status")
        if state in ("failed", "violation"):
            raise self._error(
                f"{state}: {clean_message(data.get('error', 'No reason provided.'), self._secrets)}"
            )
        if state not in ("running", "succeeded"):
            raise self._error("Unknown or missing generation status.")
        if state == "running" and not self.task_id:
            raise self._error("Running response has no task ID; do not automatically resubmit.")
        return state

    async def _receive(self, status: int, data: object, delay: float | None) -> None:
        self.http_status = status
        previous = self.task_id
        self._remember_id(data)
        if self.task_id:
            self.submission_state = "accepted"
            self.submission_unknown = False
        elif self.phase == "submitting" and status in {400, 401, 403, 404, 422, 429}:
            self.submission_state = "rejected"
            self.submission_unknown = False
        state = data.get("status") if _is_string_mapping(data) else None
        if isinstance(state, str) and state.lower() in {
            "create",
            "queued",
            "running",
            "succeeded",
            "success",
            "failed",
            "cancel",
            "violation",
        }:
            self.remote_status = state
        if self.remote_status in {"succeeded", "SUCCESS"}:
            self.submission_state = "accepted"
            self.submission_unknown = False
        # This is an awaited business boundary, not a best-effort UI notification.
        if self.task_id and previous is None and self.accepted:
            await self.accepted(self.task_id, self.remote_status)
        if (status == 429 or 500 <= status <= 599) and delay is not None and self.pressure:
            await self.pressure(delay)

    async def generate(self, request: Mapping[str, object]) -> list[torch.Tensor]:
        prompt = request.get("prompt")
        image_secrets = request.get("images")
        self._secrets = (
            self.api_key,
            *([prompt] if isinstance(prompt, str) else []),
            *(
                [item for item in image_secrets if isinstance(item, str)]
                if _is_object_list(image_secrets)
                else []
            ),
        )
        started = monotonic()
        try:
            operation = self._generate(request)
            limit = self.config.transport.task_timeout_seconds
            if limit is not None:
                operation = asyncio.wait_for(operation, timeout=limit)
            images = await self.cancellation.wait(operation)
            self.exit_reason = "completed"
            self.submission_state = "accepted"
            self.submission_unknown = False
            return images
        except TimeoutError:
            self.exit_reason = "timeout"
            raise self._error(
                "Local total task deadline reached; remote task may still be running."
            ) from None
        except BaseException:
            self.exit_reason = "interrupted" if self.cancellation.reason else "failed"
            raise
        finally:
            log_event(
                "task.finished",
                **self.log_context,
                phase=self.phase,
                exit_reason=self.exit_reason,
                submission_state=self.submission_state,
                task_id=self.task_id,
                remote_state=self.remote_status,
                remote_cancel_result=self.remote_cancel_result,
                elapsed_ms=round((monotonic() - started) * 1000),
            )

    @asynccontextmanager
    async def _sessions(self) -> AsyncIterator[tuple[aiohttp.ClientSession, aiohttp.ClientSession]]:
        # Authentication is attached to API requests, never CDN sessions.
        async with AsyncExitStack() as stack:
            if self.sessions is not None:
                yield self.sessions
            else:
                api = await stack.enter_async_context(aiohttp.ClientSession())
                cdn = await stack.enter_async_context(aiohttp.ClientSession())
                yield api, cdn

    async def _generate(self, request: Mapping[str, object]) -> list[torch.Tensor]:
        t = self.config.transport
        async with self._sessions() as (api, cdn):
            await self._notify("submitting")
            self.check_cancel()
            self.submitted = True
            self.submission_state = "unknown"
            self.submission_unknown = True
            try:
                status, data, delay = await self._json(
                    api, "POST", "/v1/api/generate", t.submit_timeout_seconds, json_body=request
                )
            except (TimeoutError, aiohttp.ClientError):
                self.submission_unknown = True
                raise self._error(
                    "Submission response lost; a remote task may already exist. "
                    "POST was not retried."
                ) from None
            await self._receive(status, data, delay)
            try:
                state = self._validate(status, data)
            except GrsaiError as exc:
                if self.submission_unknown:
                    raise self._error(
                        f"{exc} Remote task creation is uncertain; POST was not retried."
                    ) from None
                raise
            data = _response_object(data)
            backoff = min(t.poll_interval_seconds, t.retry_backoff_max_seconds)
            while state == "running":
                raw_progress = data.get("progress")
                valid_progress = (
                    isinstance(raw_progress, (int, float))
                    and not isinstance(raw_progress, bool)
                    and math.isfinite(raw_progress)
                    and 0 <= raw_progress <= 100
                )
                if not valid_progress:
                    progress = self.generation_progress
                else:
                    assert isinstance(raw_progress, (int, float))
                    progress = max(raw_progress, self.generation_progress or 0)
                    self.generation_progress = progress
                await self._notify("running", progress)
                await self._wait(t.poll_interval_seconds)
                retry_count = 0
                while True:
                    try:
                        status, next_data, delay = await self._json(
                            api,
                            "GET",
                            "/v1/api/result",
                            t.poll_request_timeout_seconds,
                            params={"id": self.task_id or ""},
                        )
                    except (TimeoutError, aiohttp.ClientError):
                        status, next_data, delay = 503, None, None
                    await self._receive(status, next_data, delay)
                    if status in RETRYABLE_HTTP_STATUSES:
                        # Explicit terminal task states take precedence over transient HTTP status.
                        if _is_string_mapping(next_data) and next_data.get("status") in (
                            "failed",
                            "violation",
                        ):
                            self._validate(status, next_data)
                        retry_count += 1
                        if retry_count > t.poll_retry_limit:
                            raise self._error("Task status query failed after bounded retries.")
                        if retry_count == 1:
                            log_event(
                                "poll.reconnecting",
                                level=logging.WARNING,
                                run_id=self.log_context.get("run_id"),
                                node_id=self.log_context.get("node_id"),
                                task_id=self.task_id,
                                http_status=status,
                            )
                        await self._notify("reconnecting")
                        await self._wait(max(backoff, delay or 0))
                        backoff = min(backoff * 2, t.retry_backoff_max_seconds)
                        continue
                    if retry_count:
                        log_event(
                            "poll.recovered",
                            run_id=self.log_context.get("run_id"),
                            node_id=self.log_context.get("node_id"),
                            task_id=self.task_id,
                            attempts=retry_count,
                        )
                    data = next_data
                    state = self._validate(status, data)
                    data = _response_object(data)
                    backoff = min(t.poll_interval_seconds, t.retry_backoff_max_seconds)
                    break
            results = data.get("results")
            if not _is_object_list(results) or not results:
                raise self._error("Succeeded response contains no images.")
            result_items = results
            urls: list[str] = []
            for index, result in enumerate(result_items, 1):
                url = result.get("url") if _is_string_mapping(result) else None
                if not isinstance(url, str):
                    raise self._error("Missing result URL.", index)
                try:
                    parsed = urlsplit(url)
                    valid = (
                        parsed.scheme in ("http", "https")
                        and parsed.hostname
                        and not parsed.username
                        and not parsed.password
                    )
                except ValueError:
                    valid = False
                if not valid or self.api_key in url:
                    raise self._error("Invalid result URL.", index)
                urls.append(url)
            images: list[torch.Tensor] = []
            await self._notify("downloading", 0, {"completed": 0, "total": len(urls)})
            for index, url in enumerate(urls, 1):
                image = await self._download(cdn, url, index)
                if self.result:
                    self._set_phase("saving")
                    await self.result(image, index, len(urls))
                images.append(image)
                await self._notify(
                    "downloading",
                    index * 100 / len(urls),
                    {"completed": index, "total": len(urls)},
                )
            await self._notify("succeeded", 100)
            return images
        raise self._error("Image download retry loop ended unexpectedly.")

    async def _download(self, session: aiohttp.ClientSession, url: str, index: int) -> torch.Tensor:
        self._set_phase("downloading")
        t = self.config.transport
        for attempt in range(t.download_retry_limit + 1):
            delay = None
            started = monotonic()
            try:
                async with session.get(
                    url, timeout=self._timeout(t.download_timeout_seconds)
                ) as response:
                    delay = retry_after(response.headers.get("Retry-After"))
                    log_event(
                        "download.response",
                        **self.log_context,
                        phase=self.phase,
                        attempt=attempt + 1,
                        task_id=self.task_id,
                        result_index=index,
                        http_status=response.status,
                        elapsed_ms=round((monotonic() - started) * 1000),
                    )
                    if not 200 <= response.status < 300:
                        if response.status not in RETRYABLE_HTTP_STATUSES:
                            raise self._error(f"Image download HTTP {response.status}.", index)
                        if attempt == t.download_retry_limit:
                            raise self._error(
                                (
                                    "Image download failed after bounded retries; "
                                    "generation was not retried."
                                ),
                                index,
                            )
                    else:
                        data = await response.read()
                        image = await run_blocking(decode_image, data)
                        self.check_cancel()
                        return image
            except ValueError as exc:
                raise self._error(str(exc), index) from None
            except (TimeoutError, aiohttp.ClientError):
                if attempt == t.download_retry_limit:
                    raise self._error(
                        (
                            "Image download failed after bounded retries; generation was not "
                            "repeated."
                        ),
                        index,
                    ) from None
            await self._wait(max(min(2**attempt, t.retry_backoff_max_seconds), delay or 0))
        raise self._error("Image download retry loop ended unexpectedly.", index)
