"""Small async HTTP client for OpenAI-compatible APIs."""

from __future__ import annotations

import asyncio
import logging
from time import monotonic
from uuid import uuid4

import httpx
from sinyuk_nodes.common.cancellation import CancellationState
from sinyuk_nodes.compat.comfy import check_interrupt


class OpenAPIRequestError(RuntimeError):
    """An API request failed without exposing credentials."""

    def __init__(self, message: str, *, submission_unknown: bool = False) -> None:
        super().__init__(message)
        self.submission_unknown = submission_unknown


_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
_ERROR_BODY_LIMIT = 500
_INTERRUPT_POLL_INTERVAL = 0.1


def _error_message(response: httpx.Response, api_key: str) -> str:
    """Return a bounded server error without request credentials."""

    body = response.text.strip().replace("\n", " ")
    if api_key:
        body = body.replace(api_key, "[REDACTED]")
    if len(body) > _ERROR_BODY_LIMIT:
        body = body[:_ERROR_BODY_LIMIT].rstrip() + "..."
    return f"HTTP {response.status_code}: {body or response.reason_phrase}"


async def _sleep_with_interrupt(delay: float) -> None:
    """Wait between retries while keeping prompt cancellation responsive."""

    remaining = delay
    while remaining > 0:
        check_interrupt()
        interval = min(_INTERRUPT_POLL_INTERVAL, remaining)
        await asyncio.sleep(interval)
        remaining -= interval


async def _request(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    *,
    headers: dict[str, str],
    payload: dict[str, object] | None = None,
    cancellation: CancellationState | None = None,
) -> httpx.Response:
    """Run one HTTP request and cancel its task when ComfyUI interrupts."""

    state = cancellation or CancellationState(check_interrupt)
    state.check()
    return await state.wait(client.request(method, url, headers=headers, json=payload))


async def _complete(
    base_url: str,
    api_key: str,
    payload: dict[str, object],
    path: str,
    cancellation: CancellationState | None = None,
) -> object:
    state = cancellation or CancellationState(check_interrupt)
    run_id = uuid4().hex[:12]
    started = monotonic()
    submission = "not_started"
    outcome = "failed"
    status: int | None = None
    logger = logging.getLogger(__name__)

    async def submit() -> object:
        nonlocal submission, outcome, status
        async with httpx.AsyncClient(timeout=httpx.Timeout(180.0, connect=15.0)) as client:
            state.check()
            submission = "unknown"
            logger.info("llm.submit run_id=%s phase=submitting attempt=1", run_id)
            response = await _request(
                client,
                "POST",
                f"{base_url}/{path}",
                headers={"Authorization": f"Bearer {api_key}"},
                payload=payload,
                cancellation=state,
            )
            status = response.status_code
            if status >= 400:
                if status in {400, 401, 403, 404, 422, 429}:
                    submission = "rejected"
                raise OpenAPIRequestError(
                    _error_message(response, api_key) + " Generation POST was not retried.",
                    submission_unknown=submission == "unknown",
                )
            result: object = response.json()
            submission = "accepted"
            outcome = "completed"
            return result

    try:
        return await state.wait(submit())
    except (httpx.HTTPError, OSError, ValueError) as exc:
        raise OpenAPIRequestError(
            "Generation response lost or invalid; submission_unknown=true. "
            "A remote request may already exist. POST was not retried.",
            submission_unknown=True,
        ) from exc
    except BaseException:
        if state.reason is not None:
            outcome = "interrupted"
        raise
    finally:
        logger.info(
            "llm.finished run_id=%s phase=submitting attempt=1 http_status=%s "
            "submission_state=%s outcome=%s elapsed_ms=%s",
            run_id,
            status,
            submission,
            outcome,
            round((monotonic() - started) * 1000),
        )


async def complete_chat(
    base_url: str,
    api_key: str,
    payload: dict[str, object],
    cancellation: CancellationState | None = None,
) -> object:
    """Submit once; a lost response must not cause duplicate generation."""
    return await _complete(base_url, api_key, payload, "chat/completions", cancellation)


async def complete_response(
    base_url: str,
    api_key: str,
    payload: dict[str, object],
    cancellation: CancellationState | None = None,
) -> object:
    """Submit a synchronous response once, without speculative idempotency."""
    return await _complete(base_url, api_key, payload, "responses", cancellation)


__all__ = [
    "OpenAPIRequestError",
    "complete_chat",
    "complete_response",
]
