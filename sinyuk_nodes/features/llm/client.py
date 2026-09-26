"""Small async HTTP client for OpenAI-compatible APIs."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from time import monotonic
from typing import TypeGuard
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


def _is_object(value: object) -> TypeGuard[dict[str, object]]:
    return isinstance(value, dict)


def _is_list(value: object) -> TypeGuard[list[object]]:
    return isinstance(value, list)


def _catalog_path() -> Path:
    try:
        from folder_paths import get_user_directory
    except ImportError:
        root = Path(os.environ.get("COMFYUI_PATH", Path.cwd())).expanduser()
        user_directory = root / "user"
    else:
        user_directory = Path(get_user_directory())
    return user_directory / "sinyuk_nodes" / "openapi_models.json"


def _read_catalog() -> dict[str, tuple[str, ...]]:
    path = _catalog_path()
    try:
        raw: object = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return {}
    if not _is_object(raw):
        return {}
    result: dict[str, tuple[str, ...]] = {}
    for base_url, models in raw.items():
        if _is_list(models):
            model_ids = [model for model in models if isinstance(model, str) and model]
            if len(model_ids) == len(models):
                result[base_url] = tuple(dict.fromkeys(model_ids))
    return result


def available_models(base_url: str) -> tuple[str, ...]:
    """Return models cached for a base URL, without making a request."""

    return _read_catalog().get(base_url, ())


def cached_model_options() -> tuple[str, ...]:
    """Return unique model IDs for the native V3 remote combo."""

    models = {model for values in _read_catalog().values() for model in values}
    return tuple(sorted(models))


def _write_catalog(base_url: str, models: tuple[str, ...]) -> None:
    path = _catalog_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    catalog = {key: list(value) for key, value in _read_catalog().items()}
    catalog[base_url] = list(models)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


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


async def fetch_models(base_url: str, api_key: str) -> tuple[str, ...]:
    """Fetch and persist model IDs from ``GET /models``."""

    url = f"{base_url}/models"
    last_error: Exception | None = None
    async with httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=15.0)) as client:
        for attempt in range(3):
            try:
                response = await _request(
                    client,
                    "GET",
                    url,
                    headers={"Authorization": f"Bearer {api_key}"},
                )
                if response.status_code >= 400:
                    error = OpenAPIRequestError(_error_message(response, api_key))
                    if response.status_code in _RETRYABLE_STATUS_CODES and attempt < 2:
                        last_error = error
                        await _sleep_with_interrupt(0.5 * (attempt + 1))
                        continue
                    raise error
                payload: object = response.json()
                if not _is_object(payload):
                    raise OpenAPIRequestError("The models response has an invalid format.")
                data_value = payload.get("data")
                if not _is_list(data_value):
                    raise OpenAPIRequestError("The models response has an invalid format.")
                data = data_value
                models_list: list[str] = []
                for item in data:
                    if _is_object(item):
                        model_id = item.get("id")
                        if isinstance(model_id, str) and model_id:
                            models_list.append(model_id)
                models = tuple(models_list)
                if not models:
                    raise OpenAPIRequestError("The models response did not contain any model IDs.")
                _write_catalog(base_url, tuple(dict.fromkeys(models)))
                return tuple(dict.fromkeys(models))
            except (httpx.TimeoutException, httpx.NetworkError, OSError) as exc:
                last_error = exc
                if attempt < 2:
                    await _sleep_with_interrupt(0.5 * (attempt + 1))
                    continue
                break
            except httpx.HTTPError as exc:
                last_error = exc
                break
            except (ValueError, OpenAPIRequestError) as exc:
                last_error = exc
                break
    raise OpenAPIRequestError(
        str(last_error)
        if isinstance(last_error, OpenAPIRequestError)
        else "Unable to load models from the OpenAI-compatible API."
    ) from last_error


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
    "available_models",
    "cached_model_options",
    "complete_chat",
    "complete_response",
    "fetch_models",
]
