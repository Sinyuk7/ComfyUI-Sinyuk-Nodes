"""Small async HTTP client for OpenAI-compatible APIs."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import TypeGuard

import httpx


class OpenAPIRequestError(RuntimeError):
    """An API request failed without exposing credentials."""


def _is_object(value: object) -> TypeGuard[dict[str, object]]:
    return isinstance(value, dict) and all(isinstance(key, str) for key in value)


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
        if isinstance(models, list) and all(isinstance(model, str) and model for model in models):
            result[base_url] = tuple(dict.fromkeys(models))
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


async def fetch_models(base_url: str, api_key: str) -> tuple[str, ...]:
    """Fetch and persist model IDs from ``GET /models``."""

    url = f"{base_url}/models"
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=15.0)) as client:
                response = await client.get(url, headers={"Authorization": f"Bearer {api_key}"})
            if response.status_code in {429, 500, 502, 503, 504} and attempt < 2:
                await asyncio.sleep(0.5 * (attempt + 1))
                continue
            response.raise_for_status()
            payload: object = response.json()
            if not _is_object(payload) or not isinstance(payload.get("data"), list):
                raise OpenAPIRequestError("The models response has an invalid format.")
            models = tuple(
                item["id"]
                for item in payload["data"]
                if _is_object(item) and isinstance(item.get("id"), str) and item["id"]
            )
            if not models:
                raise OpenAPIRequestError("The models response did not contain any model IDs.")
            _write_catalog(base_url, tuple(dict.fromkeys(models)))
            return tuple(dict.fromkeys(models))
        except (httpx.HTTPError, OSError, ValueError, OpenAPIRequestError) as exc:
            last_error = exc
            if attempt < 2 and isinstance(exc, (httpx.TimeoutException, httpx.NetworkError)):
                await asyncio.sleep(0.5 * (attempt + 1))
                continue
            break
    raise OpenAPIRequestError(
        "Unable to load models from the OpenAI-compatible API."
    ) from last_error


async def complete_chat(base_url: str, api_key: str, payload: dict[str, object]) -> object:
    """Send one chat completion request and return its decoded JSON payload."""

    url = f"{base_url}/chat/completions"
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(180.0, connect=15.0)) as client:
                response = await client.post(
                    url, headers={"Authorization": f"Bearer {api_key}"}, json=payload
                )
            if response.status_code in {429, 500, 502, 503, 504} and attempt < 2:
                await asyncio.sleep(0.5 * (attempt + 1))
                continue
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPError, OSError, ValueError) as exc:
            last_error = exc
            if attempt < 2 and isinstance(exc, (httpx.TimeoutException, httpx.NetworkError)):
                await asyncio.sleep(0.5 * (attempt + 1))
                continue
            break
    raise OpenAPIRequestError("The OpenAI-compatible chat request failed.") from last_error


__all__ = [
    "OpenAPIRequestError",
    "available_models",
    "cached_model_options",
    "complete_chat",
    "fetch_models",
]
