"""Best-effort model availability checks used only for UI hints."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TypeGuard

import aiohttp

from .errors import GrsaiError, clean_message


def _is_string_mapping(value: object) -> TypeGuard[Mapping[str, object]]:
    return isinstance(value, Mapping) and all(isinstance(key, str) for key in value)


def parse_model_status(data: object) -> tuple[bool, str]:
    if not _is_string_mapping(data):
        raise GrsaiError("Model status response must be a JSON object.")
    if type(data.get("code")) not in (int, float) or data["code"] != 0:
        raise GrsaiError(clean_message(data.get("msg", "Model status business error."), ()))
    result = data.get("data")
    if not _is_string_mapping(result) or type(result.get("status")) is not bool:
        raise GrsaiError("Model status response is invalid.")
    error = result.get("error", "")
    if not isinstance(error, str):
        raise GrsaiError("Model status error must be a string.")
    return result["status"], clean_message(error, ())


async def query_model_status(base_url: str, model: str) -> tuple[bool, str]:
    try:
        timeout = aiohttp.ClientTimeout(total=10, connect=10)
        async with (
            aiohttp.ClientSession(timeout=timeout) as session,
            session.get(
                base_url + "/client/common/getModelStatus",
                params={"model": model},
                allow_redirects=False,
            ) as response,
        ):
            if not 200 <= response.status < 300:
                raise GrsaiError(f"Model status HTTP {response.status}.")
            try:
                data = await response.json(content_type=None)
            except (ValueError, UnicodeError):
                raise GrsaiError("Model status response is not JSON.") from None
            return parse_model_status(data)
    except (TimeoutError, aiohttp.ClientError):
        raise GrsaiError("Model status query failed or timed out.") from None
