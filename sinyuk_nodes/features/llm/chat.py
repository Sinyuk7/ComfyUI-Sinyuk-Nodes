"""OpenAI-compatible multimodal chat completion behavior."""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from dataclasses import dataclass
from time import perf_counter

import torch
from sinyuk_nodes.common.cancellation import CancellationState, run_blocking
from sinyuk_nodes.compat.comfy import check_interrupt

from .client import complete_chat, complete_response
from .config import OpenAPIConfig
from .image import encode_images
from .request import (
    build_payload,
    build_responses_payload,
    provider_request_headers,
)
from .response import response_text, responses_result
from .schema import JSONSchemaDocument
from .summary import execution_summary

# Backward-compatible private imports used by the focused feature tests.
_response_text = response_text


def _responses_text(payload: object) -> str:
    return responses_result(payload)[0]


_RESPONSE_CACHE: OrderedDict[str, str] = OrderedDict()
_CACHE_LIMIT = 128
_RESPONSE_SESSIONS: dict[tuple[str, str, str, str], str] = {}


@dataclass(frozen=True)
class ChatResult:
    """Model output and a compact node-side execution summary."""

    response: str
    execution_summary: str


def _session_key(config: OpenAPIConfig, unique_id: str | int) -> tuple[str, str, str, str]:
    return (str(unique_id), config.base_url, config.model, config.api_mode)


async def execute_chat(
    config: OpenAPIConfig,
    system_prompt: str,
    prompt: str,
    images: torch.Tensor | None,
    seed: int,
    temperature: float | None = None,
    top_p: float | None = None,
    max_tokens: int | None = None,
    response_format: str = "text",
    json_schema: JSONSchemaDocument | None = None,
    image_detail: str = "high",
    reasoning_effort: str = "none",
    unique_id: str | int = "",
) -> ChatResult:
    cancellation = CancellationState(check_interrupt)
    started = perf_counter()
    cancellation.check()
    session_key = _session_key(config, unique_id)
    previous_response_id = None
    if config.session_cache and config.api_mode == "responses":
        previous_response_id = _RESPONSE_SESSIONS.get(session_key)
    else:
        _RESPONSE_SESSIONS.pop(session_key, None)
    encoded = await cancellation.wait(run_blocking(encode_images, images, image_detail))
    payload_builder = build_responses_payload if config.api_mode == "responses" else build_payload
    payload = payload_builder(
        config,
        system_prompt,
        prompt,
        encoded,
        temperature,
        top_p,
        max_tokens,
        response_format,
        json_schema,
        image_detail,
        reasoning_effort,
        previous_response_id,
    )
    fingerprint_data: dict[str, object] = {
        "base_url": config.base_url,
        "model": config.model,
        "system_prompt": system_prompt,
        "prompt": prompt,
        "image_sha256": [image.sha256 for image in encoded],
        "image_detail": image_detail,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "reasoning_effort": reasoning_effort,
        "response_format": response_format,
        "json_schema": None
        if json_schema is None
        else {"name": json_schema.name, "schema": json_schema.schema},
        # ComfyUI request/cache seed; never sent to the remote API.
        "seed": seed,
        "session_cache": config.session_cache,
        "previous_response_id": previous_response_id,
    }
    fingerprint = hashlib.sha256(
        json.dumps(
            fingerprint_data,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    if fingerprint in _RESPONSE_CACHE:
        cancellation.check()
        _RESPONSE_CACHE.move_to_end(fingerprint)
        response = _RESPONSE_CACHE[fingerprint]
        return ChatResult(
            response,
            execution_summary(
                config,
                response_format,
                json_schema,
                len(encoded),
                image_detail,
                max_tokens,
                "hit",
                round((perf_counter() - started) * 1000),
                response,
                reasoning_effort,
            ),
        )
    response_id: str | None = None
    if config.api_mode == "responses":
        extra_headers = provider_request_headers(config)
        raw_response = await complete_response(
            config.base_url,
            config.api_key,
            payload,
            cancellation,
            extra_headers or None,
        )
        result, response_id = responses_result(raw_response)
    else:
        result = response_text(
            await complete_chat(config.base_url, config.api_key, payload, cancellation)
        )
    cancellation.check()
    if config.session_cache and config.api_mode == "responses" and response_id:
        _RESPONSE_SESSIONS[session_key] = response_id
    _RESPONSE_CACHE[fingerprint] = result
    _RESPONSE_CACHE.move_to_end(fingerprint)
    while len(_RESPONSE_CACHE) > _CACHE_LIMIT:
        _RESPONSE_CACHE.popitem(last=False)
    return ChatResult(
        result,
        execution_summary(
            config,
            response_format,
            json_schema,
            len(encoded),
            image_detail,
            max_tokens,
            "miss",
            round((perf_counter() - started) * 1000),
            result,
            reasoning_effort,
        ),
    )


__all__ = ["ChatResult", "build_payload", "build_responses_payload", "execute_chat"]
