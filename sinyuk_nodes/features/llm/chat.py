"""OpenAI-compatible multimodal chat completion behavior."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from time import perf_counter

import torch
from sinyuk_nodes.common.cancellation import CancellationState, drain, run_blocking
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
ProgressReporter = Callable[[str, float], Awaitable[None]]
_PROGRESS_INTERVAL = 0.5
_VIRTUAL_PROGRESS_RATE = 2.0
_LOGGER = logging.getLogger(__name__)


async def _report(
    reporter: ProgressReporter | None,
    stage: str,
    value: float,
) -> None:
    if reporter is not None:
        await reporter(stage, value)


async def _run_with_virtual_progress(
    operation: Awaitable[object],
    cancellation: CancellationState,
    reporter: ProgressReporter | None,
) -> object:
    """Run a request while showing bounded progress during provider wait time."""

    task = asyncio.ensure_future(operation)
    started = perf_counter()
    try:
        while not task.done():
            cancellation.check()
            elapsed = perf_counter() - started
            value = min(85.0, 30.0 + elapsed * _VIRTUAL_PROGRESS_RATE)
            await _report(reporter, "Generating response", value)
            await asyncio.sleep(_PROGRESS_INTERVAL)
        cancellation.check()
        return task.result()
    except BaseException:
        if not task.done():
            task.cancel()
        await drain(task)
        raise


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
    report_progress: ProgressReporter | None = None,
) -> ChatResult:
    cancellation = CancellationState(check_interrupt)
    started = perf_counter()
    cancellation.check()
    await _report(report_progress, "Preparing", 5)
    session_key = _session_key(config, unique_id)
    previous_response_id = None
    if config.session_cache and config.api_mode == "responses":
        previous_response_id = _RESPONSE_SESSIONS.get(session_key)
    else:
        _RESPONSE_SESSIONS.pop(session_key, None)
    await _report(report_progress, "Processing images", 15)
    encoded = await cancellation.wait(run_blocking(encode_images, images, image_detail))
    for index, image in enumerate(encoded, start=1):
        _LOGGER.info(
            "llm.image model=%s detail=%s index=%s width=%s height=%s mime=%s "
            "encoded_bytes=%s data_uri_bytes=%s",
            config.model,
            image_detail,
            index,
            image.width,
            image.height,
            image.mime_type,
            len(image.encoded_bytes),
            len(image.base64_data_url.encode("ascii")),
        )
    await _report(report_progress, "Building request", 25)
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
        await _report(report_progress, "Loaded from cache", 100)
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
        raw_response = await _run_with_virtual_progress(
            complete_response(
                config.base_url,
                config.api_key,
                payload,
                cancellation,
                extra_headers or None,
            ),
            cancellation,
            report_progress,
        )
        result, response_id = responses_result(raw_response)
    else:
        raw_response = await _run_with_virtual_progress(
            complete_chat(config.base_url, config.api_key, payload, cancellation),
            cancellation,
            report_progress,
        )
        result = response_text(raw_response)
    cancellation.check()
    await _report(report_progress, "Processing result", 92)
    if config.session_cache and config.api_mode == "responses" and response_id:
        _RESPONSE_SESSIONS[session_key] = response_id
    _RESPONSE_CACHE[fingerprint] = result
    _RESPONSE_CACHE.move_to_end(fingerprint)
    while len(_RESPONSE_CACHE) > _CACHE_LIMIT:
        _RESPONSE_CACHE.popitem(last=False)
    await _report(report_progress, "Done", 100)
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
