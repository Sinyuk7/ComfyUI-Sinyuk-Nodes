"""OpenAI-compatible multimodal chat completion behavior."""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from dataclasses import dataclass
from time import perf_counter
from typing import TypeGuard

import torch
from sinyuk_nodes.compat.comfy import check_interrupt

from .client import complete_chat, complete_response
from .config import OpenAPIConfig
from .image import EncodedImage, encode_images
from .schema import JSONSchemaDocument

_RESPONSE_CACHE: OrderedDict[str, str] = OrderedDict()
_CACHE_LIMIT = 128
_REASONING_EFFORTS = {"none", "minimal", "low", "medium", "high", "xhigh", "max"}


@dataclass(frozen=True)
class ChatResult:
    """Model output and a compact node-side execution summary."""

    response: str
    execution_summary: str


def _execution_summary(
    config: OpenAPIConfig,
    response_format: str,
    json_schema: JSONSchemaDocument | None,
    image_count: int,
    image_detail: str,
    max_tokens: int | None,
    cache: str,
    elapsed_ms: int,
    response: str,
    reasoning_effort: str = "none",
) -> str:
    api = "resp" if config.api_mode == "responses" else "chat"
    fmt = {"json_schema": "JSON Schema", "json_object": "JSON Object"}.get(response_format, "Text")
    schema_name = json_schema.name if json_schema is not None else "-"
    token_limit = str(max_tokens) if max_tokens is not None else "-"
    return (
        "### Execution Summary\n\n"
        f"- **API:** `{api}`\n"
        f"- **Model:** `{config.model}`\n"
        f"- **Response format:** `{fmt}`\n"
        f"- **JSON Schema:** `{schema_name}`\n"
        f"- **Images:** `{image_count}` (`{image_detail}` detail)\n"
        f"- **Max tokens:** `{token_limit}`\n"
        f"- **Reasoning effort:** `{reasoning_effort}`\n"
        f"- **Cache:** `{cache}`\n"
        f"- **Elapsed:** `{elapsed_ms} ms`\n"
        f"- **Output length:** `{len(response)}` characters"
    )


def _content_text(value: object) -> str:
    if isinstance(value, str):
        return value
    if _is_list(value):
        parts: list[str] = []
        for item in value:
            if _is_object(item):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return ""


def _response_text(payload: object) -> str:
    if not _is_object(payload):
        raise ValueError("The chat response has an invalid format.")
    choices = payload.get("choices")
    if not _is_list(choices) or not choices or not _is_object(choices[0]):
        raise ValueError("The chat response did not contain a choice.")
    message = choices[0].get("message")
    if not _is_object(message):
        raise ValueError("The chat response did not contain a message.")
    refusal = message.get("refusal")
    if isinstance(refusal, str) and refusal.strip():
        raise ValueError(f"The model refused the request: {refusal.strip()}")
    text = _content_text(message.get("content"))
    if not text:
        raise ValueError("The chat response did not contain text content.")
    return text


def _responses_text(payload: object) -> str:
    if not _is_object(payload):
        raise ValueError("The Responses API response has an invalid format.")
    output = payload.get("output")
    if not _is_list(output):
        raise ValueError("The Responses API response did not contain output.")
    parts: list[str] = []
    for item in output:
        if not _is_object(item) or item.get("type") != "message":
            continue
        content = item.get("content")
        if not _is_list(content):
            continue
        for block in content:
            if _is_object(block):
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
    if not parts:
        raise ValueError("The Responses API response did not contain text output.")
    return "".join(parts)


def _is_object(value: object) -> TypeGuard[dict[str, object]]:
    return isinstance(value, dict)


def _is_list(value: object) -> TypeGuard[list[object]]:
    return isinstance(value, list)


def build_payload(
    config: OpenAPIConfig,
    system_prompt: str,
    prompt: str,
    images: tuple[EncodedImage, ...],
    temperature: float | None = None,
    top_p: float | None = None,
    max_tokens: int | None = None,
    response_format: str = "text",
    json_schema: JSONSchemaDocument | None = None,
    image_detail: str = "high",
    reasoning_effort: str = "none",
) -> dict[str, object]:
    if reasoning_effort not in _REASONING_EFFORTS:
        raise ValueError("Unsupported reasoning effort.")
    user_content: str | list[dict[str, object]] = prompt
    if images:
        user_content = [{"type": "text", "text": prompt}]
        for index, image in enumerate(images, start=1):
            user_content.extend(
                [
                    {"type": "text", "text": f"Image {index}:"},
                    {
                        "type": "image_url",
                        "image_url": {"url": image.base64_data_url, "detail": image_detail},
                    },
                ]
            )
    messages: list[dict[str, object]] = []
    if system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_content})
    payload: dict[str, object] = {
        "model": config.model,
        "messages": messages,
    }
    if temperature is not None:
        payload["temperature"] = temperature
    if top_p is not None:
        payload["top_p"] = top_p
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if reasoning_effort != "none":
        payload["reasoning_effort"] = reasoning_effort
    if response_format == "json_schema":
        if json_schema is None:
            raise ValueError("Connect a JSON Schema node when using JSON Schema format.")
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": json_schema.name,
                "strict": True,
                "schema": json_schema.schema,
            },
        }
    elif response_format == "json_object":
        payload["response_format"] = {"type": "json_object"}
    elif response_format != "text":
        raise ValueError("Unsupported response format.")
    return payload


def build_responses_payload(
    config: OpenAPIConfig,
    system_prompt: str,
    prompt: str,
    images: tuple[EncodedImage, ...],
    temperature: float | None = None,
    top_p: float | None = None,
    max_tokens: int | None = None,
    response_format: str = "text",
    json_schema: JSONSchemaDocument | None = None,
    image_detail: str = "high",
    reasoning_effort: str = "none",
) -> dict[str, object]:
    if reasoning_effort not in _REASONING_EFFORTS:
        raise ValueError("Unsupported reasoning effort.")
    content: str | list[dict[str, object]] = prompt
    if images:
        content = [{"type": "input_text", "text": prompt}]
        for index, image in enumerate(images, start=1):
            content.extend(
                [
                    {"type": "input_text", "text": f"Image {index}:"},
                    {
                        "type": "input_image",
                        "image_url": image.base64_data_url,
                        "detail": image_detail,
                    },
                ]
            )
    input_items: list[dict[str, object]] = []
    if system_prompt.strip():
        input_items.append({"role": "system", "content": system_prompt})
    input_items.append({"role": "user", "content": content})
    payload: dict[str, object] = {"model": config.model, "input": input_items}
    if temperature is not None:
        payload["temperature"] = temperature
    if top_p is not None:
        payload["top_p"] = top_p
    if max_tokens is not None:
        payload["max_output_tokens"] = max_tokens
    if reasoning_effort != "none":
        payload["reasoning"] = {"effort": reasoning_effort}
    if response_format == "json_schema":
        if json_schema is None:
            raise ValueError("Connect a JSON Schema node when using JSON Schema format.")
        payload["text"] = {
            "format": {
                "type": "json_schema",
                "name": json_schema.name,
                "strict": True,
                "schema": json_schema.schema,
            }
        }
    elif response_format == "json_object":
        payload["text"] = {"format": {"type": "json_object"}}
    elif response_format != "text":
        raise ValueError("Unsupported response format.")
    return payload


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
) -> ChatResult:
    started = perf_counter()
    check_interrupt()
    encoded = encode_images(images, image_detail)
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
    }
    fingerprint = hashlib.sha256(
        json.dumps(
            fingerprint_data,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    if fingerprint in _RESPONSE_CACHE:
        check_interrupt()
        _RESPONSE_CACHE.move_to_end(fingerprint)
        response = _RESPONSE_CACHE[fingerprint]
        return ChatResult(
            response,
            _execution_summary(
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
    if config.api_mode == "responses":
        result = _responses_text(await complete_response(config.base_url, config.api_key, payload))
    else:
        result = _response_text(await complete_chat(config.base_url, config.api_key, payload))
    check_interrupt()
    _RESPONSE_CACHE[fingerprint] = result
    _RESPONSE_CACHE.move_to_end(fingerprint)
    while len(_RESPONSE_CACHE) > _CACHE_LIMIT:
        _RESPONSE_CACHE.popitem(last=False)
    return ChatResult(
        result,
        _execution_summary(
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
