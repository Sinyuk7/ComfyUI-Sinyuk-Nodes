"""OpenAI-compatible multimodal chat completion behavior."""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from dataclasses import dataclass
from time import perf_counter

import torch

from .client import complete_chat, complete_response
from .config import OpenAPIConfig
from .image import EncodedImage, encode_images
from .schema import JSONSchemaDocument

_RESPONSE_CACHE: OrderedDict[str, str] = OrderedDict()
_CACHE_LIMIT = 128


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
) -> str:
    api = "resp" if config.api_mode == "responses" else "chat"
    fmt = "js" if response_format == "json_schema" else "text"
    schema_name = json_schema.name if json_schema is not None else "-"
    token_limit = str(max_tokens) if max_tokens is not None else "-"
    return (
        f"api={api} mdl={config.model} fmt={fmt} sch={schema_name} "
        f"img={image_count} det={image_detail} max={token_limit} "
        f"cache={cache} ms={elapsed_ms} out={len(response)}"
    )


def _content_text(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "".join(parts)
    return ""


def _response_text(payload: object) -> str:
    if not isinstance(payload, dict):
        raise ValueError("The chat response has an invalid format.")
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise ValueError("The chat response did not contain a choice.")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise ValueError("The chat response did not contain a message.")
    refusal = message.get("refusal")
    if isinstance(refusal, str) and refusal.strip():
        raise ValueError(f"The model refused the request: {refusal.strip()}")
    text = _content_text(message.get("content"))
    if not text:
        raise ValueError("The chat response did not contain text content.")
    return text


def _responses_text(payload: object) -> str:
    if not isinstance(payload, dict):
        raise ValueError("The Responses API response has an invalid format.")
    output = payload.get("output")
    if not isinstance(output, list):
        raise ValueError("The Responses API response did not contain output.")
    parts: list[str] = []
    for item in output:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
    if not parts:
        raise ValueError("The Responses API response did not contain text output.")
    return "".join(parts)


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
) -> dict[str, object]:
    user_content: str | list[dict[str, object]] = prompt
    if images:
        user_content = [{"type": "text", "text": prompt}]
        user_content.extend(
            {
                "type": "image_url",
                "image_url": {"url": image.base64_data_url, "detail": image_detail},
            }
            for image in images
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
) -> dict[str, object]:
    content: str | list[dict[str, object]] = prompt
    if images:
        content = [{"type": "input_text", "text": prompt}]
        content.extend(
            {"type": "input_image", "image_url": image.base64_data_url, "detail": image_detail}
            for image in images
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
) -> ChatResult:
    started = perf_counter()
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
            ),
        )
    if config.api_mode == "responses":
        result = _responses_text(await complete_response(config.base_url, config.api_key, payload))
    else:
        result = _response_text(await complete_chat(config.base_url, config.api_key, payload))
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
        ),
    )


__all__ = ["ChatResult", "build_payload", "build_responses_payload", "execute_chat"]
