"""OpenAI-compatible multimodal chat completion behavior."""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict

import torch
from sinyuk_nodes.common.llm_image import EncodedImage, encode_images

from .openapi_client import complete_chat
from .openapi_config import OpenAPIConfig

_RESPONSE_CACHE: OrderedDict[str, str] = OrderedDict()
_CACHE_LIMIT = 128


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
    text = _content_text(message.get("content"))
    if not text:
        raise ValueError("The chat response did not contain text content.")
    return text


def build_payload(
    config: OpenAPIConfig,
    system_prompt: str,
    prompt: str,
    images: tuple[EncodedImage, ...],
    temperature: float,
    top_p: float,
    max_tokens: int,
    response_format: str,
    json_schema: str,
    detail: str = "high",
) -> dict[str, object]:
    user_content: str | list[dict[str, object]] = prompt
    if images:
        user_content = [{"type": "text", "text": prompt}]
        user_content.extend(
            {
                "type": "image_url",
                "image_url": {"url": image.base64_data_url, "detail": detail},
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
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }
    if response_format == "json_object":
        payload["response_format"] = {"type": "json_object"}
    elif response_format == "json_schema":
        try:
            schema: object = json.loads(json_schema)
        except json.JSONDecodeError as exc:
            raise ValueError("JSON Schema must be valid JSON.") from exc
        if not isinstance(schema, dict):
            raise ValueError("JSON Schema must be a JSON object.")
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "response", "strict": True, "schema": schema},
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
    temperature: float,
    top_p: float,
    max_tokens: int,
    response_format: str,
    json_schema: str,
    detail: str = "high",
) -> str:
    encoded = encode_images(images, detail)
    payload = build_payload(
        config,
        system_prompt,
        prompt,
        encoded,
        temperature,
        top_p,
        max_tokens,
        response_format,
        json_schema,
        detail,
    )
    fingerprint = hashlib.sha256(
        json.dumps(
            {
                "base_url": config.base_url,
                **payload,
                "image_hashes": [image.sha256 for image in encoded],
                "seed": seed,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    if fingerprint in _RESPONSE_CACHE:
        _RESPONSE_CACHE.move_to_end(fingerprint)
        return _RESPONSE_CACHE[fingerprint]
    result = _response_text(await complete_chat(config.base_url, config.api_key, payload))
    _RESPONSE_CACHE[fingerprint] = result
    _RESPONSE_CACHE.move_to_end(fingerprint)
    while len(_RESPONSE_CACHE) > _CACHE_LIMIT:
        _RESPONSE_CACHE.popitem(last=False)
    return result


__all__ = ["build_payload", "execute_chat"]
