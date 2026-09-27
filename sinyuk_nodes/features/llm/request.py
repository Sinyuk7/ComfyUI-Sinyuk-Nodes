"""OpenAI-compatible request construction and provider policy."""

from __future__ import annotations

from typing import TypeGuard
from urllib.parse import urlsplit

from .config import OpenAPIConfig
from .image import EncodedImage
from .schema import JSONSchemaDocument

_REASONING_EFFORTS = {"none", "minimal", "low", "medium", "high", "xhigh", "max"}
_QWEN_HOSTS = frozenset({"maas.qianwenaiapi.com"})


def _is_qwen_endpoint(base_url: str) -> bool:
    hostname = urlsplit(base_url).hostname
    return hostname in _QWEN_HOSTS


def provider_request_headers(config: OpenAPIConfig) -> dict[str, str]:
    """Return provider-specific headers without changing standard payloads."""

    if (
        config.session_cache
        and config.api_mode == "responses"
        and _is_qwen_endpoint(config.base_url)
    ):
        return {"x-dashscope-session-cache": "enable"}
    return {}


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
    previous_response_id: str | None = None,
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
    if temperature is not None and temperature >= 0:
        payload["temperature"] = temperature
    if top_p is not None and top_p > 0:
        payload["top_p"] = top_p
    if max_tokens is not None and max_tokens > 0:
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
    previous_response_id: str | None = None,
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
    if previous_response_id:
        payload["previous_response_id"] = previous_response_id
    if temperature is not None and temperature >= 0:
        payload["temperature"] = temperature
    if top_p is not None and top_p > 0:
        payload["top_p"] = top_p
    if max_tokens is not None and max_tokens > 0:
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


__all__ = [
    "build_payload",
    "build_responses_payload",
    "provider_request_headers",
]
