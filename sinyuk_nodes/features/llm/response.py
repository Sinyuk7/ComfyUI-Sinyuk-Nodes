"""Response parsing for OpenAI-compatible LLM APIs."""

from __future__ import annotations

from typing import TypeGuard


def _is_object(value: object) -> TypeGuard[dict[str, object]]:
    return isinstance(value, dict)


def _is_list(value: object) -> TypeGuard[list[object]]:
    return isinstance(value, list)


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


def response_text(payload: object) -> str:
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


def responses_text(payload: object) -> str:
    text, _ = responses_result(payload)
    return text


def responses_result(payload: object) -> tuple[str, str | None]:
    if not _is_object(payload):
        raise ValueError("The Responses API response has an invalid format.")
    response_id = payload.get("id")
    if response_id is not None and not isinstance(response_id, str):
        raise ValueError("The Responses API response contained an invalid ID.")
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
    return "".join(parts), response_id


__all__ = ["response_text", "responses_result", "responses_text"]
