"""Execution summary formatting for the LLM node."""

from __future__ import annotations

from sinyuk_nodes.features.llm.config import OpenAPIConfig
from sinyuk_nodes.features.llm.schema import JSONSchemaDocument


def execution_summary(
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
    api = "Responses API" if config.api_mode == "responses" else "Chat Completions"
    fmt = {"json_schema": "JSON Schema", "json_object": "JSON Object"}.get(response_format, "Text")
    schema_name = json_schema.name if json_schema is not None else "-"
    token_limit = (
        str(max_tokens) if max_tokens is not None and max_tokens > 0 else "Provider default"
    )
    reasoning = reasoning_effort if reasoning_effort != "none" else "Provider default"
    return (
        "### Execution Summary\n\n"
        f"- **API:** `{api}`\n"
        f"- **Model:** `{config.model}`\n"
        f"- **Response format:** `{fmt}`\n"
        f"- **JSON Schema:** `{schema_name}`\n"
        f"- **Images:** `{image_count}` (`{image_detail}` detail)\n"
        f"- **Max output tokens:** `{token_limit}`\n"
        f"- **Reasoning effort:** `{reasoning}`\n"
        f"- **Cache:** `{cache}`\n"
        f"- **Elapsed:** `{elapsed_ms} ms`\n"
        f"- **Output length:** `{len(response)}` characters"
    )


__all__ = ["execution_summary"]
