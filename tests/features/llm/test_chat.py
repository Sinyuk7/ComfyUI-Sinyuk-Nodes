"""Focused tests for OpenAI-compatible configuration and request shaping."""

from __future__ import annotations

import httpx
import pytest
from comfy.model_management import InterruptProcessingException
from sinyuk_nodes.features.llm import client
from sinyuk_nodes.features.llm.chat import (
    _execution_summary,
    _response_text,
    _responses_text,
    build_payload,
    build_responses_payload,
)
from sinyuk_nodes.features.llm.config import build_config
from sinyuk_nodes.features.llm.image import EncodedImage
from sinyuk_nodes.features.llm.schema import JSONSchemaDocument, parse_json_schema

_SCHEMA = JSONSchemaDocument(
    "status_result",
    {
        "type": "object",
        "properties": {"status": {"type": "string"}},
        "required": ["status"],
        "additionalProperties": False,
    },
)


def test_custom_model_requires_explicit_custom_selection() -> None:
    config = build_config("secret", "https://example.test/v1", "custom", "vendor-model")

    assert config.model == "vendor-model"


def test_custom_model_id_is_rejected_for_preset_selection() -> None:
    with pytest.raises(ValueError, match="only valid when model is custom"):
        build_config("secret", "https://example.test/v1", "gpt-6-astra", "vendor-model")


def test_custom_selection_requires_model_id() -> None:
    with pytest.raises(ValueError, match="Enter a custom model ID"):
        build_config("secret", "https://example.test/v1", "custom", "")


def test_request_uses_standard_text_chat_shape() -> None:
    config = build_config("secret", "https://example.test/v1", "gpt-6-astra", "")

    payload = build_payload(config, "Be concise.", "Describe this.", (), 0.7, 1.0, 128, "text", "")

    assert payload["model"] == "gpt-6-astra"
    assert payload["messages"] == [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "Describe this."},
    ]
    assert "response_format" not in payload


def test_unset_sampling_parameters_are_omitted() -> None:
    config = build_config("secret", "https://example.test/v1", "gpt-6-astra", "")

    payload = build_payload(
        config,
        "",
        "Describe this.",
        (),
        temperature=None,
        top_p=None,
        max_tokens=None,
    )

    assert "temperature" not in payload
    assert "top_p" not in payload
    assert "max_tokens" not in payload


def test_image_detail_is_sent_to_each_image() -> None:
    config = build_config("secret", "https://example.test/v1", "gpt-6-astra", "")
    image = EncodedImage(
        encoded_bytes=b"abc",
        base64_data_url="data:image/jpeg;base64,abc",
        sha256="hash",
        mime_type="image/jpeg",
        width=1,
        height=1,
    )

    payload = build_payload(config, "", "Describe this.", (image,), image_detail="low")

    assert payload["messages"] == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this."},
                {"type": "text", "text": "Image 1:"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/jpeg;base64,abc", "detail": "low"},
                },
            ],
        }
    ]


def test_refusal_is_reported_separately_from_malformed_content() -> None:
    with pytest.raises(ValueError, match="The model refused the request: unsafe request"):
        _response_text({"choices": [{"message": {"content": None, "refusal": "unsafe request"}}]})


def test_chat_json_schema_is_sent_as_response_format() -> None:
    config = build_config("secret", "https://example.test/v1", "gpt-6-astra", "")
    payload = build_payload(
        config, "", "Reply.", (), response_format="json_schema", json_schema=_SCHEMA
    )
    assert payload["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "status_result", "strict": True, "schema": _SCHEMA.schema},
    }


def test_responses_json_schema_is_sent_as_text_format() -> None:
    config = build_config("secret", "https://example.test/v1", "gpt-6-astra", "")
    payload = build_responses_payload(
        config, "System.", "Reply.", (), response_format="json_schema", json_schema=_SCHEMA
    )
    assert payload["text"] == {
        "format": {
            "type": "json_schema",
            "name": "status_result",
            "strict": True,
            "schema": _SCHEMA.schema,
        }
    }


def test_json_object_format_is_supported_by_both_apis() -> None:
    config = build_config("secret", "https://example.test/v1", "gpt-6-astra", "")

    assert build_payload(config, "", "Reply.", (), response_format="json_object")[
        "response_format"
    ] == {"type": "json_object"}
    assert build_responses_payload(config, "", "Reply.", (), response_format="json_object")[
        "text"
    ] == {"format": {"type": "json_object"}}


def test_reasoning_effort_none_is_omitted_and_other_values_are_api_specific() -> None:
    config = build_config("secret", "https://example.test/v1", "gpt-6-astra", "")

    chat_default = build_payload(config, "", "Reply.", ())
    responses_default = build_responses_payload(config, "", "Reply.", ())
    assert "reasoning_effort" not in chat_default
    assert "reasoning" not in responses_default

    chat = build_payload(config, "", "Reply.", (), reasoning_effort="high")
    responses = build_responses_payload(config, "", "Reply.", (), reasoning_effort="high")
    assert chat["reasoning_effort"] == "high"
    assert responses["reasoning"] == {"effort": "high"}


def test_responses_text_extracts_output_text() -> None:
    assert (
        _responses_text(
            {"output": [{"type": "message", "content": [{"type": "output_text", "text": "OK"}]}]}
        )
        == "OK"
    )


def test_responses_payload_keeps_every_image_input() -> None:
    config = build_config("secret", "https://example.test/v1", "gpt-6-astra", "")
    images = tuple(
        EncodedImage(
            encoded_bytes=b"abc",
            base64_data_url=f"data:image/jpeg;base64,abc{index}",
            sha256=f"hash-{index}",
            mime_type="image/jpeg",
            width=1,
            height=1,
        )
        for index in range(4)
    )
    payload = build_responses_payload(config, "", "Analyze.", images)
    input_items = payload["input"]
    assert isinstance(input_items, list)
    content = input_items[0]["content"]
    assert isinstance(content, list)
    assert [item["type"] for item in content[1:]] == [
        "input_text",
        "input_image",
        "input_text",
        "input_image",
        "input_text",
        "input_image",
        "input_text",
        "input_image",
    ]
    assert content[1]["text"] == "Image 1:"
    assert content[7]["text"] == "Image 4:"


def test_schema_validation_rejects_invalid_schema_keywords() -> None:
    with pytest.raises(ValueError, match="JSON Schema is invalid"):
        parse_json_schema(
            '{"type":"object","properties":{"status":{"type":"string"}},'
            '"required":"status","additionalProperties":false}'
        )


def test_json_schema_requires_connection_for_structured_output() -> None:
    config = build_config("secret", "https://example.test/v1", "gpt-6-astra", "")
    with pytest.raises(ValueError, match="Connect a JSON Schema"):
        build_payload(config, "", "Reply.", (), response_format="json_schema")


def test_execution_summary_is_markdown() -> None:
    config = build_config("secret", "https://example.test/v1", "gpt-6-astra", "")
    summary = _execution_summary(config, "json_schema", _SCHEMA, 4, "high", 2048, "miss", 123, "{}")
    assert summary == (
        "### Execution Summary\n\n"
        "- **API:** `resp`\n"
        "- **Model:** `gpt-6-astra`\n"
        "- **Response format:** `JSON Schema`\n"
        "- **JSON Schema:** `status_result`\n"
        "- **Images:** `4` (`high` detail)\n"
        "- **Max tokens:** `2048`\n"
        "- **Reasoning effort:** `none`\n"
        "- **Cache:** `miss`\n"
        "- **Elapsed:** `123 ms`\n"
        "- **Output length:** `2` characters"
    )


@pytest.mark.anyio
async def test_http_request_propagates_comfy_interrupt(monkeypatch: pytest.MonkeyPatch) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        await client.asyncio.sleep(1)
        return httpx.Response(200, request=request)

    calls = 0

    def interrupt_after_first_check() -> None:
        nonlocal calls
        calls += 1
        if calls > 1:
            raise InterruptProcessingException()

    monkeypatch.setattr(client, "check_interrupt", interrupt_after_first_check)
    with pytest.raises(InterruptProcessingException):
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
            await client._request(
                http_client,
                "GET",
                "https://example.test",
                headers={},
            )


@pytest.mark.asyncio
@pytest.mark.parametrize("api", [client.complete_chat, client.complete_response])
@pytest.mark.parametrize("failure", ["read_timeout", "503", "429"])
async def test_generation_post_is_not_retried(monkeypatch, api, failure: str) -> None:
    attempts = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if failure == "read_timeout":
            raise httpx.ReadTimeout("response lost", request=request)
        return httpx.Response(int(failure), json={"error": "unavailable"})

    factory = httpx.AsyncClient
    monkeypatch.setattr(
        client.httpx,
        "AsyncClient",
        lambda **kwargs: factory(transport=httpx.MockTransport(handler), **kwargs),
    )
    with pytest.raises(client.OpenAPIRequestError) as error:
        await api("https://example.test", "secret", {"prompt": "private"})
    assert attempts == 1
    assert error.value.submission_unknown == (failure != "429")


@pytest.mark.asyncio
@pytest.mark.parametrize("cleanup_fails", [False, True])
async def test_latched_cancel_survives_host_reset_and_repeated_task_cancel(
    cleanup_fails: bool,
) -> None:
    import asyncio

    from sinyuk_nodes.common.cancellation import CancellationState

    signal = False
    cleanup_started = asyncio.Event()
    release_cleanup = asyncio.Event()
    cleaned = False

    def check() -> None:
        if signal:
            raise InterruptProcessingException()

    state = CancellationState(check)

    async def request() -> None:
        nonlocal cleaned
        try:
            await asyncio.Event().wait()
        finally:
            cleanup_started.set()
            await release_cleanup.wait()
            cleaned = True
            if cleanup_fails:
                raise RuntimeError("cleanup failed")

    task = asyncio.create_task(state.wait(request()))
    await asyncio.sleep(0)
    signal = True
    await asyncio.wait_for(cleanup_started.wait(), 2)
    signal = False
    task.cancel()
    await asyncio.sleep(0)
    task.cancel()
    release_cleanup.set()
    with pytest.raises(InterruptProcessingException):
        await task
    assert cleaned
    with pytest.raises(InterruptProcessingException):
        state.check()


def test_host_check_does_not_consume_interrupt() -> None:
    from comfy import model_management
    from sinyuk_nodes.compat.comfy import check_interrupt

    model_management.interrupt_current_processing(True)
    try:
        for _ in range(2):
            with pytest.raises(InterruptProcessingException):
                check_interrupt()
    finally:
        model_management.interrupt_current_processing(False)
