"""Focused tests for OpenAI-compatible configuration and request shaping."""

from __future__ import annotations

import pytest
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


def test_manual_model_takes_priority_over_dropdown() -> None:
    config = build_config(
        "secret",
        "https://example.test/v1",
        "manual-model",
        "listed-model",
        ("listed-model",),
    )

    assert config.model == "manual-model"


def test_model_selection_does_not_fall_back_to_first_cached_model() -> None:
    config = build_config(
        "secret", "https://example.test/v1", "", "auto", ("first-model", "second-model")
    )

    with pytest.raises(ValueError, match="select a model"):
        _ = config.model


def test_request_uses_standard_text_chat_shape() -> None:
    config = build_config(
        "secret", "https://example.test/v1", "", "listed-model", ("listed-model",)
    )

    payload = build_payload(config, "Be concise.", "Describe this.", (), 0.7, 1.0, 128, "text", "")

    assert payload["model"] == "listed-model"
    assert payload["messages"] == [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "Describe this."},
    ]
    assert "response_format" not in payload


def test_unset_sampling_parameters_are_omitted() -> None:
    config = build_config(
        "secret", "https://example.test/v1", "", "listed-model", ("listed-model",)
    )

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
    config = build_config(
        "secret", "https://example.test/v1", "", "listed-model", ("listed-model",)
    )
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
    config = build_config(
        "secret", "https://example.test/v1", "", "listed-model", ("listed-model",)
    )
    payload = build_payload(
        config, "", "Reply.", (), response_format="json_schema", json_schema=_SCHEMA
    )
    assert payload["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "status_result", "strict": True, "schema": _SCHEMA.schema},
    }


def test_responses_json_schema_is_sent_as_text_format() -> None:
    config = build_config(
        "secret", "https://example.test/v1", "", "listed-model", ("listed-model",)
    )
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


def test_responses_text_extracts_output_text() -> None:
    assert (
        _responses_text(
            {"output": [{"type": "message", "content": [{"type": "output_text", "text": "OK"}]}]}
        )
        == "OK"
    )


def test_responses_payload_keeps_every_image_input() -> None:
    config = build_config(
        "secret", "https://example.test/v1", "", "listed-model", ("listed-model",)
    )
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


def test_schema_validation_requires_strict_object_properties() -> None:
    with pytest.raises(ValueError, match="required"):
        parse_json_schema(
            '{"type":"object","properties":{"status":{"type":"string"}},'
            '"required":[],"additionalProperties":false}'
        )


def test_json_schema_requires_connection_for_structured_output() -> None:
    config = build_config(
        "secret", "https://example.test/v1", "", "listed-model", ("listed-model",)
    )
    with pytest.raises(ValueError, match="Connect a JSON Schema"):
        build_payload(config, "", "Reply.", (), response_format="json_schema")


def test_execution_summary_is_markdown() -> None:
    config = build_config(
        "secret", "https://example.test/v1", "", "listed-model", ("listed-model",)
    )
    summary = _execution_summary(config, "json_schema", _SCHEMA, 4, "high", 2048, "miss", 123, "{}")
    assert summary == (
        "### Execution Summary\n\n"
        "- **API:** `resp`\n"
        "- **Model:** `listed-model`\n"
        "- **Response format:** `js`\n"
        "- **JSON Schema:** `status_result`\n"
        "- **Images:** `4` (`high` detail)\n"
        "- **Max tokens:** `2048`\n"
        "- **Cache:** `miss`\n"
        "- **Elapsed:** `123 ms`\n"
        "- **Output length:** `2` characters"
    )
