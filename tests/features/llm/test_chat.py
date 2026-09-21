"""Focused tests for OpenAI-compatible configuration and request shaping."""

from __future__ import annotations

import pytest
from sinyuk_nodes.features.llm.chat import _response_text, build_payload
from sinyuk_nodes.features.llm.config import build_config
from sinyuk_nodes.features.llm.image import EncodedImage


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
