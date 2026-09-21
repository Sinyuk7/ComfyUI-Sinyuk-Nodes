"""Focused tests for OpenAI-compatible configuration and request shaping."""

from __future__ import annotations

from sinyuk_nodes.features.llm_api import build_payload
from sinyuk_nodes.features.openapi_config import build_config


def test_manual_model_takes_priority_over_dropdown() -> None:
    config = build_config(
        "secret",
        "https://example.test/v1",
        "manual-model",
        "listed-model",
        ("listed-model",),
    )

    assert config.model == "manual-model"


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
