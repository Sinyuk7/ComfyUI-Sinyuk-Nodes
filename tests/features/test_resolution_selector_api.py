from __future__ import annotations

import pytest
from sinyuk_nodes.features.resolution_selector_api import resolution_for_api_model


def test_google_resolution_uses_exact_mapping() -> None:
    assert resolution_for_api_model("Nano Banana 2", "16:9", "4K") == (5504, 3072)


def test_pro_rejects_extreme_ratio() -> None:
    with pytest.raises(ValueError, match="does not support"):
        resolution_for_api_model("Nano Banana Pro", "1:4", "1K")


def test_gpt_uses_validated_preset_size() -> None:
    assert resolution_for_api_model("GPT Image 2.5", "16:9", "4K") == (3840, 2160)


def test_gpt_4k_uses_maximum_legal_exact_ratio() -> None:
    assert resolution_for_api_model("GPT Image 2.5", "21:9", "4K") == (3808, 1632)


def test_gpt_rejects_unsupported_ratio() -> None:
    with pytest.raises(ValueError):
        resolution_for_api_model("GPT Image 2.5", "1:4", "1K")
