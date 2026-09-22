"""Tests for the pure GarmentAnalysis prompt compiler."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from sinyuk_nodes.features.garment_prompt_compiler import (
    compile_prompt,
    load_garment_analysis_context,
    load_garment_analysis_schema,
)


def _fixture_path(name: str = "GarmentAnalysis.json") -> Path:
    return Path(__file__).parents[3] / "test_data" / name


def _fixture() -> str:
    path = _fixture_path()
    return path.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "fixture_name",
    [path.name for path in sorted(_fixture_path().parent.glob("GarmentAnalysis*.json"))],
)
def test_compile_prompt_accepts_all_garment_analysis_fixtures(fixture_name: str) -> None:
    prompt = compile_prompt(_fixture_path(fixture_name).read_text(encoding="utf-8"))

    assert prompt.startswith("Use Image 1 as the fixed base subject.")
    assert "Subject context:" in prompt


def test_compile_prompt_renders_all_items_and_compact_references() -> None:
    prompt = compile_prompt(_fixture(), "Keep the background unchanged.")

    assert "Target item 1 — dress" in prompt
    assert "Target item 2" in prompt
    assert "Images 2 and 3" in prompt
    assert "Image 4" in prompt
    assert "Image 3: Simple rounded almond-toe flat" in prompt
    assert "Keep the background unchanged." in prompt


def test_compile_prompt_preserves_all_key_details_and_handles_empty_preserve() -> None:
    data = json.loads(_fixture())
    garment = data["garments"][0]
    garment["key_details"] += [
        {
            "description": "Extra critical detail.",
            "priority": "critical",
            "region": "extra",
            "source_ref": 2,
        },
        {
            "description": "Extra important detail.",
            "priority": "important",
            "region": "extra",
            "source_ref": 2,
        },
    ]
    garment["must_preserve"] = []

    prompt = compile_prompt(json.dumps(data))

    assert "Extra critical detail." in prompt
    assert "Extra important detail." in prompt
    assert "No additional preserve-critical features." in prompt
    assert ".." not in prompt


def test_compile_prompt_rejects_missing_required_fields() -> None:
    data = json.loads(_fixture())
    del data["garments"][0]["presentation"]

    with pytest.raises(ValueError, match=r"garments\[0\]\.presentation"):
        compile_prompt(json.dumps(data))


def test_bundled_schema_is_named_and_strict() -> None:
    schema = load_garment_analysis_schema()

    assert schema.name == "garment_analysis"
    assert schema.schema["additionalProperties"] is False


def test_garment_analysis_context_loads_all_protocol_parts() -> None:
    context = load_garment_analysis_context()

    assert context.system_prompt.startswith("You analyze images")
    assert context.user_prompt.startswith("Analyze the provided images")
    assert context.schema.name == "garment_analysis"
