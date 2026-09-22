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
    assert "Image 3: Simple rounded almond-toe flat" in prompt
    assert "Do not change the pose, limb positions, contact points" in prompt
    assert "three-dimensional worn construction" in prompt
    assert "Keep the background unchanged." in prompt


def test_compile_prompt_preserves_detail_order_and_derives_detail_refs() -> None:
    data = json.loads(_fixture())
    garment = data["garments"][0]
    garment["key_details"] += [
        {
            "description": "Extra critical detail.",
            "region": "extra",
            "source_ref": 2,
        },
        {
            "description": "Extra important detail.",
            "region": "extra",
            "source_ref": 4,
        },
    ]
    prompt = compile_prompt(json.dumps(data))

    assert "Extra critical detail." in prompt
    assert "Extra important detail." in prompt
    assert prompt.index("Extra critical detail.") < prompt.index("Extra important detail.")
    assert "Use Image 4 as additional detail reference." in prompt
    assert ".." not in prompt


def test_compile_prompt_rejects_missing_required_fields() -> None:
    data = json.loads(_fixture())
    del data["garments"][0]["presentation"]

    with pytest.raises(ValueError, match=r"garments\[0\]\.presentation"):
        compile_prompt(json.dumps(data))


def test_compile_prompt_rejects_old_detail_refs_field() -> None:
    data = json.loads(_fixture())
    data["garments"][0]["detail_refs"] = [2]

    with pytest.raises(ValueError):
        compile_prompt(json.dumps(data))


def test_compile_prompt_requires_v2_subject_styling_and_source_ref() -> None:
    data = json.loads(_fixture())
    del data["subject"]["styling"]
    with pytest.raises(ValueError, match="subject.styling"):
        compile_prompt(json.dumps(data))

    data = json.loads(_fixture())
    del data["garments"][0]["key_details"][0]["source_ref"]
    with pytest.raises(ValueError, match="source_ref"):
        compile_prompt(json.dumps(data))


def test_bundled_schema_is_named_and_strict() -> None:
    schema = load_garment_analysis_schema()

    assert schema.name == "garment_analysis"
    assert schema.schema["additionalProperties"] is False


def test_garment_analysis_context_loads_all_protocol_parts() -> None:
    context = load_garment_analysis_context()

    assert context.system_prompt.startswith("You analyze images")
    assert context.user_prompt.startswith("Analyze the provided images")
    assert "compatible styling anchor" in context.system_prompt
    assert "not a target garment identity reference" in context.user_prompt
    assert context.schema.name == "garment_analysis"
