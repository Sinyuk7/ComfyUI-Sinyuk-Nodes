"""Tests for the pure GarmentAnalysis prompt compiler."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from sinyuk_nodes.features.prompt_builder import (
    PromptContext,
    build_prompt,
    compile_prompt,
    load_preset_schema,
    load_prompt_context,
)


def _fixture_path(name: str = "GarmentAnalysis.json") -> Path:
    return Path(__file__).parents[3] / "test_data" / name


def _fixture() -> str:
    path = _fixture_path()
    if path.exists():
        return path.read_text(encoding="utf-8")
    return json.dumps(
        {
            "schema_version": "2.0",
            "subject": {
                "crop": "full_body",
                "pose": "standing",
                "view": "front",
                "notes": "",
                "styling": "Keep the existing layering.",
            },
            "garments": [
                {
                    "category": "dress",
                    "main_refs": [2, 3],
                    "shape": "Straight dress.",
                    "fabric_behavior": "Soft drape.",
                    "presentation": "Retain existing fit.",
                    "key_details": [
                        {"region": "collar", "description": "Rounded collar.", "source_ref": 2}
                    ],
                },
                {
                    "category": "shoes",
                    "main_refs": [3],
                    "shape": "Flat shoes.",
                    "fabric_behavior": "Matte leather.",
                    "presentation": "Follow the feet.",
                    "key_details": [
                        {
                            "region": "footwear",
                            "description": "Simple rounded almond-toe flat",
                            "source_ref": 3,
                        }
                    ],
                },
            ],
        }
    )


@pytest.mark.parametrize(
    "fixture_name",
    [path.name for path in sorted(_fixture_path().parent.glob("GarmentAnalysis*.json"))],
)
def test_compile_prompt_accepts_all_garment_analysis_fixtures(fixture_name: str) -> None:
    prompt = compile_prompt(
        _fixture_path(fixture_name).read_text(encoding="utf-8"),
        schema=load_preset_schema(),
    )

    assert prompt.startswith("Use Image 1 as the fixed base subject.")
    assert "Subject context:" in prompt


def test_compile_prompt_renders_all_items_and_compact_references() -> None:
    prompt = compile_prompt(_fixture(), schema=load_preset_schema())

    assert "Target item 1 — dress" in prompt
    assert "Target item 2" in prompt
    assert "Images 2 and 3" in prompt
    assert "Image 3: Simple rounded almond-toe flat" in prompt
    assert "Do not change the pose, limb positions, contact points" in prompt
    assert "three-dimensional worn construction" in prompt


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
    prompt = compile_prompt(json.dumps(data), schema=load_preset_schema())

    assert "Extra critical detail." in prompt
    assert "Extra important detail." in prompt
    assert prompt.index("Extra critical detail.") < prompt.index("Extra important detail.")
    assert "Use Image 4 as additional detail reference." in prompt
    assert ".." not in prompt


def test_compile_prompt_rejects_missing_required_fields() -> None:
    data = json.loads(_fixture())
    del data["garments"][0]["presentation"]

    with pytest.raises(ValueError, match=r"garments\[0\]\.presentation"):
        compile_prompt(json.dumps(data), schema=load_preset_schema())


def test_compile_prompt_rejects_old_detail_refs_field() -> None:
    data = json.loads(_fixture())
    data["garments"][0]["detail_refs"] = [2]

    with pytest.raises(ValueError):
        compile_prompt(json.dumps(data), schema=load_preset_schema())


def test_compile_prompt_requires_v2_subject_styling_and_source_ref() -> None:
    data = json.loads(_fixture())
    del data["subject"]["styling"]
    with pytest.raises(ValueError, match="subject.styling"):
        compile_prompt(json.dumps(data), schema=load_preset_schema())

    data = json.loads(_fixture())
    del data["garments"][0]["key_details"][0]["source_ref"]
    with pytest.raises(ValueError, match="source_ref"):
        compile_prompt(json.dumps(data), schema=load_preset_schema())


def test_bundled_schema_is_named_and_strict() -> None:
    schema = load_preset_schema()

    assert schema.name == "garment_replacement"
    assert schema.schema["additionalProperties"] is False


def test_garment_analysis_context_loads_all_protocol_parts() -> None:
    context = load_prompt_context()

    assert context.system_prompt.startswith("You analyze images")
    assert context.user_prompt.startswith("Analyze the provided images")
    assert "compatible styling anchor" in context.system_prompt
    assert "not a target garment identity reference" in context.user_prompt
    assert context.schema.name == "garment_replacement"


def test_context_placement_and_extra_prompt_affect_only_llm_prompts() -> None:
    external = load_prompt_context("enhancement", "External", "Focus on material details.")
    in_prompt = load_prompt_context("enhancement", "In Prompt")

    assert "Return only valid JSON matching" not in external.system_prompt
    assert "Return only valid JSON matching" in in_prompt.system_prompt
    assert "Focus on material details." in external.user_prompt
    assert "Additional instructions:" in external.user_prompt
    assert "Additional instructions:" not in in_prompt.user_prompt


def test_builder_formats_jsonschema_validation_errors() -> None:
    context = load_prompt_context()
    prompt_context = PromptContext(
        preset_id="garment.replacement",
        version=1,
        schema=context.schema,
        example=None,
        compiler_id="garment",
        template=context.template,
    )
    data = json.loads(_fixture())
    data["garments"][0]["main_refs"] = "2"

    with pytest.raises(ValueError, match=r"path: garments\[0\]\.main_refs"):
        build_prompt(json.dumps(data), context=prompt_context)


@pytest.mark.parametrize("preset", ["replacement", "enhancement"])
def test_preset_context_drives_matching_prompt(preset: str) -> None:
    context = load_prompt_context(preset)
    prompt = compile_prompt(_fixture(), schema=context.schema)
    assert context.system_prompt
    assert context.user_prompt
    assert "Target item 1" in prompt
    assert "Image 3: Simple rounded almond-toe flat" in prompt
    if preset == "enhancement":
        assert prompt.startswith("Refine the existing outfit")
        assert "Reference structure" in prompt
        assert "Replace the current outfit completely" not in prompt
    else:
        assert prompt.startswith("Use Image 1 as the fixed base subject.")


def test_builder_rejects_unknown_or_modified_schema() -> None:
    from sinyuk_nodes.features.llm.schema import JSONSchemaDocument

    with pytest.raises(ValueError, match="Unsupported garment prompt schema"):
        compile_prompt(_fixture(), schema=JSONSchemaDocument("unknown", {}))
    with pytest.raises(ValueError, match="does not match"):
        compile_prompt(_fixture(), schema=JSONSchemaDocument("garment_enhancement", {}))
    with pytest.raises(ValueError, match="Unsupported prompt preset"):
        load_prompt_context("../unknown")


def test_enhancement_omits_empty_sections_and_deduplicates_references() -> None:
    data = json.loads(_fixture())
    data["garments"] = data["garments"][:1]
    item = data["garments"][0]
    item.update(
        main_refs=[2, 2, 3],
        shape=" ",
        fabric_behavior="",
        presentation="",
        key_details=[],
    )
    prompt = compile_prompt(json.dumps(data), schema=load_preset_schema("enhancement"))
    assert "Use Images 2 and 3 as the main references." in prompt
    assert "Reference structure:" not in prompt
    assert "Material properties (" not in prompt
    assert "Source-grounded details:" not in prompt
    assert "Wearing and integration" not in prompt
    assert "No additional key details." not in prompt

    item["presentation"] = "Wear naturally from the shoulder."
    prompt = compile_prompt(json.dumps(data), schema=load_preset_schema("enhancement"))
    assert "Wearing and integration" in prompt
    assert "Wear naturally from the shoulder." in prompt


def test_enhancement_allows_no_reliably_matching_reference_items() -> None:
    data = json.loads(_fixture())
    data["garments"] = []

    prompt = compile_prompt(json.dumps(data), schema=load_preset_schema("enhancement"))

    assert prompt.startswith("Refine the existing outfit in Image 1")
    assert "Target item" not in prompt

    with pytest.raises(ValueError, match="garments cannot be empty"):
        compile_prompt(json.dumps(data), schema=load_preset_schema("replacement"))
