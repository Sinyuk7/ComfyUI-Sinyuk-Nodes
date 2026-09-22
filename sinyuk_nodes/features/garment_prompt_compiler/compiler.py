"""Pure Python compiler for GarmentAnalysis JSON."""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib.resources import files
from string import Template
from typing import TypedDict, TypeGuard

from sinyuk_nodes.features.llm.schema import JSONSchemaDocument, parse_json_schema

_PRIORITY_RANK = {"critical": 0, "important": 1, "supporting": 2}


class _KeyDetail(TypedDict):
    region: str
    description: str
    source_ref: int
    priority: str


class _Garment(TypedDict):
    id: str
    category: str
    main_refs: list[int]
    detail_refs: list[int]
    shape: str
    fabric_behavior: str
    presentation: str
    must_preserve: list[str]
    key_details: list[_KeyDetail]


class _Subject(TypedDict):
    crop: str
    pose: str
    view: str
    notes: str


class _Analysis(TypedDict):
    subject: _Subject
    garments: list[_Garment]


@dataclass(frozen=True)
class GarmentAnalysisContext:
    """Static prompts and schema for one Garment Analysis protocol."""

    system_prompt: str
    user_prompt: str
    schema: JSONSchemaDocument


def _is_object_list(value: object) -> TypeGuard[list[object]]:
    return isinstance(value, list)


def _is_object_dict(value: object) -> TypeGuard[dict[object, object]]:
    return isinstance(value, dict)


def _asset(name: str) -> str:
    return (
        files("sinyuk_nodes.features.garment_prompt_compiler")
        .joinpath(name)
        .read_text(encoding="utf-8")
    )


def load_garment_analysis_schema() -> JSONSchemaDocument:
    """Load the bundled schema used by the upstream GarmentAnalysis LLM."""

    return parse_json_schema(_asset("schema.json"), "garment_analysis")


def load_garment_analysis_context() -> GarmentAnalysisContext:
    """Load the complete static Garment Analysis protocol."""

    return GarmentAnalysisContext(
        system_prompt=_asset("templates/system_prompt.txt").strip(),
        user_prompt=_asset("templates/user_prompt.txt").strip(),
        schema=load_garment_analysis_schema(),
    )


def _require_string(value: object, path: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"Invalid GarmentAnalysis JSON: {path} must be a string.")
    return value


def _require_int(value: object, path: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 2:
        raise ValueError(f"Invalid GarmentAnalysis JSON: {path} must be an integer at least 2.")
    return value


def _require_list(value: object, path: str) -> list[object]:
    if not _is_object_list(value):
        raise ValueError(f"Invalid GarmentAnalysis JSON: {path} must be an array.")
    return value


def _require_object(value: object, path: str) -> dict[str, object]:
    if not _is_object_dict(value):
        raise ValueError(f"Invalid GarmentAnalysis JSON: {path} must be an object.")
    result: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise ValueError(f"Invalid GarmentAnalysis JSON: {path} has a non-string key.")
        result[key] = item
    return result


def _require_int_list(value: object, path: str) -> list[int]:
    values = _require_list(value, path)
    if not all(isinstance(item, int) and not isinstance(item, bool) for item in values):
        raise ValueError(f"Invalid GarmentAnalysis JSON: {path} must contain integers.")
    refs = [item for item in values if isinstance(item, int) and not isinstance(item, bool)]
    if any(ref < 2 for ref in refs):
        raise ValueError(f"Invalid GarmentAnalysis JSON: {path} references must be at least 2.")
    return refs


def _parse_detail(value: object, index: int, garment_index: int) -> _KeyDetail:
    path = f"garments[{garment_index}].key_details[{index}]"
    value_dict = _require_object(value, path)
    priority = _require_string(value_dict.get("priority"), f"{path}.priority")
    if priority not in _PRIORITY_RANK:
        raise ValueError(f"Invalid GarmentAnalysis JSON: {path}.priority is unsupported.")
    return {
        "region": _require_string(value_dict.get("region"), f"{path}.region"),
        "description": _require_string(value_dict.get("description"), f"{path}.description"),
        "source_ref": _require_int(value_dict.get("source_ref"), f"{path}.source_ref"),
        "priority": priority,
    }


def _parse_garment(value: object, index: int) -> _Garment:
    path = f"garments[{index}]"
    value_dict = _require_object(value, path)
    details = _require_list(value_dict.get("key_details"), f"{path}.key_details")
    main_refs = _require_int_list(value_dict.get("main_refs"), f"{path}.main_refs")
    if not main_refs:
        raise ValueError(f"Invalid GarmentAnalysis JSON: {path}.main_refs cannot be empty.")
    return {
        "id": _require_string(value_dict.get("id"), f"{path}.id"),
        "category": _require_string(value_dict.get("category"), f"{path}.category"),
        "main_refs": main_refs,
        "detail_refs": _require_int_list(value_dict.get("detail_refs"), f"{path}.detail_refs"),
        "shape": _require_string(value_dict.get("shape"), f"{path}.shape"),
        "fabric_behavior": _require_string(
            value_dict.get("fabric_behavior"), f"{path}.fabric_behavior"
        ),
        "presentation": _require_string(value_dict.get("presentation"), f"{path}.presentation"),
        "must_preserve": [
            _require_string(item, f"{path}.must_preserve[{item_index}]")
            for item_index, item in enumerate(
                _require_list(value_dict.get("must_preserve"), f"{path}.must_preserve")
            )
        ],
        "key_details": [
            _parse_detail(item, item_index, index) for item_index, item in enumerate(details)
        ],
    }


def _parse_analysis(raw: str) -> _Analysis:
    try:
        value: object = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("Invalid GarmentAnalysis JSON: malformed JSON.") from exc
    value_dict = _require_object(value, "root")
    schema_version = _require_string(value_dict.get("schema_version"), "schema_version")
    if schema_version != "1.0":
        raise ValueError("Invalid GarmentAnalysis JSON: schema_version must be '1.0'.")
    subject_value = _require_object(value_dict.get("subject"), "subject")
    crop = _require_string(subject_value.get("crop"), "subject.crop")
    pose = _require_string(subject_value.get("pose"), "subject.pose")
    view = _require_string(subject_value.get("view"), "subject.view")
    if crop not in {
        "full_body",
        "three_quarter_body",
        "waist_up",
        "bust_up",
        "close_up",
        "lower_body",
    }:
        raise ValueError("Invalid GarmentAnalysis JSON: subject.crop is unsupported.")
    if pose not in {
        "standing",
        "walking",
        "seated",
        "kneeling",
        "crouching",
        "reclining",
        "lying",
        "dynamic",
        "other",
    }:
        raise ValueError("Invalid GarmentAnalysis JSON: subject.pose is unsupported.")
    if view not in {"front", "three_quarter_front", "side", "three_quarter_back", "back", "mixed"}:
        raise ValueError("Invalid GarmentAnalysis JSON: subject.view is unsupported.")
    subject: _Subject = {
        "crop": crop,
        "pose": pose,
        "view": view,
        "notes": _require_string(subject_value.get("notes"), "subject.notes"),
    }
    garments = _require_list(value_dict.get("garments"), "garments")
    if not garments:
        raise ValueError("Invalid GarmentAnalysis JSON: garments cannot be empty.")
    return {
        "subject": subject,
        "garments": [_parse_garment(item, i) for i, item in enumerate(garments)],
    }


def _format_refs(refs: list[int]) -> str:
    if len(refs) == 1:
        return f"Image {refs[0]}"
    if len(refs) == 2:
        return f"Images {refs[0]} and {refs[1]}"
    return f"Images {', '.join(str(ref) for ref in refs[:-1])}, and {refs[-1]}"


def _render_subject(subject: _Subject) -> str:
    result = f"Crop: {subject['crop']}. Pose: {subject['pose']}. View: {subject['view']}."
    notes = subject["notes"].strip()
    return f"{result} Pose and visibility notes: {notes}" if notes else result


def _render_item_title(index: int, garment: _Garment) -> str:
    category = garment["category"].strip()
    return f"Target item {index} — {category}" if category != "other" else f"Target item {index}"


def _render_item(index: int, garment: _Garment) -> str:
    details = sorted(garment["key_details"], key=lambda item: _PRIORITY_RANK[item["priority"]])
    detail_lines = [
        f"- {detail['priority'].capitalize()} — {detail['region']} — "
        f"Image {detail['source_ref']}: {detail['description'].strip().rstrip('.。')}."
        for detail in details
    ]
    key_details = "\n".join(detail_lines) if detail_lines else "No additional key details."
    preserve = (
        "; ".join(item.strip().rstrip(".。") for item in garment["must_preserve"]) + "."
        if garment["must_preserve"]
        else "No additional preserve-critical features."
    )
    references = f"Use {_format_refs(garment['main_refs'])} as the main reference."
    if garment["detail_refs"]:
        references += (
            f" Use {_format_refs(garment['detail_refs'])} as additional detail reference(s)."
        )
    return (
        Template(_asset("templates/outfit_item.txt"))
        .safe_substitute(
            item_title=_render_item_title(index, garment),
            references=references,
            shape=garment["shape"],
            fabric_behavior=garment["fabric_behavior"],
            must_preserve=preserve,
            key_details=key_details,
            presentation=garment["presentation"],
        )
        .strip()
    )


def compile_prompt(analysis_json: str, extra_prompt: str = "") -> str:
    """Compile one validated GarmentAnalysis JSON document into a prompt."""

    analysis = _parse_analysis(analysis_json)
    item_blocks: list[str] = []
    for index, garment in enumerate(analysis["garments"], start=1):
        item_blocks.append(_render_item(index, garment))
    return (
        Template(_asset("templates/garment_replacement.txt"))
        .safe_substitute(
            outfit_items="\n\n".join(item_blocks),
            subject_context=_render_subject(analysis["subject"]),
            extra_prompt=extra_prompt.strip(),
        )
        .strip()
    )


__all__ = [
    "GarmentAnalysisContext",
    "compile_prompt",
    "load_garment_analysis_context",
    "load_garment_analysis_schema",
]
