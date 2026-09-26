"""Preset-backed prompt context loading and prompt compilation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib.resources import files
from string import Template
from typing import TypedDict, TypeGuard

from sinyuk_nodes.features.llm.schema import (
    JSONSchemaDocument,
    parse_json_schema,
    validate_json,
)


class _KeyDetail(TypedDict):
    region: str
    description: str
    source_ref: int


class _Garment(TypedDict):
    category: str
    main_refs: list[int]
    shape: str
    fabric_behavior: str
    presentation: str
    key_details: list[_KeyDetail]


class _Subject(TypedDict):
    crop: str
    pose: str
    view: str
    notes: str
    styling: str


class _Analysis(TypedDict):
    subject: _Subject
    garments: list[_Garment]


@dataclass(frozen=True)
class LoadedPromptContext:
    """Loaded prompts and schema for one prompt preset."""

    preset_id: str
    version: int
    compiler_id: str
    system_prompt: str
    user_prompt: str
    schema: JSONSchemaDocument | None
    example: str | None
    template: str | None


@dataclass(frozen=True)
class PromptContext:
    """Preset contract passed from a context node to a prompt builder."""

    preset_id: str
    version: int
    schema: JSONSchemaDocument | None
    example: str | None
    compiler_id: str
    template: str | None


_PRESET_PATHS = {
    "garment.replacement": "garment/replacement",
    "garment.enhancement": "garment/enhancement",
}


def _is_object_list(value: object) -> TypeGuard[list[object]]:
    return isinstance(value, list)


def _is_object_dict(value: object) -> TypeGuard[dict[object, object]]:
    return isinstance(value, dict)


def _asset(name: str, preset: str) -> str:
    return (
        files("sinyuk_nodes.features.prompt_builder")
        .joinpath("presets", preset, name)
        .read_text(encoding="utf-8")
    )


def _optional_asset(name: str, preset: str) -> str | None:
    try:
        return _asset(name, preset)
    except FileNotFoundError:
        return None


def _normalize_preset_id(preset: str) -> str:
    normalized = preset.strip().lower()
    aliases = {
        "replacement": "garment.replacement",
        "enhancement": "garment.enhancement",
    }
    return aliases.get(normalized, normalized)


def _preset_path(preset_id: str) -> str:
    try:
        return _PRESET_PATHS[preset_id]
    except KeyError as exc:
        raise ValueError(f"Unsupported prompt preset: {preset_id}.") from exc


def _load_manifest(preset_id: str, preset_path: str) -> tuple[int, str]:
    raw = _optional_asset("manifest.json", preset_path)
    if raw is None:
        raise ValueError(f"Preset {preset_id} is missing manifest.json.")
    try:
        value: object = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid manifest.json for preset {preset_id}.") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Invalid manifest.json for preset {preset_id}: expected an object.")
    manifest_id = value.get("id")
    if manifest_id != preset_id:
        raise ValueError(f"Invalid manifest.json for preset {preset_id}: id must match the preset.")
    version = value.get("version", 1)
    compiler = value.get("compiler", value.get("compiler_id", "template"))
    if not isinstance(version, int) or isinstance(version, bool) or version < 1:
        raise ValueError(f"Invalid manifest.json for preset {preset_id}: version must be positive.")
    if not isinstance(compiler, str) or not compiler.strip():
        raise ValueError(f"Invalid manifest.json for preset {preset_id}: compiler is required.")
    return version, compiler.strip()


def load_preset_schema(preset: str = "garment.replacement") -> JSONSchemaDocument:
    """Load the JSON contract bundled with one prompt preset."""

    preset_id = _normalize_preset_id(preset)
    preset_path = _preset_path(preset_id)
    return parse_json_schema(_asset("schema.json", preset_path), preset_id.replace(".", "_"))


def load_prompt_context(
    preset: str = "garment.replacement",
    schema_placement: str = "External",
    extra_prompt: str = "",
) -> LoadedPromptContext:
    """Load one preset contract and compose its LLM-facing prompts."""

    preset_id = _normalize_preset_id(preset)
    preset_path = _preset_path(preset_id)
    version, compiler_id = _load_manifest(preset_id, preset_path)
    try:
        schema = load_preset_schema(preset_id)
    except FileNotFoundError:
        schema = None
    system_prompt = _asset("system.txt", preset_path).strip()
    user_prompt = _asset("user.txt", preset_path).strip()
    example = _optional_asset("example.json", preset_path)
    if example is not None:
        try:
            example_value: object = json.loads(example)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid example.json for preset {preset_id}.") from exc
        if schema is not None:
            validate_json(example_value, schema)
        example = json.dumps(example_value, ensure_ascii=False, indent=2)
    if schema_placement == "In Prompt":
        schema_text = (
            json.dumps(schema.schema, ensure_ascii=False, indent=2) if schema is not None else ""
        )
        prompt_parts = [
            "Return only valid JSON matching the following JSON Schema.",
            "Do not include Markdown fences or additional text.",
        ]
        if schema_text:
            prompt_parts.append(schema_text)
        if example is not None:
            prompt_parts.append(f"Example output:\n{example}")
        system_prompt = "\n\n".join(
            part for part in (system_prompt, "\n".join(prompt_parts)) if part
        )
    elif schema_placement != "External":
        raise ValueError(f"Unsupported schema placement: {schema_placement}.")
    extra = extra_prompt.strip()
    if extra:
        user_prompt = "\n\n".join(
            part for part in (user_prompt, f"Additional instructions:\n{extra}") if part
        )
    template = _optional_asset("template.txt", preset_path)
    return LoadedPromptContext(
        preset_id,
        version,
        compiler_id,
        system_prompt,
        user_prompt,
        schema,
        example,
        template,
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


def _reject_unknown(value: dict[str, object], allowed: set[str], path: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(
            f"Invalid GarmentAnalysis JSON: {path} has unsupported field(s): {', '.join(unknown)}."
        )


def _require_int_list(value: object, path: str) -> list[int]:
    values = _require_list(value, path)
    if not all(isinstance(item, int) and not isinstance(item, bool) for item in values):
        raise ValueError(f"Invalid GarmentAnalysis JSON: {path} must contain integers.")
    refs = [item for item in values if isinstance(item, int) and not isinstance(item, bool)]
    if any(ref < 2 for ref in refs):
        raise ValueError(f"Invalid GarmentAnalysis JSON: {path} references must be at least 2.")
    return list(dict.fromkeys(refs))


def _parse_detail(value: object, index: int, garment_index: int) -> _KeyDetail:
    path = f"garments[{garment_index}].key_details[{index}]"
    value_dict = _require_object(value, path)
    _reject_unknown(value_dict, {"region", "description", "source_ref"}, path)
    return {
        "region": _require_string(value_dict.get("region"), f"{path}.region"),
        "description": _require_string(value_dict.get("description"), f"{path}.description"),
        "source_ref": _require_int(value_dict.get("source_ref"), f"{path}.source_ref"),
    }


def _parse_garment(value: object, index: int) -> _Garment:
    path = f"garments[{index}]"
    value_dict = _require_object(value, path)
    _reject_unknown(
        value_dict,
        {"category", "main_refs", "shape", "fabric_behavior", "presentation", "key_details"},
        path,
    )
    details = _require_list(value_dict.get("key_details"), f"{path}.key_details")
    main_refs = _require_int_list(value_dict.get("main_refs"), f"{path}.main_refs")
    if not main_refs:
        raise ValueError(f"Invalid GarmentAnalysis JSON: {path}.main_refs cannot be empty.")
    return {
        "category": _require_string(value_dict.get("category"), f"{path}.category"),
        "main_refs": main_refs,
        "shape": _require_string(value_dict.get("shape"), f"{path}.shape"),
        "fabric_behavior": _require_string(
            value_dict.get("fabric_behavior"), f"{path}.fabric_behavior"
        ),
        "presentation": _require_string(value_dict.get("presentation"), f"{path}.presentation"),
        "key_details": [
            _parse_detail(item, item_index, index) for item_index, item in enumerate(details)
        ],
    }


def _parse_analysis(raw: str, *, allow_empty_garments: bool = False) -> _Analysis:
    try:
        value: object = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("Invalid GarmentAnalysis JSON: malformed JSON.") from exc
    value_dict = _require_object(value, "root")
    _reject_unknown(value_dict, {"schema_version", "subject", "garments"}, "root")
    schema_version = _require_string(value_dict.get("schema_version"), "schema_version")
    if schema_version != "2.0":
        raise ValueError("Invalid GarmentAnalysis JSON: schema_version must be '2.0'.")
    subject_value = _require_object(value_dict.get("subject"), "subject")
    _reject_unknown(subject_value, {"crop", "pose", "view", "notes", "styling"}, "subject")
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
        "styling": _require_string(subject_value.get("styling"), "subject.styling"),
    }
    garments = _require_list(value_dict.get("garments"), "garments")
    if not garments and not allow_empty_garments:
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
    styling = subject["styling"].strip()
    if notes:
        result += f" Pose and visibility notes: {notes}"
    if styling:
        result += f" Compatible styling context: {styling}"
    return result


def _render_item_title(index: int, garment: _Garment) -> str:
    category = garment["category"].strip()
    return f"Target item {index} — {category}" if category != "other" else f"Target item {index}"


def _render_item(index: int, garment: _Garment, preset: str) -> str:
    detail_lines = [
        f"- {detail['region']} — "
        f"Image {detail['source_ref']}: {detail['description'].strip().rstrip('.。')}."
        for detail in garment["key_details"]
    ]
    key_details = "\n".join(detail_lines) if detail_lines else "No additional key details."
    detail_refs: list[int] = []
    for detail in garment["key_details"]:
        ref = detail["source_ref"]
        if ref not in garment["main_refs"] and ref not in detail_refs:
            detail_refs.append(ref)
    main_label = "reference" if len(garment["main_refs"]) == 1 else "references"
    references = f"Use {_format_refs(garment['main_refs'])} as the main {main_label}."
    if detail_refs:
        detail_label = "reference" if len(detail_refs) == 1 else "references"
        references += f" Use {_format_refs(detail_refs)} as additional detail {detail_label}."
    values = {
        "item_title": _render_item_title(index, garment),
        "references": references,
        "shape": garment["shape"],
        "fabric_behavior": garment["fabric_behavior"],
        "key_details": key_details,
        "presentation": garment["presentation"],
    }
    template = _asset("templates/outfit_item.txt", preset)
    if preset.rsplit("/", maxsplit=1)[-1] == "enhancement":
        values["key_details"] = "\n".join(detail_lines)
        # Each optional section belongs to one field in the bundled item template.
        template = "\n\n".join(
            block
            for block in template.split("\n\n")
            if not any(
                "${" + field + "}" in block and not values[field].strip()
                for field in ("shape", "fabric_behavior", "key_details", "presentation")
            )
        )
    return Template(template).substitute(values).strip()


def compile_prompt(
    analysis_json: str,
    *,
    schema: JSONSchemaDocument,
    preset: str | None = None,
    template: str | None = None,
) -> str:
    """Compile one validated GarmentAnalysis JSON document into a prompt."""

    preset = preset or {
        "garment_replacement": "replacement",
        "garment_enhancement": "enhancement",
    }.get(schema.name)
    if preset is None:
        raise ValueError(f"Unsupported garment prompt schema: {schema.name}.")
    if schema.schema != load_preset_schema(f"garment.{preset}").schema:
        raise ValueError("Analysis Schema does not match the bundled preset.")
    preset_path = f"garment/{preset}"
    analysis = _parse_analysis(analysis_json, allow_empty_garments=preset == "enhancement")
    item_blocks: list[str] = []
    for index, garment in enumerate(analysis["garments"], start=1):
        item_blocks.append(_render_item(index, garment, preset_path))
    return (
        Template(template or _asset("template.txt", f"garment/{preset}"))
        .safe_substitute(
            outfit_items="\n\n".join(item_blocks),
            subject_context=_render_subject(analysis["subject"]),
        )
        .strip()
    )


def build_prompt(llm_response: str, *, context: PromptContext) -> str:
    """Build a prompt from a preset context and a structured LLM response."""

    if context.schema is None:
        raise ValueError("Prompt Builder requires the preset JSON Schema.")
    if context.compiler_id == "garment" and context.template is None:
        raise ValueError("Prompt Builder requires the preset Prompt template.")
    try:
        value: object = json.loads(llm_response)
    except json.JSONDecodeError as exc:
        raise ValueError("Prompt Builder received malformed JSON from the LLM.") from exc
    validate_json(value, context.schema)
    if context.compiler_id == "garment":
        return compile_prompt(
            json.dumps(value, ensure_ascii=False),
            schema=context.schema,
            preset=context.preset_id.rsplit(".", maxsplit=1)[-1],
            template=context.template,
        )
    if context.compiler_id == "template":
        if context.template is None:
            raise ValueError("Prompt Builder requires a template for the template compiler.")
        rendered = Template(context.template).safe_substitute(
            json=json.dumps(value, ensure_ascii=False, indent=2),
            response=llm_response,
        )
        return rendered.strip()
    raise ValueError(f"Unsupported prompt compiler: {context.compiler_id}.")


__all__ = [
    "LoadedPromptContext",
    "PromptContext",
    "build_prompt",
    "compile_prompt",
    "load_preset_schema",
    "load_prompt_context",
]
