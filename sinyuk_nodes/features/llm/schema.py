"""Load and validate schemas for Structured Outputs."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_JSON_TYPES = {"array", "boolean", "integer", "null", "number", "object", "string"}


@dataclass(frozen=True)
class JSONSchemaDocument:
    """A named JSON Schema ready to be sent to an LLM API."""

    name: str
    schema: dict[str, object]


def _validate_schema_node(value: object, path: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"JSON Schema {path} must be an object.")
    schema_type = value.get("type")
    if schema_type is not None and not (
        isinstance(schema_type, str)
        and schema_type in _JSON_TYPES
        or isinstance(schema_type, list)
        and schema_type
        and all(isinstance(item, str) and item in _JSON_TYPES for item in schema_type)
    ):
        raise ValueError(f"JSON Schema {path}.type must be a supported JSON type.")
    properties = value.get("properties")
    if properties is not None:
        if not isinstance(properties, dict) or not all(isinstance(key, str) for key in properties):
            raise ValueError(f"JSON Schema {path}.properties must be an object.")
        if value.get("additionalProperties") is not False:
            raise ValueError(
                f"JSON Schema {path} must set additionalProperties to false for strict output."
            )
        required = value.get("required")
        if not isinstance(required, list) or set(required) != set(properties):
            raise ValueError(
                f"JSON Schema {path}.required must contain every property exactly "
                "for strict output."
            )
        for key, child in properties.items():
            _validate_schema_node(child, f"{path}.properties.{key}")
    items = value.get("items")
    if items is not None:
        _validate_schema_node(items, f"{path}.items")
    any_of = value.get("anyOf")
    if any_of is not None:
        if not isinstance(any_of, list) or not any_of:
            raise ValueError(f"JSON Schema {path}.anyOf must be a non-empty array.")
        for index, child in enumerate(any_of):
            _validate_schema_node(child, f"{path}.anyOf[{index}]")


def parse_json_schema(raw: str, name: str = "") -> JSONSchemaDocument:
    """Parse and validate one strict Structured Outputs schema."""

    try:
        value: object = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("JSON Schema must be valid JSON.") from exc
    if not isinstance(value, dict):
        raise ValueError("JSON Schema must be a JSON object.")
    if value.get("type") != "object":
        raise ValueError("The root JSON Schema type must be object for strict output.")
    resolved_name = name.strip() or "schema"
    if not _NAME_PATTERN.fullmatch(resolved_name):
        raise ValueError("Schema name must be 1-64 letters, numbers, underscores, or hyphens.")
    _validate_schema_node(value, "root")
    return JSONSchemaDocument(resolved_name, value)


def load_json_schema(source: str, source_kind: str, name: str = "") -> JSONSchemaDocument:
    """Load a schema from inline JSON or a local file path."""

    if source_kind == "file":
        path = Path(source).expanduser()
        try:
            raw = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ValueError(f"Unable to read JSON Schema file: {path}") from exc
        resolved_name = name.strip() or path.stem
    elif source_kind == "inline":
        raw = source
        resolved_name = name.strip() or "schema"
    else:
        raise ValueError("Schema source must be file or inline.")
    return parse_json_schema(raw, resolved_name)


__all__ = ["JSONSchemaDocument", "load_json_schema", "parse_json_schema"]
