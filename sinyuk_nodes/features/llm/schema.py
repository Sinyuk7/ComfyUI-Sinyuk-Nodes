"""Load and validate schemas for Structured Outputs."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TypeGuard

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError, ValidationError

_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def _is_object(value: object) -> TypeGuard[dict[str, object]]:
    return isinstance(value, dict)


@dataclass(frozen=True)
class JSONSchemaDocument:
    """A named JSON Schema ready to be sent to an LLM API."""

    name: str
    schema: dict[str, object]


def parse_json_schema(raw: str, name: str = "") -> JSONSchemaDocument:
    """Parse and validate one strict Structured Outputs schema."""

    try:
        value: object = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("JSON Schema must be valid JSON.") from exc
    if not _is_object(value):
        raise ValueError("JSON Schema must be a JSON object.")
    if value.get("type") != "object":
        raise ValueError("The root JSON Schema type must be object for strict output.")
    resolved_name = name.strip() or "schema"
    if not _NAME_PATTERN.fullmatch(resolved_name):
        raise ValueError("Schema name must be 1-64 letters, numbers, underscores, or hyphens.")
    try:
        Draft202012Validator.check_schema(value)
    except SchemaError as exc:
        raise ValueError("JSON Schema is invalid.") from exc
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


def format_validation_error(error: ValidationError) -> str:
    """Format a jsonschema error for a ComfyUI node message."""

    path = "".join(
        f"[{part}]" if isinstance(part, int) else f".{part}" for part in error.absolute_path
    ).lstrip(".")
    return (
        "LLM response does not match preset schema:\n"
        f"path: {path or '<root>'}\n"
        f"error: {error.message}"
    )


def validate_json(value: object, document: JSONSchemaDocument) -> None:
    """Validate decoded JSON with the jsonschema library."""

    validator = Draft202012Validator(document.schema)
    error = next(validator.iter_errors(value), None)
    if error is not None:
        raise ValueError(format_validation_error(error)) from error


__all__ = [
    "JSONSchemaDocument",
    "load_json_schema",
    "parse_json_schema",
    "format_validation_error",
    "validate_json",
]
