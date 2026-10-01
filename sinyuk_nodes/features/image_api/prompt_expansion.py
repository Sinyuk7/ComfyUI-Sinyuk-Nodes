"""Validation and expansion for Image API generation prompts."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import TypeGuard

PLACEHOLDER_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


def _is_object_list(value: object) -> TypeGuard[list[object]]:
    return isinstance(value, list)


def _string_list(value: object, name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not _is_object_list(value):
        raise ValueError(f"{name} must be a STRING list.")
    values: Sequence[object] = value
    if len(values) == 1 and _is_object_list(values[0]):
        values = values[0]
    if any(not item or not isinstance(item, str) or not item.strip() for item in values):
        raise ValueError(f"Every {name} item must be a nonempty string.")
    return tuple(item for item in values if isinstance(item, str))


def expand_prompts(
    prompt: object,
    mode: object,
    generation_count: object,
    placeholder: object,
    variants: object,
) -> tuple[str, ...]:
    """Return final prompts while preserving multiline variant text verbatim."""
    if not isinstance(prompt, str):
        raise ValueError("Prompt must be a string.")
    if not isinstance(mode, str) or mode not in {"Repeat", "Variants"}:
        raise ValueError("Generation Mode must be Repeat or Variants.")
    if type(generation_count) is not int or not 1 <= generation_count <= 10:
        raise ValueError("Generation Count must be an integer from 1 to 10.")
    if mode == "Repeat":
        return (prompt,) * generation_count

    if not isinstance(placeholder, str) or not PLACEHOLDER_NAME.fullmatch(placeholder):
        raise ValueError("Placeholder must match [A-Za-z_][A-Za-z0-9_]*.")
    values = _string_list(variants, "Variants")
    if not 1 <= len(values) <= 10:
        raise ValueError("Variants must contain from 1 to 10 strings.")
    target = f"[{placeholder}]"
    if target not in prompt:
        raise ValueError(f"Prompt does not contain the placeholder {target}.")
    return tuple(prompt.replace(target, value) for value in values)
