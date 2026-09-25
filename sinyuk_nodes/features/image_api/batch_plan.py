"""Pure positional image grouping and Base-major prompt variants."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import TypeGuard

from .references import ReferenceSet, normalize_reference


def _is_object_list(value: object) -> TypeGuard[list[object]]:
    return isinstance(value, list)


def _is_reference_mapping(value: object) -> TypeGuard[Mapping[str, object]]:
    return isinstance(value, Mapping)


@dataclass(frozen=True)
class TaskSpec:
    task_index: int
    base_index: int
    prompt_index: int


@dataclass(frozen=True)
class BatchPlan:
    columns: tuple[ReferenceSet, ...]
    prompts: tuple[str, ...]
    prompt_source: str
    base_count: int

    @property
    def total(self) -> int:
        return self.base_count * len(self.prompts)

    def tasks(self) -> Iterator[TaskSpec]:
        for base in range(1, self.base_count + 1):
            for prompt in range(1, len(self.prompts) + 1):
                yield TaskSpec((base - 1) * len(self.prompts) + prompt, base, prompt)

    def input_index(self, column: int, base: int) -> int:
        return 0 if len(self.columns[column].images) == 1 else base - 1


def prompt_variants(prompt: object, prompts: object) -> tuple[tuple[str, ...], str]:
    if prompts is None:
        prompts = []
    if not _is_object_list(prompts):
        raise ValueError("Prompts must be a STRING list or one wrapped prompt array.")
    prompt_values: Sequence[object] = prompts
    if len(prompt_values) == 1 and _is_object_list(prompt_values[0]):
        prompt_values = prompt_values[0]
    if any(not isinstance(p, str) for p in prompt_values):
        raise ValueError("Every Prompt Variant must be a string.")
    string_prompts = tuple(p for p in prompt_values if isinstance(p, str))
    if prompt_values:
        return string_prompts, "prompts"
    if not _is_object_list(prompt) or len(prompt) != 1 or not isinstance(prompt[0], str):
        raise ValueError("Prompt must contain exactly one string when Prompts is empty.")
    return (prompt[0],), "prompt"


def plan_batch(
    references: object,
    prompt: object,
    prompts: object = None,
    *,
    local_limit: int = 10,
    model_limit: int | None = None,
    check_cancel: Callable[[], None] = lambda: None,
) -> BatchPlan:
    if not _is_reference_mapping(references) or not references:
        raise ValueError("Connect at least Reference 1.")
    names: list[tuple[int, str]] = []
    raw_names: list[object] = list(references.keys())
    for raw_name in raw_names:
        if not isinstance(raw_name, str):
            raise ValueError("Invalid Reference socket name.")
        name = raw_name
        match = re.fullmatch(r"reference_([1-9][0-9]*)", name)
        if not match:
            raise ValueError("Invalid Reference socket name.")
        names.append((int(match[1]), name))
    names.sort()
    if [i for i, _ in names] != list(range(1, len(names) + 1)):
        raise ValueError(
            "Reference sockets must be consecutive from Reference 1; a connection is missing."
        )
    if len(names) > local_limit:
        raise ValueError(
            f"Reference count exceeds local safety limit ({local_limit}); not a provider limit."
        )
    if model_limit is not None and len(names) > model_limit:
        raise ValueError(
            f"Reference count exceeds the configured official model limit ({model_limit})."
        )
    columns = tuple(normalize_reference(references[name], check_cancel) for _, name in names)
    count = max(len(c.images) for c in columns)
    for index, column in enumerate(columns, 1):
        if len(column.images) not in (1, count):
            raise ValueError(
                f"reference_{index} contains {len(column.images)} images; expected 1 or {count}."
            )
    variants, source = prompt_variants(prompt, prompts)
    return BatchPlan(columns, variants, source, count)


def validate_options(concurrency: object, prefix: object, api_key: str) -> None:
    if type(concurrency) is not int or not 2 <= concurrency <= 10:
        raise ValueError("Max concurrency must be an integer from 2 to 10.")
    if (
        not isinstance(prefix, str)
        or not prefix.strip()
        or prefix != prefix.strip()
        or len(prefix) > 100
        or any(ord(c) < 32 for c in prefix)
        or any(c in prefix for c in '/\\<>:"|?*')
        or prefix in {".", ".."}
        or prefix.endswith(".")
        or api_key in prefix
    ):
        raise ValueError("Output prefix must be a safe filename fragment without credentials.")
