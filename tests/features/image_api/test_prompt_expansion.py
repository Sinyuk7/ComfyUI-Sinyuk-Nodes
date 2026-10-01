from __future__ import annotations

import pytest
from sinyuk_nodes.features.image_api.prompt_expansion import expand_prompts


def test_repeat_preserves_generation_inputs() -> None:
    assert expand_prompts("draw", "Repeat", 3, "pose", None) == ("draw", "draw", "draw")


def test_variants_replace_exact_placeholder_and_preserve_multiline_text() -> None:
    values = ["standing\nfront view", "sitting [other]"]
    assert expand_prompts("A [pose] subject", "Variants", 1, "pose", values) == (
        "A standing\nfront view subject",
        "A sitting [other] subject",
    )


def test_variants_replace_every_occurrence() -> None:
    assert expand_prompts("[pose] / [pose]", "Variants", 1, "pose", ["standing"]) == (
        "standing / standing",
    )


@pytest.mark.parametrize(
    ("prompt", "mode", "count", "placeholder", "variants", "message"),
    [
        ("draw", "Repeat", 0, "pose", None, "Generation Count"),
        ("draw", "Variants", 1, "bad-name", ["x"], "Placeholder"),
        ("draw", "Variants", 1, "pose", [], "Variants"),
        ("draw", "Variants", 1, "pose", ["x"], "does not contain"),
        ("draw [pose]", "Variants", 1, "pose", ["   "], "Variants"),
    ],
)
def test_prompt_expansion_rejects_invalid_inputs(
    prompt: str,
    mode: str,
    count: int,
    placeholder: str,
    variants: list[str] | None,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        expand_prompts(prompt, mode, count, placeholder, variants)
