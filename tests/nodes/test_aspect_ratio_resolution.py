from __future__ import annotations

import pytest
from sinyuk_nodes.nodes.aspect_ratio_resolution import (
    AspectRatioResolutionNode,
    infer_aspect_ratio,
)


@pytest.mark.parametrize(
    ("width", "height", "expected"),
    [
        (2100, 3100, "2:3"),
        (3000, 3000, "1:1"),
        (1080, 1920, "9:16"),
        (1920, 1080, "16:9"),
        (4000, 5000, "4:5"),
        (5000, 4000, "5:4"),
        (6000, 4000, "3:2"),
        (100, 1000, "9:16"),
        (1000, 100, "21:9"),
    ],
)
def test_infers_nearest_standard_ratio(width: int, height: int, expected: str) -> None:
    assert infer_aspect_ratio(width, height)[0] == expected


def test_resolution_node_uses_default_multiple_of_32() -> None:
    schema = AspectRatioResolutionNode.define_schema()
    schema.validate()

    output = AspectRatioResolutionNode.execute(width=2100, height=3100)

    assert output.result == (832, 1248, "2:3")


def test_invalid_source_dimensions_fail_without_image() -> None:
    with pytest.raises(ValueError, match="positive"):
        AspectRatioResolutionNode.execute(width=0, height=0)
