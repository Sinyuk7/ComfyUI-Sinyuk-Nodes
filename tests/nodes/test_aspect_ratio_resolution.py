from __future__ import annotations

import pytest
import torch
from sinyuk_nodes.nodes.aspect_ratio_resolution import (
    AspectRatioResolutionNode,
    exact_aspect_ratio,
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
        (100, 1000, "1:3"),
        (1000, 100, "3:1"),
    ],
)
def test_infers_nearest_standard_ratio(width: int, height: int, expected: str) -> None:
    assert infer_aspect_ratio(width, height)[0] == expected


def test_resolution_node_original_mode_outputs_source_dimensions() -> None:
    schema = AspectRatioResolutionNode.define_schema()
    schema.validate()
    assert schema.display_name == "Image Ratio"

    output = AspectRatioResolutionNode.execute(image=torch.zeros((1, 3100, 2100, 3)))

    assert output.result == ("2:3", "21:31", 2100, 3100)


@pytest.mark.parametrize(
    ("width", "height", "expected"),
    [
        (1920, 1080, ("16:9", "16:9", 1920, 1080)),
        (2100, 900, ("21:9", "7:3", 2100, 900)),
        (1920, 1088, ("16:9", "30:17", 1920, 1088)),
    ],
)
def test_original_mode_preserves_source_dimensions_and_ratios(width, height, expected) -> None:
    output = AspectRatioResolutionNode.execute(image=torch.zeros((1, height, width, 3)))

    assert output.result == expected


def test_outputs_nearest_and_exact_ratio_independently() -> None:
    output = AspectRatioResolutionNode.execute(image=torch.zeros((1, 900, 2100, 3)))

    assert output.result[:2] == ("21:9", "7:3")


def test_total_size_mode_calculates_dimensions() -> None:
    output = AspectRatioResolutionNode.execute(
        image=torch.zeros((1, 3100, 2100, 3)), mode="Total Size"
    )

    assert output.result == ("2:3", "21:31", 832, 1248)


def test_exact_ratio_reduces_arbitrary_positive_dimensions() -> None:
    assert exact_aspect_ratio(30, 17) == "30:17"
    with pytest.raises(ValueError, match="positive"):
        exact_aspect_ratio(0, 9)


def test_invalid_source_dimensions_fail_without_image() -> None:
    with pytest.raises(ValueError, match="Connect an image"):
        AspectRatioResolutionNode.execute()


def test_schema_has_no_legacy_source_or_dimension_inputs() -> None:
    input_ids = [item.id for item in AspectRatioResolutionNode.define_schema().inputs]

    assert input_ids == ["image", "mode", "megapixels", "step"]
