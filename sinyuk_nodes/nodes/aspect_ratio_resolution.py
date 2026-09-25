"""Infer a standard image ratio and calculate a generation resolution."""

# ComfyUI V3 stubs model execute as ``**kwargs`` while runtime dispatch accepts
# typed node signatures. This suppression is limited to the node adapter.
# pyright: reportIncompatibleMethodOverride=false

from __future__ import annotations

import logging
import math

import torch
from sinyuk_nodes.compat.comfy import io

LOGGER = logging.getLogger(__name__)

ASPECT_RATIOS: dict[str, tuple[int, int]] = {
    "1:1": (1, 1),
    "4:5": (4, 5),
    "3:4": (3, 4),
    "2:3": (2, 3),
    "9:16": (9, 16),
    "5:4": (5, 4),
    "4:3": (4, 3),
    "3:2": (3, 2),
    "16:9": (16, 9),
    "21:9": (21, 9),
}


def infer_aspect_ratio(source_width: int, source_height: int) -> tuple[str, float]:
    """Return the closest standard ratio and its relative snap error."""
    if source_width <= 0 or source_height <= 0:
        raise ValueError("Source width and height must both be positive.")

    input_ratio = source_width / source_height
    name, (ratio_width, ratio_height) = min(
        ASPECT_RATIOS.items(),
        key=lambda item: abs(math.log(input_ratio / (item[1][0] / item[1][1]))),
    )
    best_distance = abs(math.log(input_ratio / (ratio_width / ratio_height)))
    return name, math.exp(best_distance) - 1


def resolution_for_ratio(aspect_ratio: str, megapixels: float, multiple: int) -> tuple[int, int]:
    """Apply ComfyUI's ResolutionSelector calculation to a standard ratio."""
    if megapixels <= 0:
        raise ValueError("Megapixels must be greater than zero.")
    if multiple <= 0:
        raise ValueError("Multiple must be greater than zero.")

    ratio_width, ratio_height = ASPECT_RATIOS[aspect_ratio]
    total_pixels = megapixels * 1024 * 1024
    scale = math.sqrt(total_pixels / (ratio_width * ratio_height))
    width = round(ratio_width * scale / multiple) * multiple
    height = round(ratio_height * scale / multiple) * multiple
    return max(width, multiple), max(height, multiple)


class AspectRatioResolutionNode(io.ComfyNode):
    """Infer a standard ratio from an image or source dimensions."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Sinyuk.AspectRatioResolution",
            display_name="Aspect Ratio Resolution",
            category="Sinyuk/Utilities",
            description=(
                "Infer the nearest standard aspect ratio and calculate a generation resolution."
            ),
            inputs=[
                io.Image.Input(
                    "image",
                    display_name="Image",
                    tooltip="Optional source image. When connected, its dimensions take priority.",
                    optional=True,
                ),
                io.Int.Input(
                    "width",
                    default=0,
                    min=0,
                    tooltip="Source width, used when no image is connected.",
                    optional=True,
                ),
                io.Int.Input(
                    "height",
                    default=0,
                    min=0,
                    tooltip="Source height, used when no image is connected.",
                    optional=True,
                ),
                io.Float.Input(
                    "megapixels",
                    default=1.0,
                    min=0.1,
                    max=16.0,
                    step=0.1,
                    tooltip="Target output megapixels, matching ComfyUI ResolutionSelector.",
                ),
                io.Int.Input(
                    "multiple",
                    default=32,
                    min=8,
                    max=128,
                    step=8,
                    tooltip="Round output width and height to this multiple.",
                ),
            ],
            outputs=[
                io.Int.Output("width", tooltip="Calculated output width."),
                io.Int.Output("height", tooltip="Calculated output height."),
                io.String.Output("aspect_ratio", tooltip="Inferred standard aspect ratio."),
            ],
        )

    @classmethod
    def execute(
        cls,
        image: torch.Tensor | None = None,
        width: int = 0,
        height: int = 0,
        megapixels: float = 1.0,
        multiple: int = 32,
    ) -> io.NodeOutput:
        if image is not None:
            if image.ndim < 3:
                raise ValueError("Input image must have height and width dimensions.")
            source_height = int(image.shape[-3])
            source_width = int(image.shape[-2])
        else:
            source_width = width
            source_height = height

        aspect_ratio, snap_error = infer_aspect_ratio(source_width, source_height)
        if snap_error > 0.20:
            LOGGER.warning(
                "Input aspect ratio %s:%s is far from standard ratio %s (snap error %.1f%%).",
                source_width,
                source_height,
                aspect_ratio,
                snap_error * 100,
            )

        output_width, output_height = resolution_for_ratio(aspect_ratio, megapixels, multiple)
        return io.NodeOutput(output_width, output_height, aspect_ratio)


__all__ = ["ASPECT_RATIOS", "AspectRatioResolutionNode", "infer_aspect_ratio"]
