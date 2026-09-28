"""Infer a standard image ratio and calculate a generation resolution."""

# ComfyUI V3 stubs model execute as ``**kwargs`` while runtime dispatch accepts
# typed node signatures. This suppression is limited to the node adapter.
# pyright: reportIncompatibleMethodOverride=false

from __future__ import annotations

import logging
import math
from math import gcd

import torch
from sinyuk_nodes.compat.comfy import io

LOGGER = logging.getLogger(__name__)

STANDARD_ASPECT_RATIOS: dict[str, tuple[int, int]] = {
    "1:1": (1, 1),
    "4:5": (4, 5),
    "3:4": (3, 4),
    "2:3": (2, 3),
    "9:16": (9, 16),
    "1:3": (1, 3),
    "1:2": (1, 2),
    "9:21": (9, 21),
    "5:4": (5, 4),
    "4:3": (4, 3),
    "3:2": (3, 2),
    "16:9": (16, 9),
    "21:9": (21, 9),
    "2:1": (2, 1),
    "3:1": (3, 1),
}

SNAP_WARNING_THRESHOLD = 0.20


def infer_aspect_ratio(source_width: int, source_height: int) -> tuple[str, float]:
    """Return the closest standard ratio and its relative snap error."""
    if source_width <= 0 or source_height <= 0:
        raise ValueError("Source width and height must both be positive.")

    input_ratio = source_width / source_height
    name, (ratio_width, ratio_height) = min(
        STANDARD_ASPECT_RATIOS.items(),
        key=lambda item: abs(math.log(input_ratio / (item[1][0] / item[1][1]))),
    )
    best_distance = abs(math.log(input_ratio / (ratio_width / ratio_height)))
    return name, math.exp(best_distance) - 1


def exact_aspect_ratio(source_width: int, source_height: int) -> str:
    """Return the exact input ratio reduced to its smallest positive integers."""
    if source_width <= 0 or source_height <= 0:
        raise ValueError("Source width and height must both be positive.")
    divisor = gcd(source_width, source_height)
    return f"{source_width // divisor}:{source_height // divisor}"


def resolution_for_ratio(aspect_ratio: str, megapixels: float, step: int) -> tuple[int, int]:
    """Apply ComfyUI's ResolutionSelector calculation to a standard ratio."""
    if megapixels <= 0:
        raise ValueError("Megapixels must be greater than zero.")
    if step <= 0:
        raise ValueError("Step must be greater than zero.")

    ratio_width, ratio_height = STANDARD_ASPECT_RATIOS[aspect_ratio]
    total_pixels = megapixels * 1024 * 1024
    scale = math.sqrt(total_pixels / (ratio_width * ratio_height))
    width = round(ratio_width * scale / step) * step
    height = round(ratio_height * scale / step) * step
    return max(width, step), max(height, step)


class AspectRatioResolutionNode(io.ComfyNode):
    """Read image ratios and calculate output dimensions using the selected mode."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Sinyuk.AspectRatioResolution",
            display_name="Image Ratio",
            category="Sinyuk/Utilities",
            description=(
                "Infer nearest and exact image ratios, with optional megapixel-based "
                "resolution calculation."
            ),
            inputs=[
                io.Image.Input(
                    "image",
                    display_name="Image",
                    tooltip="Image used to read the source dimensions.",
                ),
                io.Combo.Input(
                    "mode",
                    options=["Original", "Total Size"],
                    default="Original",
                    display_name="Mode",
                    tooltip=(
                        "Original outputs the source dimensions. Total Size calculates dimensions "
                        "from Megapixels and Step."
                    ),
                ),
                io.Float.Input(
                    "megapixels",
                    default=1.0,
                    min=0.1,
                    max=16.0,
                    step=0.1,
                    advanced=True,
                    tooltip=(
                        "Used only in Total Size mode; target total size in megapixels. "
                        "Ignored in Original mode."
                    ),
                ),
                io.Int.Input(
                    "step",
                    display_name="Step",
                    default=32,
                    min=8,
                    max=128,
                    step=8,
                    advanced=True,
                    tooltip=(
                        "Used only in Total Size mode; round calculated dimensions to this step. "
                        "Ignored in Original mode."
                    ),
                ),
            ],
            outputs=[
                io.String.Output(
                    "image_ratio",
                    display_name="Image Ratio",
                    tooltip="Nearest product standard ratio; suitable as a generation preset.",
                ),
                io.String.Output(
                    "exact_ratio",
                    display_name="Exact Ratio",
                    tooltip="Exact input ratio reduced to smallest integers; e.g. 2100x900 → 7:3.",
                ),
                io.Int.Output(
                    "width",
                    tooltip="Source width in Original mode; calculated width in Total Size mode.",
                ),
                io.Int.Output(
                    "height",
                    tooltip="Source height in Original mode; calculated height in Total Size mode.",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        image: torch.Tensor | None = None,
        mode: str = "Original",
        megapixels: float = 1.0,
        step: int = 32,
    ) -> io.NodeOutput:
        if image is None:
            raise ValueError("Connect an image to read its dimensions.")
        if image.ndim < 3:
            raise ValueError("Input image must have height and width dimensions.")
        source_height = int(image.shape[-3])
        source_width = int(image.shape[-2])

        aspect_ratio, snap_error = infer_aspect_ratio(source_width, source_height)
        if snap_error > SNAP_WARNING_THRESHOLD:
            LOGGER.warning(
                "Input aspect ratio %s:%s is far from standard ratio %s (snap error %.1f%%).",
                source_width,
                source_height,
                aspect_ratio,
                snap_error * 100,
            )

        if mode == "Original":
            output_width, output_height = source_width, source_height
        elif mode == "Total Size":
            output_width, output_height = resolution_for_ratio(aspect_ratio, megapixels, step)
        else:
            raise ValueError("Mode must be Original or Total Size.")
        return io.NodeOutput(
            aspect_ratio,
            exact_aspect_ratio(source_width, source_height),
            output_width,
            output_height,
        )


__all__ = [
    "STANDARD_ASPECT_RATIOS",
    "SNAP_WARNING_THRESHOLD",
    "AspectRatioResolutionNode",
    "exact_aspect_ratio",
    "infer_aspect_ratio",
]
