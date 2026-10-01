"""Resolution mappings and validation for API image models."""

from __future__ import annotations

from collections.abc import Mapping
from math import gcd, isqrt
from types import MappingProxyType
from typing import Final

GOOGLE_RATIOS: Final[tuple[str, ...]] = (
    "1:1",
    "1:4",
    "1:8",
    "2:3",
    "3:2",
    "3:4",
    "4:1",
    "4:3",
    "4:5",
    "5:4",
    "8:1",
    "9:16",
    "16:9",
    "21:9",
)
GOOGLE_STANDARD_RATIOS: Final[tuple[str, ...]] = (
    "1:1",
    "2:3",
    "3:2",
    "3:4",
    "4:3",
    "4:5",
    "5:4",
    "9:16",
    "16:9",
    "21:9",
)
GOOGLE_TIERS: Final[tuple[str, ...]] = ("0.5K", "1K", "2K", "4K")
GOOGLE_PRO_TIERS: Final[tuple[str, ...]] = ("1K", "2K", "4K")
MODEL_NAMES: Final[tuple[str, ...]] = (
    "Nano Banana 2",
    "Nano Banana Pro",
    "GPT Image 2.5",
)

_GOOGLE_BASE: Mapping[str, tuple[int, int]] = {
    "1:1": (1024, 1024),
    "2:3": (848, 1264),
    "3:2": (1264, 848),
    "3:4": (896, 1200),
    "4:3": (1200, 896),
    "4:5": (928, 1152),
    "5:4": (1152, 928),
    "9:16": (768, 1376),
    "16:9": (1376, 768),
    "21:9": (1584, 672),
}
_GOOGLE_EXTREME: Mapping[str, tuple[int, int]] = {
    "1:4": (512, 2048),
    "1:8": (384, 3072),
    "4:1": (2048, 512),
    "8:1": (3072, 384),
}


def _scaled(mapping: Mapping[str, tuple[int, int]], factor: float) -> dict[str, tuple[int, int]]:
    return {
        ratio: (round(width * factor), round(height * factor))
        for ratio, (width, height) in mapping.items()
    }


GOOGLE_RESOLUTIONS: Final[Mapping[str, Mapping[str, tuple[int, int]]]] = MappingProxyType(
    {
        "0.5K": MappingProxyType({**_scaled(_GOOGLE_BASE, 0.5), **_scaled(_GOOGLE_EXTREME, 0.5)}),
        "1K": MappingProxyType({**_GOOGLE_BASE, **_GOOGLE_EXTREME}),
        "2K": MappingProxyType(_scaled({**_GOOGLE_BASE, **_GOOGLE_EXTREME}, 2)),
        "4K": MappingProxyType(_scaled({**_GOOGLE_BASE, **_GOOGLE_EXTREME}, 4)),
    }
)

_GPT_COMMON_RESOLUTIONS: Final[dict[str, tuple[int, int]]] = {
    "1K:1:1": (1024, 1024),
    "1K:5:4": (1120, 896),
    "1K:4:3": (1152, 864),
    "1K:3:2": (1248, 832),
    "1K:16:9": (1280, 720),
    "1K:21:9": (1456, 624),
    "1K:4:5": (896, 1120),
    "1K:3:4": (864, 1152),
    "1K:2:3": (832, 1248),
    "1K:9:16": (720, 1280),
    "2K:1:1": (2048, 2048),
    "2K:5:4": (2240, 1792),
    "2K:4:3": (2304, 1728),
    "2K:3:2": (2496, 1664),
    "2K:16:9": (2560, 1440),
    "2K:21:9": (3024, 1296),
    "2K:4:5": (1792, 2240),
    "2K:3:4": (1728, 2304),
    "2K:2:3": (1664, 2496),
    "2K:9:16": (1440, 2560),
}


def _max_gpt_dimensions(aspect_ratio: str) -> tuple[int, int]:
    """Find the largest exact-ratio GPT size allowed by the API limits."""
    ratio_width, ratio_height = (int(value) for value in aspect_ratio.split(":"))
    divisor = gcd(ratio_width, ratio_height)
    ratio_width //= divisor
    ratio_height //= divisor
    unit = 16
    max_scale = min(3840 // (ratio_width * unit), 3840 // (ratio_height * unit))
    max_scale = min(max_scale, isqrt(8_294_400 // (ratio_width * ratio_height * unit**2)))
    width, height = ratio_width * unit * max_scale, ratio_height * unit * max_scale
    if width * height < 655_360:
        raise ValueError(f"No legal GPT Image 2.5 size exists for aspect ratio {aspect_ratio}.")
    return width, height


_GPT_RESOLUTIONS: Final[Mapping[str, tuple[int, int]]] = MappingProxyType(
    {
        **_GPT_COMMON_RESOLUTIONS,
        **{f"4K:{ratio}": _max_gpt_dimensions(ratio) for ratio in GOOGLE_STANDARD_RATIOS},
    }
)


def _validate_dimensions(width: int, height: int) -> tuple[int, int]:
    if width <= 0 or height <= 0:
        raise ValueError("GPT Image 2.5 width and height must be positive.")
    if width % 16 or height % 16:
        raise ValueError("GPT Image 2.5 width and height must be multiples of 16.")
    if max(width, height) > 3840:
        raise ValueError("GPT Image 2.5 width and height must each be at most 3840.")
    if max(width, height) / min(width, height) > 3:
        raise ValueError("GPT Image 2.5 aspect ratio must be at most 3:1.")
    pixels = width * height
    if pixels < 655_360 or pixels > 8_294_400:
        raise ValueError("GPT Image 2.5 total pixels must be between 655,360 and 8,294,400.")
    return width, height


def resolution_for_api_model(
    model: str,
    aspect_ratio: str,
    resolution: str,
) -> tuple[int, int]:
    """Return dimensions for a model, ratio, and resolution tier."""
    if model == "GPT Image 2.5":
        if aspect_ratio not in GOOGLE_STANDARD_RATIOS:
            raise ValueError("GPT Image 2.5 supports standard aspect ratios only.")
        try:
            dimensions = _GPT_RESOLUTIONS[f"{resolution}:{aspect_ratio}"]
        except KeyError as error:
            raise ValueError(
                f"GPT Image 2.5 does not support resolution tier {resolution}."
            ) from error
        return _validate_dimensions(*dimensions)
    if model == "Nano Banana 2":
        allowed_ratios, allowed_tiers = GOOGLE_RATIOS, GOOGLE_TIERS
    elif model == "Nano Banana Pro":
        allowed_ratios, allowed_tiers = GOOGLE_STANDARD_RATIOS, GOOGLE_PRO_TIERS
    else:
        raise ValueError(f"Unsupported API image model: {model}.")
    if aspect_ratio not in allowed_ratios:
        raise ValueError(f"{model} does not support aspect ratio {aspect_ratio}.")
    if resolution not in allowed_tiers:
        raise ValueError(f"{model} does not support resolution tier {resolution}.")
    return GOOGLE_RESOLUTIONS[resolution][aspect_ratio]


__all__ = [
    "GOOGLE_RATIOS",
    "GOOGLE_STANDARD_RATIOS",
    "GOOGLE_TIERS",
    "GOOGLE_PRO_TIERS",
    "MODEL_NAMES",
    "GOOGLE_RESOLUTIONS",
    "resolution_for_api_model",
]
