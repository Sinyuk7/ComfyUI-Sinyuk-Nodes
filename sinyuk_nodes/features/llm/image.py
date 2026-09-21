"""Preprocess ComfyUI images before they are uploaded to an LLM API."""

from __future__ import annotations

import base64
import hashlib
from dataclasses import dataclass
from io import BytesIO
from typing import Literal

import numpy as np
import torch
from PIL import Image

ImageDetail = Literal["auto", "low", "high", "original"]

# GPT-5.6-oriented client preprocessing limits.
_PATCH_SIZE = 32
_DETAIL_LIMITS: dict[str, tuple[int, int | None]] = {
    "low": (512, None),
    "high": (2048, 2500),
}
_RANGE_EPSILON = 1e-6


@dataclass(frozen=True)
class EncodedImage:
    """One LLM-upload image and its encoded payload fingerprint."""

    encoded_bytes: bytes
    base64_data_url: str
    sha256: str
    mime_type: str
    width: int
    height: int


def _to_pil(image: torch.Tensor, index: int) -> Image.Image:
    if image.ndim != 3 or image.shape[-1] not in {1, 3, 4}:
        raise ValueError(f"Input image {index} must have shape [H, W, 1|3|4].")
    if image.shape[0] < 1 or image.shape[1] < 1:
        raise ValueError(f"Input image {index} must have non-empty dimensions.")
    if not image.is_floating_point():
        raise ValueError(f"Input image {index} must contain finite floating-point values.")

    pixels = image.detach().to(device="cpu", dtype=torch.float32)
    if not bool(torch.isfinite(pixels).all()):
        raise ValueError(f"Input image {index} must contain finite floating-point values.")
    minimum = float(pixels.min().item())
    maximum = float(pixels.max().item())
    if minimum < -_RANGE_EPSILON or maximum > 1 + _RANGE_EPSILON:
        raise ValueError(f"Input image {index} must contain values in the range [0, 1].")

    pixel_array = (pixels.clamp(0, 1).numpy() * 255).round().astype(np.uint8)
    if pixel_array.shape[-1] == 1:
        return Image.fromarray(pixel_array[..., 0], mode="L").convert("RGB")
    if pixel_array.shape[-1] == 4:
        return Image.fromarray(pixel_array, mode="RGBA")
    return Image.fromarray(pixel_array, mode="RGB")


def _patches(width: int, height: int) -> int:
    return ((width + _PATCH_SIZE - 1) // _PATCH_SIZE) * ((height + _PATCH_SIZE - 1) // _PATCH_SIZE)


def _resize_for_detail(image: Image.Image, detail: ImageDetail) -> Image.Image:
    limits = _DETAIL_LIMITS.get(detail)
    if limits is None:
        return image

    max_dimension, patch_budget = limits
    scale = min(1.0, max_dimension / max(image.width, image.height))
    limited_width = max(1, int(image.width * scale))
    limited_height = max(1, int(image.height * scale))

    if patch_budget is not None:
        patches = _patches(limited_width, limited_height)
        if patches > patch_budget:
            scale *= (patch_budget / patches) ** 0.5

    width = max(1, int(image.width * scale))
    height = max(1, int(image.height * scale))
    if patch_budget is not None:
        while _patches(width, height) > patch_budget:
            scale *= 0.999
            width = max(1, int(image.width * scale))
            height = max(1, int(image.height * scale))

    if (width, height) == image.size:
        return image
    return image.resize((width, height), Image.Resampling.LANCZOS)


def _detail(value: str) -> ImageDetail:
    if value == "auto":
        return "auto"
    if value == "low":
        return "low"
    if value == "high":
        return "high"
    if value == "original":
        return "original"
    raise ValueError("Image detail must be one of: auto, low, high, original.")


def encode_image(image: torch.Tensor, detail: str = "high", index: int = 1) -> EncodedImage:
    """Encode one ComfyUI image for an LLM request without writing to disk."""

    selected_detail = _detail(detail)
    pil_image = _resize_for_detail(_to_pil(image, index), selected_detail)
    has_alpha = pil_image.mode == "RGBA"

    buffer = BytesIO()
    if has_alpha:
        mime_type = "image/png"
        pil_image.save(buffer, format="PNG", optimize=False, compress_level=6)
    else:
        mime_type = "image/jpeg"
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")
        pil_image.save(buffer, format="JPEG", quality=90, optimize=True, progressive=False)

    data = buffer.getvalue()
    return EncodedImage(
        encoded_bytes=data,
        base64_data_url=f"data:{mime_type};base64," + base64.b64encode(data).decode("ascii"),
        sha256=hashlib.sha256(data).hexdigest(),
        mime_type=mime_type,
        width=pil_image.width,
        height=pil_image.height,
    )


def encode_images(images: torch.Tensor | None, detail: str = "high") -> tuple[EncodedImage, ...]:
    """Encode a ComfyUI IMAGE batch for LLM upload."""

    if images is None:
        return ()
    if images.ndim != 4 or images.shape[-1] not in {1, 3, 4} or images.shape[0] < 1:
        raise ValueError("Images must have shape [B, H, W, 1|3|4].")
    selected_detail = _detail(detail)
    return tuple(
        encode_image(image, selected_detail, index) for index, image in enumerate(images, 1)
    )


__all__ = ["EncodedImage", "ImageDetail", "encode_image", "encode_images"]
