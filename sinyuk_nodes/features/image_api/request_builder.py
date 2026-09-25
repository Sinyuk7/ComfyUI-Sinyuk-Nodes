"""Pure request construction; no authentication or network side effects."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TypeGuard

from .config import Config
from .runninghub_config import RunningHubCatalog


def _is_parameter_mapping(value: object) -> TypeGuard[Mapping[str, object]]:
    return isinstance(value, Mapping)


def _is_string_sequence(value: object) -> TypeGuard[Sequence[str]]:
    if not isinstance(value, Sequence) or isinstance(value, str):
        return False
    values: Sequence[object] = value
    return all(isinstance(item, str) for item in values)


def normalize_key(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("API key must not be empty.")
    key = value.strip()
    if any(ord(c) < 32 or ord(c) == 127 for c in key):
        raise ValueError("API key contains invalid control characters.")
    return key


def build_request(
    model: str,
    prompt: object,
    parameters: object,
    encoded_images: object,
    config: Config,
) -> dict[str, object]:
    profile = config.profile(model)
    if not isinstance(prompt, str):
        raise ValueError("Exactly one string prompt is required.")
    if not _is_parameter_mapping(parameters):
        raise ValueError("Model parameters must be an object.")
    if not _is_string_sequence(encoded_images) or not 1 <= len(encoded_images) <= 10:
        raise ValueError("A request requires 1 to 10 encoded reference images.")
    if any(not image for image in encoded_images):
        raise ValueError("A request requires 1 to 10 encoded reference images.")
    request: dict[str, object] = {
        "model": model,
        "prompt": prompt,
        "images": list(encoded_images),
        "replyType": "async",
    }
    for name, rule in profile.parameters.items():
        value = parameters.get(name)
        if not isinstance(value, str) or value not in rule.values:
            raise ValueError(
                f"Invalid or missing {name} for the selected model; update the workflow explicitly."
            )
        request[name] = value
    return request


def build_runninghub_request(
    model: str,
    prompt: object,
    parameters: object,
    image_urls: object,
    catalog: RunningHubCatalog,
) -> dict[str, object]:
    profile = catalog.profile(model)
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Exactly one nonempty string prompt is required.")
    if not _is_parameter_mapping(parameters):
        raise ValueError("Model parameters must be an object.")
    if not _is_string_sequence(image_urls) or not 1 <= len(image_urls) <= 10:
        raise ValueError("RunningHub requires 1 to 10 reference image URLs.")
    if any(not url for url in image_urls):
        raise ValueError("RunningHub image URLs must be nonempty strings.")
    request: dict[str, object] = {
        "prompt": prompt,
        "imageUrls": list(image_urls),
        **profile.fixed,
    }
    for name, rule in profile.parameters.items():
        value = parameters.get(name)
        if not isinstance(value, str) or value not in rule.values:
            raise ValueError(
                f"Invalid or missing {name} for the selected model; update the workflow explicitly."
            )
        if name == "aspectRatio" and value == "auto":
            continue
        request[name] = value
    return request
