"""Input validation and provider selection for Image API node workflows."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeGuard

from .api_settings import RuntimeAPIConfig, require_api_config
from .config import get_config
from .runninghub_config import get_runninghub_catalog

if TYPE_CHECKING:
    from .config import Profile


def scalar(value: object, name: str) -> object:
    if not isinstance(value, list) or len(value) != 1:
        raise ValueError(
            f"{name} must contain exactly one value; scalar broadcasting is not supported."
        )
    return value[0]


def _is_string_object(value: object) -> TypeGuard[dict[str, object]]:
    return isinstance(value, dict) and all(isinstance(key, str) for key in value)


def normalize_inputs(
    api_config: object, model: object, prompt: object
) -> tuple[RuntimeAPIConfig, str, str, dict[str, object]]:
    settings = require_api_config(scalar(api_config, "api_config"))
    text = scalar(prompt, "prompt")
    if not isinstance(text, str):
        raise ValueError("Prompt must be a string.")
    if not _is_string_object(model) or "model" not in model:
        raise ValueError("Invalid dynamic model input; update the workflow explicitly.")
    selected = scalar(model["model"], "model")
    if not isinstance(selected, str):
        raise ValueError("Model must be a string.")
    parameters: dict[str, object] = {}
    for name, value in model.items():
        if name != "model":
            parameters[name] = scalar(value, name)
    return settings, selected, text, parameters


def provider_profile(settings: RuntimeAPIConfig, model: str) -> Profile:
    if settings.provider == "grsai":
        if model.startswith("rh:"):
            raise ValueError("Select a GRSAI model for the connected API Config.")
        return get_config().profile(model)
    if not model.startswith("rh:"):
        raise ValueError("Select a RunningHub model for the connected API Config.")
    return get_runninghub_catalog().profile(model)


def model_profiles() -> dict[str, Profile]:
    return {**get_config().models, **get_runninghub_catalog().models}
