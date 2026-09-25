"""Validated, immutable configuration loaded once per ComfyUI process."""

from __future__ import annotations

import json
import math
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import TypeGuard
from urllib.parse import urlsplit


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class Parameter:
    options: tuple[tuple[str, str], ...]
    default: str

    @property
    def values(self) -> tuple[str, ...]:
        return tuple(value for value, _ in self.options)


@dataclass(frozen=True)
class Profile:
    family: str
    parameters: Mapping[str, Parameter]
    max_reference_images: int | None = None
    reference_limit_source: str | None = None


@dataclass(frozen=True)
class Transport:
    poll_interval_seconds: float = 3
    task_timeout_seconds: float | None = None
    connect_timeout_seconds: float = 15
    submit_timeout_seconds: float = 120
    poll_request_timeout_seconds: float = 60
    download_timeout_seconds: float = 120
    retry_backoff_max_seconds: float = 30
    poll_retry_limit: int = 3
    download_retry_limit: int = 3
    image_encoding: str = "base64_png"


@dataclass(frozen=True)
class Config:
    base_url: str
    default_model: str
    transport: Transport
    models: Mapping[str, Profile]
    batch_reference_limit: int = 10

    def profile(self, model: str) -> Profile:
        if model not in self.models:
            raise ConfigError("Unknown or disabled model; update the workflow explicitly.")
        return self.models[model]

    def public_catalog(self) -> dict[str, object]:
        return {
            "base_url": self.base_url,
            "default_model": self.default_model,
            "batch_reference_limit": self.batch_reference_limit,
            "models": {
                model: {
                    "family": profile.family,
                    "parameters": {
                        name: {
                            "default": param.default,
                            "options": [
                                {"value": value, "label": label} for value, label in param.options
                            ],
                        }
                        for name, param in profile.parameters.items()
                    },
                }
                for model, profile in self.models.items()
            },
        }


def _is_string_object(value: object) -> TypeGuard[dict[str, object]]:
    # The parser receives JSON objects, whose keys are always strings.
    return isinstance(value, dict)


def _is_object_list(value: object) -> TypeGuard[list[object]]:
    return isinstance(value, list)


def _object(
    value: object, allowed: Collection[str], required: Collection[str], where: str
) -> dict[str, object]:
    if not _is_string_object(value) or set(value) - set(allowed) or set(required) - set(value):
        raise ConfigError(f"Invalid fields in {where}.")
    return value


def _text(value: object, where: str) -> str:
    if not isinstance(value, str) or not value.strip() or any(ord(c) < 32 for c in value):
        raise ConfigError(f"Expected nonempty text in {where}.")
    return value


def normalize_base_url(value: object, where: str = "base_url") -> str:
    base = _text(value, where).rstrip("/")
    url = urlsplit(base)
    if (
        url.scheme not in {"http", "https"}
        or not url.hostname
        or url.username
        or url.password
        or url.path
        or url.query
        or url.fragment
    ):
        raise ConfigError(
            f"{where} must be an HTTP(S) host root without credentials, path or query."
        )
    try:
        _ = url.port
    except ValueError:
        raise ConfigError(f"Invalid {where} port.") from None
    return base


def _number(values: Mapping[str, object], key: str, default: float) -> float:
    value = values.get(key, default)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ConfigError(f"{key} must be a number.")
    return float(value)


def _optional_number(values: Mapping[str, object], key: str) -> float | None:
    value = values.get(key)
    if value is None:
        return None
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ConfigError(f"{key} must be a number.")
    return float(value)


def _integer(values: Mapping[str, object], key: str, default: int) -> int:
    value = values.get(key, default)
    if type(value) is not int:
        raise ConfigError(f"{key} must be an integer.")
    return value


def _string(values: Mapping[str, object], key: str, default: str) -> str:
    value = values.get(key, default)
    if not isinstance(value, str):
        raise ConfigError(f"{key} must be a string.")
    return value


def parse_config(raw: object) -> Config:
    fields = {
        "schema_version",
        "base_url",
        "default_model",
        "transport",
        "parameter_presets",
        "profiles",
        "model_groups",
    }
    raw = _object(raw, fields | {"batch_reference_limit"}, fields, "configuration")
    local_limit = raw.get("batch_reference_limit", 10)
    if type(local_limit) is not int or not 1 <= local_limit <= 10:
        raise ConfigError(
            "batch_reference_limit must be an integer from 1 to 10 (shared provider limit)."
        )
    if type(raw["schema_version"]) is not int or raw["schema_version"] != 1:
        raise ConfigError("Unsupported schema_version; expected 1.")
    base = normalize_base_url(raw["base_url"])
    network = _object(raw["transport"], Transport.__dataclass_fields__, (), "transport")
    transport = Transport(
        poll_interval_seconds=_number(network, "poll_interval_seconds", 3.0),
        task_timeout_seconds=_optional_number(network, "task_timeout_seconds"),
        connect_timeout_seconds=_number(network, "connect_timeout_seconds", 15.0),
        submit_timeout_seconds=_number(network, "submit_timeout_seconds", 120.0),
        poll_request_timeout_seconds=_number(network, "poll_request_timeout_seconds", 60.0),
        download_timeout_seconds=_number(network, "download_timeout_seconds", 120.0),
        retry_backoff_max_seconds=_number(network, "retry_backoff_max_seconds", 30.0),
        poll_retry_limit=_integer(network, "poll_retry_limit", 3),
        download_retry_limit=_integer(network, "download_retry_limit", 3),
        image_encoding=_string(network, "image_encoding", "base64_png"),
    )
    for name in Transport.__dataclass_fields__:
        value = getattr(transport, name)
        if name.endswith("seconds"):
            if name == "task_timeout_seconds" and value is None:
                continue
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ConfigError(f"{name} must be a positive finite number.")
    for name in ("poll_retry_limit", "download_retry_limit"):
        value = getattr(transport, name)
        if type(value) is not int or not 0 <= value <= 3:
            raise ConfigError(f"{name} must be an integer from 0 to 3.")
    if transport.image_encoding not in {"base64_png", "data_url_png"}:
        raise ConfigError("Unsupported image_encoding.")
    presets = raw["parameter_presets"]
    if not _is_string_object(presets) or not presets:
        raise ConfigError("parameter_presets must be a nonempty object.")
    options: dict[str, tuple[tuple[str, str], ...]] = {}
    for name, entries in presets.items():
        _text(name, "preset name")
        if not _is_object_list(entries) or not entries:
            raise ConfigError("Each preset must contain options.")
        pairs: list[tuple[str, str]] = []
        for entry in entries:
            entry = _object(entry, {"value", "label"}, {"value", "label"}, "preset option")
            pairs.append(
                (_text(entry["value"], "option value"), _text(entry["label"], "option label"))
            )
        if len({value for value, _ in pairs}) != len(pairs):
            raise ConfigError("Duplicate option values in preset.")
        options[name] = tuple(pairs)
    raw_profiles = raw["profiles"]
    if not _is_string_object(raw_profiles) or not raw_profiles:
        raise ConfigError("profiles must be a nonempty object.")
    profiles: dict[str, Profile] = {}
    family_fields = {
        "nano_banana": {"aspectRatio", "imageSize"},
        "gpt_image": {"aspectRatio", "quality"},
    }
    for name, profile in raw_profiles.items():
        _text(name, "profile name")
        profile = _object(
            profile,
            {"family", "parameters", "max_reference_images", "reference_limit_source"},
            {"family", "parameters"},
            "profile",
        )
        limit = profile.get("max_reference_images")
        source = profile.get("reference_limit_source")
        if limit is not None:
            if type(limit) is not int or limit < 1:
                raise ConfigError("max_reference_images must be a positive integer.")
            source = _text(source, "official reference_limit_source")
        elif source is not None:
            raise ConfigError("reference_limit_source requires max_reference_images.")
        family = _text(profile["family"], "family")
        if family not in family_fields:
            raise ConfigError("Unsupported profile family.")
        params = _object(
            profile["parameters"],
            family_fields[family],
            family_fields[family],
            "profile parameters",
        )
        parsed: dict[str, Parameter] = {}
        for field, rule in params.items():
            rule = _object(rule, {"preset", "default"}, {"preset", "default"}, "parameter rule")
            preset = _text(rule["preset"], "preset reference")
            default = _text(rule["default"], "parameter default")
            if preset not in options or default not in {v for v, _ in options[preset]}:
                raise ConfigError("Missing preset or invalid parameter default.")
            parsed[field] = Parameter(options[preset], default)
        profiles[name] = Profile(family, MappingProxyType(parsed), limit, source)
    groups = raw["model_groups"]
    if not _is_object_list(groups) or not groups:
        raise ConfigError("model_groups must be a nonempty list.")
    models: dict[str, Profile] = {}
    seen: set[str] = set()
    for group in groups:
        group = _object(
            group, {"enabled", "models", "profile"}, {"enabled", "models", "profile"}, "model group"
        )
        profile = _text(group["profile"], "profile reference")
        if type(group["enabled"]) is not bool or profile not in profiles:
            raise ConfigError("Invalid enabled flag or missing profile.")
        group_models = group["models"]
        if not _is_object_list(group_models) or not group_models:
            raise ConfigError("Each model group must contain models.")
        for model in group_models:
            model = _text(model, "model")
            if model in seen:
                raise ConfigError("Duplicate model in configuration.")
            seen.add(model)
            if group["enabled"]:
                models[model] = profiles[profile]
    default = _text(raw["default_model"], "default_model")
    if default not in models:
        raise ConfigError("default_model must be enabled.")
    return Config(base, default, transport, MappingProxyType(models), local_limit)


def load_config(directory: Path | None = None) -> Config:
    directory = directory or Path(__file__).parent
    path = directory / "image_api_config.json"
    if not path.exists():
        path = directory / "config.example.json"
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise ConfigError(f"Cannot read valid JSON from {path.name}.") from None
    return parse_config(raw)


@lru_cache(maxsize=1)
def get_config() -> Config:
    return load_config()
