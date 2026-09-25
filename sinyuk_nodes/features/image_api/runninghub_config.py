"""Validated local catalog for the curated RunningHub image-editing models."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import TypeGuard

from .config import ConfigError, Parameter, normalize_base_url


@dataclass(frozen=True)
class RunningHubModel:
    label: str
    channel: str
    endpoint: str
    parameters: Mapping[str, Parameter]
    fixed: Mapping[str, str]
    family: str = "runninghub"
    max_reference_images: int = 10
    reference_limit_source: str = "local shared provider limit"


@dataclass(frozen=True)
class RunningHubCatalog:
    base_url: str
    default_model: str
    models: Mapping[str, RunningHubModel]

    def profile(self, model: str) -> RunningHubModel:
        if model not in self.models:
            raise ConfigError("Unknown RunningHub model; update the workflow explicitly.")
        return self.models[model]

    def public_catalog(self) -> dict[str, object]:
        return {
            "base_url": self.base_url,
            "default_model": self.default_model,
            "models": {
                model: {
                    "label": profile.label,
                    "family": profile.family,
                    "parameters": {
                        name: {
                            "default": parameter.default,
                            "options": [
                                {"value": value, "label": label}
                                for value, label in parameter.options
                            ],
                        }
                        for name, parameter in profile.parameters.items()
                    },
                }
                for model, profile in self.models.items()
            },
        }


def _is_string_object(value: object) -> TypeGuard[dict[str, object]]:
    # Catalog files are decoded from JSON, which guarantees string object keys.
    return isinstance(value, dict)


def _is_object_list(value: object) -> TypeGuard[list[object]]:
    return isinstance(value, list)


def _object(value: object, allowed: set[str], required: set[str]) -> dict[str, object]:
    if not _is_string_object(value) or set(value) - allowed or required - set(value):
        raise ConfigError("Invalid RunningHub catalog fields.")
    return value


def parse_runninghub_catalog(raw: object) -> RunningHubCatalog:
    raw = _object(
        raw,
        {"schema_version", "base_url", "default_model", "models"},
        {"schema_version", "base_url", "default_model", "models"},
    )
    if raw["schema_version"] != 1 or type(raw["schema_version"]) is not int:
        raise ConfigError("Unsupported RunningHub catalog schema.")
    base_url = normalize_base_url(raw["base_url"], "RunningHub base_url")
    models: dict[str, RunningHubModel] = {}
    model_items = raw["models"]
    if not _is_object_list(model_items) or not model_items:
        raise ConfigError("RunningHub models must be a nonempty list.")
    allowed_parameters = {"aspectRatio", "resolution", "quality"}
    for entry in model_items:
        item = _object(
            entry,
            {"id", "label", "channel", "endpoint", "parameters", "fixed"},
            {"id", "label", "channel", "endpoint", "parameters"},
        )
        model = item["id"]
        if not isinstance(model, str) or not model.startswith("rh:") or model in models:
            raise ConfigError("Invalid or duplicate RunningHub model ID.")
        channel_value = item["channel"]
        if channel_value not in {"economy", "stable"}:
            raise ConfigError("Invalid RunningHub channel.")
        channel = "economy" if channel_value == "economy" else "stable"
        endpoint = item["endpoint"]
        if (
            not isinstance(endpoint, str)
            or not endpoint.startswith("/openapi/v2/")
            or ".." in endpoint
        ):
            raise ConfigError("Invalid RunningHub endpoint.")
        parameters = item["parameters"]
        if (
            not _is_string_object(parameters)
            or not parameters
            or set(parameters) - allowed_parameters
        ):
            raise ConfigError("Invalid RunningHub parameters.")
        parsed: dict[str, Parameter] = {}
        for name, rule in parameters.items():
            rule = _object(rule, {"default", "options"}, {"default", "options"})
            if set(rule) != {"default", "options"}:
                raise ConfigError("Invalid RunningHub parameter rule.")
            options = rule["options"]
            if (
                not _is_object_list(options)
                or not options
                or any(not isinstance(v, str) or not v for v in options)
            ):
                raise ConfigError("Invalid RunningHub parameter options.")
            string_options = [value for value in options if isinstance(value, str)]
            default_option = rule["default"]
            if (
                len(options) != len(set(string_options))
                or not isinstance(default_option, str)
                or default_option not in string_options
            ):
                raise ConfigError("Invalid RunningHub parameter default.")
            parsed[name] = Parameter(
                tuple((value, value) for value in string_options), default_option
            )
        fixed = item.get("fixed", {})
        if not _is_string_object(fixed) or set(fixed) - {"background", "outputFormat"}:
            raise ConfigError("Invalid RunningHub fixed parameters.")
        if fixed and fixed != {"background": "opaque", "outputFormat": "png"}:
            raise ConfigError("Unsupported RunningHub fixed parameters.")
        fixed_values = {"background": "opaque", "outputFormat": "png"} if fixed else {}
        label = item["label"]
        if not isinstance(label, str) or not label.strip():
            raise ConfigError("Invalid RunningHub model label.")
        models[model] = RunningHubModel(
            label.strip(),
            channel,
            endpoint,
            MappingProxyType(parsed),
            MappingProxyType(fixed_values),
        )
    default = raw["default_model"]
    if not isinstance(default, str) or default not in models:
        raise ConfigError("RunningHub default_model must be enabled.")
    return RunningHubCatalog(base_url, default, MappingProxyType(models))


@lru_cache(maxsize=1)
def get_runninghub_catalog() -> RunningHubCatalog:
    path = Path(__file__).with_name("runninghub_catalog.json")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise ConfigError("Cannot read valid JSON from runninghub_catalog.json.") from None
    return parse_runninghub_catalog(raw)
