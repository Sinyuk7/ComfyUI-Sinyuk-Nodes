"""Validated runtime credentials and endpoint settings."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

from .config import Config, normalize_base_url
from .request_builder import normalize_key


def _optional_secret(value: object, name: str) -> str | None:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string.")
    secret = value.strip()
    if any(ord(char) < 32 or ord(char) == 127 for char in secret):
        raise ValueError(f"{name} contains invalid control characters.")
    return secret or None


@dataclass(frozen=True)
class RuntimeAPIConfig:
    api_key: str
    base_url: str
    token: str | None = None
    provider: Literal["grsai", "runninghub"] = "grsai"

    def apply(self, config: Config) -> Config:
        return replace(config, base_url=self.base_url)


def build_provider_config(
    api_key: object,
    base_url: object,
    token: object,
    provider: str,
    grsai_base_url: str,
    runninghub_base_url: str,
) -> RuntimeAPIConfig:
    key = normalize_key(api_key)
    if provider not in {"grsai", "runninghub"}:
        raise ValueError("Provider must be GRSAI or RunningHub.")
    if not isinstance(base_url, str):
        raise ValueError("Base URL must be a string.")
    default = grsai_base_url if provider == "grsai" else runninghub_base_url
    if not default:
        raise ValueError("Provider default Base URL is unavailable.")
    endpoint = normalize_base_url(base_url.strip() or default, "Base URL")
    selected_provider: Literal["grsai", "runninghub"] = (
        "grsai" if provider == "grsai" else "runninghub"
    )
    return RuntimeAPIConfig(
        key,
        endpoint,
        _optional_secret(token, "Token") if provider == "grsai" else None,
        selected_provider,
    )


def require_api_config(value: object) -> RuntimeAPIConfig:
    if not isinstance(value, RuntimeAPIConfig):
        raise ValueError("Connect an API Config node.")
    return value
