"""Validated configuration for an OpenAI-compatible endpoint."""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlsplit


def normalize_base_url(value: str) -> str:
    """Validate and normalize an OpenAI-compatible API base URL."""

    if not isinstance(value, str):
        raise ValueError("Base URL must be a string.")
    base_url = value.strip().rstrip("/")
    parsed = urlsplit(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("Base URL must be an HTTP(S) URL.")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("Base URL must not contain credentials, a query, or a fragment.")
    return base_url


@dataclass(frozen=True)
class OpenAPIConfig:
    """Runtime values emitted by the API Config node."""

    api_key: str
    base_url: str
    model_input: str
    model_selection: str
    available_models: tuple[str, ...]

    @property
    def model(self) -> str:
        """Return the manual model when present, otherwise the selected model."""

        manual = self.model_input.strip()
        if manual:
            return manual
        selected = self.model_selection.strip()
        if selected and selected != "auto":
            return selected
        if self.available_models:
            return self.available_models[0]
        raise ValueError("Enter a model or select a model after the model list has been loaded.")


def build_config(
    api_key: str,
    base_url: str,
    model_input: str,
    model_selection: str,
    available_models: tuple[str, ...],
) -> OpenAPIConfig:
    """Validate node values and create a runtime configuration."""

    if not isinstance(api_key, str) or not api_key.strip():
        raise ValueError("API Key is required.")
    if not isinstance(model_input, str) or not isinstance(model_selection, str):
        raise ValueError("Model values must be strings.")
    return OpenAPIConfig(
        api_key=api_key.strip(),
        base_url=normalize_base_url(base_url),
        model_input=model_input,
        model_selection=model_selection,
        available_models=tuple(dict.fromkeys(available_models)),
    )


__all__ = ["OpenAPIConfig", "build_config", "normalize_base_url"]
