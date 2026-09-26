"""Validated configuration for an OpenAI-compatible endpoint."""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlsplit

OPENAPI_MODELS: tuple[str, ...] = (
    "gpt-6-astra",
    "gpt-5.6-sol",
    "gpt-5.6-terra",
    "gpt-5.6-luna",
    "gpt-5.5",
    "custom",
)


def normalize_base_url(value: str) -> str:
    """Validate and normalize an OpenAI-compatible API base URL."""

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
    model_selection: str
    custom_model_id: str
    api_mode: str = "responses"

    @property
    def model(self) -> str:
        """Return the manual model when present, otherwise the selected model."""

        selected = self.model_selection.strip()
        if selected == "custom":
            custom = self.custom_model_id.strip()
            if not custom:
                raise ValueError("Enter a custom model ID.")
            return custom
        if selected in OPENAPI_MODELS:
            return selected
        raise ValueError("Select a supported model or choose custom.")


def build_config(
    api_key: str,
    base_url: str,
    model_selection: str,
    custom_model_id: str,
    api_mode: str = "responses",
) -> OpenAPIConfig:
    """Validate node values and create a runtime configuration."""

    if not api_key.strip():
        raise ValueError("API Key is required.")
    if api_mode not in {"responses", "chat_completions"}:
        raise ValueError("API mode must be responses or chat_completions.")
    if model_selection not in OPENAPI_MODELS:
        raise ValueError("Select a supported model or choose custom.")
    if model_selection == "custom" and not custom_model_id.strip():
        raise ValueError("Enter a custom model ID.")
    if model_selection != "custom" and custom_model_id.strip():
        raise ValueError("Custom model ID is only valid when model is custom.")
    return OpenAPIConfig(
        api_key=api_key.strip(),
        base_url=normalize_base_url(base_url),
        model_selection=model_selection,
        custom_model_id=custom_model_id.strip(),
        api_mode=api_mode,
    )


__all__ = ["OPENAPI_MODELS", "OpenAPIConfig", "build_config", "normalize_base_url"]
