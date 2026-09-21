"""ComfyUI node that owns OpenAI-compatible connection settings."""

from __future__ import annotations

from sinyuk_nodes.compat.comfy import ComfyAPI, io
from sinyuk_nodes.features.llm.client import (
    OpenAPIRequestError,
    available_models,
    fetch_models,
)
from sinyuk_nodes.features.llm.config import build_config

OPENAPI_CONFIG = io.Custom("OPENAPI_CONFIG")


class OpenAPIConfigNode(io.ComfyNode):
    """Expose credentials, endpoint, and model selection as one workflow value."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Sinyuk.OpenAPIConfig",
            display_name="API Config",
            category="Sinyuk/LLM",
            description="Configure an OpenAI-compatible API endpoint and model.",
            inputs=[
                io.Combo.Input(
                    "provider",
                    options=["openapi"],
                    default="openapi",
                    display_name="Provider",
                    tooltip="OpenAI-compatible API provider.",
                ),
                io.Combo.Input(
                    "api_mode",
                    options=["Responses API", "Chat Completions"],
                    default="Responses API",
                    display_name="API Mode",
                    tooltip="Responses API is recommended for new integrations.",
                ),
                io.String.Input(
                    "api_key",
                    default="",
                    display_name="API Key",
                    socketless=True,
                    tooltip="API key sent to the configured endpoint.",
                ),
                io.String.Input(
                    "base_url",
                    default="https://api.openai.com/v1",
                    display_name="Base URL",
                    socketless=True,
                    tooltip="HTTP(S) base URL for the OpenAI-compatible API.",
                ),
                io.String.Input(
                    "model_input",
                    default="",
                    display_name="Model ID",
                    socketless=True,
                    tooltip="Optional model ID. This value takes priority over the model dropdown.",
                ),
                io.Combo.Input(
                    "model_selection",
                    options=["auto"],
                    default="auto",
                    display_name="Model",
                    tooltip="Select a cached model. The list is synchronized by the backend.",
                    remote=io.RemoteOptions(
                        route="/sinyuk/openapi/models",
                        refresh_button=False,
                    ),
                ),
            ],
            outputs=[
                OPENAPI_CONFIG.Output(
                    "config",
                    display_name="API Config",
                    tooltip="Validated OpenAI-compatible API configuration.",
                )
            ],
            hidden=[io.Hidden.unique_id],
        )

    @classmethod
    async def execute(
        cls,
        provider: str,
        api_mode: str,
        api_key: str,
        base_url: str,
        model_input: str,
        model_selection: str,
    ) -> io.NodeOutput:
        if provider != "openapi":
            raise ValueError("Only the openapi provider is supported.")
        api_modes = {"Responses API": "responses", "Chat Completions": "chat_completions"}
        normalized_mode = api_modes.get(api_mode, api_mode)
        config = build_config(api_key, base_url, model_input, model_selection, (), normalized_mode)
        models = available_models(config.base_url)
        if models:
            config = build_config(
                api_key, base_url, model_input, model_selection, models, normalized_mode
            )
        models = config.available_models
        if not models:
            await ComfyAPI().execution.set_progress(0, 1, node_id=str(cls.hidden.unique_id))
            try:
                models = await fetch_models(config.base_url, config.api_key)
            except OpenAPIRequestError:
                if not config.model_input.strip():
                    raise
                models = ()
            await ComfyAPI().execution.set_progress(1, 1, node_id=str(cls.hidden.unique_id))
            config = build_config(
                api_key, base_url, model_input, model_selection, models, normalized_mode
            )
        return io.NodeOutput(config)


__all__ = ["OPENAPI_CONFIG", "OpenAPIConfigNode"]
