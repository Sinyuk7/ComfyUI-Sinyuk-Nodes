"""ComfyUI node that owns OpenAI-compatible connection settings."""

# See the V3 adapter-boundary note in ``nodes.llm.chat``.
# pyright: reportIncompatibleMethodOverride=false

from __future__ import annotations

from sinyuk_nodes.compat.comfy import check_interrupt, io
from sinyuk_nodes.features.llm.config import OPENAPI_MODELS, build_config

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
                io.String.Input(
                    "base_url",
                    default="https://api.openai.com/v1",
                    display_name="Base URL",
                    socketless=True,
                    tooltip="HTTP(S) base URL for the OpenAI-compatible API.",
                ),
                io.Combo.Input(
                    "model_selection",
                    options=list(OPENAPI_MODELS),
                    default=OPENAPI_MODELS[0],
                    display_name="Model",
                    tooltip="Select a preset model or choose custom.",
                ),
                io.String.Input(
                    "custom_model_id",
                    default="",
                    display_name="Custom Model ID",
                    socketless=True,
                    tooltip="Required only when Model is set to custom.",
                ),
                io.String.Input(
                    "api_key",
                    default="",
                    display_name="API Key",
                    socketless=True,
                    tooltip="API key sent to the configured endpoint.",
                ),
                io.Combo.Input(
                    "api_mode",
                    options=["Responses API", "Chat Completions"],
                    default="Responses API",
                    display_name="API Mode",
                    tooltip="Responses API is recommended for new integrations.",
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
        model_selection: str,
        custom_model_id: str,
    ) -> io.NodeOutput:
        check_interrupt()
        if provider != "openapi":
            raise ValueError("Only the openapi provider is supported.")
        api_modes = {"Responses API": "responses", "Chat Completions": "chat_completions"}
        normalized_mode = api_modes.get(api_mode, api_mode)
        config = build_config(api_key, base_url, model_selection, custom_model_id, normalized_mode)
        return io.NodeOutput(config)


__all__ = ["OPENAPI_CONFIG", "OpenAPIConfigNode"]
