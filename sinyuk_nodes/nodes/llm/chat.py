"""ComfyUI node adapter for OpenAI-compatible chat completions."""

# ComfyUI V3 stubs currently reject heterogeneous input subclasses, optional
# ``None`` defaults, and async execute signatures accepted by runtime dispatch.
# These are adapter-boundary typing defects in the external stubs.
# pyright: reportArgumentType=false, reportIncompatibleMethodOverride=false

from __future__ import annotations

import torch
from sinyuk_nodes.compat.comfy import io
from sinyuk_nodes.features.llm.chat import execute_chat
from sinyuk_nodes.features.llm.config import OpenAPIConfig
from sinyuk_nodes.features.llm.schema import JSONSchemaDocument

from .config import OPENAPI_CONFIG
from .schema import JSON_SCHEMA

_RESPONSE_FORMAT_VALUES = {
    "Text": "text",
    "JSON Schema": "json_schema",
}


class LLMAPINode(io.ComfyNode):
    """Send text and optional images to the model selected by API Config."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Sinyuk.LLMAPI",
            display_name="LLM API",
            category="Sinyuk/LLM",
            description="Send a prompt and optional images to an OpenAI-compatible vision model.",
            is_output_node=True,
            inputs=[
                OPENAPI_CONFIG.Input(
                    "api_config", display_name="API Config", tooltip="Connect an API Config node."
                ),
                io.String.Input(
                    "system_prompt",
                    default="",
                    multiline=True,
                    display_name="System Prompt",
                    tooltip="Optional system instruction.",
                    optional=True,
                ),
                io.String.Input(
                    "prompt",
                    default="",
                    multiline=True,
                    display_name="User Prompt",
                    tooltip="User prompt sent to the model.",
                ),
                io.Combo.Input(
                    "response_format",
                    options=list(_RESPONSE_FORMAT_VALUES),
                    default="Text",
                    display_name="Response Format",
                    tooltip=(
                        "Text or JSON Schema Structured Outputs format requested from the model."
                    ),
                ),
                JSON_SCHEMA.Input(
                    "json_schema",
                    display_name="JSON Schema",
                    tooltip="Connect a validated JSON Schema node when using JSON Schema format.",
                    optional=True,
                ),
                io.Image.Input(
                    "images",
                    display_name="Images",
                    tooltip="Optional input images for a vision-capable model.",
                    optional=True,
                ),
                io.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=2**31 - 1,
                    control_after_generate=True,
                    display_name="Seed",
                    tooltip="Seed used for ComfyUI execution and cache identity.",
                    advanced=True,
                ),
                io.Combo.Input(
                    "image_detail",
                    options=["auto", "low", "high", "original"],
                    default="high",
                    display_name="Image Detail",
                    tooltip="Detail level used when preparing images for the LLM API.",
                    advanced=True,
                ),
                io.Float.Input(
                    "temperature",
                    default=None,
                    min=0.0,
                    max=2.0,
                    step=0.01,
                    display_name="Temperature",
                    tooltip="Optional sampling temperature sent to the API.",
                    optional=True,
                    advanced=True,
                ),
                io.Float.Input(
                    "top_p",
                    default=None,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    display_name="Top P",
                    tooltip="Optional nucleus sampling probability sent to the API.",
                    optional=True,
                    advanced=True,
                ),
                io.Int.Input(
                    "max_tokens",
                    default=2048,
                    min=512,
                    max=1_000_000,
                    display_name="Max Tokens",
                    tooltip=(
                        "Maximum number of output tokens. Defaults to 2048 for structured "
                        "garment analysis."
                    ),
                    optional=True,
                    advanced=True,
                ),
            ],
            outputs=[
                io.String.Output(
                    "response", display_name="Response", tooltip="Text returned by the model."
                ),
                io.String.Output(
                    "execution_summary",
                    display_name="Execution Summary",
                    tooltip="Markdown-formatted node-side request and response diagnostics.",
                ),
            ],
        )

    @classmethod
    async def execute(
        cls,
        api_config: OpenAPIConfig,
        system_prompt: str,
        prompt: str,
        images: torch.Tensor | None = None,
        image_detail: str = "high",
        response_format: str = "text",
        json_schema: JSONSchemaDocument | None = None,
        seed: int = 0,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
    ) -> io.NodeOutput:
        response_format = _RESPONSE_FORMAT_VALUES.get(response_format, response_format)
        result = await execute_chat(
            api_config,
            system_prompt,
            prompt,
            images,
            seed,
            temperature,
            top_p,
            max_tokens,
            response_format,
            json_schema,
            image_detail,
        )
        return io.NodeOutput(result.response, result.execution_summary)


__all__ = ["LLMAPINode"]
