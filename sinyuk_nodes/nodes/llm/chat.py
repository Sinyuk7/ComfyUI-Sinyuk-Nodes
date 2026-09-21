"""ComfyUI node adapter for OpenAI-compatible chat completions."""

from __future__ import annotations

import torch
from sinyuk_nodes.compat.comfy import io
from sinyuk_nodes.features.llm.chat import execute_chat
from sinyuk_nodes.features.llm.config import OpenAPIConfig

from .config import OPENAPI_CONFIG

_RESPONSE_FORMAT_VALUES = {
    "Text": "text",
    "JSON Schema": "json_schema",
    "JSON Object (Deprecated)": "json_object",
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
                    display_name="Prompt",
                    tooltip="User prompt sent to the model.",
                ),
                io.Image.Input(
                    "images",
                    display_name="Images",
                    tooltip="Optional input images for a vision-capable model.",
                    optional=True,
                ),
                io.Combo.Input(
                    "image_detail",
                    options=["auto", "low", "high", "original"],
                    default="high",
                    display_name="Image Detail",
                    tooltip="Detail level used when preparing images for the LLM API.",
                ),
                io.Combo.Input(
                    "response_format",
                    options=list(_RESPONSE_FORMAT_VALUES),
                    default="Text",
                    display_name="Response Format",
                    tooltip=(
                        "Text, JSON Schema, or JSON Object (Deprecated) format requested "
                        "from the model."
                    ),
                ),
                io.String.Input(
                    "json_schema",
                    default="",
                    multiline=True,
                    display_name="JSON Schema",
                    tooltip="JSON object schema used when response format is json_schema.",
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
                    default=None,
                    min=1,
                    max=1_000_000,
                    display_name="Max Tokens",
                    tooltip="Optional maximum number of output tokens (sent as max_tokens).",
                    optional=True,
                    advanced=True,
                ),
            ],
            outputs=[
                io.String.Output(
                    "response", display_name="Response", tooltip="Text returned by the model."
                )
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
        json_schema: str = "",
        seed: int = 0,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
    ) -> io.NodeOutput:
        if not isinstance(api_config, OpenAPIConfig):
            raise ValueError("Connect an API Config node.")
        response_format = _RESPONSE_FORMAT_VALUES.get(response_format, response_format)
        response = await execute_chat(
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
        return io.NodeOutput(response)


__all__ = ["LLMAPINode"]
