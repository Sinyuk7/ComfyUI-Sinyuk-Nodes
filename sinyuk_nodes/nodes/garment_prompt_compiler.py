"""ComfyUI nodes for GarmentAnalysis schemas and prompt compilation."""

# ComfyUI's current V3 stubs type ``execute`` as synchronous ``**kwargs``;
# runtime dispatch also supports typed async signatures used by these nodes.
# The mismatch is external to this package and is isolated at this adapter boundary.
# pyright: reportIncompatibleMethodOverride=false

from __future__ import annotations

from sinyuk_nodes.compat.comfy import io
from sinyuk_nodes.features.garment_prompt_compiler import (
    compile_prompt,
    load_garment_analysis_context,
)
from sinyuk_nodes.features.llm.schema import JSONSchemaDocument

from .llm.schema import JSON_SCHEMA


class GarmentAnalysisContextNode(io.ComfyNode):
    """Provide the complete Garment Analysis protocol to an upstream LLM node."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Sinyuk.GarmentAnalysisContext",
            display_name="Garment Analysis Context",
            category="Sinyuk/Garment",
            description=(
                "Provide the system prompt, user prompt, and JSON Schema for garment analysis."
            ),
            inputs=[
                io.Combo.Input(
                    "preset",
                    options=["Replacement", "Enhancement"],
                    default="Replacement",
                    display_name="Preset",
                )
            ],
            outputs=[
                io.String.Output("system_prompt", display_name="System Prompt"),
                io.String.Output("user_prompt", display_name="User Prompt"),
                JSON_SCHEMA.Output("schema", display_name="JSON Schema"),
            ],
        )

    @classmethod
    def execute(cls, preset: str = "Replacement") -> io.NodeOutput:
        preset_id = preset.lower()
        context = load_garment_analysis_context(preset_id)
        return io.NodeOutput(context.system_prompt, context.user_prompt, context.schema)


class GarmentPromptCompiler(io.ComfyNode):
    """Compile GarmentAnalysis JSON into a deterministic image editing prompt."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Sinyuk.GarmentPromptCompiler",
            display_name="Garment Prompt Builder",
            category="Sinyuk/Garment",
            description=(
                "Build an image editing prompt from analysis JSON and its matching Schema."
            ),
            inputs=[
                io.String.Input(
                    "analysis_json",
                    display_name="GarmentAnalysis JSON",
                    multiline=True,
                    optional=True,
                    tooltip="JSON response produced by the upstream garment analysis LLM.",
                ),
                JSON_SCHEMA.Input(
                    "schema",
                    display_name="Analysis Schema",
                    tooltip="The schema emitted by the matching Garment Analysis Context.",
                ),
                io.String.Input(
                    "extra_prompt",
                    display_name="Extra Prompt",
                    multiline=True,
                    optional=True,
                    tooltip="Optional instructions appended to the compiled prompt.",
                ),
            ],
            outputs=[io.String.Output("prompt", display_name="Prompt")],
        )

    @classmethod
    def execute(
        cls,
        analysis_json: str = "",
        schema: JSONSchemaDocument | None = None,
        extra_prompt: str = "",
    ) -> io.NodeOutput:
        if schema is None:
            raise ValueError("Garment Prompt Builder requires the matching Analysis Schema.")
        return io.NodeOutput(compile_prompt(analysis_json, extra_prompt, schema=schema))


__all__ = ["GarmentAnalysisContextNode", "GarmentPromptCompiler"]
