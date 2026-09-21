"""ComfyUI nodes for GarmentAnalysis schemas and prompt compilation."""

# ComfyUI's current V3 stubs type ``execute`` as synchronous ``**kwargs``;
# runtime dispatch also supports typed async signatures used by these nodes.
# The mismatch is external to this package and is isolated at this adapter boundary.
# pyright: reportIncompatibleMethodOverride=false

from __future__ import annotations

from sinyuk_nodes.compat.comfy import io
from sinyuk_nodes.features.garment_prompt_compiler import (
    compile_prompt,
    load_garment_analysis_schema,
)

from .llm.schema import JSON_SCHEMA


class GarmentAnalysisSchemaNode(io.ComfyNode):
    """Provide the bundled GarmentAnalysis schema to an upstream LLM node."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Sinyuk.GarmentAnalysisSchema",
            display_name="Garment Analysis Schema",
            category="Sinyuk/Garment",
            description="Provide the bundled GarmentAnalysis JSON Schema to an LLM API node.",
            outputs=[JSON_SCHEMA.Output("schema", display_name="JSON Schema")],
        )

    @classmethod
    def execute(cls) -> io.NodeOutput:
        return io.NodeOutput(load_garment_analysis_schema())


class GarmentPromptCompiler(io.ComfyNode):
    """Compile GarmentAnalysis JSON into a deterministic replacement prompt."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Sinyuk.GarmentPromptCompiler",
            display_name="Garment Prompt Compiler",
            category="Sinyuk/Garment",
            description=(
                "Compile a GarmentAnalysis JSON response into a garment replacement prompt."
            ),
            inputs=[
                io.String.Input(
                    "analysis_json",
                    display_name="GarmentAnalysis JSON",
                    multiline=True,
                    optional=True,
                    tooltip="JSON response produced by the upstream garment analysis LLM.",
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
    def execute(cls, analysis_json: str = "", extra_prompt: str = "") -> io.NodeOutput:
        return io.NodeOutput(compile_prompt(analysis_json, extra_prompt))


__all__ = ["GarmentAnalysisSchemaNode", "GarmentPromptCompiler"]
