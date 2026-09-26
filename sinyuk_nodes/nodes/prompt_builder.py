"""ComfyUI nodes for preset-backed prompt contexts and prompt compilation."""

# ComfyUI's current V3 stubs type ``execute`` as synchronous ``**kwargs``;
# runtime dispatch also supports typed async signatures used by these nodes.
# The mismatch is external to this package and is isolated at this adapter boundary.
# pyright: reportIncompatibleMethodOverride=false

from __future__ import annotations

from sinyuk_nodes.compat.comfy import io
from sinyuk_nodes.features.prompt_builder import (
    PromptContext,
    build_prompt,
    load_prompt_context,
)

from .llm.schema import JSON_SCHEMA

PROMPT_CONTEXT = io.Custom("PROMPT_CONTEXT")


class PromptContextNode(io.ComfyNode):
    """Load a prompt preset and prepare its LLM-facing context."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Sinyuk.PromptContext",
            display_name="Prompt Context",
            category="Sinyuk/Prompt",
            description=(
                "Load a preset and prepare prompts and its contract for a downstream "
                "Prompt Builder."
            ),
            inputs=[
                io.Combo.Input(
                    "preset",
                    options=["garment.replacement", "garment.enhancement"],
                    default="garment.replacement",
                    display_name="Preset",
                ),
                io.Combo.Input(
                    "schema_placement",
                    options=["External", "In Prompt"],
                    default="External",
                    display_name="Schema Placement",
                ),
                io.String.Input(
                    "extra_prompt",
                    default="",
                    multiline=True,
                    optional=True,
                    display_name="Extra Prompt",
                    tooltip="Additional instructions appended to the preset User Prompt.",
                ),
            ],
            outputs=[
                io.String.Output("system_prompt", display_name="System Prompt"),
                io.String.Output("user_prompt", display_name="User Prompt"),
                JSON_SCHEMA.Output("schema", display_name="JSON Schema"),
                PROMPT_CONTEXT.Output("prompt_context", display_name="Prompt Context"),
            ],
        )

    @classmethod
    def execute(
        cls,
        preset: str = "Replacement",
        schema_placement: str = "External",
        extra_prompt: str = "",
    ) -> io.NodeOutput:
        loaded = load_prompt_context(preset, schema_placement, extra_prompt)
        prompt_context = PromptContext(
            preset_id=loaded.preset_id,
            version=loaded.version,
            schema=loaded.schema,
            example=loaded.example,
            compiler_id=loaded.compiler_id,
            template=loaded.template,
        )
        return io.NodeOutput(
            loaded.system_prompt,
            loaded.user_prompt,
            loaded.schema,
            prompt_context,
        )


class PromptBuilderNode(io.ComfyNode):
    """Validate structured LLM output and compile a final image prompt."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Sinyuk.PromptBuilder",
            display_name="Prompt Builder",
            category="Sinyuk/Prompt",
            description=("Build a final prompt from a preset context and an LLM response."),
            inputs=[
                PROMPT_CONTEXT.Input(
                    "prompt_context",
                    display_name="Prompt Context",
                    tooltip="Connect the matching Prompt Context preset.",
                ),
                io.String.Input(
                    "llm_response",
                    display_name="LLM Response",
                    multiline=True,
                    tooltip="Structured JSON response produced by any upstream LLM node.",
                ),
            ],
            outputs=[io.String.Output("prompt", display_name="Prompt")],
        )

    @classmethod
    def execute(
        cls,
        prompt_context: PromptContext,
        llm_response: str = "",
    ) -> io.NodeOutput:
        if prompt_context.compiler_id != "garment":
            raise ValueError(f"Unsupported prompt compiler: {prompt_context.compiler_id}.")
        return io.NodeOutput(build_prompt(llm_response, context=prompt_context))


__all__ = [
    "PromptBuilderNode",
    "PromptContextNode",
    "PROMPT_CONTEXT",
]
