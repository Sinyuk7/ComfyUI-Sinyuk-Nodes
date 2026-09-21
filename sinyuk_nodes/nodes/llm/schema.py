"""ComfyUI node for loading a strict JSON Schema."""

# See the V3 adapter-boundary note in ``nodes.llm.chat``.
# pyright: reportIncompatibleMethodOverride=false

from __future__ import annotations

from sinyuk_nodes.compat.comfy import io
from sinyuk_nodes.features.llm.schema import JSONSchemaDocument, load_json_schema

JSON_SCHEMA = io.Custom("JSON_SCHEMA")


class JSONSchemaNode(io.ComfyNode):
    """Load and validate a schema from a file or an inline JSON input."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Sinyuk.JSONSchema",
            display_name="JSON Schema",
            category="Sinyuk/LLM",
            description="Load and validate a strict Structured Outputs JSON Schema.",
            inputs=[
                io.Combo.Input(
                    "source_kind",
                    options=["File", "Inline JSON"],
                    default="File",
                    display_name="Source",
                ),
                io.String.Input(
                    "file_path",
                    default="",
                    display_name="File Path",
                    optional=True,
                    tooltip="Local JSON file path used when Source is File.",
                ),
                io.String.Input(
                    "json_text",
                    default="",
                    multiline=True,
                    display_name="JSON",
                    optional=True,
                    tooltip="Inline JSON Schema used when Source is Inline JSON.",
                ),
                io.String.Input(
                    "schema_name",
                    default="",
                    display_name="Schema Name",
                    optional=True,
                    tooltip="Optional API name. Defaults to the file name or schema.",
                ),
            ],
            outputs=[
                JSON_SCHEMA.Output(
                    "schema",
                    display_name="JSON Schema",
                    tooltip="Validated schema for an LLM API node.",
                )
            ],
        )

    @classmethod
    async def execute(
        cls,
        source_kind: str,
        file_path: str = "",
        json_text: str = "",
        schema_name: str = "",
    ) -> io.NodeOutput:
        source = file_path if source_kind == "File" else json_text
        kind = "file" if source_kind == "File" else "inline"
        schema: JSONSchemaDocument = load_json_schema(source, kind, schema_name)
        return io.NodeOutput(schema)


__all__ = ["JSON_SCHEMA", "JSONSchemaNode"]
