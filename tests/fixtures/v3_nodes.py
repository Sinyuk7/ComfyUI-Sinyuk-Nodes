"""Reference-only V3 nodes used to test the ComfyUI boundary."""

from __future__ import annotations

from sinyuk_nodes.compat.comfy import io


class EchoStringReferenceNode(io.ComfyNode):
    """Minimal V3 node that is intentionally absent from the production registry."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SinyukTest.EchoStringReference",
            display_name="Sinyuk Test Echo String",
            category="Sinyuk/Test",
            inputs=[io.String.Input("text")],
            outputs=[io.String.Output()],
        )

    @classmethod
    def execute(cls, text: str) -> io.NodeOutput:
        return io.NodeOutput(text)
