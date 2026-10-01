"""ComfyUI node for API image model resolutions."""

from __future__ import annotations

from sinyuk_nodes.compat.comfy import io
from sinyuk_nodes.features.resolution_selector_api import (
    GOOGLE_RATIOS,
    MODEL_NAMES,
    resolution_for_api_model,
)


class ResolutionSelectorAPINode(io.ComfyNode):
    """Select a Google model tier or validate a GPT Image custom size."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Sinyuk.ResolutionSelectorAPI",
            display_name="Resolution Selector(API)",
            category="Sinyuk/Utilities",
            description="Select API image model dimensions with model-specific validation.",
            inputs=[
                io.Combo.Input("model", options=MODEL_NAMES, default="Nano Banana 2"),
                io.Combo.Input("aspect_ratio", options=GOOGLE_RATIOS, default="1:1"),
                io.Combo.Input("resolution", options=("0.5K", "1K", "2K", "4K"), default="1K"),
            ],
            outputs=[
                io.Int.Output("width", tooltip="Selected or validated output width."),
                io.Int.Output("height", tooltip="Selected or validated output height."),
            ],
        )

    @classmethod
    def execute(
        cls,
        model: str = "Nano Banana 2",
        aspect_ratio: str = "1:1",
        resolution: str = "1K",
    ) -> io.NodeOutput:
        if model != "GPT Image 2.5" and resolution == "0.5K" and model == "Nano Banana Pro":
            raise ValueError("Nano Banana Pro does not support 0.5K.")
        return io.NodeOutput(*resolution_for_api_model(model, aspect_ratio, resolution))


__all__ = ["ResolutionSelectorAPINode"]
