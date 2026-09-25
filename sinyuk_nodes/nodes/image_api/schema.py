"""Shared schema inputs for single and batch Image API generation."""

from __future__ import annotations

from ...compat.comfy import io
from ...features.image_api.workflow import model_profiles

PARAMETER_TOOLTIPS = {
    "aspectRatio": (
        "Output aspect ratio supported by the selected model. Auto lets the provider infer it."
    ),
    "imageSize": "Output resolution tier supported by the selected model.",
    "resolution": "Output resolution tier supported by the selected model.",
    "quality": "Provider quality or processing tier supported by the selected model.",
}


def model_input_options() -> list[io.DynamicCombo.Option]:
    options: list[io.DynamicCombo.Option] = []
    for model, profile in model_profiles().items():
        inputs: list[io.Input] = []
        for name, parameter in profile.parameters.items():
            label = {
                "aspectRatio": "Aspect ratio"
                if profile.family in {"nano_banana", "runninghub"}
                else "Image size",
                "imageSize": "Resolution",
                "resolution": "Resolution",
                "quality": "Quality",
            }[name]
            inputs.append(
                io.Combo.Input(
                    name,
                    options=list(parameter.values),
                    default=parameter.default,
                    display_name=label,
                    tooltip=PARAMETER_TOOLTIPS[name],
                )
            )
        options.append(io.DynamicCombo.Option(model, inputs))
    return options
