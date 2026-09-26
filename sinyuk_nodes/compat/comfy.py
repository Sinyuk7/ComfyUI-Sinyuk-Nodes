"""Single import boundary for the ComfyUI V3 API."""

from __future__ import annotations

from comfy_api.latest import ComfyAPI, ComfyExtension, io, ui


def check_interrupt() -> None:
    """Raise ComfyUI's interrupt exception when the current prompt was cancelled."""

    from comfy import model_management

    if model_management.processing_interrupted():
        raise model_management.InterruptProcessingException()


__all__ = ["ComfyAPI", "ComfyExtension", "check_interrupt", "io", "ui"]
