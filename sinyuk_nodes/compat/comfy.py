"""Single import boundary for the ComfyUI V3 API."""

from __future__ import annotations

from comfy_api.latest import ComfyAPI, ComfyExtension, io, ui


def check_interrupt() -> None:
    """Raise ComfyUI's interrupt exception when the current prompt was cancelled."""

    from comfy import model_management

    model_management.throw_exception_if_processing_interrupted()


__all__ = ["ComfyAPI", "ComfyExtension", "check_interrupt", "io", "ui"]
