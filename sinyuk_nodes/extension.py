"""ComfyUI V3 extension entry point."""

from __future__ import annotations

from .compat.comfy import ComfyExtension, io
from .features.image_api.config import get_config
from .features.image_api.diagnostics import initialize_diagnostics
from .nodes.image_api.host import install_host
from .registry import get_node_list


class SinyukNodesExtension(ComfyExtension):
    """Expose the explicitly registered Sinyuk node classes."""

    async def on_load(self) -> None:
        """Initialize extension-owned services."""

        get_config()
        initialize_diagnostics()
        install_host()

    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        """Return the node classes made available by this extension."""

        return get_node_list()


async def comfy_entrypoint() -> SinyukNodesExtension:
    """Create the ComfyUI extension instance."""

    return SinyukNodesExtension()


__all__ = ["SinyukNodesExtension", "comfy_entrypoint"]
