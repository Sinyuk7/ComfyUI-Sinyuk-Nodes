"""ComfyUI V3 extension entry point."""

from __future__ import annotations

from .compat.comfy import ComfyExtension, io
from .features.openapi_client import cached_model_options
from .registry import get_node_list


class SinyukNodesExtension(ComfyExtension):
    """Expose the explicitly registered Sinyuk node classes."""

    async def on_load(self) -> None:
        """Expose the cached model catalog to the native V3 remote combo."""

        from aiohttp import web
        from server import PromptServer

        @PromptServer.instance.routes.get("/sinyuk/openapi/models")
        async def get_models(_request: web.Request) -> web.Response:
            return web.json_response(["auto", *cached_model_options()])

    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        """Return the node classes made available by this extension."""

        return get_node_list()


async def comfy_entrypoint() -> SinyukNodesExtension:
    """Create the ComfyUI extension instance."""

    return SinyukNodesExtension()


__all__ = ["SinyukNodesExtension", "comfy_entrypoint"]
