"""Explicit node registry.

Keep node registration visible and reviewable. New node families should expose
their classes from this module instead of relying on import-time discovery.
"""

from __future__ import annotations

from .compat.comfy import io

# The initial scaffold intentionally exposes no nodes.
ALL_NODES: list[type[io.ComfyNode]] = []


def get_node_list() -> list[type[io.ComfyNode]]:
    """Return a fresh list of registered node classes."""

    return list(ALL_NODES)


__all__ = ["ALL_NODES", "get_node_list"]
