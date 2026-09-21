"""Explicit node registry.

Keep node registration visible and reviewable. New node families should expose
their classes from this module instead of relying on import-time discovery.
"""

from __future__ import annotations

from .compat.comfy import io
from .nodes.garment_prompt_compiler import GarmentAnalysisSchemaNode, GarmentPromptCompiler
from .nodes.llm.chat import LLMAPINode
from .nodes.llm.config import OpenAPIConfigNode
from .nodes.llm.schema import JSONSchemaNode

# Keep registration explicit so the published node surface remains reviewable.
ALL_NODES: list[type[io.ComfyNode]] = [
    OpenAPIConfigNode,
    JSONSchemaNode,
    GarmentAnalysisSchemaNode,
    GarmentPromptCompiler,
    LLMAPINode,
]


def get_node_list() -> list[type[io.ComfyNode]]:
    """Return a fresh list of registered node classes."""

    return list(ALL_NODES)


__all__ = ["ALL_NODES", "get_node_list"]
