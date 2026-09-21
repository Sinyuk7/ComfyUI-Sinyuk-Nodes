"""Tests for the extension and explicit registry contract."""

from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path

from sinyuk_nodes.extension import SinyukNodesExtension, comfy_entrypoint
from sinyuk_nodes.registry import ALL_NODES, get_node_list


def test_registry_is_explicit() -> None:
    assert [node.__name__ for node in ALL_NODES] == [
        "OpenAPIConfigNode",
        "JSONSchemaNode",
        "GarmentAnalysisSchemaNode",
        "GarmentPromptCompiler",
        "LLMAPINode",
    ]
    assert get_node_list() == ALL_NODES
    assert get_node_list() is not ALL_NODES


def test_entrypoint_returns_v3_extension() -> None:
    extension = asyncio.run(comfy_entrypoint())
    assert isinstance(extension, SinyukNodesExtension)
    assert asyncio.run(extension.get_node_list()) == ALL_NODES


def test_root_entrypoint_is_discoverable() -> None:
    root = Path(__file__).parents[1] / "__init__.py"
    spec = importlib.util.spec_from_file_location("sinyuk_nodes_root", root)
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.__version__ == "0.1.0"
    assert callable(module.comfy_entrypoint)
