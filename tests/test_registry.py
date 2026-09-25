"""Tests for the extension and explicit registry contract."""

from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path

from sinyuk_nodes.extension import SinyukNodesExtension, comfy_entrypoint
from sinyuk_nodes.nodes.garment_prompt_compiler import (
    GarmentAnalysisContextNode,
    GarmentPromptCompiler,
)
from sinyuk_nodes.nodes.image_api.batch import BatchImageGenerate, ImageAPILoadImagesFromFolder
from sinyuk_nodes.nodes.image_api.config import ImageAPIConfig
from sinyuk_nodes.nodes.image_api.generate import ImageGenerate
from sinyuk_nodes.nodes.llm.chat import LLMAPINode
from sinyuk_nodes.registry import ALL_NODES, get_node_list


def test_registry_is_explicit() -> None:
    assert [node.__name__ for node in ALL_NODES] == [
        "AspectRatioResolutionNode",
        "OpenAPIConfigNode",
        "JSONSchemaNode",
        "GarmentAnalysisContextNode",
        "GarmentPromptCompiler",
        "LLMAPINode",
        "ImageAPIConfig",
        "ImageGenerate",
        "ImageAPILoadImagesFromFolder",
        "BatchImageGenerate",
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


def test_garment_analysis_context_node_returns_protocol_outputs() -> None:
    schema = GarmentAnalysisContextNode.define_schema()
    schema.validate()
    output = GarmentAnalysisContextNode.execute()

    assert len(output.result) == 3
    assert output.result[0]
    assert output.result[1]
    assert output.result[2].name == "garment_replacement"


def test_llm_api_schema_groups_prompts_before_structured_output_options() -> None:
    inputs = LLMAPINode.define_schema().inputs

    assert [input.id for input in inputs[:6]] == [
        "api_config",
        "system_prompt",
        "prompt",
        "response_format",
        "json_schema",
        "images",
    ]
    assert inputs[2].display_name == "User Prompt"


def test_image_api_nodes_use_the_module_namespace() -> None:
    assert [
        ImageAPIConfig.define_schema().node_id,
        ImageGenerate.define_schema().node_id,
        ImageAPILoadImagesFromFolder.define_schema().node_id,
        BatchImageGenerate.define_schema().node_id,
    ] == [
        "Sinyuk.ImageAPI.Config",
        "Sinyuk.ImageAPI.Generate",
        "Sinyuk.ImageAPI.LoadFolder",
        "Sinyuk.ImageAPI.BatchGenerate",
    ]


def test_enhancement_context_connects_to_builder() -> None:
    import json

    output = GarmentAnalysisContextNode.execute("Enhancement")
    schema = output.result[2]
    GarmentPromptCompiler.define_schema().validate()
    analysis = json.dumps(
        {
            "schema_version": "2.0",
            "subject": {
                "crop": "waist_up",
                "pose": "standing",
                "view": "front",
                "notes": "",
                "styling": "",
            },
            "garments": [
                {
                    "category": "bag",
                    "main_refs": [2],
                    "shape": "Small bag",
                    "fabric_behavior": "Leather",
                    "presentation": "Shoulder worn",
                    "key_details": [],
                }
            ],
        }
    )
    result = GarmentPromptCompiler.execute(analysis_json=analysis, schema=schema)
    assert result.result[0].startswith("Enhance the outfit")
