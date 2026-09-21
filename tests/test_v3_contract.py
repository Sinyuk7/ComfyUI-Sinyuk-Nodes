"""Tests for the minimum ComfyUI V3 node contract."""

from __future__ import annotations

from sinyuk_nodes.compat.comfy import io
from tests.fixtures.v3_nodes import EchoStringReferenceNode


def test_reference_node_is_a_v3_node() -> None:
    assert issubclass(EchoStringReferenceNode, io.ComfyNode)


def test_reference_node_schema_is_valid() -> None:
    schema = EchoStringReferenceNode.define_schema()

    assert isinstance(schema, io.Schema)
    assert schema.node_id == "SinyukTest.EchoStringReference"
    schema.validate()


def test_reference_node_returns_node_output() -> None:
    output = EchoStringReferenceNode.execute("hello")

    assert isinstance(output, io.NodeOutput)
    assert output.result == ("hello",)
