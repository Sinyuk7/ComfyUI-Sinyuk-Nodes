from __future__ import annotations

import json

import pytest
import torch
from sinyuk_nodes.features.image_api.batch_plan import BatchPlan
from sinyuk_nodes.features.image_api.batch_storage import BatchStore
from sinyuk_nodes.features.image_api.references import ReferenceSet, Source


@pytest.mark.asyncio
async def test_batch_saves_outputs_and_manifest_in_task_order(tmp_path) -> None:
    image = torch.zeros((1, 2, 2, 3))
    source = Source("input.png", None, 1, "0" * 64)
    reference = ReferenceSet((image,), (source,))
    plan = BatchPlan((reference,), ("first", "second"), "prompts", 1)
    store = BatchStore(tmp_path, plan, "model", {}, 2, "result", "secret")

    await store.update(1, status="succeeded")
    await store.save(1, image, 1, 1)
    await store.update(2, status="succeeded")
    await store.save(2, image, 1, 1)
    outputs = await store.finish("succeeded")

    manifest = json.loads(store.manifest.read_text(encoding="utf-8"))
    assert len(outputs) == 2
    assert [task["task_index"] for task in manifest["tasks"]] == [1, 2]
    assert [task["outputs"][0]["flat_output_index"] for task in manifest["tasks"]] == [0, 1]
    assert manifest["summary"]["saved_images"] == 2
