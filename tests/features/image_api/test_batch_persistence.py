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


@pytest.mark.asyncio
async def test_cancelled_save_finishes_before_final_manifest(tmp_path, monkeypatch) -> None:
    import asyncio
    import threading

    image = torch.zeros((1, 2, 2, 3))
    source = Source("input.png", None, 1, "0" * 64)
    plan = BatchPlan((ReferenceSet((image,), (source,)),), ("prompt",), "prompts", 1)
    store = BatchStore(tmp_path, plan, "model", {}, 2, "result", "secret")
    await store.update(
        1, status="running", submitted=True, remote_task_id="known", submission_state="accepted"
    )
    started = threading.Event()
    release = threading.Event()
    original = store._save_image

    def slow_save(destination, tensor) -> None:
        started.set()
        assert release.wait(3)
        original(destination, tensor)

    monkeypatch.setattr(store, "_save_image", slow_save)
    saving = asyncio.create_task(store.save(1, image, 1, 1))
    try:
        async with asyncio.timeout(2):
            while not started.is_set():
                await asyncio.sleep(0.01)
        saving.cancel()
        await asyncio.sleep(0)
        saving.cancel()
        final = asyncio.create_task(store.finish("interrupted", interrupted_tasks=True))
        await asyncio.sleep(0.02)
        assert not final.done()
    finally:
        release.set()
    with pytest.raises(asyncio.CancelledError):
        await saving
    await final
    manifest = json.loads(store.manifest.read_text())
    task = manifest["tasks"][0]
    assert task["remote_task_id"] == "known"
    assert task["submission_state"] == "accepted"
    assert task["status"] == "interrupted"
    assert len(task["outputs"]) == 1
    assert (store.path / task["outputs"][0]["file"]).is_file()


@pytest.mark.asyncio
async def test_batch_interrupt_stops_new_submissions_and_keeps_remote_ids(tmp_path) -> None:
    import asyncio
    from dataclasses import replace

    from aiohttp import web
    from comfy.model_management import InterruptProcessingException
    from sinyuk_nodes.features.image_api.batch_runner import BatchRunner
    from sinyuk_nodes.features.image_api.config import get_config

    submissions = 0
    interrupted = False

    async def submit(request: web.Request) -> web.Response:
        nonlocal submissions
        await request.read()
        submissions += 1
        return web.json_response({"id": f"remote-{submissions}", "status": "running"})

    def check() -> None:
        if interrupted:
            raise InterruptProcessingException()

    app = web.Application()
    app.router.add_post("/v1/api/generate", submit)
    server = web.AppRunner(app)
    await server.setup()
    try:
        await web.TCPSite(server, "127.0.0.1", 0).start()
        config = get_config()
        config = replace(
            config,
            base_url=f"http://127.0.0.1:{server.addresses[0][1]}",
            transport=replace(config.transport, poll_interval_seconds=60),
        )
        image = torch.zeros((1, 2, 2, 3))
        source = Source("input.png", None, 1, "0" * 64)
        plan = BatchPlan(
            (ReferenceSet((image,), (source,)),), ("first", "second", "third"), "prompts", 1
        )
        batch = BatchRunner(
            config,
            "test-key",
            plan,
            config.default_model,
            {k: v.default for k, v in config.profile(config.default_model).parameters.items()},
            2,
            "result",
            tmp_path,
            check,
        )
        task = asyncio.create_task(batch.run())
        async with asyncio.timeout(3):
            while sum(bool(t["remote_task_id"]) for t in batch.store.state["tasks"]) < 2:
                await asyncio.sleep(0.01)
        interrupted = True
        with pytest.raises(InterruptProcessingException):
            await asyncio.wait_for(task, 3)
        interrupted = False
        with pytest.raises(InterruptProcessingException):
            batch.check_cancel()
        manifest = json.loads(batch.store.manifest.read_text())
        assert submissions == 2
        assert manifest["status"] == "interrupted"
        assert [t["status"] for t in manifest["tasks"]] == ["interrupted", "interrupted", "pending"]
        assert {t["remote_task_id"] for t in manifest["tasks"][:2]} == {"remote-1", "remote-2"}
        assert all(t["submission_state"] == "accepted" for t in manifest["tasks"][:2])
        assert all(t["remote_cancel_result"] == "not_verified" for t in manifest["tasks"][:2])
    finally:
        await server.cleanup()


@pytest.mark.asyncio
@pytest.mark.parametrize("stop", ["fatal", "cooldown"])
async def test_runninghub_rechecks_submission_gate_after_upload(
    tmp_path, monkeypatch, stop
) -> None:
    import time

    import aiohttp
    from sinyuk_nodes.features.image_api.batch_plan import TaskSpec
    from sinyuk_nodes.features.image_api.batch_runner import BatchRunner
    from sinyuk_nodes.features.image_api.config import get_config
    from sinyuk_nodes.features.image_api.runninghub_client import RunningHubClient
    from sinyuk_nodes.features.image_api.runninghub_config import get_runninghub_catalog

    image = torch.zeros((1, 2, 2, 3))
    source = Source("input.png", None, 1, "0" * 64)
    plan = BatchPlan((ReferenceSet((image,), (source,)),), ("prompt",), "prompts", 1)
    catalog = get_runninghub_catalog()
    batch = BatchRunner(
        get_config(),
        "test-key",
        plan,
        catalog.default_model,
        {k: v.default for k, v in catalog.profile(catalog.default_model).parameters.items()},
        2,
        "result",
        tmp_path,
        provider="runninghub",
    )
    submitted = False

    async def upload(client, files):
        if stop == "fatal":
            batch.fatal = "another worker failed"
        else:
            await batch._pressure(0.05)
        return ["https://cdn.test/reference.png"]

    async def generate(client, request):
        nonlocal submitted
        assert time.monotonic() >= batch.cooldown_until
        submitted = True
        return []

    monkeypatch.setattr(batch, "_runninghub_urls", upload)
    monkeypatch.setattr(RunningHubClient, "generate", generate)
    async with aiohttp.ClientSession() as api, aiohttp.ClientSession() as cdn:
        await batch._task(TaskSpec(1, 1, 1), (api, cdn))
    assert submitted == (stop == "cooldown")
