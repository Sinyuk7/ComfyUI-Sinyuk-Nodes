from __future__ import annotations

import pytest
import torch
from sinyuk_nodes.features.image_api.client import GrsaiClient
from sinyuk_nodes.features.image_api.config import get_config


@pytest.mark.asyncio
async def test_single_generation_submits_polls_and_downloads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = GrsaiClient(get_config(), "test-key")
    responses = iter(
        [
            (200, {"id": "task-1", "status": "running", "progress": 20}, None),
            (
                200,
                {
                    "id": "task-1",
                    "status": "succeeded",
                    "results": [{"url": "https://cdn.test/image.png"}],
                },
                None,
            ),
        ]
    )
    requests: list[tuple[str, str]] = []

    async def fake_json(
        _session: object, method: str, path: str, _timeout: float, **kwargs: object
    ) -> tuple[int, object, None]:
        requests.append((method, path))
        return next(responses)

    async def fake_wait(_seconds: float) -> None:
        return None

    async def fake_download(_session: object, _url: str, _index: int) -> torch.Tensor:
        return torch.zeros((1, 2, 2, 3))

    monkeypatch.setattr(client, "_json", fake_json)
    monkeypatch.setattr(client, "_wait", fake_wait)
    monkeypatch.setattr(client, "_download", fake_download)

    result = await client.generate({"model": "test", "prompt": "draw", "images": ["x"]})

    assert len(result) == 1
    assert requests == [("POST", "/v1/api/generate"), ("GET", "/v1/api/result")]
    assert client.task_id == "task-1"


@pytest.mark.asyncio
async def test_runninghub_real_multipart_poll_download_and_borrowed_sessions() -> None:
    from dataclasses import replace
    from io import BytesIO

    import aiohttp
    from aiohttp import web
    from PIL import Image
    from sinyuk_nodes.features.image_api.runninghub_client import RunningHubClient

    calls: list[str] = []
    png = BytesIO()
    Image.new("RGB", (2, 2)).save(png, format="PNG")
    content = png.getvalue()
    base = ""

    async def upload(request: web.Request) -> web.Response:
        calls.append("upload")
        assert request.headers["Authorization"] == "Bearer test-key"
        form = await request.multipart()
        part = await form.next()
        assert isinstance(part, aiohttp.BodyPartReader)
        assert part.name == "file"
        assert bytes(await part.read()) == content
        return web.json_response({"code": 0, "data": {"download_url": base + "/image"}})

    async def submit(request: web.Request) -> web.Response:
        calls.append("submit")
        assert (await request.json())["images"] == [base + "/image"]
        return web.json_response({"taskId": "remote-1", "status": "CREATE"})

    async def query(request: web.Request) -> web.Response:
        calls.append("query")
        assert (await request.json())["taskId"] == "remote-1"
        return web.json_response(
            {"taskId": "remote-1", "status": "SUCCESS", "results": [{"url": base + "/image"}]}
        )

    async def download(request: web.Request) -> web.Response:
        calls.append("download")
        assert "Authorization" not in request.headers
        return web.Response(body=content, content_type="image/png")

    app = web.Application()
    app.router.add_post("/openapi/v2/media/upload/binary", upload)
    app.router.add_post("/generate", submit)
    app.router.add_post("/openapi/v2/query", query)
    app.router.add_get("/image", download)
    runner = web.AppRunner(app)
    await runner.setup()
    try:
        await web.TCPSite(runner, "127.0.0.1", 0).start()
        base = f"http://127.0.0.1:{runner.addresses[0][1]}"
        config = get_config()
        config = replace(
            config, base_url=base, transport=replace(config.transport, poll_interval_seconds=0)
        )
        async with aiohttp.ClientSession() as api, aiohttp.ClientSession() as cdn:
            client = RunningHubClient(config, "test-key", endpoint="/generate", sessions=(api, cdn))
            urls = await client.upload_images([content])
            images = await client.generate({"prompt": "test", "images": urls})
            assert not api.closed and not cdn.closed
            assert images[0].shape == (1, 2, 2, 3)
            assert client.submission_state == "accepted"
            assert client.remote_status == "SUCCESS"
            assert client.exit_reason == "completed"
        # Exercise client-owned sessions too, without patching the transport boundary.
        client = RunningHubClient(config, "test-key", endpoint="/generate")
        await client.generate({"images": urls})
        assert calls == ["upload", "submit", "query", "download", "submit", "query", "download"]
    finally:
        await runner.cleanup()


@pytest.mark.asyncio
@pytest.mark.parametrize("stop", ["deadline", "cancel", "disconnect"])
async def test_submit_interruption_preserves_unknown_without_retry(stop: str) -> None:
    import asyncio
    from dataclasses import replace

    from aiohttp import web
    from sinyuk_nodes.features.image_api.errors import GrsaiError

    accepted = asyncio.Event()
    release = asyncio.Event()
    count = 0

    async def submit(request: web.Request) -> web.Response:
        nonlocal count
        await request.read()
        count += 1
        accepted.set()
        if stop == "disconnect":
            assert request.transport is not None
            request.transport.close()
            return web.Response()
        await release.wait()
        return web.json_response({"id": "late-id", "status": "running"})

    app = web.Application()
    app.router.add_post("/v1/api/generate", submit)
    runner = web.AppRunner(app)
    await runner.setup()
    try:
        await web.TCPSite(runner, "127.0.0.1", 0).start()
        config = get_config()
        config = replace(
            config,
            base_url=f"http://127.0.0.1:{runner.addresses[0][1]}",
            transport=replace(
                config.transport, task_timeout_seconds=0.2 if stop == "deadline" else None
            ),
        )
        client = GrsaiClient(config, "test-key")
        task = asyncio.create_task(client.generate({"prompt": "test"}))
        await asyncio.wait_for(accepted.wait(), 2)
        if stop == "cancel":
            task.cancel()
        with pytest.raises(asyncio.CancelledError if stop == "cancel" else GrsaiError):
            await task
        assert count == 1
        assert client.submission_unknown
        assert client.submission_state == "unknown"
        assert (
            client.exit_reason
            == {"deadline": "timeout", "cancel": "interrupted", "disconnect": "failed"}[stop]
        )
    finally:
        release.set()
        await runner.cleanup()


@pytest.mark.asyncio
@pytest.mark.parametrize("owner", ["single", "batch"])
async def test_cdn_download_respects_environment_proxy(owner, monkeypatch, tmp_path) -> None:
    from io import BytesIO

    from aiohttp import web
    from PIL import Image
    from sinyuk_nodes.features.image_api.batch_plan import BatchPlan
    from sinyuk_nodes.features.image_api.batch_runner import BatchRunner
    from sinyuk_nodes.features.image_api.references import ReferenceSet, Source

    buffer = BytesIO()
    Image.new("RGB", (2, 2)).save(buffer, format="PNG")
    received = []

    async def proxy(request: web.Request) -> web.Response:
        assert "Authorization" not in request.headers
        received.append(request.raw_path)
        return web.Response(body=buffer.getvalue(), content_type="image/png")

    app = web.Application()
    app.router.add_route("*", "/{path:.*}", proxy)
    server = web.AppRunner(app)
    await server.setup()
    try:
        await web.TCPSite(server, "127.0.0.1", 0).start()
        monkeypatch.setenv("http_proxy", f"http://127.0.0.1:{server.addresses[0][1]}")
        monkeypatch.setenv("no_proxy", "")
        monkeypatch.delenv("NO_PROXY", raising=False)
        config = get_config()
        url = "http://cdn.invalid/image.png"
        if owner == "single":
            client = GrsaiClient(config, "test-key")
            async with client._sessions() as (_, cdn):
                image = await client._download(cdn, url, 1)
                assert image.shape == (1, 2, 2, 3)
        else:

            async def generate(client, request):
                assert client.sessions is not None
                image = await client._download(client.sessions[1], url, 1)
                assert client.result is not None
                await client.result(image, 1, 1)
                return [image]

            monkeypatch.setattr(GrsaiClient, "generate", generate)
            source = Source("input.png", None, 1, "0" * 64)
            references = ReferenceSet((torch.zeros((1, 2, 2, 3)),), (source,))
            plan = BatchPlan((references,), ("prompt",), "prompts", 1)
            parameters = {
                k: v.default for k, v in config.profile(config.default_model).parameters.items()
            }
            batch = BatchRunner(
                config, "test-key", plan, config.default_model, parameters, 2, "result", tmp_path
            )
            images, _ = await batch.run()
            assert len(images) == 1
        assert received == [url]
    finally:
        await server.cleanup()
