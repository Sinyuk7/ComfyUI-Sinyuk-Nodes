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
