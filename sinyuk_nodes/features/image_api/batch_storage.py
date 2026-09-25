"""Single-owner in-memory manifest and atomic files, without background writers."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import tempfile
from collections.abc import Callable, Mapping
from contextlib import suppress
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import NotRequired, TypedDict
from uuid import uuid4

import torch
from PIL import Image

from .batch_plan import BatchPlan


class OutputRecord(TypedDict):
    result_index: int
    file: str
    flat_output_index: int | None


class TaskRecord(TypedDict):
    task_index: int
    base_index: int
    prompt_index: int
    status: str
    remote_task_id: str | None
    remote_status: str | None
    submitted: bool
    error: str | None
    outputs: list[OutputRecord]
    submission_state: NotRequired[str]


class BatchSummary(TypedDict):
    succeeded_tasks: int
    failed_tasks: int
    pending_tasks: int
    interrupted_tasks: int
    saved_images: int


class BatchManifest(TypedDict):
    schema_version: int
    batch_id: str
    started_at: str
    finished_at: str | None
    status: str
    provider: str
    model: str
    endpoint: str | None
    parameters: dict[str, str]
    reference_count: int
    base_count: int
    prompt_count: int
    total_tasks: int
    prompt_source: str
    max_concurrency: int
    output_prefix: str
    bases: list[dict[str, object]]
    prompts: list[dict[str, object]]
    tasks: list[TaskRecord]
    error: NotRequired[str | None]
    summary: NotRequired[BatchSummary]


class StorageError(RuntimeError):
    pass


def timestamp() -> str:
    return datetime.now(UTC).isoformat()


def sanitized(value: object, key: str) -> object:
    if isinstance(value, str):
        return value.replace(key, "[redacted]")
    if isinstance(value, dict):
        return {k: sanitized(v, key) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitized(v, key) for v in value]
    return value


class BatchStore:
    def __init__(
        self,
        output_root: Path,
        plan: BatchPlan,
        model: str,
        parameters: Mapping[str, str],
        concurrency: int,
        prefix: str,
        api_key: str,
        provider: str = "grsai",
        endpoint: str | None = None,
    ) -> None:
        self.prefix = prefix
        self.api_key = api_key
        self.lock = asyncio.Lock()
        self.images: dict[tuple[int, int], torch.Tensor] = {}
        self.path = (
            Path(output_root)
            / "image_api"
            / (datetime.now(UTC).strftime("%Y%m%d_%H%M%S_") + uuid4().hex[:12])
        )
        try:
            self.path.mkdir(parents=True, exist_ok=False)
        except OSError:
            raise StorageError(
                "Cannot create batch output directory; no generation was submitted."
            ) from None
        bases: list[dict[str, object]] = []
        for base in range(1, plan.base_count + 1):
            mapping: list[dict[str, object]] = []
            for col, column in enumerate(plan.columns):
                index = plan.input_index(col, base)
                mapping.append(
                    {
                        "reference": col + 1,
                        "input_index": index + 1,
                        "reused": len(column.images) == 1,
                        "source": asdict(column.sources[index]),
                    }
                )
            bases.append({"base_index": base, "references": mapping})
        task_records: list[TaskRecord] = [
            {
                **asdict(task),
                "status": "pending",
                "remote_task_id": None,
                "remote_status": None,
                "submitted": False,
                "error": None,
                "outputs": [],
            }
            for task in plan.tasks()
        ]
        self.state: BatchManifest = {
            "schema_version": 1,
            "batch_id": self.path.name,
            "started_at": timestamp(),
            "finished_at": None,
            "status": "planned",
            "provider": provider,
            "model": model,
            "endpoint": endpoint,
            "parameters": dict(parameters),
            "reference_count": len(plan.columns),
            "base_count": plan.base_count,
            "prompt_count": len(plan.prompts),
            "total_tasks": plan.total,
            "prompt_source": plan.prompt_source,
            "max_concurrency": concurrency,
            "output_prefix": prefix,
            "bases": bases,
            "prompts": [
                {"prompt_index": i, "sha256": hashlib.sha256(p.encode()).hexdigest()}
                for i, p in enumerate(plan.prompts, 1)
            ],
            "tasks": task_records,
        }
        self._write()

    @property
    def manifest(self) -> Path:
        return self.path / "manifest.json"

    def _atomic(self, destination: Path, write: Callable[[object], object]) -> None:
        temporary = None
        try:
            fd, temporary = tempfile.mkstemp(prefix=".image-api-", dir=self.path)
            with os.fdopen(fd, "wb") as stream:
                write(stream)
                stream.flush()
            os.replace(temporary, destination)
        except (OSError, ValueError):
            raise StorageError(
                "Batch storage unavailable; already saved files were retained."
            ) from None
        finally:
            if temporary:
                with suppress(OSError):
                    Path(temporary).unlink(missing_ok=True)

    def _write(self) -> None:
        data = json.dumps(sanitized(self.state, self.api_key), ensure_ascii=False, indent=2).encode(
            "utf-8"
        )
        self._atomic(self.manifest, lambda stream: stream.write(data))

    async def update(self, index: int, **changes: object) -> None:
        async with self.lock:
            self.state["tasks"][index - 1].update(changes)
            self._write()

    async def save(
        self, task_index: int, image: torch.Tensor, result_index: int, result_count: int
    ) -> None:
        async with self.lock:
            suffix = f"_{result_index:02d}" if result_count > 1 else ""
            name = f"{self.prefix}_{task_index:03d}{suffix}.png"
            pixels = (image[0].detach().cpu().float().numpy() * 255).round().astype("uint8")
            self._atomic(
                self.path / name, lambda stream: Image.fromarray(pixels).save(stream, format="PNG")
            )
            self.images[(task_index, result_index)] = image
            self.state["tasks"][task_index - 1]["outputs"].append(
                {
                    "result_index": result_index,
                    "file": name,
                    "flat_output_index": None,
                }
            )
            self._write()

    async def finish(
        self, status: str, error: str | None = None, interrupted_tasks: bool = False
    ) -> list[torch.Tensor]:
        async with self.lock:
            self.state.update(status=status, finished_at=timestamp(), error=error)
            if interrupted_tasks:
                for task in self.state["tasks"]:
                    if task["status"] == "running":
                        task["status"] = "interrupted"
                        if task["submitted"] and not task["remote_task_id"]:
                            task["submission_state"] = "unknown"
            if not interrupted_tasks and status != "interrupted":
                for flat, (task, result) in enumerate(sorted(self.images)):
                    for output in self.state["tasks"][task - 1]["outputs"]:
                        if output["result_index"] == result:
                            output["flat_output_index"] = flat
            tasks = self.state["tasks"]
            self.state["summary"] = {
                "succeeded_tasks": sum(t["status"] == "succeeded" for t in tasks),
                "failed_tasks": sum(
                    t["status"] in {"failed", "partial", "submission_unknown"} for t in tasks
                ),
                "pending_tasks": sum(t["status"] == "pending" for t in tasks),
                "interrupted_tasks": sum(t["status"] == "interrupted" for t in tasks),
                "saved_images": len(self.images),
            }
            self._write()
        return [self.images[key] for key in sorted(self.images)]
