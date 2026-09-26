"""Bounded task lifecycles with submit-once transport and incremental persistence."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import sys
import time
from collections.abc import Awaitable, Callable, Mapping
from contextlib import suppress
from pathlib import Path
from typing import Literal

import aiohttp
import torch
from sinyuk_nodes.common.cancellation import CancellationState, drain, finish_owned, run_blocking

from .batch_plan import BatchPlan, TaskSpec, validate_options
from .batch_storage import BatchStore, StorageError
from .client import Config, GrsaiClient
from .diagnostics import log_event, new_run_id
from .errors import GrsaiError, clean_message
from .images import encode_file_payloads, encode_image_files, validate_reference_files
from .request_builder import build_request, build_runninghub_request, normalize_key
from .runninghub_client import RunningHubClient
from .runninghub_config import get_runninghub_catalog

RUNNINGHUB_UPLOAD_CONCURRENCY = 8


class BatchRunner:
    def __init__(
        self,
        config: Config,
        key: str,
        plan: BatchPlan,
        model: str,
        parameters: Mapping[str, str],
        concurrency: int,
        prefix: str,
        output_root: Path,
        check_cancel: Callable[[], None] = lambda: None,
        progress: Callable[[dict[str, object]], Awaitable[None]] | None = None,
        run_id: str | None = None,
        node_id: str | None = None,
        provider: Literal["grsai", "runninghub"] = "grsai",
    ) -> None:
        self.config = config
        self.key = normalize_key(key)
        validate_options(concurrency, prefix, self.key)
        if provider not in {"grsai", "runninghub"}:
            raise ValueError("Unsupported image API provider.")
        if provider == "grsai":
            for prompt in plan.prompts:
                build_request(model, prompt, parameters, ["pending"], config)
        else:
            catalog = get_runninghub_catalog()
            for prompt in plan.prompts:
                build_runninghub_request(model, prompt, parameters, ["pending"], catalog)
        self.plan, self.model, self.parameters = plan, model, parameters
        self.provider = provider
        self.runninghub_profile = (
            get_runninghub_catalog().profile(model) if provider == "runninghub" else None
        )
        self.cancellation = CancellationState(check_cancel)
        self.concurrency, self.check_cancel, self.progress = (
            concurrency,
            self.cancellation.check,
            progress,
        )
        self.encode_lock = asyncio.Lock()
        self.store = BatchStore(
            output_root,
            plan,
            model,
            parameters,
            concurrency,
            prefix,
            self.key,
            provider,
            self.runninghub_profile.endpoint if self.runninghub_profile else "/v1/api/generate",
        )
        self.run_id = run_id or new_run_id()
        self.node_id = node_id
        self.submitted = False
        self.cooldown_until = 0
        self.fatal: str | None = None
        self.encoded: dict[tuple[int, int], bytes] = {}
        self.uploaded: dict[str, asyncio.Task[str]] = {}
        self.upload_lock = asyncio.Lock()
        self.upload_semaphore = asyncio.Semaphore(RUNNINGHUB_UPLOAD_CONCURRENCY)
        self.remaining = {base: len(plan.prompts) for base in range(1, plan.base_count + 1)}
        self.active: dict[int, dict[str, object]] = {}

    async def _emit(self, stage: str) -> None:
        if self.progress:
            tasks = self.store.state["tasks"]
            success = sum(t["status"] == "succeeded" for t in tasks)
            failed = sum(t["status"] in {"failed", "partial", "submission_unknown"} for t in tasks)
            active = [self.active[index] for index in sorted(self.active)]
            reported = [
                item["progress"]
                for item in active
                if item.get("stage") == "running"
                and type(item.get("progress")) in (int, float)
                and 0 <= item["progress"] <= 100
            ]
            overall_progress = (
                ((success + failed) + sum(value / 100 for value in reported))
                * 100
                / self.plan.total
            )
            with suppress(Exception):
                await self.progress(
                    {
                        "stage": stage,
                        "base_count": self.plan.base_count,
                        "prompt_count": len(self.plan.prompts),
                        "total": self.plan.total,
                        "reference_count": len(self.plan.columns),
                        "model": self.model,
                        "concurrency": self.concurrency,
                        "completed": success + failed,
                        "overall_progress": min(100, overall_progress),
                        "success": success,
                        "failed": failed,
                        "running": len(active),
                        "active": active,
                        "directory": str(self.store.path).replace(self.key, "[redacted]"),
                        "manifest": str(self.store.manifest).replace(self.key, "[redacted]"),
                    }
                )

    async def _pressure(self, delay: float) -> None:
        self.cooldown_until = max(self.cooldown_until, time.monotonic() + delay)

    async def _wait_to_submit(self) -> None:
        while True:
            self.check_cancel()
            remaining = self.cooldown_until - time.monotonic()
            if self.fatal or remaining <= 0:
                return
            await asyncio.sleep(min(remaining, 0.2))

    async def _images(self, base: int) -> list[bytes]:
        images: list[bytes] = []
        for col, column in enumerate(self.plan.columns):
            index = self.plan.input_index(col, base)
            cache_key = col, index
            async with self.encode_lock:
                self.check_cancel()
                if cache_key not in self.encoded:
                    files = await run_blocking(
                        encode_image_files,
                        [column.images[index]],
                        enforce_size_limits=self.provider != "runninghub",
                    )
                    self.check_cancel()
                    self.encoded[cache_key] = files[0]
            images.append(self.encoded[cache_key])
        return images

    async def _runninghub_upload(self, client: RunningHubClient, content: bytes, index: int) -> str:
        async with self.upload_semaphore:
            return await client.upload_image(client.sessions[0], content, index)

    async def _runninghub_url(self, client: RunningHubClient, content: bytes, index: int) -> str:
        digest = hashlib.sha256(content).hexdigest()
        async with self.upload_lock:
            upload = self.uploaded.get(digest)
            if upload is None:
                upload = asyncio.create_task(self._runninghub_upload(client, content, index))
                self.uploaded[digest] = upload
        try:
            return await asyncio.shield(upload)
        except BaseException:
            if upload.done():
                async with self.upload_lock:
                    if self.uploaded.get(digest) is upload:
                        self.uploaded.pop(digest, None)
            raise

    async def _runninghub_urls(self, client: RunningHubClient, files: list[bytes]) -> list[str]:
        return await asyncio.gather(
            *(
                self._runninghub_url(client, content, index)
                for index, content in enumerate(files, 1)
            )
        )

    def _release(self, base: int) -> None:
        self.remaining[base] -= 1
        if not self.remaining[base]:
            for col, column in enumerate(self.plan.columns):
                if len(column.images) > 1:
                    self.encoded.pop((col, base - 1), None)

    async def _task(
        self,
        spec: TaskSpec,
        sessions: tuple[aiohttp.ClientSession, aiohttp.ClientSession],
    ) -> None:
        client = None

        async def task_progress(
            stage: str, value: int | float | None, _task_id: str | None, details: dict[str, object]
        ) -> None:
            self.active[spec.task_index] = {
                "task_index": spec.task_index,
                "stage": stage,
                "progress": value,
                **details,
            }
            await self._emit("running")

        async def accepted(task_id: str, remote_status: str | None) -> None:
            await self.store.update(
                spec.task_index, remote_task_id=task_id, remote_status=remote_status, submitted=True
            )

        async def result(image: torch.Tensor, index: int, count: int) -> None:
            await self.store.save(spec.task_index, image, index, count)

        try:
            files = await self._images(spec.base_index)
            validate_reference_files(files, self.provider != "runninghub")
            # Re-check pressure and fatal state after image preparation.
            await self._wait_to_submit()
            if self.fatal:
                return
            self.check_cancel()
            await self.store.update(spec.task_index, status="running")
            common = {
                "accepted": accepted,
                "result": result,
                "pressure": self._pressure,
                "sessions": sessions,
                "log_context": {
                    "run_id": self.run_id,
                    "node_id": self.node_id,
                    "batch_id": self.store.path.name,
                    "task_index": spec.task_index,
                },
            }
            if self.provider == "runninghub":
                build_runninghub_request(
                    self.model,
                    self.plan.prompts[spec.prompt_index - 1],
                    self.parameters,
                    ["pending"] * len(files),
                    get_runninghub_catalog(),
                )
                client = RunningHubClient(
                    self.config,
                    self.key,
                    self.check_cancel,
                    task_progress,
                    endpoint=self.runninghub_profile.endpoint,
                    **common,
                )
                client._set_phase("uploading")
                urls = await self._runninghub_urls(client, files)
                request = build_runninghub_request(
                    self.model,
                    self.plan.prompts[spec.prompt_index - 1],
                    self.parameters,
                    urls,
                    get_runninghub_catalog(),
                )
            else:
                encoded = encode_file_payloads(files, self.config.transport.image_encoding)
                request = build_request(
                    self.model,
                    self.plan.prompts[spec.prompt_index - 1],
                    self.parameters,
                    encoded,
                    self.config,
                )
                client = GrsaiClient(
                    self.config, self.key, self.check_cancel, task_progress, **common
                )
            await self._wait_to_submit()
            if self.fatal:
                await self.store.update(spec.task_index, status="pending")
                return
            self.check_cancel()
            await client.generate(request)
            await self.store.update(
                spec.task_index, status="succeeded", remote_status=client.remote_status
            )
        except GrsaiError as exc:
            error = clean_message(str(exc), (self.key, *self.plan.prompts))
            if client and client.http_status in {401, 403}:
                self.fatal = error
            outputs = self.store.state["tasks"][spec.task_index - 1]["outputs"]
            state = (
                "partial"
                if outputs
                else ("submission_unknown" if client and client.submission_unknown else "failed")
            )
            await self.store.update(
                spec.task_index,
                status=state,
                error=error,
                remote_status=client.remote_status if client else None,
            )
            log_event(
                "batch.task_failed",
                level=logging.ERROR,
                run_id=self.run_id,
                node_id=self.node_id,
                batch_id=self.store.path.name,
                task_index=spec.task_index,
                task_id=client.task_id if client else None,
                status=state,
                error=error,
            )
        except (StorageError, ValueError) as exc:
            self.fatal = clean_message(str(exc), (self.key, *self.plan.prompts))
            raise
        finally:
            if client:
                self.submitted |= client.submitted
                unwinding = sys.exc_info()[0] is not None
                try:
                    await finish_owned(
                        self.store.update(
                            spec.task_index,
                            submitted=client.submitted,
                            remote_status=client.remote_status,
                            remote_task_id=client.task_id,
                            phase=client.phase,
                            exit_reason=client.exit_reason,
                            submission_state=client.submission_state,
                            remote_cancel_result=client.remote_cancel_result,
                        )
                    )
                except StorageError:
                    if not unwinding:
                        raise
            self.active.pop(spec.task_index, None)
            self._release(spec.base_index)
        await self._emit("running")

    async def _execute(self) -> None:
        iterator = iter(self.plan.tasks())

        async def worker(sessions: tuple[aiohttp.ClientSession, aiohttp.ClientSession]) -> None:
            while not self.fatal:
                await self._wait_to_submit()
                if self.fatal:
                    break
                spec = next(iterator, None)
                if spec is None:
                    break
                await self._task(spec, sessions)

        async with aiohttp.ClientSession() as api, aiohttp.ClientSession() as cdn:
            workers = [
                asyncio.create_task(worker((api, cdn)))
                for _ in range(min(self.concurrency, self.plan.total))
            ]
            try:
                await asyncio.gather(*workers)
            finally:
                log_event("batch.cleanup.started", run_id=self.run_id)
                owned = [*workers, *self.uploaded.values()]
                for task in owned:
                    if not task.done() and not task.cancelling():
                        task.cancel()
                await drain(asyncio.gather(*owned, return_exceptions=True))
                log_event("batch.cleanup.completed", run_id=self.run_id)

    async def run(self) -> tuple[list[torch.Tensor], str]:
        started = time.monotonic()
        log_event(
            "batch.started",
            run_id=self.run_id,
            node_id=self.node_id,
            batch_id=self.store.path.name,
            model=self.model,
            tasks=self.plan.total,
            concurrency=self.concurrency,
        )
        await self._emit("planned")
        try:
            await self.cancellation.wait(self._execute())
        except BaseException as exc:
            interrupted = self.cancellation.reason is not None
            status = "interrupted" if interrupted else "failed"
            error = clean_message(str(exc), (self.key, *self.plan.prompts))
            cleanup = asyncio.create_task(self.store.finish(status, error, interrupted_tasks=True))
            await drain(cleanup)
            if not cleanup.cancelled() and cleanup.exception() is not None:
                log_event("batch.cleanup.failed", level=logging.ERROR, run_id=self.run_id)
            await drain(asyncio.create_task(self._emit(status)))
            log_event(
                "batch.interrupted" if interrupted else "batch.failed",
                level=logging.WARNING if interrupted else logging.ERROR,
                run_id=self.run_id,
                node_id=self.node_id,
                batch_id=self.store.path.name,
                error=error,
                elapsed_ms=round((time.monotonic() - started) * 1000),
            )
            raise
        finally:
            self.encoded.clear()
            self.uploaded.clear()
        tasks = self.store.state["tasks"]
        status = (
            "succeeded"
            if all(t["status"] == "succeeded" for t in tasks)
            else ("partial" if self.store.images else "failed")
        )
        images = await self.store.finish(status, self.fatal)
        await self._emit(status)
        summary = self.store.state["summary"]
        log_event(
            "batch.completed",
            level=logging.INFO if status == "succeeded" else logging.WARNING,
            run_id=self.run_id,
            node_id=self.node_id,
            batch_id=self.store.path.name,
            status=status,
            succeeded=summary["succeeded_tasks"],
            failed=summary["failed_tasks"],
            outputs=summary["saved_images"],
            elapsed_ms=round((time.monotonic() - started) * 1000),
        )
        return images, str(self.store.manifest)
