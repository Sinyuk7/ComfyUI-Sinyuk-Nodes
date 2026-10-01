"""Non-persistent bounded execution for Image API prompt batches."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Literal

import aiohttp
import torch

from .client import Config, GrsaiClient
from .images import encode_file_payloads
from .request_builder import build_request, build_runninghub_request
from .runninghub_client import RunningHubClient
from .runninghub_config import get_runninghub_catalog


@dataclass(frozen=True)
class GenerationFailure:
    task_index: int
    prompt: str
    error: str
    task_id: str | None


@dataclass(frozen=True)
class GenerationBatchResult:
    images: tuple[torch.Tensor, ...]
    prompts: tuple[str, ...]
    failures: tuple[GenerationFailure, ...]
    submitted: bool


Progress = Callable[[dict[str, object]], Awaitable[None]]


class GenerateRunner:
    def __init__(
        self,
        config: Config,
        key: str,
        prompts: tuple[str, ...],
        model: str,
        parameters: Mapping[str, object],
        files: list[bytes],
        concurrency: int,
        check_cancel: Callable[[], None],
        progress: Progress | None = None,
        provider: Literal["grsai", "runninghub"] = "grsai",
        endpoint: str = "/v1/api/generate",
        mask_url: str | None = None,
    ) -> None:
        if type(concurrency) is not int or not 1 <= concurrency <= 100:
            raise ValueError("Max concurrency must be an integer from 1 to 100.")
        if not prompts:
            raise ValueError("At least one generation prompt is required.")
        self.config = config
        self.key = key
        self.prompts = prompts
        self.model = model
        self.parameters = parameters
        self.files = files
        self.concurrency = min(concurrency, len(prompts))
        self.check_cancel = check_cancel
        self.progress = progress
        self.provider = provider
        self.endpoint = endpoint
        self.mask_url = mask_url
        self.completed = 0
        self.submitted = False
        self._lock = asyncio.Lock()

    async def _emit(self, failures: list[GenerationFailure]) -> None:
        if self.progress is None:
            return
        async with self._lock:
            completed = self.completed
            payload = {
                "stage": "running" if completed < len(self.prompts) else "completed",
                "completed": completed,
                "total": len(self.prompts),
                "success": len(self.prompts) - len(failures),
                "failed": len(failures),
                "running": len(self.prompts) - completed,
                "overall_progress": completed * 100 / len(self.prompts),
            }
        await self.progress(payload)

    async def _task(
        self,
        index: int,
        prompt: str,
        sessions: tuple[aiohttp.ClientSession, aiohttp.ClientSession],
        images: list[list[torch.Tensor]],
        failures: list[GenerationFailure],
    ) -> None:
        client = None

        async def result(image: torch.Tensor, _result_index: int, _result_count: int) -> None:
            images[index].append(image)

        try:
            self.check_cancel()
            if self.provider == "runninghub":
                catalog = get_runninghub_catalog()
                client = RunningHubClient(
                    self.config,
                    self.key,
                    self.check_cancel,
                    accepted=None,
                    result=result,
                    sessions=sessions,
                    endpoint=self.endpoint,
                )
                urls = await client.upload_images(self.files)
                request = build_runninghub_request(
                    self.model, prompt, self.parameters, urls, catalog, self.mask_url
                )
            else:
                request = build_request(
                    self.model,
                    prompt,
                    self.parameters,
                    encode_file_payloads(self.files, self.config.transport.image_encoding),
                    self.config,
                    self.mask_url,
                )
                client = GrsaiClient(
                    self.config,
                    self.key,
                    self.check_cancel,
                    accepted=None,
                    result=result,
                    sessions=sessions,
                )
            self.submitted = True
            await client.generate(request)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            failures.append(
                GenerationFailure(index + 1, prompt, str(exc), client.task_id if client else None)
            )
        finally:
            async with self._lock:
                self.completed += 1
            await self._emit(failures)

    async def run(self) -> GenerationBatchResult:
        images: list[list[torch.Tensor]] = [[] for _ in self.prompts]
        failures: list[GenerationFailure] = []
        iterator = iter(enumerate(self.prompts))
        semaphore = asyncio.Semaphore(self.concurrency)

        async with aiohttp.ClientSession() as api, aiohttp.ClientSession(trust_env=True) as cdn:

            async def worker() -> None:
                while True:
                    async with semaphore:
                        item = next(iterator, None)
                        if item is None:
                            return
                        index, prompt = item
                        await self._task(index, prompt, (api, cdn), images, failures)

            await asyncio.gather(*(worker() for _ in range(self.concurrency)))

        output_images: list[torch.Tensor] = []
        output_prompts: list[str] = []
        for prompt, task_images in zip(self.prompts, images, strict=True):
            output_images.extend(task_images)
            output_prompts.extend([prompt] * len(task_images))
        return GenerationBatchResult(
            tuple(output_images), tuple(output_prompts), tuple(failures), self.submitted
        )
