"""ComfyUI V3 adapter for one ordered, non-cached generation request."""

from __future__ import annotations

import asyncio
import logging
import time

from comfy_api.latest import io
from sinyuk_nodes.common.cancellation import CancellationState, run_blocking
from sinyuk_nodes.compat.comfy import check_interrupt

from ...features.image_api.client import GrsaiClient
from ...features.image_api.config import get_config
from ...features.image_api.diagnostics import log_event, new_run_id
from ...features.image_api.errors import clean_message
from ...features.image_api.images import (
    encode_file_payloads,
    encode_image_files,
    validate_reference_files,
)
from ...features.image_api.request_builder import build_request, build_runninghub_request
from ...features.image_api.runninghub_client import RunningHubClient
from ...features.image_api.runninghub_config import get_runninghub_catalog
from ...features.image_api.workflow import normalize_inputs, provider_profile
from .config import APIConfigType
from .schema import model_input_options


class ImageGenerate(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        config = get_config()
        options = model_input_options()
        return io.Schema(
            node_id="Sinyuk.ImageAPI.Generate",
            display_name="Image API Generate",
            category="Image API",
            description="Generate images with a compatible asynchronous image API.",
            search_aliases=["GRSAI Image Generate", "RunningHub Image Generate"],
            inputs=[
                APIConfigType.Input(
                    "api_config",
                    display_name="Config",
                    tooltip="Connect an API Config node.",
                ),
                io.DynamicCombo.Input(
                    "model",
                    display_name="Model",
                    options=options,
                    extra_dict={"default": config.default_model},
                    tooltip=(
                        "Select a curated edit model for the connected Provider. "
                        "Availability checks are advisory only."
                    ),
                ),
                io.String.Input(
                    "prompt",
                    display_name="Prompt",
                    default="",
                    multiline=True,
                    dynamic_prompts=False,
                    tooltip=(
                        "Instructions for editing or generating from the connected reference "
                        "images."
                    ),
                ),
                io.Image.Input(
                    "images",
                    display_name="Images",
                    tooltip="Required reference images sent together in order (1-10 PNG images).",
                ),
            ],
            outputs=[
                io.Image.Output(
                    "images",
                    display_name="Images",
                    is_output_list=True,
                    tooltip=(
                        "Generated images in provider result order. Connect to Preview Image, "
                        "Save Image, or image processing nodes."
                    ),
                )
            ],
            hidden=[io.Hidden.unique_id, io.Hidden.extra_pnginfo],
            is_input_list=True,
            not_idempotent=True,
        )

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        return float("nan")

    @classmethod
    async def execute(cls, api_config=None, model=None, prompt=None, images=None):
        from comfy import model_management

        from .host import execution_ui

        settings, selected, text, parameters = normalize_inputs(api_config, model, prompt)
        config = settings.apply(get_config())
        profile = provider_profile(settings, selected)
        run_id = new_run_id()
        started = time.monotonic()
        node_id = str(cls.hidden.unique_id)
        log_event("generation.started", run_id=run_id, node_id=node_id, model=selected)
        # Validate cheap fields before encoding potentially large images.
        client = None
        ui = None
        interrupted = False

        cancellation = CancellationState(check_interrupt)
        check_cancel = cancellation.check

        try:
            check_cancel()
            enforce_size_limits = settings.provider != "runninghub"
            files = await cancellation.wait(
                run_blocking(encode_image_files, images, enforce_size_limits=enforce_size_limits)
            )
            if not files:
                raise ValueError("Connect 1 to 10 reference images.")
            validate_reference_files(files, enforce_size_limits)
            ui = execution_ui(cls.hidden, config, settings.token, settings.provider)
            if settings.provider == "runninghub":
                build_runninghub_request(
                    selected, text, parameters, ["pending"] * len(files), get_runninghub_catalog()
                )
                client = RunningHubClient(
                    config,
                    settings.api_key,
                    check_cancel,
                    ui.progress,
                    endpoint=profile.endpoint,
                    log_context={"run_id": run_id, "node_id": node_id},
                )
                urls = await client.upload_images(files)
                request = build_runninghub_request(
                    selected, text, parameters, urls, get_runninghub_catalog()
                )
            else:
                encoded = encode_file_payloads(files, config.transport.image_encoding)
                request = build_request(selected, text, parameters, encoded, config)
                client = GrsaiClient(
                    config,
                    settings.api_key,
                    check_cancel,
                    ui.progress,
                    log_context={"run_id": run_id, "node_id": node_id},
                )
            result = await client.generate(request)
            log_event(
                "generation.succeeded",
                run_id=run_id,
                node_id=node_id,
                task_id=client.task_id,
                outputs=len(result),
                elapsed_ms=round((time.monotonic() - started) * 1000),
            )
            return io.NodeOutput(result)
        except (model_management.InterruptProcessingException, asyncio.CancelledError):
            interrupted = True
            if ui:
                ui.stale()
            log_event(
                "generation.interrupted",
                level=logging.WARNING,
                run_id=run_id,
                node_id=node_id,
                task_id=client.task_id if client else None,
                elapsed_ms=round((time.monotonic() - started) * 1000),
            )
            raise
        except Exception as exc:
            log_event(
                "generation.failed",
                level=logging.ERROR,
                run_id=run_id,
                node_id=node_id,
                task_id=client.task_id if client else None,
                error_type=type(exc).__name__,
                error=clean_message(str(exc), (settings.api_key, text)),
                elapsed_ms=round((time.monotonic() - started) * 1000),
            )
            raise
        finally:
            # Scheduling failure must never replace the generation result or its original error.
            if (
                client
                and client.submitted
                and not interrupted
                and not model_management.processing_interrupted()
            ):
                ui.refresh_balance()
