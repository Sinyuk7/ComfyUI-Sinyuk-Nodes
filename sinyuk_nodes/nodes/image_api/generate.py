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
from ...features.image_api.generate_runner import GenerateRunner
from ...features.image_api.images import (
    encode_file_payloads,
    encode_image_files,
    image_size,
    prepare_mask,
    validate_reference_files,
)
from ...features.image_api.prompt_expansion import expand_prompts
from ...features.image_api.request_builder import build_request, build_runninghub_request
from ...features.image_api.runninghub_client import RunningHubClient, upload_mask_via_runninghub
from ...features.image_api.runninghub_config import get_runninghub_catalog
from ...features.image_api.workflow import normalize_inputs, provider_profile, scalar
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
                io.Combo.Input(
                    "generation_mode",
                    options=["Repeat", "Variants"],
                    default="Repeat",
                    display_name="Generation Mode",
                    tooltip="Repeat the same generation inputs or expand one prompt placeholder.",
                ),
                io.Int.Input(
                    "generation_count",
                    default=1,
                    min=1,
                    max=10,
                    display_name="Generation Count",
                    tooltip="Number of independent requests in Repeat mode.",
                ),
                io.String.Input(
                    "placeholder",
                    default="frame",
                    display_name="Placeholder",
                    tooltip="Name used to form the exact [name] placeholder in Variants mode.",
                ),
                io.String.Input(
                    "variants",
                    optional=True,
                    force_input=True,
                    multiline=True,
                    display_name="Variants",
                    tooltip="STRING list of complete replacement values for the placeholder.",
                ),
                io.Int.Input(
                    "max_concurrency",
                    default=10,
                    min=1,
                    max=100,
                    display_name="Max Concurrency",
                    tooltip="Maximum independent generation requests running at once.",
                ),
                io.Image.Input(
                    "images",
                    display_name="Images",
                    tooltip="Required reference images sent together in order (1-10 PNG images).",
                ),
                io.Mask.Input(
                    "mask",
                    optional=True,
                    display_name="MASK",
                    tooltip=(
                        "Optional edit mask for GPT Image 2.5. It applies only to the first "
                        "input image and may be resized proportionally."
                    ),
                ),
                io.String.Input(
                    "external_aspect_ratio",
                    display_name="Aspect Ratio Override",
                    optional=True,
                    force_input=True,
                    tooltip=(
                        "Optional logical aspect ratio override, such as 16:9. When connected, "
                        "it takes precedence over the model's manual aspect ratio."
                    ),
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
                ),
                io.String.Output(
                    "expanded_prompt",
                    display_name="Expanded Prompt",
                    is_output_list=True,
                    tooltip="Final prompt corresponding to each successfully returned image.",
                ),
            ],
            hidden=[io.Hidden.unique_id, io.Hidden.extra_pnginfo],
            is_input_list=True,
            not_idempotent=True,
        )

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        return float("nan")

    @classmethod
    async def execute(
        cls,
        api_config=None,
        model=None,
        prompt=None,
        images=None,
        mask=None,
        external_aspect_ratio=None,
        generation_mode=None,
        generation_count=None,
        placeholder=None,
        variants=None,
        max_concurrency=None,
    ):
        from comfy import model_management

        from .host import execution_ui

        settings, selected, text, parameters = normalize_inputs(api_config, model, prompt)
        mode = scalar(generation_mode or ["Repeat"], "generation_mode")
        count = scalar(generation_count or [1], "generation_count")
        placeholder_name = scalar(placeholder or ["frame"], "placeholder")
        concurrency = scalar(max_concurrency or [10], "max_concurrency")
        prompts = expand_prompts(text, mode, count, placeholder_name, variants)
        task_prompt = next(iter(prompts), None)
        if task_prompt is None:
            raise ValueError("At least one generation prompt is required.")
        if external_aspect_ratio is not None:
            if settings.provider != "runninghub":
                raise ValueError(
                    "External aspect ratio is currently supported only for RunningHub."
                )
            external_ratio = scalar(external_aspect_ratio, "external_aspect_ratio")
            if not isinstance(external_ratio, str) or not external_ratio.strip():
                raise ValueError("Connected external aspect ratio must be a nonempty string.")
            parameters["aspectRatio"] = external_ratio
        config = settings.apply(get_config())
        profile = provider_profile(settings, selected)
        run_id = new_run_id()
        started = time.monotonic()
        node_id = str(cls.hidden.unique_id)
        log_event("generation.started", run_id=run_id, node_id=node_id, model=selected)
        # Validate cheap fields before encoding potentially large images.
        client = None
        submitted = False
        ui = execution_ui(cls.hidden, config, settings.token, settings.provider)
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
            mask_url: str | None = None
            if mask is not None:
                supports_mask = (
                    settings.provider == "grsai" and profile.family == "gpt_image"
                ) or (
                    settings.provider == "runninghub" and selected.startswith("rh:gpt-image-2.5-")
                )
                if not supports_mask:
                    raise ValueError("The selected model does not support a native MASK.")
                mask_value = scalar(mask, "mask") if isinstance(mask, list) else mask
                target_size = await cancellation.wait(run_blocking(image_size, files[0]))
                mask_png = await cancellation.wait(
                    run_blocking(prepare_mask, mask_value, target_size)
                )
                mask_url = await upload_mask_via_runninghub(
                    mask_png,
                    get_runninghub_catalog().base_url,
                    settings.api_key,
                    config,
                    check_cancel,
                )
            if len(prompts) > 1:
                if settings.provider == "runninghub":
                    build_runninghub_request(
                        selected,
                        task_prompt,
                        parameters,
                        ["pending"] * len(files),
                        get_runninghub_catalog(),
                        mask_url,
                    )
                else:
                    encoded = encode_file_payloads(files, config.transport.image_encoding)
                    build_request(selected, prompts[0], parameters, encoded, config, mask_url)
                runner = GenerateRunner(
                    config,
                    settings.api_key,
                    prompts,
                    selected,
                    parameters,
                    files,
                    concurrency,
                    check_cancel,
                    ui.batch_progress,
                    provider=settings.provider,
                    endpoint=profile.endpoint
                    if settings.provider == "runninghub"
                    else "/v1/api/generate",
                    mask_url=mask_url,
                )
                batch = await runner.run()
                submitted = batch.submitted
                for failure in batch.failures:
                    log_event(
                        "generation.task_failed",
                        level=logging.WARNING,
                        run_id=run_id,
                        node_id=node_id,
                        task_index=failure.task_index,
                        task_id=failure.task_id,
                        error=clean_message(failure.error, (settings.api_key, failure.prompt)),
                    )
                if not batch.images:
                    raise ValueError(
                        f"All {len(prompts)} generation tasks failed; "
                        "no valid images were returned."
                    )
                result = list(batch.images)
                result_prompts = list(batch.prompts)
            else:
                if settings.provider == "runninghub":
                    build_runninghub_request(
                        selected,
                        task_prompt,
                        parameters,
                        ["pending"] * len(files),
                        get_runninghub_catalog(),
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
                        selected,
                        task_prompt,
                        parameters,
                        urls,
                        get_runninghub_catalog(),
                        mask_url,
                    )
                else:
                    encoded = encode_file_payloads(files, config.transport.image_encoding)
                    request = build_request(
                        selected, task_prompt, parameters, encoded, config, mask_url
                    )
                    client = GrsaiClient(
                        config,
                        settings.api_key,
                        check_cancel,
                        ui.progress,
                        log_context={"run_id": run_id, "node_id": node_id},
                    )
                result = await client.generate(request)
                submitted = client.submitted
                result_prompts = [task_prompt] * len(result)
            log_event(
                "generation.succeeded",
                run_id=run_id,
                node_id=node_id,
                task_id=client.task_id if client else None,
                outputs=len(result),
                elapsed_ms=round((time.monotonic() - started) * 1000),
            )
            return io.NodeOutput(result, result_prompts)
        except (model_management.InterruptProcessingException, asyncio.CancelledError):
            interrupted = True
            await ui.progress("interrupted", None, client.task_id if client else None)
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
            await ui.progress("failed", None, client.task_id if client else None)
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
                (client and client.submitted or submitted)
                and not interrupted
                and not model_management.processing_interrupted()
            ):
                ui.refresh_balance()
