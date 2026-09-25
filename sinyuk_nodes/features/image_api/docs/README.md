# Image API feature

This feature provides asynchronous image generation through GRSAI and RunningHub, single-task and bounded batch workflows, and ordered folder references. ComfyUI schemas and execution adapters live in `sinyuk_nodes/nodes/image_api`; provider requests, image conversion, batch planning, and persistence live beside this document.

The packaged `config.example.json` supplies GRSAI defaults. Local overrides use `image_api_config.json` in this module directory. RunningHub's curated model catalog is `runninghub_catalog.json`. These files are initialized from this repository; settings from the former standalone extension are not imported.

The browser extension in `web/` provides provider selection, task progress, batch progress, balance status, and model availability hints. The feature logs lifecycle diagnostics in the ComfyUI user directory under `image_api/logs/`. A stopped ComfyUI prompt cancels local waits; accepted provider tasks may continue remotely.
