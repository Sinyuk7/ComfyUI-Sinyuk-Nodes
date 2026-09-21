# ComfyUI-Sinyuk-Nodes

An extensible ComfyUI V3 custom node pack by Sinyuk.

This repository provides a focused OpenAI-compatible vision LLM workflow for
ComfyUI V3. Connection settings and model selection live in an `API Config`
node, and prompts are sent through the `LLM API` node.

## Status

The published nodes are `API Config` and `LLM API`.

The project supports the ComfyUI V3 API only. Legacy V1 registration is not
supported.

## Architecture

```text
ComfyUI
  -> compat / extension / nodes
  -> features
  -> common
```

`compat/` is the only import boundary for the ComfyUI API. Node adapters belong
in `nodes/`; feature logic should remain independent of concrete node classes;
LLM image upload preprocessing lives in `common/llm_image.py`.

Node registration is explicit in `sinyuk_nodes/registry.py`. Directory scanning,
reflection-based discovery, and legacy `NODE_CLASS_MAPPINGS` registration are not
used.

## Development

Install development tools into the environment used by ComfyUI:

```bash
python -m pip install -r requirements-dev.txt
```

Set `COMFYUI_PATH` to a ComfyUI checkout when running tests that import the V3
API:

```bash
COMFYUI_PATH=/path/to/ComfyUI python -m pytest
COMFYUI_PATH=/path/to/ComfyUI pyright
ruff check .
ruff format --check .
```

Published node IDs are workflow compatibility contracts and must remain stable.
Changes to node schemas, inputs, outputs, or behavior should be documented in
the repository's canonical documentation when that durable information is
needed.
