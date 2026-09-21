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
LLM image upload preprocessing lives in `features/llm/image.py`.

Node registration is explicit in `sinyuk_nodes/registry.py`. Directory scanning,
reflection-based discovery, and legacy `NODE_CLASS_MAPPINGS` registration are not
used.

## Development

The repository does not install ComfyUI itself. The ComfyUI checkout supplies
the V3 API and runtime packages such as PyTorch. Run quality checks with the
same Python environment that runs ComfyUI, and point both `COMFYUI_PATH` and
`PYTHONPATH` at that checkout:

```bash
export COMFYUI_PATH=/path/to/ComfyUI
export PYTHONPATH="$COMFYUI_PATH:$PWD"

"$PYTHON" -m pip install -r requirements-dev.txt -r requirements.txt
"$PYTHON" -m ruff check .
"$PYTHON" -m ruff format --check .
"$PYTHON" -m pyright
"$PYTHON" -m pytest
```

If ComfyUI is installed into another environment, replace the `PYTHON` value
with that environment's interpreter. Do not commit a machine-specific
ComfyUI path; `COMFYUI_PATH` is intentionally supplied by each checkout or CI
job.

```bash
COMFYUI_PATH=/path/to/ComfyUI \
PYTHONPATH=/path/to/ComfyUI:$PWD \
/path/to/ComfyUI/.venv/bin/python -m pytest
```

Published node IDs are workflow compatibility contracts and must remain stable.
Changes to node schemas, inputs, outputs, or behavior should be documented in
the repository's canonical documentation when that durable information is
needed.
