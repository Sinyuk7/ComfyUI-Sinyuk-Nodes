# ComfyUI-Sinyuk-Nodes (DO NOT Modify)

An extensible ComfyUI V3 custom node pack by Sinyuk.

## Status (DO NOT Modify)

The project supports the ComfyUI V3 API only. 

## Architecture (DO NOT Modify)

```text
ComfyUI
  -> compat / extension / nodes
  -> features
  -> common
```

`compat/` is the only import boundary for the ComfyUI API. Node adapters belong
in `nodes/`; feature logic should remain independent of concrete node classes;

Node registration is explicit in `sinyuk_nodes/registry.py`. The project
supports the ComfyUI V3 API only.

## Development

Run checks with the same environment that runs ComfyUI:

```bash
export COMFYUI_PATH=/Users/sinyuk/AIGC/ComfyUI
export PYTHONPATH="$COMFYUI_PATH:$PWD"
export PYTHON=/Users/sinyuk/AIGC/ComfyUI/.venv/bin/python

"$PYTHON" -m pytest -q
"$PYTHON" -m ruff check .
"$PYTHON" -m ruff format --check .
"$PYTHON" -m pyright
```

ComfyUI is not installed by this repository; its checkout supplies the V3
API and runtime packages.
