# Python Standards

## Checks

Use the ComfyUI environment; Pyright does not read the shell's `PYTHON` variable:

```bash
export COMFYUI_PATH=/Users/sinyuk/AIGC/ComfyUI
export PYTHONPATH="$COMFYUI_PATH:$PWD"
export PYTHON="$COMFYUI_PATH/.venv/bin/python"
"$PYTHON" -m ruff check .
"$PYTHON" -m ruff format --check .
"$PYTHON" -m pyright --pythonpath "$PYTHON"
"$PYTHON" -m pytest
```

Run changed-module tests first, then full checks when shared code or framework boundaries change.

## Types and validation

Every module uses `from __future__ import annotations`; public functions have explicit types and parameterized containers. Prefer dataclasses, `TypedDict`, `Literal`, `Enum`, and `Protocol` for known shapes. Keep untyped JSON or SDK responses as `object` at the boundary, validate them with a `TypeGuard`, and convert them to concrete values immediately. Do not use `Any`, `cast()`, or type-ignore comments to silence errors. Validate external input at runtime and preserve exception chains when translating errors.

## ComfyUI V3 and ownership

Use only `io.ComfyNode`, `io.Schema`, `io.NodeOutput`, and `ComfyExtension`. Published `node_id` values are stable. Nodes adapt schemas and inputs, call feature code, and return `NodeOutput`; business logic belongs in `features/`. Shared code belongs in `common/` only after multiple independent features need the same implementation. Dependencies flow `nodes → features → common`, while ComfyUI-specific imports stay in `compat/`, `nodes/`, `extension.py`, or `registry.py`.

ComfyUI's current V3 stubs describe `execute` as synchronous `**kwargs` and type heterogeneous input lists too narrowly. Runtime V3 accepts the typed async node signatures used here. Keep any Pyright diagnostic suppression for that mismatch limited to node adapter files; never suppress diagnostics in feature or common code.

## Quality and scope

Use `pathlib.Path`, Ruff import order, focused meaningful tests, and mocks for network calls. Make the smallest coherent change, preserve existing contracts, and avoid speculative nodes, utilities, abstractions, or documentation.
