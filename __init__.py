"""ComfyUI entry point for Sinyuk Nodes."""

from __future__ import annotations

import sys
from pathlib import Path

# ComfyUI loads custom-node ``__init__.py`` by file path and does not add the
# loaded directory to ``sys.path``. Make the repository package importable in
# that loader while keeping normal installed imports unchanged.
_PACKAGE_ROOT = str(Path(__file__).resolve().parent)
if _PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, _PACKAGE_ROOT)

# ruff: noqa: E402 - the path must be adjusted before importing the package.
from sinyuk_nodes.extension import comfy_entrypoint

__version__ = "0.1.0"
WEB_DIRECTORY = "./sinyuk_nodes/features/image_api/web"

__all__ = ["__version__", "WEB_DIRECTORY", "comfy_entrypoint"]
